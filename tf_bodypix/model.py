import logging
import re
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

try:
    import tfjs_graph_converter
except ImportError:
    tfjs_graph_converter = None


from tf_bodypix.bodypix_js_utils.decode_part_map import (
    to_mask_tensor
)

from tf_bodypix.bodypix_js_utils.part_channels import PART_CHANNELS

from tf_bodypix.bodypix_js_utils.output_rendering_util import (
    RAINBOW_PART_COLORS
)

from tf_bodypix.bodypix_js_utils.util import (
    get_bodypix_input_resolution_height_and_width,
    pad_and_resize_to,
    scale_and_crop_to_input_tensor_shape,
    scaleAndFlipPoses,
    Padding
)

from tf_bodypix.bodypix_js_utils.types import Pose

from tf_bodypix.bodypix_js_utils.multi_person.decode_multiple_poses import (
    decodeMultiplePoses
)


LOGGER = logging.getLogger(__name__)


PART_CHANNEL_INDEX_BY_NAME = {
    name: index
    for index, name in enumerate(PART_CHANNELS)
}


DEFAULT_RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR


ImageSize = namedtuple('ImageSize', ('height', 'width'))


T_Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]


class ModelArchitectureNames:
    MOBILENET_V1 = 'mobilenet_v1'
    RESNET_50 = 'resnet50'


VALID_MODEL_ARCHITECTURE_NAMES = {
    ModelArchitectureNames.MOBILENET_V1,
    ModelArchitectureNames.RESNET_50
}


# see https://github.com/tensorflow/tfjs-models/blob/body-pix-v2.0.4/body-pix/src/resnet.ts
IMAGE_NET_MEAN = [-123.15, -115.90, -103.06]


class DictPredictWrapper:
    def __init__(
        self,
        wrapped: Callable[[np.ndarray], Union[dict, list]],
        output_names: List[str]
    ):
        self.wrapped = wrapped
        self.output_names = output_names

    def __call__(self, *args, **kwargs):
        result = self.wrapped(*args, **kwargs)
        if isinstance(result, list):
            return dict(zip(self.output_names, result))
        return result


class BodyPixArchitecture(ABC):
    def __init__(self, architecture_name: str):
        self.architecture_name = architecture_name

    @abstractmethod
    def __call__(self, image: np.ndarray) -> dict:
        pass


class MobileNetBodyPixPredictWrapper(BodyPixArchitecture):
    def __init__(self, predict_fn: Callable[[np.ndarray], dict]):
        super().__init__(ModelArchitectureNames.MOBILENET_V1)
        self.predict_fn = predict_fn

    def __call__(self, image: np.ndarray) -> dict:
        if len(image.shape) == 3:
            image = image[tf.newaxis, ...]
        return self.predict_fn(
            tf.keras.applications.mobilenet.preprocess_input(image)
        )


class ResNet50BodyPixPredictWrapper(BodyPixArchitecture):
    def __init__(self, predict_fn: Callable[[np.ndarray], dict]):
        super().__init__(ModelArchitectureNames.RESNET_50)
        self.predict_fn = predict_fn

    def __call__(self, image: np.ndarray) -> dict:
        image = np.add(image, np.array(IMAGE_NET_MEAN))
        # Note: tf.keras.applications.resnet50.preprocess_input is rotating the image as well?
        if len(image.shape) == 3:
            image = image[tf.newaxis, ...]
        image = tf.cast(image, tf.float32)
        LOGGER.debug('image.shape: %s (%s)', image.shape, image.dtype)
        predictions = self.predict_fn(
            tf.constant(image)
        )
        return predictions


def get_colored_part_mask_for_segmentation(
    part_segmentation: np.ndarray,
    part_colors: Optional[List[T_Color]] = None,
    default_color: Optional[T_Color] = None
):
    _part_colors = (
        part_colors if part_colors is not None
        else RAINBOW_PART_COLORS
    )
    part_colors_array = np.asarray(_part_colors)
    if default_color is None:
        default_color = (0, 0, 0)
    # np.take will take the last value if the index is -1
    part_colors_with_default_array = np.append(
        part_colors_array,
        np.asarray([default_color]),
        axis=-2
    )
    LOGGER.debug('part_colors_with_default_array.shape: %s', part_colors_with_default_array.shape)
    LOGGER.debug('part_segmentation.shape: %s', part_segmentation.shape)
    part_segmentation_colored = np.take(
        part_colors_with_default_array,
        part_segmentation,
        axis=-2
    )
    LOGGER.debug('part_segmentation_colored.shape: %s', part_segmentation_colored.shape)
    return part_segmentation_colored


def is_all_part_names(part_names: Optional[List[str]]) -> bool:
    if not part_names:
        return True
    part_names_set = set(part_names)
    if len(part_names_set) == len(PART_CHANNELS):
        return True
    return False


def get_filtered_part_segmentation(
    part_segmentation: np.ndarray,
    part_names: List[str] = None
):
    if is_all_part_names(part_names):
        return part_segmentation
    assert part_names
    part_names_set = set(part_names)
    part_filter_mask = np.asarray([
        (
            part_index
            if part_name in part_names_set
            else -1
        )
        for part_index, part_name in enumerate(PART_CHANNELS)
    ])
    LOGGER.debug('part_filter_mask: %s', part_filter_mask)
    return part_filter_mask[part_segmentation]


class BodyPixResultWrapper:
    def __init__(
            self,
            segments_logits: np.ndarray,
            part_heatmap_logits: np.ndarray,
            heatmap_logits: Optional[np.ndarray],
            short_offsets: Optional[np.ndarray],
            long_offsets: Optional[np.ndarray],
            part_offsets: Optional[np.ndarray],
            displacement_fwd: Optional[np.ndarray],
            displacement_bwd: Optional[np.ndarray],
            output_stride: int,
            original_size: ImageSize,
            model_input_size: ImageSize,
            padding: Padding):
        self.segments_logits = segments_logits
        self.part_heatmap_logits = part_heatmap_logits
        self.heatmap_logits = heatmap_logits
        self.short_offsets = short_offsets
        self.long_offsets = long_offsets
        self.part_offsets = part_offsets
        self.displacement_fwd = displacement_fwd
        self.displacement_bwd = displacement_bwd
        self.output_stride = output_stride
        self.original_size = original_size
        self.model_input_size = model_input_size
        self.padding = padding

    def _get_scaled_scores(
        self,
        logits: np.ndarray,
        resize_method: str = DEFAULT_RESIZE_METHOD
    ) -> np.ndarray:
        return scale_and_crop_to_input_tensor_shape(
            logits,
            self.original_size.height,
            self.original_size.width,
            self.model_input_size.height,
            self.model_input_size.width,
            padding=self.padding,
            apply_sigmoid_activation=True,
            resize_method=resize_method
        )

    def get_scaled_segment_scores(self, **kwargs) -> np.ndarray:
        return self._get_scaled_scores(self.segments_logits, **kwargs)

    def get_scaled_part_heatmap_scores(self, **kwargs) -> np.ndarray:
        return self._get_scaled_scores(self.part_heatmap_logits, **kwargs)

    def get_scaled_part_segmentation(
        self,
        mask: np.ndarray = None,
        part_names: List[str] = None,
        outside_mask_value: int = -1,
        resize_method: str = DEFAULT_RESIZE_METHOD
    ) -> np.ndarray:
        scaled_part_heatmap_argmax = np.argmax(
            self.get_scaled_part_heatmap_scores(resize_method=resize_method),
            -1
        )
        LOGGER.debug('scaled_part_heatmap_argmax.shape: %s', scaled_part_heatmap_argmax.shape)
        if part_names:
            scaled_part_heatmap_argmax = get_filtered_part_segmentation(
                scaled_part_heatmap_argmax,
                part_names
            )
            LOGGER.debug(
                'scaled_part_heatmap_argmax.shape (filtered): %s',
                scaled_part_heatmap_argmax.shape
            )
        if mask is not None:
            LOGGER.debug('mask.shape: %s', mask.shape)
            return np.where(
                np.squeeze(mask, axis=-1),
                scaled_part_heatmap_argmax,
                np.asarray([outside_mask_value])
            )
        return scaled_part_heatmap_argmax

    def get_mask(
        self,
        threshold: float,
        resize_method: str = DEFAULT_RESIZE_METHOD,
        **kwargs
    ) -> np.ndarray:
        return to_mask_tensor(
            self.get_scaled_segment_scores(resize_method=resize_method),
            threshold,
            **kwargs
        )

    def get_part_mask(
        self,
        mask: np.ndarray,
        part_names: List[str] = None,
        resize_method: str = DEFAULT_RESIZE_METHOD
    ) -> np.ndarray:
        if is_all_part_names(part_names):
            return mask
        part_segmentation = self.get_scaled_part_segmentation(
            mask, part_names=part_names, resize_method=resize_method
        )
        part_mask = np.where(
            np.expand_dims(part_segmentation, -1) >= 0,
            mask,
            0
        )
        LOGGER.debug('part_mask.shape: %s', part_mask.shape)
        return part_mask

    def get_colored_part_mask(
        self,
        mask: np.ndarray,
        part_colors: List[T_Color] = None,
        part_names: List[str] = None,
        resize_method: str = DEFAULT_RESIZE_METHOD
    ) -> np.ndarray:
        part_segmentation = self.get_scaled_part_segmentation(
            mask, part_names=part_names, resize_method=resize_method
        )
        return get_colored_part_mask_for_segmentation(
            part_segmentation,
            part_colors=part_colors
        )

    def get_poses(self) -> List[Pose]:
        assert self.heatmap_logits is not None
        assert self.short_offsets is not None
        assert self.displacement_fwd is not None
        assert self.displacement_bwd is not None
        poses = decodeMultiplePoses(
            scoresBuffer=np.asarray(self.heatmap_logits[0]),
            offsetsBuffer=np.asarray(self.short_offsets[0]),
            displacementsFwdBuffer=np.asarray(self.displacement_fwd[0]),
            displacementsBwdBuffer=np.asarray(self.displacement_bwd[0]),
            outputStride=self.output_stride,
            maxPoseDetections=2
        )
        scaled_poses = scaleAndFlipPoses(
            poses,
            height=self.original_size.height,
            width=self.original_size.width,
            inputResolutionHeight=self.model_input_size.height,
            inputResolutionWidth=self.model_input_size.width,
            padding=self.padding,
            flipHorizontal=False
        )
        return scaled_poses


class BodyPixModelWrapper:
    def __init__(
            self,
            predict_fn: Callable[[np.ndarray], Dict[str, Any]],
            output_stride: int,
            internal_resolution: float = 0.5):
        self.predict_fn = predict_fn
        self.internal_resolution = internal_resolution
        self.output_stride = output_stride

    def get_bodypix_input_size(self, original_size: ImageSize) -> ImageSize:
        return ImageSize(
            *get_bodypix_input_resolution_height_and_width(
                self.internal_resolution, self.output_stride,
                original_size.height, original_size.width
            )
        )

    def get_padded_and_resized(
        self, image: np.ndarray, model_input_size: ImageSize
    ) -> Tuple[np.ndarray, Padding]:
        LOGGER.debug(
            'pad_and_resize_to: image.shape=%s, model_input_size=%s',
            image.shape, model_input_size
        )
        return pad_and_resize_to(
            image,
            model_input_size.height,
            model_input_size.width
        )

    def find_optional_tensor_in_map(
        self,
        tensor_map: Dict[str, np.ndarray],
        name: str
    ) -> Optional[np.ndarray]:
        if name in tensor_map:
            return tensor_map[name]
        for key, value in tensor_map.items():
            if name in key:
                return value
        return None

    def find_required_tensor_in_map(
        self,
        tensor_map: Dict[str, np.ndarray],
        name: str
    ) -> np.ndarray:
        value = self.find_optional_tensor_in_map(tensor_map, name)
        if value is not None:
            return value
        raise ValueError('tensor with name %r not found in %s' % (
            name, tensor_map.keys()
        ))

    def predict_single(self, image: np.ndarray) -> BodyPixResultWrapper:
        original_size = ImageSize(*image.shape[:2])
        model_input_size = self.get_bodypix_input_size(original_size)
        model_input_image, padding = self.get_padded_and_resized(image, model_input_size)

        tensor_map = self.predict_fn(model_input_image)

        LOGGER.debug('tensor_map type: %s', type(tensor_map))
        LOGGER.debug('tensor_map keys: %s', tensor_map.keys())
        LOGGER.debug('tensor_map shapes: %s', {
            k: t.shape
            for k, t in tensor_map.items()
        })

        return BodyPixResultWrapper(
            segments_logits=self.find_required_tensor_in_map(
                tensor_map, 'float_segments'
            ),
            part_heatmap_logits=self.find_required_tensor_in_map(
                tensor_map, 'float_part_heatmaps'
            ),
            heatmap_logits=self.find_required_tensor_in_map(
                tensor_map, 'float_heatmaps'
            ),
            short_offsets=self.find_required_tensor_in_map(
                tensor_map, 'float_short_offsets'
            ),
            long_offsets=self.find_required_tensor_in_map(
                tensor_map, 'float_long_offsets'
            ),
            part_offsets=self.find_required_tensor_in_map(
                tensor_map, 'float_part_offsets'
            ),
            displacement_fwd=self.find_required_tensor_in_map(
                tensor_map, 'displacement_fwd'
            ),
            displacement_bwd=self.find_required_tensor_in_map(
                tensor_map, 'displacement_bwd'
            ),
            original_size=original_size,
            model_input_size=model_input_size,
            output_stride=self.output_stride,
            padding=padding
        )


def get_structured_output_names(structured_outputs: List[tf.Tensor]) -> List[str]:
    return [
        tensor.name.replace(':0', '')
        for tensor in structured_outputs
    ]


def to_number_of_dimensions(data: np.ndarray, dimension_count: int) -> np.ndarray:
    while len(data.shape) > dimension_count:
        data = data[0]
    while len(data.shape) < dimension_count:
        data = np.expand_dims(data, axis=0)
    return data


def load_tflite_model(model_path: str):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    LOGGER.debug('input_details: %s', input_details)
    input_names = [item['name'] for item in input_details]
    LOGGER.debug('input_names: %s', input_names)
    input_details_map = dict(zip(input_names, input_details))

    output_details = interpreter.get_output_details()
    LOGGER.debug('output_details: %s', output_details)
    output_names = [item['name'] for item in output_details]
    LOGGER.debug('output_names: %s', output_names)

    try:
        image_input = input_details_map['image']
    except KeyError:
        assert len(input_details_map) == 1
        image_input = list(input_details_map.values())[0]
    input_shape = image_input['shape']
    LOGGER.debug('input_shape: %s', input_shape)

    def predict(image_data: np.ndarray):
        nonlocal input_shape
        image_data = to_number_of_dimensions(image_data, len(input_shape))
        LOGGER.debug('tflite predict, image_data.shape=%s (%s)', image_data.shape, image_data.dtype)
        height, width, *_ = image_data.shape
        if tuple(image_data.shape) != tuple(input_shape):
            LOGGER.info('resizing input tensor: %s -> %s', tuple(input_shape), image_data.shape)
            interpreter.resize_tensor_input(image_input['index'], list(image_data.shape))
            interpreter.allocate_tensors()
            input_shape = image_data.shape
        interpreter.set_tensor(image_input['index'], image_data)
        if 'image_size' in input_details_map:
            interpreter.set_tensor(
                input_details_map['image_size']['index'],
                np.array([height, width], dtype=np.float)
            )

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        return {
            item['name']: interpreter.get_tensor(item['index'])
            for item in output_details
        }
    return predict


def load_using_saved_model_and_get_predict_function(model_path):
    loaded = tf.saved_model.load(model_path)
    LOGGER.debug('loaded: %s', loaded)
    LOGGER.debug('signature keys: %s', list(loaded.signatures.keys()))
    infer = loaded.signatures["serving_default"]
    LOGGER.info('structured_outputs: %s', infer.structured_outputs)
    return infer


def load_using_tfjs_graph_converter_and_get_predict_function(
    model_path: str
) -> Callable[[np.ndarray], dict]:
    if tfjs_graph_converter is None:
        raise ImportError('tfjs_graph_converter required')
    graph = tfjs_graph_converter.api.load_graph_model(model_path)
    tf_fn = tfjs_graph_converter.api.graph_to_function_v2(graph)
    return DictPredictWrapper(
        tf_fn,
        get_structured_output_names(tf_fn.structured_outputs)
    )


def load_model_and_get_predict_function(
    model_path: str
) -> Callable[[np.ndarray], dict]:
    if model_path.endswith('.tflite'):
        return load_tflite_model(model_path)
    try:
        return load_using_saved_model_and_get_predict_function(model_path)
    except OSError:
        return load_using_tfjs_graph_converter_and_get_predict_function(model_path)


def get_output_stride_from_model_path(model_path: str) -> int:
    match = re.search(r'stride(\d+)|_(\d+)_quant', model_path)
    if not match:
        raise ValueError('cannot extract output stride from model path: %r' % model_path)
    return int(match.group(1) or match.group(2))


def get_architecture_from_model_path(model_path: str) -> str:
    model_path_lower = model_path.lower()
    if 'mobilenet' in model_path_lower:
        return ModelArchitectureNames.MOBILENET_V1
    if 'resnet' in model_path_lower:
        return ModelArchitectureNames.RESNET_50
    raise ValueError('cannot extract model architecture from model path: %r' % model_path)


def load_model(
    model_path: str,
    output_stride: int = None,
    architecture_name: str = None,
    **kwargs
):
    if not output_stride:
        output_stride = get_output_stride_from_model_path(model_path)
    if not architecture_name:
        architecture_name = get_architecture_from_model_path(model_path)
    predict_fn = load_model_and_get_predict_function(model_path)
    architecture_wrapper: BodyPixArchitecture
    if architecture_name == ModelArchitectureNames.MOBILENET_V1:
        architecture_wrapper = MobileNetBodyPixPredictWrapper(predict_fn)
    elif architecture_name == ModelArchitectureNames.RESNET_50:
        architecture_wrapper = ResNet50BodyPixPredictWrapper(predict_fn)
    else:
        ValueError('unsupported architecture: %s' % architecture_name)
    return BodyPixModelWrapper(
        architecture_wrapper,
        output_stride=output_stride,
        **kwargs
    )

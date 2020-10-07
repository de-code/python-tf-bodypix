import logging
import re
from collections import namedtuple
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf

try:
    import tfjs_graph_converter
except ImportError:
    tfjs_graph_converter = None


from tf_bodypix.bodypix_js_utils.decode_part_map import (
    to_mask_tensor
)

from tf_bodypix.bodypix_js_utils.util import (
    get_bodypix_input_resolution_height_and_width,
    pad_and_resize_to,
    scale_and_crop_to_input_tensor_shape,
    Padding
)


LOGGER = logging.getLogger(__name__)


ImageSize = namedtuple('ImageSize', ('height', 'width'))


class ModelArchitectureNames:
    MOBILENET_V1 = 'mobilenet_v1'
    RESNET_50 = 'resnet50'


VALID_MODEL_ARCHITECTURE_NAMES = {
    ModelArchitectureNames.MOBILENET_V1,
    ModelArchitectureNames.RESNET_50
}


# see https://github.com/tensorflow/tfjs-models/blob/body-pix-v2.0.4/body-pix/src/resnet.ts
IMAGE_NET_MEAN = [-123.15, -115.90, -103.06]


class DictPredictWrapper(Callable[[np.ndarray], dict]):
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


class BodyPixArchitecture(Callable[[np.ndarray], dict]):
    def __init__(self, architecture_name: str):
        self.architecture_name = architecture_name


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


class BodyPixResultWrapper:
    def __init__(
            self,
            segments_logits: np.ndarray,
            original_size: ImageSize,
            model_input_size: ImageSize,
            padding: Padding):
        self.segments_logits = segments_logits
        self.original_size = original_size
        self.model_input_size = model_input_size
        self.padding = padding

    def get_scaled_segment_scores(self) -> np.ndarray:
        return scale_and_crop_to_input_tensor_shape(
            self.segments_logits,
            self.original_size.height,
            self.original_size.width,
            self.model_input_size.height,
            self.model_input_size.width,
            padding=self.padding,
            apply_sigmoid_activation=True
        )

    def get_mask(self, threshold: float) -> np.ndarray:
        return to_mask_tensor(
            self.get_scaled_segment_scores(),
            threshold
        )


class BodyPixModelWrapper:
    def __init__(self, predict_fn: Callable[[np.ndarray], Dict[str, Any]]):
        self.predict_fn = predict_fn
        self.internal_resolution = 0.5
        self.output_stride = 16

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
        return pad_and_resize_to(
            image,
            model_input_size.height,
            model_input_size.width
        )

    def predict_single(self, image: np.ndarray) -> BodyPixResultWrapper:
        original_size = ImageSize(*image.shape[:2])
        model_input_size = self.get_bodypix_input_size(original_size)
        model_input_image, padding = self.get_padded_and_resized(image, model_input_size)

        tensor_map = self.predict_fn(model_input_image)

        LOGGER.debug('tensor_map type: %s', type(tensor_map))
        LOGGER.debug('tensor_map keys: %s', tensor_map.keys())

        return BodyPixResultWrapper(
            segments_logits=tensor_map['float_segments'],
            original_size=original_size,
            model_input_size=model_input_size,
            padding=padding
        )


def get_structured_output_names(structured_outputs: List[tf.Tensor]) -> List[str]:
    return [
        tensor.name.replace(':0', '')
        for tensor in structured_outputs
    ]


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
    try:
        return load_using_saved_model_and_get_predict_function(model_path)
    except OSError:
        return load_using_tfjs_graph_converter_and_get_predict_function(model_path)


def get_output_stride_from_model_path(model_path: str) -> int:
    match = re.search(r'stride(\d+)', model_path)
    if not match:
        raise ValueError('cannot extract output stride from model path: %r' % model_path)
    return int(match.group(1))


def get_architecture_from_model_path(model_path: str) -> int:
    model_path_lower = model_path.lower()
    if 'mobilenet' in model_path_lower:
        return ModelArchitectureNames.MOBILENET_V1
    if 'resnet50' in model_path_lower:
        return ModelArchitectureNames.RESNET_50
    raise ValueError('cannot extract model architecture from model path: %r' % model_path)


def load_model(
    model_path: str,
    output_stride: int = None,
    architecture_name: str = None
):
    if not output_stride:
        output_stride = get_output_stride_from_model_path(model_path)
    if not architecture_name:
        architecture_name = get_architecture_from_model_path(model_path)
    predict_fn = load_model_and_get_predict_function(model_path)
    if architecture_name == ModelArchitectureNames.MOBILENET_V1:
        architecture_wrapper = MobileNetBodyPixPredictWrapper(predict_fn)
    elif architecture_name == ModelArchitectureNames.RESNET_50:
        architecture_wrapper = ResNet50BodyPixPredictWrapper(predict_fn)
    else:
        ValueError('unsupported architecture: %s' % architecture_name)
    return BodyPixModelWrapper(
        architecture_wrapper
    )

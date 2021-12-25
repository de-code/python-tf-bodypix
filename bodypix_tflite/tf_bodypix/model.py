import logging
import numpy as np
from typing import Tuple, Union
from collections import namedtuple
import tflite_runtime.interpreter as tflite

from tf_bodypix.bodypix_js_utils.part_channels import PART_CHANNELS
from tf_bodypix.bodypix_js_utils.util import scaleAndFlipPoses, Padding
from tf_bodypix.bodypix_js_utils.multi_person.decode_multiple_poses import decodeMultiplePoses


LOGGER = logging.getLogger(__name__)


PART_CHANNEL_INDEX_BY_NAME = {
    name: index
    for index, name in enumerate(PART_CHANNELS)
}

# ImageSize = namedtuple('ImageSize', ('height', 'width'))


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


def get_poses(heatmap_logits, short_offsets, displacement_fwd, displacement_bwd, image_height, image_width,
              padding, output_stride=16, model_input_height=209, model_input_width=321):
    assert heatmap_logits is not None
    assert short_offsets is not None
    assert displacement_fwd is not None
    assert displacement_bwd is not None

    poses = decodeMultiplePoses(
        scoresBuffer=np.asarray(heatmap_logits[0]),
        offsetsBuffer=np.asarray(short_offsets[0]),
        displacementsFwdBuffer=np.asarray(displacement_fwd[0]),
        displacementsBwdBuffer=np.asarray(displacement_bwd[0]),
        outputStride=output_stride,
        maxPoseDetections=2
    )
    scaled_poses = scaleAndFlipPoses(
        poses,
        height=image_height,
        width=image_width,
        inputResolutionHeight=model_input_height,
        inputResolutionWidth=model_input_width,
        padding=padding,
        flipHorizontal=False
    )

    return scaled_poses


def to_number_of_dimensions(data: np.ndarray, dimension_count: int) -> np.ndarray:
    while len(data.shape) > dimension_count:
        data = data[0]
    while len(data.shape) < dimension_count:
        data = np.expand_dims(data, axis=0)
    return data


def load_tflite_model(model_path: str):
    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    print('input_details: %s', input_details)
    input_names = [item['name'] for item in input_details]
    print('input_names: %s', input_names)
    input_details_map = dict(zip(input_names, input_details))

    output_details = interpreter.get_output_details()
    print('output_details: %s', output_details)
    output_names = [item['name'] for item in output_details]
    print('output_names: %s', output_names)

    try:
        image_input = input_details_map['image']
    except KeyError:
        assert len(input_details_map) == 1
        image_input = list(input_details_map.values())[0]
    input_shape = image_input['shape']
    print('input_shape: %s', input_shape)

    def predict(image_data: np.ndarray):
        nonlocal input_shape
        image_data = to_number_of_dimensions(image_data, len(input_shape))
        print('tflite predict, image_data.shape=%s (%s)', image_data.shape, image_data.dtype)
        height, width, *_ = image_data.shape
        if tuple(image_data.shape) != tuple(input_shape):
            print('resizing input tensor: %s -> %s', tuple(input_shape), image_data.shape)
            interpreter.resize_tensor_input(image_input['index'], list(image_data.shape))
            interpreter.allocate_tensors()
            input_shape = image_data.shape
        interpreter.set_tensor(image_input['index'], image_data)
        if 'image_size' in input_details_map:
            interpreter.set_tensor(
                input_details_map['image_size']['index'],
                np.array([height, width], dtype=np.float_)
            )

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        return {
            item['name']: interpreter.get_tensor(item['index'])
            for item in output_details
        }
    return predict

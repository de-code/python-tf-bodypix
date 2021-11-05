import logging
from collections import namedtuple

import numpy as np
import tensorflow as tf

try:
    from cv2 import cv2
except ImportError:
    cv2 = None


LOGGER = logging.getLogger(__name__)


ImageSize = namedtuple('ImageSize', ('height', 'width'))


ImageArray = np.ndarray


def require_opencv():
    if cv2 is None:
        raise ImportError('OpenCV is required')


def box_blur_image(image: np.ndarray, blur_size: int) -> np.ndarray:
    if not blur_size:
        return image
    require_opencv()
    if len(image.shape) == 4:
        image = image[0]
    result = cv2.blur(np.asarray(image), (blur_size, blur_size))
    if len(result.shape) == 2:
        result = np.expand_dims(result, axis=-1)
    result = result.astype(np.float32)
    return result


def get_image_size(image: np.ndarray):
    height, width, *_ = image.shape
    return ImageSize(height=height, width=width)


def resize_image_to(image: np.ndarray, size: ImageSize) -> np.ndarray:
    if get_image_size(image) == size:
        LOGGER.debug('image has already desired size: %s', size)
        return image

    return tf.image.resize([image], (size.height, size.width))[0]


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    # see https://www.scivision.dev/numpy-image-bgr-to-rgb/
    return image[..., ::-1]


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return bgr_to_rgb(image)

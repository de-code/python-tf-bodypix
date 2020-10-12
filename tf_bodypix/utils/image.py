import logging
from collections import namedtuple

import numpy as np
import tensorflow as tf


LOGGER = logging.getLogger(__name__)


ImageSize = namedtuple('ImageSize', ('height', 'width'))

ImageArray = np.ndarray


def get_image_size(image: ImageArray):
    height, width, *_ = image.shape
    return ImageSize(height=height, width=width)


def resize_image_to(image: ImageArray, size: ImageSize) -> ImageArray:
    if get_image_size(image) == size:
        LOGGER.debug('image has already desired size: %s', size)
        return image

    return tf.image.resize([image], (size.height, size.width))[0]

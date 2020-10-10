import os
import logging
from contextlib import contextmanager
from functools import partial
from typing import Callable

import numpy as np
import tensorflow as tf


LOGGER = logging.getLogger(__name__)


T_OutputSink = Callable[[np.ndarray], None]


def write_image_to(image_array: np.ndarray, path: str):
    LOGGER.info('writing image to: %r', path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tf.keras.preprocessing.image.save_img(path, image_array)


@contextmanager
def get_image_file_output_sink(path: str) -> T_OutputSink:
    yield partial(write_image_to, path=path)


def get_show_image_output_sink() -> T_OutputSink:
    from tf_bodypix.utils.opencv import ShowImageSink  # pylint: disable=import-outside-toplevel
    return ShowImageSink('image')

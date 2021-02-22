import os
import logging
from contextlib import contextmanager
from functools import partial
from typing import Callable, Iterator

import numpy as np
import tensorflow as tf

# pylint: disable=import-outside-toplevel


LOGGER = logging.getLogger(__name__)


T_OutputSink = Callable[[np.ndarray], None]


def write_image_to(image_array: np.ndarray, path: str):
    LOGGER.info('writing image to: %r', path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tf.keras.preprocessing.image.save_img(path, image_array)


def get_v4l2_output_sink(device_name: str) -> T_OutputSink:
    from tf_bodypix.utils.v4l2 import VideoLoopbackImageSink
    return VideoLoopbackImageSink(device_name)


@contextmanager
def get_image_file_output_sink(path: str) -> Iterator[T_OutputSink]:
    yield partial(write_image_to, path=path)


def get_image_output_sink_for_path(path: str) -> T_OutputSink:
    if path.startswith('/dev/video'):
        return get_v4l2_output_sink(path)
    return get_image_file_output_sink(path)


def get_show_image_output_sink() -> T_OutputSink:
    from tf_bodypix.utils.opencv import ShowImageSink
    return ShowImageSink(
        window_name='image',
        window_title='tf-bodypix'
    )

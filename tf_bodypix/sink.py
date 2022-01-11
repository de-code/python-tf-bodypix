import logging
from contextlib import contextmanager
from functools import partial
from typing import Callable, ContextManager, Iterator

import numpy as np

from tf_bodypix.utils.image import write_image_to

# pylint: disable=import-outside-toplevel


LOGGER = logging.getLogger(__name__)


T_OutputSink = Callable[[np.ndarray], None]


def get_v4l2_output_sink(device_name: str) -> ContextManager[T_OutputSink]:
    from tf_bodypix.utils.v4l2 import VideoLoopbackImageSink
    return VideoLoopbackImageSink(device_name)


@contextmanager
def get_image_file_output_sink(path: str) -> Iterator[T_OutputSink]:
    yield partial(write_image_to, path=path)


def get_image_output_sink_for_path(path: str) -> ContextManager[T_OutputSink]:
    if path.startswith('/dev/video'):
        return get_v4l2_output_sink(path)
    return get_image_file_output_sink(path)


def get_show_image_output_sink() -> ContextManager[T_OutputSink]:
    from tf_bodypix.utils.opencv import ShowImageSink
    return ShowImageSink(
        window_name='image',
        window_title='tf-bodypix'
    )

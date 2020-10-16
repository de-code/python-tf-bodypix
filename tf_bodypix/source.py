import logging
import os
import re
from contextlib import contextmanager
from hashlib import md5
from queue import Queue
from threading import Thread
from typing import ContextManager, Iterable

import tensorflow as tf

from tf_bodypix.utils.image import resize_image_to, ImageSize, ImageArray


# pylint: disable=import-outside-toplevel


LOGGER = logging.getLogger(__name__)


T_ImageSource = ContextManager[Iterable[ImageArray]]


def get_file(file_path: str) -> str:
    if os.path.exists(file_path):
        return file_path
    local_path = tf.keras.utils.get_file(
        md5(file_path.encode('utf-8')).hexdigest() + '-' + os.path.basename(file_path),
        file_path
    )
    return local_path


def get_webcam_number(path: str) -> int:
    match = re.match(r'(?:/dev/video|webcam:)(\d+)', path)
    if not match:
        return None
    return int(match.group(1))


def get_webcam_image_source(webcam_number: int, **kwargs) -> T_ImageSource:
    from tf_bodypix.utils.opencv import get_webcam_image_source as _get_webcam_image_source
    return _get_webcam_image_source(webcam_number, **kwargs)


@contextmanager
def get_simple_image_source(
    path: str,
    image_size: ImageSize = None,
    **_
) -> T_ImageSource:
    local_image_path = get_file(path)
    LOGGER.debug('local_image_path: %r', local_image_path)
    image = tf.keras.preprocessing.image.load_img(
        local_image_path
    )
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    if image_size is not None:
        image_array = resize_image_to(image_array, image_size)
    yield [image_array]


def get_image_source(path: str, **kwargs) -> T_ImageSource:
    webcam_number = get_webcam_number(path)
    if webcam_number is not None:
        return get_webcam_image_source(webcam_number, **kwargs)
    return get_simple_image_source(path, **kwargs)


class ThreadedImageSource:
    def __init__(self, image_source: T_ImageSource, queue_size: int = 1):
        self.image_source = image_source
        self.image_source_iterator = None
        self.queue = Queue(queue_size)
        self.thread = None
        self.stopped = False

    def __enter__(self):
        self.stopped = False
        self.thread = Thread(target=self.run)
        self.image_source_iterator = iter(self.image_source.__enter__())
        self.thread.start()
        LOGGER.info('using threaded image source')
        return self

    def __exit__(self, *args, **kwargs):
        self.stop()
        self.image_source.__exit__(*args, **kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        LOGGER.debug('reading from queue, qsize: %d', self.queue.qsize())
        return self.queue.get()

    def stop(self):
        self.stopped = True
        self.thread.join()

    def run(self):
        while not self.stopped:
            try:
                data = next(self.image_source_iterator)
            except StopIteration:
                self.stopped = True
                return
            self.queue.put(data)


def get_threaded_image_source(image_source: T_ImageSource) -> T_ImageSource:
    return ThreadedImageSource(image_source)

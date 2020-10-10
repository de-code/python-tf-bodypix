import logging
import os
import re
from hashlib import md5
from typing import Iterable

import numpy as np
import tensorflow as tf


LOGGER = logging.getLogger(__name__)


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


def get_webcam_image_source(webcam_number: int) -> Iterable[np.ndarray]:
    import cv2  # pylint: disable=import-outside-toplevel

    cam = cv2.VideoCapture(webcam_number)
    try:
        while True:
            _, image_array = cam.read()
            yield image_array
    finally:
        cam.release()


def get_simple_image_source(path: str) -> Iterable[np.ndarray]:
    local_image_path = get_file(path)
    LOGGER.debug('local_image_path: %r', local_image_path)
    image = tf.keras.preprocessing.image.load_img(
        local_image_path
    )
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    yield image_array


def get_image_source(path: str) -> Iterable[np.ndarray]:
    webcam_number = get_webcam_number(path)
    if webcam_number is not None:
        return get_webcam_image_source(webcam_number)
    return get_simple_image_source(path)

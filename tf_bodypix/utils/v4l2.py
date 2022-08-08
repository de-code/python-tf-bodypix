import logging
import os

from pyfakewebcam import FakeWebcam

import numpy as np
import cv2


# pylint: disable=protected-access


LOGGER = logging.getLogger(__name__)


def create_fakewebcam(
    device_name: str,
    preferred_width: int,
    preferred_height: int
) -> FakeWebcam:
    fakewebcam_instance = FakeWebcam(
        device_name,
        width=preferred_width,
        height=preferred_height
    )
    fakewebcam_settings = fakewebcam_instance._settings
    actual_width = fakewebcam_settings.fmt.pix.width
    actual_height = fakewebcam_settings.fmt.pix.height
    if actual_height != preferred_height or actual_width != preferred_width:
        LOGGER.warning(
            'unable to set virtual webcam resolution, using: width=%d, height=%d',
            actual_width, actual_height
        )
        fakewebcam_instance._buffer = np.zeros(
            (actual_height, 2 * actual_width),
            dtype=np.uint8
        )
        fakewebcam_instance._yuv = np.zeros(
            (actual_height, actual_width, 3),
            dtype=np.uint8
        )
        fakewebcam_instance._ones = np.ones(
            (actual_height, actual_width, 1),
            dtype=np.uint8
        )
    return fakewebcam_instance


def close_fakewebcam(fakewebcam_instance: FakeWebcam):
    os.close(fakewebcam_instance._video_device)


class VideoLoopbackImageSink:
    def __init__(self, device_name: str):
        self.device_name = device_name
        self.fakewebcam_instance = None
        self.width = None
        self.height = None

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        if self.fakewebcam_instance is not None:
            close_fakewebcam(self.fakewebcam_instance)

    def initialize_fakewebcam(self, preferred_width: int, preferred_height: int):
        fakewebcam_instance = create_fakewebcam(
            self.device_name,
            preferred_width=preferred_width,
            preferred_height=preferred_height
        )
        self.fakewebcam_instance = fakewebcam_instance
        self.width = fakewebcam_instance._settings.fmt.pix.width
        self.height = fakewebcam_instance._settings.fmt.pix.height

    def __call__(self, image_array: np.ndarray):
        image_array = np.asarray(image_array).astype(np.uint8)
        height, width, *_ = image_array.shape
        if self.fakewebcam_instance is None:
            LOGGER.info('initializing, width=%d, height=%d', width, height)
            self.initialize_fakewebcam(
                preferred_width=width,
                preferred_height=height
            )
        if height != self.height or width != self.width:
            LOGGER.info('resizing to: width=%d, height=%d', self.width, self.height)
            image_array = cv2.resize(
                image_array,
                (self.width, self.height),
                interpolation=cv2.INTER_AREA
            )
            LOGGER.info('resized image_array.shape=%s', image_array.shape)
        assert self.fakewebcam_instance is not None
        self.fakewebcam_instance.schedule_frame(image_array)

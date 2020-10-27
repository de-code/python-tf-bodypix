import logging
from contextlib import contextmanager
from typing import ContextManager, Iterable

import cv2
import numpy as np

from tf_bodypix.utils.image import ImageSize, resize_image_to, bgr_to_rgb, rgb_to_bgr


LOGGER = logging.getLogger(__name__)


def iter_read_video_images(
    video_capture: cv2.VideoCapture,
    image_size: ImageSize = None
) -> Iterable[np.ndarray]:
    is_first = True
    while True:
        _, image_array = video_capture.read()
        LOGGER.debug('cam image_array.shape: %s', image_array.shape)
        if is_first:
            LOGGER.info(
                'received webcam image shape: %s (requested: %s)',
                image_array.shape, image_size
            )
        if image_size:
            image_array = resize_image_to(image_array, image_size)
        yield bgr_to_rgb(image_array)
        is_first = False


@contextmanager
def get_webcam_image_source(
    webcam_number: int,
    image_size: ImageSize = None,
    fourcc: str = None,
    fps: int = None
) -> ContextManager[Iterable[np.ndarray]]:
    cam = cv2.VideoCapture(webcam_number)
    if fourcc:
        LOGGER.info('setting camera fourcc to %r', fourcc)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if image_size:
        LOGGER.info('attempting to set camera image size to: %s', image_size)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, image_size.width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size.height)
    if fps:
        LOGGER.info('setting camera fps to %r', fps)
        cam.set(cv2.CAP_PROP_FPS, fps)
    actual_camera_image_size = ImageSize(
        width=cam.get(cv2.CAP_PROP_FRAME_WIDTH),
        height=cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    actual_fps = cam.get(cv2.CAP_PROP_FPS)
    LOGGER.info(
        'camera reported image size: %s (%s fps)',
        actual_camera_image_size, actual_fps
    )
    try:
        yield iter_read_video_images(cam)
    finally:
        LOGGER.debug('releasing video capture: %s', webcam_number)
        cam.release()


class ShowImageSink:
    def __init__(self, window_name: str):
        self.window_name = window_name

    def __enter__(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 600, 600)
        return self

    def __exit__(self, *_, **__):
        cv2.destroyAllWindows()

    def __call__(self, image_array: np.ndarray):
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) <= 0:
            LOGGER.info('window closed')
            raise KeyboardInterrupt('window closed')
        image_array = np.asarray(image_array).astype(np.uint8)
        cv2.imshow(self.window_name, rgb_to_bgr(image_array))
        cv2.waitKey(1)

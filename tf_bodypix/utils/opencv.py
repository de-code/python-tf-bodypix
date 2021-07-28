import logging
from collections import deque
from contextlib import contextmanager
from time import monotonic, sleep
from typing import Callable, ContextManager, Deque, Iterable, Iterator, Union

import cv2
import numpy as np

from tf_bodypix.utils.io import get_file
from tf_bodypix.utils.image import (
    ImageSize, ImageArray, bgr_to_rgb, rgb_to_bgr,
    get_image_size
)


LOGGER = logging.getLogger(__name__)


DEFAULT_WEBCAM_FOURCC = 'MJPG'


def iter_read_raw_video_images(
    video_capture: cv2.VideoCapture,
    repeat: bool = False,
    is_stopped: Callable[[], bool] = None
) -> Iterable[ImageArray]:
    while is_stopped is None or not is_stopped():
        grabbed, image_array = video_capture.read()
        if not grabbed:
            LOGGER.info('video end reached')
            if not repeat:
                return
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            grabbed, image_array = video_capture.read()
            if not grabbed:
                LOGGER.info('unable to rewind video')
                return
        yield image_array


def iter_resize_video_images(
    video_images: Iterable[ImageArray],
    image_size: ImageSize = None,
    interpolation: int = cv2.INTER_LINEAR
) -> Iterable[ImageArray]:
    is_first = True
    for image_array in video_images:
        LOGGER.debug('video image_array.shape: %s', image_array.shape)
        if is_first:
            LOGGER.info(
                'received video image shape: %s (requested: %s)',
                image_array.shape, image_size
            )
            is_first = False
        if image_size and get_image_size(image_array) != image_size:
            image_array = cv2.resize(
                image_array,
                (image_size.width, image_size.height),
                interpolation=interpolation
            )
        yield image_array


def iter_convert_video_images_to_rgb(
    video_images: Iterable[ImageArray]
) -> Iterable[ImageArray]:
    return (bgr_to_rgb(image_array) for image_array in video_images)


def iter_delay_video_images_to_fps(
    video_images: Iterable[ImageArray],
    fps: float = None
) -> Iterable[np.ndarray]:
    if not fps or fps <= 0:
        LOGGER.info('no fps requested, providing images from source (without delay)')
        yield from video_images
        return
    desired_frame_time = 1 / fps
    LOGGER.info(
        'limiting frame rate to %.3f fsp (%.1f ms per frame)',
        fps, desired_frame_time * 1000
    )
    last_frame_time = None
    frame_times: Deque[float] = deque(maxlen=10)
    current_fps = 0.0
    additional_frame_adjustment = 0.0
    end_frame_time = monotonic()
    video_images_iterator = iter(video_images)
    while True:
        start_frame_time = end_frame_time
        # attempt to retrieve the next frame (that may vary in time)
        try:
            image_array = next(video_images_iterator)
        except StopIteration:
            return
        # wait time until delivery in order to achieve a similar fps
        current_time = monotonic()
        if last_frame_time:
            desired_wait_time = (
                desired_frame_time
                - (current_time - last_frame_time)
                + additional_frame_adjustment
            )
            if desired_wait_time > 0:
                LOGGER.debug(
                    'sleeping for desired fps: %s (desired_frame_time: %s, fps: %.3f)',
                    desired_wait_time, desired_frame_time, current_fps
                )
                sleep(desired_wait_time)
        last_frame_time = monotonic()
        # emit the frame (post processing may add to the overall)
        yield image_array
        end_frame_time = monotonic()
        frame_time = end_frame_time - start_frame_time
        additional_frame_adjustment = desired_frame_time - frame_time
        frame_times.append(frame_time)
        current_fps = 1 / (sum(frame_times) / len(frame_times))


def iter_read_video_images(
    video_capture: cv2.VideoCapture,
    image_size: ImageSize = None,
    interpolation: int = cv2.INTER_LINEAR,
    repeat: bool = True,
    fps: float = None
) -> Iterable[np.ndarray]:
    video_images: Iterable[np.ndarray]
    video_images = iter_read_raw_video_images(video_capture, repeat=repeat)
    video_images = iter_delay_video_images_to_fps(video_images, fps)
    video_images = iter_resize_video_images(
        video_images, image_size=image_size, interpolation=interpolation
    )
    video_images = iter_convert_video_images_to_rgb(video_images)
    return video_images


@contextmanager
def get_video_image_source(  # pylint: disable=too-many-locals
    path: Union[str, int],
    image_size: ImageSize = None,
    download: bool = True,
    fps: float = None,
    fourcc: str = None,
    buffer_size: int = None,
    **_
) -> Iterator[Iterable[ImageArray]]:
    local_path: Union[str, int]
    if isinstance(path, str):
        local_path = get_file(path, download=download)
    else:
        local_path = path
    if local_path != path:
        LOGGER.info('loading video: %r (downloaded from %r)', local_path, path)
    else:
        LOGGER.info('loading video: %r', path)
    video_capture = cv2.VideoCapture(local_path)
    if fourcc:
        LOGGER.info('setting video fourcc to %r', fourcc)
        video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if buffer_size:
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    if image_size:
        LOGGER.info('attempting to set video image size to: %s', image_size)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, image_size.width)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size.height)
    if fps:
        LOGGER.info('attempting to set video fps to %r', fps)
        video_capture.set(cv2.CAP_PROP_FPS, fps)
    actual_image_size = ImageSize(
        width=video_capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        height=video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    actual_fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    LOGGER.info(
        'video reported image size: %s (%s fps, %s frames)',
        actual_image_size, actual_fps, frame_count
    )
    try:
        yield iter_read_video_images(
            video_capture,
            image_size=image_size,
            fps=fps if fps is not None else actual_fps
        )
    finally:
        LOGGER.debug('releasing video capture: %s', path)
        video_capture.release()


def get_webcam_image_source(
    path: Union[str, int],
    fourcc: str = None,
    buffer_size: int = 1,
    **kwargs
) -> ContextManager[Iterable[ImageArray]]:
    if fourcc is None:
        fourcc = DEFAULT_WEBCAM_FOURCC
    return get_video_image_source(path, fourcc=fourcc, buffer_size=buffer_size, **kwargs)


class ShowImageSink:
    def __init__(
        self,
        window_name: str,
        window_title: str = ''
    ):
        self.window_name = window_name
        self.window_title = window_title
        self.was_opened = False

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        if self.was_opened:
            cv2.destroyWindow(self.window_name)

    @property
    def is_closed(self):
        if not self.was_opened:
            return False
        cv2.waitKey(1)
        return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) <= 0

    def create_window(self, image_size: ImageSize):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, image_size.width, image_size.height)
        if self.window_title:
            cv2.setWindowTitle(self.window_name, self.window_title)
        self.was_opened = True

    def __call__(self, image_array: np.ndarray):
        if self.is_closed:
            LOGGER.info('window closed')
            raise KeyboardInterrupt('window closed')
        image_array = np.asarray(image_array).astype(np.uint8)
        if not self.was_opened:
            self.create_window(get_image_size(image_array))
        cv2.imshow(self.window_name, rgb_to_bgr(image_array))

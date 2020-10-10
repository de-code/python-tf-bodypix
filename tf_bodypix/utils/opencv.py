import logging

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


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
        cv2.imshow(self.window_name, image_array)
        cv2.waitKey(1)

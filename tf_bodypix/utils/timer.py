import logging
from time import time


LOGGER = logging.getLogger(__name__)


class LoggingTimer:
    def __init__(self, min_interval: float = 1):
        self.min_interval = min_interval
        self.next_log_time = None
        self.frame_start_time = None
        self.frame_durations = []

    def start(self):
        current_time = time()
        self.next_log_time = current_time + self.min_interval

    def on_frame_start(self):
        self.frame_start_time = time()

    def on_frame_end(self):
        frame_end_time = time()
        self.frame_durations.append(frame_end_time - self.frame_start_time)
        self.check_log(frame_end_time)

    def check_log(self, current_time: float):
        if self.frame_durations and current_time >= self.next_log_time:
            LOGGER.info(
                '%0.3fs per frame (%d frames)',
                sum(self.frame_durations) / len(self.frame_durations),
                len(self.frame_durations)
            )
            self.frame_durations = []
            self.next_log_time = current_time + self.min_interval

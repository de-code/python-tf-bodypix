import logging
from time import time
from typing import Dict, List, Optional


LOGGER = logging.getLogger(__name__)


def _mean(a: List[float]) -> float:
    if not a:
        return 0
    return sum(a) / len(a)


class LoggingTimer:
    def __init__(self, min_interval: float = 1):
        self.min_interval = min_interval
        self.interval_start_time: Optional[float] = None
        self.frame_start_time: Optional[float] = None
        self.frame_durations: List[float] = []
        self.step_durations_map: Dict[Optional[str], List[float]] = {}
        self.current_step_name: Optional[str] = None
        self.current_step_start_time: Optional[float] = None
        self.ordered_step_names: List[Optional[str]] = []

    def start(self):
        current_time = time()
        self.interval_start_time = current_time

    def _set_current_step_name(self, step_name: str, current_time: float = None):
        if step_name == self.current_step_name:
            return
        if current_time is None:
            current_time = time()
        assert self.current_step_start_time is not None
        duration = current_time - self.current_step_start_time
        if duration > 0 or self.current_step_name:
            self.step_durations_map.setdefault(self.current_step_name, []).append(
                duration
            )
            if self.current_step_name not in self.ordered_step_names:
                self.ordered_step_names.append(self.current_step_name)
            self.current_step_name = step_name
        self.current_step_start_time = current_time

    def on_frame_start(self, initial_step_name: str = None):
        self.frame_start_time = time()
        self.current_step_name = initial_step_name
        self.current_step_start_time = self.frame_start_time

    def on_step_start(self, step_name: str):
        if step_name == self.current_step_name:
            return
        self._set_current_step_name(step_name)

    def on_step_end(self):
        self._set_current_step_name(None)

    def on_frame_end(self):
        frame_end_time = time()
        self._set_current_step_name(None, current_time=frame_end_time)
        self.frame_durations.append(frame_end_time - self.frame_start_time)
        self.check_log(frame_end_time)

    def check_log(self, current_time: float):
        assert self.interval_start_time is not None
        interval_duration = current_time - self.interval_start_time
        if self.frame_durations and interval_duration >= self.min_interval:
            step_info = ', '.join([
                '%s=%0.3f' % (step_name, _mean(
                    self.step_durations_map.get(step_name, [])
                ))
                for step_name in self.ordered_step_names
            ])
            LOGGER.info(
                '%0.3fs per frame (%0.1ffps%s)',
                _mean(self.frame_durations),
                len(self.frame_durations) / interval_duration,
                ', ' + step_info if step_info else ''
            )
            self.frame_durations.clear()
            self.step_durations_map.clear()
            self.ordered_step_names.clear()
            self.interval_start_time = current_time

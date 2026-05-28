"""util/FPSRollingAverage.py: Helper class to facilitate calculation of fps rolling averages."""

import numpy as np
import torch
import warnings
from time import perf_counter

from Logging import Logger


class FPSRollingAverage:
    """Class to calculate the rolling average of the FPS. The FPS is calculated as the inverse of the frametime."""
    def __init__(self, window_size=10):
        self._measure_cuda = True
        self._fps_window = window_size
        self._frame_times = np.full((window_size,), np.nan, dtype=np.float32)
        self._frame_times_index = 0
        self._frametime_mean_units = 'ms'
        self._frametime_std_units = 'ms'
        self._frametime_last_units = 'ms'
        self._start_time_cuda: torch.cuda.Event | None = None
        self._start_time: float | None = None

    def enable_cuda(self):
        self._measure_cuda = True

    def disable_cuda(self):
        self._measure_cuda = False

    def start_timer(self):
        """Starts the internal timer for FPS calculation."""
        if self._measure_cuda:
            self._start_time_cuda = torch.cuda.Event(enable_timing=True)
            self._start_time_cuda.record(torch.cuda.default_stream())
        else:
            self._start_time = perf_counter()

    def update(self):
        """Updates the FPS calculation with the current frametime."""
        if self._measure_cuda:
            if self._start_time_cuda is None:
                Logger.log_warning('FPSRollingAverage: start_timer() must be called before update()')
                frame_time = np.nan
            else:
                end_time = torch.cuda.Event(enable_timing=True)
                end_time.record(torch.cuda.default_stream())
                end_time.synchronize()
                frame_time = self._start_time_cuda.elapsed_time(end_time) / 1000.0  # Convert to seconds
                self._start_time_cuda = None
        else:
            if self._start_time is None:
                Logger.log_warning('FPSRollingAverage: start_timer() must be called before update()')
                frame_time = np.nan
            else:
                end_time = perf_counter()
                frame_time = end_time - self._start_time
                self._start_time = None

        # Store frame time and update index
        self._frame_times[self._frame_times_index] = frame_time
        self._frame_times_index = (self._frame_times_index + 1) % self._fps_window

    @property
    def fps_mean(self):
        """Returns the mean FPS over the last window_size frames."""
        with warnings.catch_warnings():  # If all NaN, ignore the warning raised by numpy and just return NaN
            warnings.simplefilter("ignore")
            return 1.0 / np.nanmean(self._frame_times)

    @property
    def fps_stdev(self):
        """Returns the standard deviation of the FPS over the last window_size frames."""
        with warnings.catch_warnings():  # If all NaN, ignore the warning raised by numpy and just return NaN
            warnings.simplefilter("ignore")
            return np.nanstd(1.0 / self._frame_times)

    @property
    def fps_last(self):
        """Returns the FPS of the last frame."""
        with warnings.catch_warnings():  # If all NaN, ignore the warning raised by numpy and just return NaN
            warnings.simplefilter("ignore")
            return 1.0 / self._frame_times[(self._frame_times_index - 1) % self._fps_window]

    @property
    def frametime_mean(self) -> str:
        """Returns the mean frametime over the last window_size frames."""
        with warnings.catch_warnings():  # If all NaN, ignore the warning raised by numpy and just return NaN
            warnings.simplefilter("ignore")
            frametime = np.nanmean(self._frame_times)

        if frametime is np.nan:
            return 'N/A'

        # Change units, but keep a buffer zone to avoid changing back and forth when at the threshold
        if frametime > 1.5:
            self._frametime_mean_units = 's'
        elif 0.002 < frametime < 0.5:
            self._frametime_mean_units = 'ms'
        elif frametime < 0.0005:
            self._frametime_mean_units = 'µs'

        if self._frametime_mean_units == 's':
            return f'{frametime:.2f} s'
        if self._frametime_mean_units == 'ms':
            return f'{frametime * 1e3:.1f} ms'
        return f'{frametime * 1e6:.0f} µs'

    @property
    def frametime_stdev(self) -> str:
        """Returns the standard deviation of the frametime over the last window_size frames."""
        with warnings.catch_warnings():  # If all NaN, ignore the warning raised by numpy and just return NaN
            warnings.simplefilter("ignore")
            frametime = np.nanstd(self._frame_times)

        if frametime is np.nan:
            return 'N/A'

        # Change units, but keep a buffer zone to avoid changing back and forth when at the threshold
        if frametime > 5.0:
            self._frametime_std_units = 's'
        elif 0.002 < frametime < 0.2:
            self._frametime_std_units = 'ms'
        elif frametime < 0.0001:
            self._frametime_std_units = 'µs'

        if self._frametime_std_units == 's':
            return f'{frametime:.2f} s'
        if self._frametime_std_units == 'ms':
            return f'{frametime * 1e3:.1f} ms'
        return f'{frametime * 1e6:.0f} µs'

    @property
    def frametime_last(self) -> str:
        """Returns the frametime of the last frame."""
        with warnings.catch_warnings():  # If all NaN, ignore the warning raised by numpy and just return NaN
            warnings.simplefilter("ignore")
            frametime = self._frame_times[(self._frame_times_index - 1) % self._fps_window]

        if frametime is np.nan:
            return 'N/A'

        # Change units, but keep a buffer zone to avoid changing back and forth when at the threshold
        if frametime > 5.0:
            self._frametime_last_units = 's'
        elif 0.002 < frametime < 0.2:
            self._frametime_last_units = 'ms'
        elif frametime < 0.0001:
            self._frametime_last_units = 'µs'

        if self._frametime_last_units == 's':
            return f'{frametime:.2f} s'
        if self._frametime_last_units == 'ms':
            return f'{frametime * 1e3:.1f} ms'
        return f'{frametime * 1e6:.0f} µs'

    @property
    def stats(self):
        """Returns the mean and standard deviation of the FPS over the last window_size frames."""
        return {
            'fps': self.fps_mean,
            'fps_std': self.fps_stdev,
            'fps_last': self.fps_last,
            'frametime_mean': self.frametime_mean,
            'frametime_std': self.frametime_stdev,
            # 'frametime_last': self.frametime_last,
        }

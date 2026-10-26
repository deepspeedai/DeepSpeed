# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import importlib.util
import time
from numpy import mean
from deepspeed.utils.logging import print_dist
from deepspeed.accelerator import get_accelerator

FORWARD_MICRO_TIMER = 'fwd_microstep'
FORWARD_GLOBAL_TIMER = 'fwd'
BACKWARD_MICRO_TIMER = 'bwd_microstep'
BACKWARD_GLOBAL_TIMER = 'bwd'
BACKWARD_INNER_MICRO_TIMER = 'bwd_inner_microstep'
BACKWARD_INNER_GLOBAL_TIMER = 'bwd_inner'
BACKWARD_REDUCE_MICRO_TIMER = 'bwd_allreduce_microstep'
BACKWARD_REDUCE_GLOBAL_TIMER = 'bwd_allreduce'
STEP_MICRO_TIMER = 'step_microstep'
STEP_GLOBAL_TIMER = 'step'
TIME_EPSILON = 1e-6


class SynchronizedWallClockTimer:
    """Group of timers. Borrowed from Nvidia Megatron code"""

    class Timer:
        """Timer."""

        def __init__(self, name):
            self.name_ = name
            self.started_ = False
            self.use_host_timer = get_accelerator().use_host_timers()
            self.start_event = None
            self.stop_event = None
            self.start_time = 0.0
            self.end_time = 0.0
            # Reading device events waits for the device, so the latest interval is
            # measured when it is first needed instead of in stop().
            self.read_pending = False
            self.latest_elapsed_msec = 0.0
            self.total_elapsed_msec = 0.0
            self.num_intervals = 0

        def start(self):
            """Start the timer."""
            assert not self.started_, f"{self.name_} timer has already been started"
            # Measure the previous interval before its events are recorded again.
            self._read_latest()
            if self.use_host_timer:
                self.start_time = time.time()
            else:
                if self.start_event is None:
                    event_class = get_accelerator().Event
                    self.start_event = event_class(enable_timing=True)
                self.start_event.record()
            self.started_ = True

        def stop(self, reset=False, record=False):
            """Stop the timer."""
            assert self.started_, "timer is not started"
            if self.use_host_timer:
                self.end_time = time.time()
            else:
                if self.stop_event is None:
                    event_class = get_accelerator().Event
                    self.stop_event = event_class(enable_timing=True)
                self.stop_event.record()
            self.read_pending = True
            self.started_ = False

        def _read_latest(self):
            if not self.read_pending:
                return
            if self.use_host_timer:
                self.latest_elapsed_msec = (self.end_time - self.start_time) * 1000.0
            else:
                get_accelerator().current_stream().wait_event(self.stop_event)
                self.stop_event.synchronize()
                self.latest_elapsed_msec = self.start_event.elapsed_time(self.stop_event)
            self.total_elapsed_msec += self.latest_elapsed_msec
            self.num_intervals += 1
            self.read_pending = False

        def reset(self):
            """Reset timer."""
            self.started_ = False
            self.read_pending = False
            self.latest_elapsed_msec = 0.0
            self.total_elapsed_msec = 0.0
            self.num_intervals = 0

        def _elapsed(self, reset, total):
            # A timer still running here was not stopped by its caller, so end its interval.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            self._read_latest()
            if total:
                elapsed_ = self.total_elapsed_msec
            else:
                elapsed_ = self.latest_elapsed_msec
            # Reset the elapsed time
            if reset:
                self.reset()
            return elapsed_

        def elapsed(self, reset=True):
            """Elapsed time in ms of the latest start/stop interval."""
            return self._elapsed(reset, total=False)

        def elapsed_total(self, reset=True):
            """Total elapsed time in ms of all intervals since the last reset."""
            return self._elapsed(reset, total=True)

        def mean(self):
            """Mean elapsed time in ms of all intervals since the last reset."""
            self.elapsed(reset=False)
            if self.num_intervals == 0:
                return 0.0
            return self.total_elapsed_msec / self.num_intervals

    def __init__(self):
        self.timers = {}

    def get_timers(self):
        return self.timers

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    @staticmethod
    def memory_usage():
        alloc = "mem_allocated: {:.4f} GB".format(get_accelerator().memory_allocated() / (1024 * 1024 * 1024))
        max_alloc = "max_mem_allocated: {:.4f} GB".format(get_accelerator().max_memory_allocated() /
                                                          (1024 * 1024 * 1024))
        cache = "cache_allocated: {:.4f} GB".format(get_accelerator().memory_cached() / (1024 * 1024 * 1024))
        max_cache = "max_cache_allocated: {:.4f} GB".format(get_accelerator().max_memory_cached() /
                                                            (1024 * 1024 * 1024))
        return " | {} | {} | {} | {}".format(alloc, max_alloc, cache, max_cache)

    def log(self, names, normalizer=1.0, reset=True, memory_breakdown=False, ranks=None):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = "time (ms)"
        for name in names:
            if name in self.timers:
                elapsed_time = (self.timers[name].elapsed_total(reset=reset) / normalizer)
                string += " | {}: {:.2f}".format(name, elapsed_time)

        # timers logging should be independent of the global log level it's already conditional on wall_clock_breakdown being True, so using use_logger=False will always print the stats
        print_dist(string, ranks=ranks or [0])

    def get_mean(self, names, normalizer=1.0, reset=True):
        """Get the mean of a group of timers."""
        assert normalizer > 0.0
        means = {}
        for name in names:
            if name in self.timers:
                elapsed_time = (self.timers[name].mean() * 1000.0 / normalizer)
                means[name] = elapsed_time
        return means


class NoopTimer:

    class Timer:

        def start(self):
            ...

        def reset(self):
            ...

        def stop(self, **kwargs):
            ...

        def elapsed(self, **kwargs):
            return 0

        def elapsed_total(self, **kwargs):
            return 0

        def mean(self):
            return 0

    def __init__(self):
        self.timer = self.Timer()

    def __call__(self, name):
        return self.timer

    def get_timers(self):
        return {}

    def log(self, names, normalizer=1.0, reset=True, memory_breakdown=False, ranks=None):
        ...

    def get_mean(self, names, normalizer=1.0, reset=True):
        ...


class ThroughputTimer:

    def __init__(self, config, batch_size, start_step=2, steps_per_output=None, monitor_memory=False, logging_fn=None):
        from deepspeed.utils import logger
        self.config = config
        self.start_time = 0
        self.end_time = 0
        self.started = False
        self.batch_size = 1 if batch_size is None else batch_size
        self.start_step = start_step
        self.epoch_count = 0
        self.micro_step_count = 0
        self.global_step_count = 0
        self.total_elapsed_time = 0
        self.step_elapsed_time = 0
        self.steps_per_output = steps_per_output
        self.monitor_memory = monitor_memory
        self.logging = logging_fn
        if self.logging is None:
            self.logging = logger.info
        self.initialized = False

        if self.monitor_memory and importlib.util.find_spec("psutil") is None:
            raise ImportError("Unable to import 'psutils', please install package")

    def update_epoch_count(self):
        self.epoch_count += 1
        self.micro_step_count = 0

    def _init_timer(self):
        self.initialized = True

    def start(self):
        if not self.config.enabled:
            return
        self._init_timer()
        self.started = True
        if self.global_step_count >= self.start_step:
            if self.config.synchronized:
                get_accelerator().synchronize()
            self.start_time = time.time()

    def _is_report_boundary(self):
        if self.steps_per_output is None:
            return False
        return self.global_step_count % self.steps_per_output == 0

    def stop(self, global_step=False, report_speed=True):
        if not self.config.enabled or not self.started:
            return
        self.started = False
        self.micro_step_count += 1
        if global_step:
            self.global_step_count += 1

        if self.start_time > 0:
            if self.config.synchronized:
                get_accelerator().synchronize()
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.total_elapsed_time += duration
            self.step_elapsed_time += duration

            if global_step:
                if report_speed and self._is_report_boundary():
                    self.logging(
                        "epoch={}/micro_step={}/global_step={}, RunningAvgSamplesPerSec={}, CurrSamplesPerSec={}, "
                        "MemAllocated={}GB, MaxMemAllocated={}GB".format(
                            self.epoch_count,
                            self.micro_step_count,
                            self.global_step_count,
                            self.avg_samples_per_sec(),
                            self.batch_size / (self.step_elapsed_time + TIME_EPSILON),
                            round(get_accelerator().memory_allocated() / 1024**3, 2),
                            round(get_accelerator().max_memory_allocated() / 1024**3, 2),
                        ))
                    if self.monitor_memory:
                        import psutil
                        virt_mem = psutil.virtual_memory()
                        swap = psutil.swap_memory()
                        self.logging("epoch={}/micro_step={}/global_step={}, vm %: {}, swap %: {}".format(
                            self.epoch_count,
                            self.micro_step_count,
                            self.global_step_count,
                            virt_mem.percent,
                            swap.percent,
                        ))
                self.step_elapsed_time = 0

    def avg_samples_per_sec(self):
        if self.global_step_count > 0:
            total_step_offset = self.global_step_count - self.start_step
            avg_time_per_step = self.total_elapsed_time / total_step_offset
            # training samples per second
            return self.batch_size / avg_time_per_step
        return float("-inf")


def trim_mean(data, trim_percent):
    """Compute the trimmed mean of a list of numbers.

    Args:
        data (list): List of numbers.
        trim_percent (float): Percentage of data to trim.

    Returns:
        float: Trimmed mean.
    """
    assert 0.0 <= trim_percent <= 1.0
    n = len(data)
    # Account for edge case of empty list
    if len(data) == 0:
        return 0
    # sorted(), not data.sort(): CommsLogger passes its stored latency/algbw/busbw
    # lists straight in, and sorting them in place reorders each independently and
    # destroys the index correspondence between them.
    data = sorted(data)
    k = int(round(n * (trim_percent)))
    return mean(data[k:n - k])

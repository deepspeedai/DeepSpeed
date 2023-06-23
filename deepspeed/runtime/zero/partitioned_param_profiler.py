# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass
from deepspeed.utils import log_dist


class PartitionedParameterProfiler(object):

    @dataclass
    class EventCounter:
        name: str
        count: int
        num_elem: int

        def reset(self):
            self.count = 0
            self.num_elem = 0

        def increment(self, numel):
            self.count += 1
            self.num_elem += numel

    def __init__(self, timers):
        self.timers = timers
        self.event_counters = {}

    def reset_events(self):
        for event_ctr in self.event_counters.values():
            event_ctr.reset()

    def start_event(self, name):
        if self.timers is None:
            return

        if name not in self.event_counters:
            self.event_counters[name] = __class__.EventCounter(name=name, count=0, num_elem=0)
        self.timers(name).start()

    def stop_event(self, name, num_elem):
        if self.timers is None:
            return
        assert name in self.event_counters, f'unknown event {name}'
        self.event_counters[name].increment(num_elem)
        self.timers(name).stop()

    def _log_timers(self):
        if self.timers is None:
            return
        self.timers.log(names=list(self.event_counters.keys()))

    def _log_event_counters(self):
        for event_ctr in self.event_counters.values():
            log_dist(
                f'{event_ctr.name}: count = {event_ctr.count}, numel = {event_ctr.num_elem}',
                #f'{event_ctr.name}: time = {self._log_timers()},count = {event_ctr.count}, numel = {event_ctr.num_elem}',
                ranks=[0])

    def log_events(self):
        self._log_event_counters()
        self._log_timers()

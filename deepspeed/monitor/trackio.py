# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .utils import check_trackio_availability
from .monitor import Monitor
import deepspeed.comm as dist


class TrackioMonitor(Monitor):

    def __init__(self, trackio_config):
        super().__init__(trackio_config)
        check_trackio_availability()
        import trackio
        self.enabled = trackio_config.enabled
        self.project = trackio_config.project
        if self.enabled and dist.get_rank() == 0:
            trackio.init(project=self.project)

    def log(self, data, step=None):
        if self.enabled and dist.get_rank() == 0:
            import trackio
            return trackio.log(data, step=step)

    def write_events(self, event_list):
        if self.enabled and dist.get_rank() == 0:
            for event in event_list:
                label = event[0]
                value = event[1]
                step = event[2]
                self.log({label: value}, step=step)
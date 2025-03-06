# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import TYPE_CHECKING, Any, Tuple, List, Dict, Optional

from .utils import check_comet_availability
from .monitor import Monitor

import deepspeed.comm as dist

if TYPE_CHECKING:
    import comet_ml
    from .config import CometConfig

Name = str
Value = Any
GlobalSamples = int
Event = Tuple[Name, Value, GlobalSamples]


class CometMonitor(Monitor):

    def __init__(self, comet_config: "CometConfig"):
        super().__init__(comet_config)
        check_comet_availability()
        import comet_ml

        self.enabled = comet_config.enabled
        self._samples_log_interval = comet_config.samples_log_interval
        self._experiment: Optional["comet_ml.ExperimentBase"] = None

        if self.enabled and dist.get_rank() == 0:
            self._experiment = comet_ml.start(
                api_key=comet_config.api_key,
                project=comet_config.project,
                workspace=comet_config.workspace,
                experiment_key=comet_config.experiment_key,
                mode=comet_config.mode,
                online=comet_config.online,
            )

            if comet_config.experiment_name is not None:
                self._experiment.set_name(comet_config.experiment_name)

        self._events_log_scheduler = EventsLogScheduler(comet_config.samples_log_interval)

    @property
    def experiment(self) -> Optional["comet_ml.ExperimentBase"]:
        return self._experiment

    @property
    def samples_log_interval(self) -> int:
        return self._samples_log_interval

    def write_events(self, event_list: List[Event]) -> None:
        if not self.enabled or dist.get_rank() != 0:
            return None

        for event in event_list:
            name = event[0]
            value = event[1]
            engine_global_samples = event[2]

            if self._events_log_scheduler.needs_logging(name, engine_global_samples):
                self._experiment.__internal_api__log_metric__(
                    name=name,
                    value=value,
                    step=engine_global_samples,
                )


class EventsLogScheduler:

    def __init__(self, samples_log_interval: int):
        self._samples_log_interval = samples_log_interval
        self._last_logged_events_samples: Dict[str, int] = {}

    def needs_logging(self, name: str, current_sample: int) -> bool:
        if name not in self._last_logged_events_samples:
            self._last_logged_events_samples[name] = current_sample
            return True

        last_logged_sample = self._last_logged_events_samples[name]
        samples_delta = current_sample - last_logged_sample

        if samples_delta >= self._samples_log_interval:
            self._last_logged_events_samples[name] = current_sample
            return True

        return False

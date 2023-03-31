# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Support different forms of monitoring such as wandb and tensorboard
"""

from abc import ABC, abstractmethod
import deepspeed.comm as dist


class Monitor(ABC):

    @abstractmethod
    def __init__(self, monitor_config):
        self.monitor_config = monitor_config

    @abstractmethod
    def write_events(self, event_list):
        pass


from .wandb import WandbMonitor
from .tensorboard import TensorBoardMonitor
from .csv_monitor import csvMonitor


class MonitorMaster(Monitor):

    def __init__(self, monitor_config):
        super().__init__(monitor_config)
        self.tb_monitor = None
        self.wandb_monitor = None
        self.csv_monitor = None
        self.enabled = monitor_config.enabled

        if dist.get_rank() == 0:
            if monitor_config.tensorboard.enabled:
                self.tb_monitor = TensorBoardMonitor(monitor_config.tensorboard)
            if monitor_config.wandb.enabled:
                self.wandb_monitor = WandbMonitor(monitor_config.wandb)
            if monitor_config.csv_monitor.enabled:
                self.csv_monitor = csvMonitor(monitor_config.csv_monitor)

    def write_events(self, event_list):
        if dist.get_rank() == 0:
            if self.tb_monitor is not None:
                self.tb_monitor.write_events(event_list)
            if self.wandb_monitor is not None:
                self.wandb_monitor.write_events(event_list)
            if self.csv_monitor is not None:
                self.csv_monitor.write_events(event_list)

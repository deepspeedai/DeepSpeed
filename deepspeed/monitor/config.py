"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from pydantic import BaseModel
from .constants import *


class MonitorConfig(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        use_enum_values = True
        extra = 'forbid'


class TensorBoardConfig(MonitorConfig):
    enabled: bool = TENSORBOARD_ENABLED_DEFAULT
    output_path: str = TENSORBOARD_OUTPUT_PATH_DEFAULT
    job_name: str = TENSORBOARD_JOB_NAME_DEFAULT


class WandbConfig(MonitorConfig):
    enabled: bool = WANDB_ENABLED_DEFAULT
    group: str = WANDB_GROUP_NAME_DEFAULT
    team: str = WANDB_TEAM_NAME_DEFAULT
    project: str = WANDB_PROJECT_NAME_DEFAULT


class CSVConfig(MonitorConfig):
    enabled: bool = CSV_MONITOR_ENABLED_DEFAULT
    output_path: str = CSV_MONITOR_OUTPUT_PATH_DEFAULT
    job_name: str = CSV_MONITOR_JOB_NAME_DEFAULT


class DeepSpeedMonitorConfig:
    def __init__(self, ds_config):
        self.tensorboard_enabled = 'tensorboard' in ds_config
        self.wandb_enabled = 'wandb' in ds_config
        self.csv_monitor_enabled = 'csv_monitor' in ds_config

        if self.tensorboard_enabled:
            self.tensorboard_config = TensorBoardConfig(**ds_config['tensorboard'])
        if self.wandb_enabled:
            self.wandb_config = WandbConfig(**ds_config['wandb'])
        if self.csv_monitor_enabled:
            self.csv_monitor_config = CSVConfig(**ds_config['csv_monitor'])

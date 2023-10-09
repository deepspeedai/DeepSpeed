# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .constants import *
from ..pydantic_v1 import BaseModel


class CommsConfig(BaseModel):

    class Config:
        validate_all = True
        validate_assignment = True
        use_enum_values = True
        extra = 'forbid'


class CommsLoggerConfig(CommsConfig):
    enabled: bool = COMMS_LOGGER_ENABLED_DEFAULT
    prof_all: bool = COMMS_LOGGER_PROF_ALL_DEFAULT
    prof_ops: list = COMMS_LOGGER_PROF_OPS_DEFAULT
    verbose: bool = COMMS_LOGGER_VERBOSE_DEFAULT
    debug: bool = COMMS_LOGGER_DEBUG_DEFAULT


class DeepSpeedCommsConfig:

    def __init__(self, ds_config):
        self.comms_logger_enabled = 'comms_logger' in ds_config

        if self.comms_logger_enabled:
            self.comms_logger = CommsLoggerConfig(**ds_config['comms_logger'])

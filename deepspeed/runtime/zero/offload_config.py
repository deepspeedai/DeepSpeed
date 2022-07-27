"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from pydantic import Field, validator
from enum import Enum
from pathlib import Path
from deepspeed.runtime.config_utils import DeepSpeedConfigModel


class OffloadDeviceEnum(str, Enum):
    none = "none"
    cpu = "cpu"
    nvme = "nvme"


class DeepSpeedZeroOffloadParamConfig(DeepSpeedConfigModel):
    device: OffloadDeviceEnum = OffloadDeviceEnum.none
    nvme_path: Path = None
    buffer_count: int = Field(5, ge=0)
    buffer_size: int = Field(1e8, ge=0)
    max_in_cpu: int = Field(1e9, ge=0)
    pin_memory: bool = False


class DeepSpeedZeroOffloadOptimizerConfig(DeepSpeedConfigModel):
    device: OffloadDeviceEnum = OffloadDeviceEnum.none
    nvme_path: Path = None
    buffer_count: int = Field(4, ge=0)
    pin_memory: bool = False
    pipeline_read: bool = False
    pipeline_write: bool = False
    fast_init: bool = False

    @validator("pipeline_read", "pipeline_write", always=True)
    def set_pipeline(cls, field_value, values):
        values["pipeline"] = field_value or values.get("pipeline", False)
        return field_value

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from enum import Enum
from pathlib import Path
from deepspeed.pydantic_v1 import Field, validator
from deepspeed.runtime.config_utils import DeepSpeedConfigModel, pp_int


class OffloadDeviceEnum(str, Enum):
    """ Enum for valid offload devices """
    none = "none"
    cpu = "cpu"
    nvme = "nvme"


class DeepSpeedZeroOffloadParamConfig(DeepSpeedConfigModel):
    """ Set options for parameter offload. Valid only with stage 3. """

    device: OffloadDeviceEnum = "none"
    """
    Device memory to offload model parameters. Supported options are `cpu` and
    `nvme`.
    """

    nvme_path: Path = None
    """ Filesystem path for NVMe device for parameter offloading. """

    buffer_count: int = Field(5, ge=0)
    """ Number of buffers in buffer pool for parameter offloading to NVMe. """

    buffer_size: int = Field(pp_int(1e8), ge=0)
    """ Size of buffers in buffer pool for parameter offloading to NVMe. """

    max_in_cpu: int = Field(pp_int(1e9), ge=0)
    """
    Number of parameter elements to maintain in CPU memory when offloading to
    NVMe is enabled.
    """

    pin_memory: bool = False
    """
    Offload to page-locked CPU memory. This could boost throughput at the cost
    of extra memory overhead.
    """


class DeepSpeedZeroOffloadOptimizerConfig(DeepSpeedConfigModel):
    """ Set options for optimizer offload. Valid with stage 1, 2, and 3. """

    device: OffloadDeviceEnum = "none"
    """
    Device memory to offload optimizer state. Supported options are `cpu` and
    `nvme`. Optimizer computation is offload to CPU regardless of device option.
    """

    nvme_path: Path = None
    """ Filesystem path for NVMe device for optimizer state offloading. """

    buffer_count: int = Field(4, ge=0)
    """
    Number of buffers in buffer pool for optimizer state offloading to NVMe.
    This should be at least the number of states maintained per parameter by
    the optimizer. For example, Adam optimizer has 4 states (parameter,
    gradient, momentum, and variance).
    """

    pin_memory: bool = False
    """
    Offload to page-locked CPU memory. This could boost throughput at the cost
    of extra memory overhead.
    """

    pipeline_read: bool = False
    """
    For tile-based optimizer step processing, overlap read of next tile with
    computation of current tile. Used in ZeRO-Infinity.
    """

    pipeline_write: bool = False
    """
    For tile-based optimizer step processing, overlap write of previous tile
    with computation of current tile.
    """

    fast_init: bool = False
    """ Enable fast optimizer initialization when offloading to NVMe. """

    @validator("pipeline_read", "pipeline_write", always=True)
    def set_pipeline(cls, field_value, values):
        values["pipeline"] = field_value or values.get("pipeline", False)
        return field_value

    ratio: float = Field(1.0, ge=0.0, le=1.0)
    """ Percentage of offloaded optimizer states to CPU Adam. Only valid with ZeRO Stage 3."""

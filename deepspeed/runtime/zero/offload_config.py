# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from enum import Enum
from pathlib import Path
from pydantic import Field, model_validator
from typing import Optional, Union

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

    nvme_path: Optional[Path] = None
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

    nvme_path: Optional[Path] = None
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

    ratio: float = Field(1.0, ge=0.0, le=1.0)
    """ Percentage of offloaded optimizer states to CPU Adam. Only valid with ZeRO Stage 3."""

    @model_validator(mode="after")
    def set_pipeline(self):
        pipeline = self.pipeline_read or self.pipeline_write
        self.__dict__["pipeline"] = pipeline
        return self


class ZenFlowConfig(DeepSpeedConfigModel):
    """Configuration options for ZenFlow optimization module."""

    topk_ratio: float = Field(0.1, ge=0.0, le=1.0)
    """Ratio of top-k important gradient columns to retain (range: 0.0 to 1.0)."""

    select_strategy: str = "auto"
    """Strategy for selecting important gradient indices.
    Options: "auto", "step", or "epoch"."""

    select_interval: Union[str, int] = "auto"
    """Interval at which to reselect important gradient indices.
    Can be "auto" or a fixed integer step/epoch interval."""

    update_interval: Union[str, int] = "auto"
    """Interval for applying accumulated unimportant gradients to model parameters.
    Can be "auto" or a fixed integer step interval."""

    overlap_step: bool = False
    """Whether to overlap CPU-side optimizer steps with forward/backward computation."""

    offload: bool = False
    """Whether to offload selective optimizer states to CPU to save memory."""

    auto_ratio: float = Field(0.99, ge=0.0, le=1.0)
    """Threshold used in the "auto" strategy to determine update_interval."""

    full_warm_up_rounds: int = 0
    """Number of initial rounds during which all gradients are fully updated (no selection)."""

    steps_per_epoch: Optional[int] = Field(
        default=None,
        description=
        "Number of steps per epoch. This field is initialized during execution and should not be set by users.",
        exclude=True)

    @model_validator(mode="after")
    def validate_fields(self):
        if self.select_strategy not in ["auto", "step", "epoch"]:
            raise ValueError('select_strategy must be one of "auto", "step", or "epoch"')

        if isinstance(self.select_interval, str) and self.select_interval != "auto":
            raise ValueError('If select_interval is a string, it must be "auto"')

        if isinstance(self.update_interval, str) and self.update_interval != "auto":
            raise ValueError('If update_interval is a string, it must be "auto"')

        if not isinstance(self.full_warm_up_rounds, int):
            raise ValueError('full_warm_up_rounds must be an integer')

        return self


class OffloadStateTypeEnum(str, Enum):
    """ Enum for internal buffer types """
    optim_states = "optim_states"
    hp_params = "hp_params"
    lp_params = "lp_params"
    lp_grads = "lp_grads"
    contiguous_grad_buffer = "contiguous_grad_buffer"

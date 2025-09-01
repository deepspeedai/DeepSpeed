# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from pydantic import Field, model_validator
from typing import Optional, Union

from deepspeed.runtime.config_utils import DeepSpeedConfigModel


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

    pt_reserved_cores_perc: float = Field(0.5, ge=0.0, le=1.0)
    """Number of cores reserved for pytorch threads,
       the remaining cores will be used by zenflow optimizer workers"""

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

        if not isinstance(self.pt_reserved_cores_perc, float):
            raise ValueError('pt_reserved_cores_perc must be a float')

        return self

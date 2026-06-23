# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
from pydantic import field_validator, model_validator
from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from .fp16.loss_scaler import (
    INITIAL_LOSS_SCALE,
    SCALE_WINDOW,
    DELAYED_SHIFT,
    CONSECUTIVE_HYSTERESIS,
    MIN_LOSS_SCALE,
)

#########################################
# BFLOAT16 support
#########################################
# BFLOAT16 feature. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
BFLOAT16_FORMAT = '''
BFLOAT16 parameters should be of the format:
"bf16": {
  "enabled": true,
  "immediate_grad_update": false,
  "check_grad_overflow": false
}
'''
BFLOAT16 = "bf16"
BFLOAT16_OLD = "bfloat16"  # keeping for backwards compatibility


def get_bfloat16_config(param_dict):
    bf16_config_dict = param_dict.get(BFLOAT16, None)
    if bf16_config_dict is None:
        bf16_config_dict = param_dict.get(BFLOAT16_OLD, {})
    return DeepSpeedBF16Config(**bf16_config_dict)


class DeepSpeedBF16Config(DeepSpeedConfigModel):
    """
    For bfloat16 configuration
    """

    enabled: bool = False
    """
    Enable bfloat16 mixed-precision training/inference
    """

    immediate_grad_update: bool = False
    """
    Apply gradient updates immediately rather than delayed.
    """

    check_grad_overflow: bool = False
    """
    Check for gradient overflows and underflows
    """

    bf16_master_weights_and_grads: bool = False
    """
    Maintain master weights/gradients in bf16 precision for ZeRO optimizer.
    """

    bf16_optimizer_states: bool = False
    """
    Keep optimizer states in bf16 (only valid when bf16_master_weights_and_grads is enabled).
    """


#########################################
# FP16 support
#########################################
# FP16 feature. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
FP16_FORMAT = '''
FP16 parameters should be of the format:
"fp16": {
  "enabled": true,
  "auto_cast": false,
  "loss_scale": 0,
  "initial_scale_power": 16,
  "loss_scale_window": 1000,
  "hysteresis": 2,
  "consecutive_hysteresis": false,
  "min_loss_scale": 1
}
'''
FP16 = "fp16"


def get_float16_config(param_dict):
    fp16_config_dict = param_dict.get(FP16, {})
    return DeepSpeedFP16Config(**fp16_config_dict)


class DeepSpeedFP16Config(DeepSpeedConfigModel):
    """
    For float16 configuration
    """

    enabled: bool = False
    """
    Enable fp16 mixed-precision training/inference
    """

    auto_cast: bool = False
    """
    Automatically cast inputs to fp16
    """

    loss_scale: float = 0
    """
    Loss scaling value. Default value of 0 means dynamic loss scaling instead of static loss scale.
    """

    @field_validator("loss_scale", mode="before")
    @classmethod
    def _validate_loss_scale(cls, v):
        if isinstance(v, bool):
            raise ValueError("fp16.loss_scale must be a number, not bool")
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError("fp16.loss_scale must be a number")
        if not math.isfinite(v):
            raise ValueError("fp16.loss_scale must be a finite number (not inf/-inf/nan)")
        if v < 0:
            raise ValueError("fp16.loss_scale must be >= 0 (0 enables dynamic loss scaling)")
        return v

    initial_scale_power: int = 16
    """
    For dynamic loss scaling, set initial loss scale to 2^{initial_scale_power}.
    """

    loss_scale_window: int = 1000
    """
    Iteration intervals for raising/lowering dynamic loss scale value.
    """

    hysteresis: int = 2
    """
    Delay shift in dynamic loss scaling.
    """

    consecutive_hysteresis: bool = False
    """
    Refill hysteresis if iteration does not overflow/underflow.
    """

    min_loss_scale: int = 1
    """
    Minimum dynamic loss scale value.
    """

    fp16_master_weights_and_grads: bool = False
    """
    Maintain master weights in optimizer state as fp16 instead of fp32 (valid with DeepSpeedCPUAdam only).
    """

    @field_validator("loss_scale_window", "min_loss_scale", mode="before")
    @classmethod
    def _reject_non_integer_scale_params(cls, v, info):
        # Pydantic coerces bool to int (True -> 1, False -> 0) and floats to int,
        # so a bool or non-finite value would silently pass the positivity check
        # in _validate_dynamic_loss_scale_params. Reject those here before coercion.
        field = f"fp16.{info.field_name}"
        if isinstance(v, bool):
            raise ValueError(f"{field} must be an integer, not bool")
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"{field} must be a finite number (not inf/-inf/nan)")
        try:
            int(v)
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be an integer")
        return v

    @model_validator(mode="after")
    def _validate_dynamic_loss_scale_params(self):
        # loss_scale_window and min_loss_scale only take effect when dynamic loss
        # scaling is active, i.e. fp16 is enabled and loss_scale == 0 (see
        # DeepSpeedEngine.dynamic_loss_scale). Validating them otherwise would
        # reject valid static-loss-scale configs that carry unused values.
        if self.enabled and self.loss_scale == 0:
            # loss_scale_window is used as `stable_interval % scale_window` in
            # DynamicLossScaler.update_scale, so 0 raises ZeroDivisionError.
            if self.loss_scale_window <= 0:
                raise ValueError(
                    "fp16.loss_scale_window must be > 0 when dynamic loss scaling is enabled (loss_scale=0)")
            # min_loss_scale is the loss-scale floor, which collapses if <= 0.
            if self.min_loss_scale <= 0:
                raise ValueError("fp16.min_loss_scale must be > 0 when dynamic loss scaling is enabled (loss_scale=0)")
        return self

    def initial_dynamic_scale(self):
        return 2**self.initial_scale_power

    def dynamic_loss_scale_args(self):
        return {
            INITIAL_LOSS_SCALE: 2**self.initial_scale_power,
            SCALE_WINDOW: self.loss_scale_window,
            DELAYED_SHIFT: self.hysteresis,
            CONSECUTIVE_HYSTERESIS: self.consecutive_hysteresis,
            MIN_LOSS_SCALE: self.min_loss_scale,
        }

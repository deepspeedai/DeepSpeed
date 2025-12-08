# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

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

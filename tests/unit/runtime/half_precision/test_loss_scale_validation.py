# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace
import pytest
import torch

from deepspeed.runtime.fp16.loss_scaler import (
    CONSECUTIVE_HYSTERESIS,
    DELAYED_SHIFT,
    INITIAL_LOSS_SCALE,
    MIN_LOSS_SCALE,
    SCALE_WINDOW,
    CreateLossScaler,
    LossScaleConfig,
)
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3


def test_loss_scale_config_rejects_non_finite_static_loss_scale():
    with pytest.raises(ValueError, match="fp16.loss_scale must be finite"):
        LossScaleConfig(low_precision_dtype=torch.float16,
                        dynamic_loss_scale=False,
                        static_loss_scale=float("inf"),
                        dynamic_loss_args=None)


def test_create_loss_scaler_rejects_non_finite_dynamic_init_scale():
    dynamic_loss_args = {
        INITIAL_LOSS_SCALE: float("inf"),
        SCALE_WINDOW: 1000,
        DELAYED_SHIFT: 2,
        CONSECUTIVE_HYSTERESIS: False,
        MIN_LOSS_SCALE: 1.0,
    }
    with pytest.raises(ValueError, match="dynamic_loss_args\\['init_scale'\\] must be finite"):
        CreateLossScaler(torch.float16, static_loss_scale=0, dynamic_scaling=True, dynamic_loss_args=dynamic_loss_args)


def test_stage1_override_loss_scale_validates_values():
    optimizer = SimpleNamespace(external_loss_scale=None, custom_loss_scaler=False)
    with pytest.raises(ValueError, match="loss_scale must be finite"):
        DeepSpeedZeroOptimizer.override_loss_scale(optimizer, float("inf"))

    DeepSpeedZeroOptimizer.override_loss_scale(optimizer, 256.0)
    assert optimizer.custom_loss_scaler is True
    assert optimizer.external_loss_scale == 256.0


def test_stage3_set_loss_scale_validates_values():
    optimizer = SimpleNamespace(loss_scaler=SimpleNamespace(cur_scale=1.0))
    with pytest.raises(ValueError, match="loss_scale must be greater than 0"):
        DeepSpeedZeroOptimizer_Stage3._set_loss_scale(optimizer, 0)

    DeepSpeedZeroOptimizer_Stage3._set_loss_scale(optimizer, 128.0)
    assert optimizer.loss_scaler.cur_scale == 128.0

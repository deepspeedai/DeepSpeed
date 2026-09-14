# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import logging

import pytest
import torch

from deepspeed.runtime.fp16.loss_scaler import CreateLossScaler, DynamicLossScaler, LossScaler


@pytest.fixture
def warnings(monkeypatch):
    recorded = []
    monkeypatch.setattr("deepspeed.utils.logging.logger.log", lambda level, msg, *a, **kw: recorded.append(
        (level, msg)))
    return recorded


def test_static_scaler_reports_the_skipped_step(warnings):
    LossScaler(scale=1.0).update_scale(overflow=True)

    assert len(warnings) == 1
    level, msg = warnings[0]
    assert level == logging.WARNING
    assert "OVERFLOW" in msg and "weights were not updated" in msg


def test_static_scaler_is_quiet_without_overflow(warnings):
    LossScaler(scale=1.0).update_scale(overflow=False)

    assert warnings == []


def test_dynamic_scaler_is_not_double_reported(warnings, monkeypatch):
    """DynamicLossScaler overrides update_scale and logs the skip itself."""
    monkeypatch.setattr("deepspeed.comm.get_rank", lambda: 0)
    scaler = DynamicLossScaler(init_scale=2**16,
                               scale_window=1000,
                               min_scale=1,
                               delayed_shift=1,
                               consecutive_hysteresis=False)
    scaler.update_scale(overflow=True)

    assert not [msg for _, msg in warnings if "weights were not updated" in msg]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.half])
def test_non_dynamic_configs_get_the_reporting_scaler(dtype):
    """bf16 and fp32 never get a DynamicLossScaler, so the static path is the only one."""
    scaler = CreateLossScaler(dtype=dtype, static_loss_scale=1.0, dynamic_scaling=False, dynamic_loss_args=None)

    assert not isinstance(scaler, DynamicLossScaler)

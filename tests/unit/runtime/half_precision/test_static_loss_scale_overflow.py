# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import logging
from unittest import mock

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import FusedAdamBuilder
from deepspeed.runtime.fp16.loss_scaler import CreateLossScaler, DynamicLossScaler, LossScaler
from deepspeed.utils.logging import logger
from unit.common import DistributedTest


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


@pytest.mark.parametrize("precision,zero_stage,optimizer_type,grad_accum_dtype,wrapper", [
    ("bf16", 0, "SGD", None, "FP16_UnfusedOptimizer"),
    ("bf16", 0, "Adam", None, "FP16_Optimizer"),
    ("bf16", 1, "SGD", "fp32", "BF16_Optimizer"),
    ("fp16", 0, "SGD", None, "FP16_UnfusedOptimizer"),
    ("fp16", 0, "Adam", None, "FP16_Optimizer"),
])
class TestFixedScaleSkippedStepIsReported(DistributedTest):
    """Optimizer wrappers outside the ZeRO optimizers skip non-finite steps without a LossScaler to report them."""
    world_size = 1

    def test_skipped_step_warns(self, precision, zero_stage, optimizer_type, grad_accum_dtype, wrapper):
        if precision == "fp16" and not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported on this accelerator")
        if precision == "bf16" and not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported on this accelerator")
        if optimizer_type == "Adam":
            # FP16_Optimizer is only selected for FusedAdam, and __compatible_ops__ is true on hosts
            # where the op still fails to build, so gate on the build itself.
            try:
                FusedAdamBuilder().load()
            except Exception as e:
                pytest.skip(f"FusedAdam cannot be built here: {type(e).__name__}")
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": optimizer_type,
                "params": {
                    "lr": 0.1
                }
            },
            "zero_optimization": {
                "stage": zero_stage
            },
            "zero_allow_untested_optimizer": True,
        }
        if precision == "bf16":
            config["bf16"] = {"enabled": True}
            dtype = torch.bfloat16
        else:
            config["fp16"] = {"enabled": True, "loss_scale": 1.0}
            dtype = torch.float16
        if grad_accum_dtype is not None:
            config["data_types"] = {"grad_accum_dtype": grad_accum_dtype}
        model = torch.nn.Linear(4, 4)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        try:
            assert type(engine.optimizer).__name__ == wrapper
            for scale in (1.0, float("inf")):
                recorded = []
                before = [parameter.detach().clone() for parameter in engine.parameters()]
                previous_skips = engine.skipped_steps
                inputs = torch.ones(1, 4, device=engine.device, dtype=dtype)
                with mock.patch.object(logger, "log", lambda level, msg, *a, **kw: recorded.append((level, msg))):
                    engine.backward(engine(inputs).float().sum() * scale)
                    engine.step()
                skipped = engine.skipped_steps - previous_skips
                unchanged = all(torch.equal(b, a) for b, a in zip(before, engine.parameters()))
                # Pins #8415: the skip must be visible at the default WARNING level, not only counted.
                reported = [
                    msg for level, msg in recorded if level == logging.WARNING and "weights were not updated" in msg
                ]
                assert skipped == int(scale == float("inf"))
                assert unchanged == bool(skipped)
                assert len(reported) == skipped
        finally:
            engine.destroy()

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.utils import safe_set_full_grad
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader


class _NoTruthTest(torch.Tensor):
    """A flag that fails if anything tests its truth value, which would synchronize the host."""

    def __bool__(self):
        raise AssertionError("the offload overflow flag was truth-tested")


def _offload_engine(zero_stage, check_grad_overflow):
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4,
                "torch_adam": True
            }
        },
        "bf16": {
            "enabled": True,
            "check_grad_overflow": check_grad_overflow
        },
        "zero_optimization": {
            "stage": zero_stage,
            "offload_optimizer": {
                "device": "cpu"
            }
        },
    }
    model = SimpleModel(10)
    engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
    return engine


def _step_with_grad(engine, batch, grad_value):
    engine.backward(engine(batch[0], batch[1]))
    for p in engine.parameters():
        safe_set_full_grad(p, torch.full_like(p, grad_value))
    engine.step()


@pytest.mark.parametrize("zero_stage", [1, 2])
class TestOffloadOverflowFlag(DistributedTest):
    world_size = 2

    def test_flag_accumulates_on_device(self, zero_stage):
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported")
        engine = _offload_engine(zero_stage, check_grad_overflow=True)
        optimizer = engine.optimizer
        original = DeepSpeedZeroOptimizer._has_inf_or_nan
        DeepSpeedZeroOptimizer._has_inf_or_nan = staticmethod(lambda x: original(x).as_subclass(_NoTruthTest))
        try:
            optimizer.reset_cpu_buffers()
            device = get_accelerator().current_device_name()
            for value in (0.5, -2.0):
                optimizer.update_offload_overflow_tracker(torch.full((4, ), value, device=device))
            assert torch.is_tensor(optimizer.local_overflow)
            assert not optimizer.has_overflow()

            optimizer.update_offload_overflow_tracker(torch.tensor([1.0, float("inf")], device=device))
            optimizer.update_offload_overflow_tracker(torch.full((4, ), 1.0, device=device))
            assert optimizer.has_overflow()

            optimizer.reset_cpu_buffers()
            assert optimizer.local_overflow is False
            assert not optimizer.has_overflow()
        finally:
            DeepSpeedZeroOptimizer._has_inf_or_nan = original

    def test_overflow_still_skips_step(self, zero_stage):
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported")
        engine = _offload_engine(zero_stage, check_grad_overflow=True)
        loader = random_dataloader(model=engine,
                                   total_samples=2,
                                   hidden_dim=10,
                                   device=engine.device,
                                   dtype=torch.bfloat16)
        batches = list(loader)
        _step_with_grad(engine, batches[0], 0.01)
        assert engine.skipped_steps == 0
        _step_with_grad(engine, batches[1], float("nan"))
        assert engine.skipped_steps == 1

    def test_no_check_when_disabled(self, zero_stage):
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported")
        engine = _offload_engine(zero_stage, check_grad_overflow=False)
        original = DeepSpeedZeroOptimizer._has_inf_or_nan

        def unexpected(x):
            raise AssertionError("gradients were checked for overflow with check_grad_overflow=False")

        DeepSpeedZeroOptimizer._has_inf_or_nan = staticmethod(unexpected)
        try:
            loader = random_dataloader(model=engine,
                                       total_samples=1,
                                       hidden_dim=10,
                                       device=engine.device,
                                       dtype=torch.bfloat16)
            _step_with_grad(engine, next(iter(loader)), float("inf"))
            assert engine.skipped_steps == 0
            assert engine.optimizer.local_overflow is False
        finally:
            DeepSpeedZeroOptimizer._has_inf_or_nan = original

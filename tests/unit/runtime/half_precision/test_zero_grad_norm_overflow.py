# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import deepspeed

from deepspeed.runtime.zenflow.zenflow_config import ZenFlowConfig
from deepspeed.runtime.zenflow.zenflow_stage_1_and_2 import (
    ZenFlowZeroOptimizer,
    ZenFlowZeroOptimizerParallel,
)
from deepspeed.utils import safe_set_full_grad
from unit.common import DistributedTest


@pytest.mark.parametrize("zero_stage", [1, 2])
@pytest.mark.parametrize("gradient_clipping", [0.0, 1.0])
class TestZeROFP32InvalidNorm(DistributedTest):
    world_size = 2

    def test_invalid_gradients_are_rejected(self, zero_stage, gradient_clipping):
        torch.manual_seed(42)
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_clipping": gradient_clipping,
            "zero_optimization": {
                "stage": zero_stage
            },
            "zero_allow_untested_optimizer": True,
        }
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
        try:
            assert engine.optimizer.dtype == torch.float32
            for value in (2.0, float("inf"), float("nan")):
                previous_skips = engine.skipped_steps
                inputs = torch.ones(1, 4, device=engine.device)
                engine.backward(engine(inputs).sum())
                for parameter in engine.parameters():
                    safe_set_full_grad(parameter, torch.full_like(parameter, value))
                engine.step()
                assert engine.skipped_steps - previous_skips == int(not math.isfinite(value))
                assert all(bool(torch.isfinite(parameter).all()) for parameter in engine.parameters())
        finally:
            engine.destroy()


@pytest.mark.parametrize(
    "dtype,check_grad_overflow,reject_invalid",
    [(torch.float32, False, True), (torch.float16, True, True), (torch.bfloat16, False, False),
     (torch.bfloat16, True, True)],
)
@pytest.mark.parametrize("norm", [2.0, -1.0, float("inf"), float("nan")])
def test_zenflow_overlap_invalid_norm(dtype, check_grad_overflow, reject_invalid, norm):
    # Exercise the actual overlap step's admission logic without starting a worker.
    # Optimizer-process arithmetic is covered separately by integration tests.
    optimizer = SimpleNamespace(
        dtype=dtype,
        check_grad_overflow=check_grad_overflow,
        loss_scale=1.0,
        cpu_offload=True,
        micro_step=0,
        full_warm_up_rounds=1,
        check_overflow=Mock(),
        scaled_global_norm=Mock(return_value=norm),
        _update_scale=Mock(),
        zero_grad=Mock(),
        reset_cpu_buffers=Mock(),
        timers=Mock(),
        get_overlap_step_state=Mock(return_value=0),
        zenflow_cpu_optimizer_step=Mock(),
        wait_last_update_and_copy=Mock(),
    )
    optimizer_class = ZenFlowZeroOptimizer.create(ZenFlowConfig(overlap_step=True))
    assert optimizer_class is ZenFlowZeroOptimizerParallel
    optimizer_class.step(optimizer)

    overflow = reject_invalid and (norm < 0 or not math.isfinite(norm))
    assert optimizer.overflow == overflow
    optimizer._update_scale.assert_called_once_with(overflow)
    if overflow:
        optimizer.zero_grad.assert_called_once_with(set_to_none=True)
        optimizer.reset_cpu_buffers.assert_called_once()
        optimizer.zenflow_cpu_optimizer_step.assert_not_called()
        optimizer.wait_last_update_and_copy.assert_not_called()
        assert optimizer._global_grad_norm == float("inf")
    else:
        optimizer.zero_grad.assert_not_called()
        optimizer.reset_cpu_buffers.assert_not_called()
        optimizer.zenflow_cpu_optimizer_step.assert_called_once_with(0, norm)
        optimizer.wait_last_update_and_copy.assert_called_once()

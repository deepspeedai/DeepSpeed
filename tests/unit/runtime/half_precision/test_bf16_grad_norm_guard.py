# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""End-to-end ZeRO stage 1 BF16 steps for the two edge gradient norms.

A fully masked micro-batch gives all-zero gradients and a norm of exactly 0: the step must still run.
A non-finite loss gives a non-finite norm: the step must be skipped and the parameters left unchanged.

Both go through `deepspeed.initialize`, which builds `BF16_Optimizer` for ZeRO stage 1 with bf16
parameters and fp32 gradient accumulation.
"""

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from unit.common import DistributedTest
from unit.simple_model import SimpleModel


@pytest.mark.skipif(not get_accelerator().is_bf16_supported(), reason="bf16 is not supported on this accelerator")
class TestBF16GradNormGuard(DistributedTest):
    world_size = 1

    def _engine(self):
        hidden_dim = 8
        model = SimpleModel(hidden_dim)
        config = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_clipping": 1.0,
            "zero_allow_untested_optimizer": True,
            "bf16": {
                "enabled": True
            },
            "data_types": {
                "grad_accum_dtype": "fp32"
            },
            "zero_optimization": {
                "stage": 1
            },
            "optimizer": {
                "type": "SGD",
                "params": {
                    "lr": 0.1
                }
            },
        }
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
        assert isinstance(engine.optimizer, BF16_Optimizer)
        x = torch.randn(2, hidden_dim, dtype=torch.bfloat16, device=engine.device)
        y = torch.randint(hidden_dim, (2, ), device=engine.device)
        return engine, x, y

    def test_an_all_zero_gradient_step_is_not_an_error(self):
        # A fully masked micro-batch gives exactly zero gradients; the step must still run.
        engine, x, y = self._engine()
        engine.backward(engine(x, y) * 0.0)
        engine.step()
        assert float(engine.get_global_grad_norm()) == 0.0
        assert engine.was_step_applied()
        assert engine.skipped_steps == 0

    def test_a_non_finite_loss_skips_the_step_and_leaves_params_unchanged(self):
        engine, x, y = self._engine()
        before = [p.detach().clone() for p in engine.module.parameters()]
        engine.backward(engine(x, y) * float("inf"))
        engine.step()
        assert not engine.was_step_applied()
        assert engine.skipped_steps == 1
        for p, b in zip(engine.module.parameters(), before):
            assert torch.equal(p, b), "no update may be applied from a non-finite norm"

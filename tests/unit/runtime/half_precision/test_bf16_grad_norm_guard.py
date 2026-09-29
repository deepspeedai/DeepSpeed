# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""What `BF16_Optimizer.step` does with a gradient norm it cannot clip from.

`get_global_norm_of_tensors` does not raise on a non-finite norm; it writes -1, and
`get_norm_with_moe_layers` returns -1 for the same reason. Handed that, the clip coefficient
`max_norm / (global_norm + eps)` is negative, so clipping flips the sign of every gradient
rather than shrinking it. A bare `assert all_groups_norm > 0.` stood in the way, which said
nothing about the cause and also rejected a norm of exactly zero: a micro-batch whose labels
are all masked produces one, and clipping from it is a no-op, not an error.

The last two tests go through `deepspeed.initialize`, which builds `BF16_Optimizer` for ZeRO stage 1
with bf16 parameters and fp32 gradient accumulation.
"""

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.utils import clip_tensors_by_global_norm, mask_nan_or_inf_with_val_inplace
from unit.common import DistributedTest
from unit.simple_model import SimpleModel


def test_a_non_finite_norm_is_reported_as_minus_one():
    """The premise: this is the value the norm helpers hand `step`, not an exception."""
    for bad in (float("inf"), float("-inf"), float("nan")):
        norm = torch.tensor(bad)
        mask_nan_or_inf_with_val_inplace(norm, device=norm.device)
        assert norm.item() == -1.0


def test_clipping_from_minus_one_flips_every_gradient():
    """Why the guard has to exist at all, stated as the behaviour it prevents."""
    gradient = torch.full((4, ), 2.0)
    clip_tensors_by_global_norm(input_tensors=[gradient], max_norm=1.0, global_norm=-1.0)
    assert (gradient < 0).all(), "a negative clip coefficient negates the gradient"


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

    def test_a_non_finite_norm_raises_with_a_reason(self):
        engine, x, y = self._engine()
        before = [p.detach().clone() for p in engine.module.parameters()]
        engine.backward(engine(x, y) * float("inf"))
        with pytest.raises(RuntimeError, match="not finite"):
            engine.step()
        for p, b in zip(engine.module.parameters(), before):
            assert torch.equal(p, b), "no update may be applied from a non-finite norm"

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""The fp16 unfused optimizer's clip norm has to include the expert gradients.

`step` dropped the expert half of the shared/expert split and `step_fused_lamb` spent it on the
overflow check only, so the norm the clip coefficient is computed from was built from the shared
parameters alone. `FP16_Optimizer` folds it in through `get_norm_with_moe_layers`.
"""

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.utils import groups
from unit.common import DistributedTest
from unit.simple_model import SimpleMoEModel


class LegacyStepOptimizer(torch.optim.SGD):
    """SGD that also accepts the legacy fused step signature used by step_fused_lamb."""

    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None):
        return super().step(closure)


def _expert_group_name():
    name = "ep_size_1"
    if name not in groups._get_expert_parallel_group_dict():
        groups._create_expert_and_data_parallel(1)
    return name


def _optimizer(named_groups, has_moe_layers, fused_lamb_legacy):
    shared = torch.nn.Parameter(torch.zeros(4, dtype=torch.float16))
    expert = torch.nn.Parameter(torch.zeros(4, dtype=torch.float16))
    expert.allreduce = False  # what is_moe_param reads
    expert.group_name = _expert_group_name()
    if named_groups:
        # What the engine builds for its own optimizer: a split, named expert group.
        param_groups = [{'params': [shared]}, {'params': [expert], 'moe': True, 'name': expert.group_name}]
    else:
        # A client optimizer over model.parameters(): one unnamed group mixing both kinds.
        param_groups = [shared, expert]
    optimizer = FP16_UnfusedOptimizer(LegacyStepOptimizer(param_groups, lr=0.0),
                                      static_loss_scale=1.0,
                                      fused_lamb_legacy=fused_lamb_legacy,
                                      has_moe_layers=has_moe_layers,
                                      verbose=False)
    shared.grad = torch.full_like(shared, 3.0)
    expert.grad = torch.full_like(expert, 4.0)
    return optimizer


class TestUnfusedMoEClipNorm(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("fused_lamb_legacy", [False, True])
    @pytest.mark.parametrize("named_groups", [False, True])
    def test_expert_gradients_reach_the_clip_norm(self, named_groups, fused_lamb_legacy):
        # Four elements of 3.0 and four of 4.0: 6.0 shared alone, 10.0 with the experts.
        optimizer = _optimizer(named_groups, has_moe_layers=True, fused_lamb_legacy=fused_lamb_legacy)
        optimizer.step()
        assert optimizer._global_grad_norm == pytest.approx(10.0, rel=1e-6)

    @pytest.mark.parametrize("fused_lamb_legacy", [False, True])
    def test_a_model_without_experts_is_unchanged(self, fused_lamb_legacy):
        optimizer = _optimizer(named_groups=True, has_moe_layers=False, fused_lamb_legacy=fused_lamb_legacy)
        optimizer.step()
        assert optimizer._global_grad_norm == pytest.approx(6.0, rel=1e-6)

    @pytest.mark.skipif(not get_accelerator().is_fp16_supported(), reason="fp16 is not supported on this accelerator")
    def test_engine_step_with_client_optimizer(self):
        hidden_dim = 16
        model = SimpleMoEModel(hidden_dim, num_experts=1)
        config = {
            "train_micro_batch_size_per_gpu": 8,
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
                "loss_scale": 1.0
            }
        }
        engine, _, _, _ = deepspeed.initialize(config=config,
                                               model=model,
                                               optimizer=torch.optim.AdamW(model.parameters()),
                                               dist_init_required=False)
        x = torch.randn(8, 4, hidden_dim, dtype=torch.float16, device=engine.device)
        y = torch.randint(hidden_dim, (8, ), device=engine.device)
        engine.backward(engine(x, y))
        expected = torch.stack([p.grad.float().norm() for p in engine.module.parameters()
                                if p.grad is not None]).norm().item()
        engine.step()
        assert engine.get_global_grad_norm() == pytest.approx(expected, rel=1e-3)

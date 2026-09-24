# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""The fp16 unfused optimizer's clip norm has to include the expert gradients.

`FP16_UnfusedOptimizer` already splits each group into shared and expert gradients, and the split
exists so the expert half can be reduced over the expert-parallel group rather than the data-parallel
one. `step` dropped that half and `step_fused_lamb` spent it on the overflow check only, so the norm
the clip coefficient is computed from was built from the shared parameters alone. `FP16_Optimizer`
folds it in through `get_norm_with_moe_layers`.

CPU-only and single-rank: the expert-parallel group is `None` here, so `get_norm_with_moe_layers`
runs as the arithmetic it is and no collective is involved. What that arithmetic is worth across
ranks is the fused optimizer's contract, already covered; what is under test is whether the expert
gradients reach it at all.
"""

import math

import pytest
import torch

from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.utils import groups


@pytest.fixture(autouse=True)
def single_rank_expert_group(monkeypatch):
    """One rank, so the expert-parallel group is nothing to reduce over.

    `get_global_norm_of_tensors` skips the collective when the group is None, which leaves the
    plain p-norm. Patched rather than initialized because registering a real group needs a
    process group, and what is under test is which tensors reach the norm.
    """
    monkeypatch.setattr(groups, "_get_expert_parallel_group", lambda name: None)


class _NoOverflow:

    def check(self, param_groups=None):
        return False

    def check_using_norm(self, norm_group, reduce_overflow=True):
        return False


class _Optimizer:
    """Records the gradients it is stepped with, and nothing else."""

    def __init__(self, param_groups):
        self.param_groups = param_groups

    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None):
        self.scale = scale


class _Unfused:
    """The parts of FP16_UnfusedOptimizer that `step` touches, borrowing the real methods."""

    step = FP16_UnfusedOptimizer.step
    step_fused_lamb = FP16_UnfusedOptimizer.step_fused_lamb
    unscale_and_clip_grads = FP16_UnfusedOptimizer.unscale_and_clip_grads
    # Fetched rather than referenced so the test still collects against a build without it,
    # where the point is the number `step` reports, not an import error.
    _norm_with_experts = getattr(FP16_UnfusedOptimizer, "_norm_with_experts", None)

    def __init__(self, shared_grad, expert_grad, has_moe_layers, fused_lamb_legacy=False):
        shared = torch.nn.Parameter(torch.zeros(4, dtype=torch.float16))
        expert = torch.nn.Parameter(torch.zeros(4, dtype=torch.float16))
        expert.allreduce = False  # what is_moe_param reads
        shared.grad = torch.full_like(shared, shared_grad)
        expert.grad = torch.full_like(expert, expert_grad)

        self.fp16_groups = [[shared], [expert]]
        self.fp32_groups = [[p.detach().float().requires_grad_(True)] for group in self.fp16_groups for p in group]
        self.fp32_groups = [[self.fp16_groups[i][0].detach().float()] for i in range(2)]
        self.optimizer = _Optimizer([{
            'params': self.fp32_groups[0]
        }, {
            'params': self.fp32_groups[1],
            'moe': True,
            'name': 'ep_size_1'
        }])
        self.overflow_checker = _NoOverflow()
        self.fused_lamb_legacy = fused_lamb_legacy
        self.has_moe_layers = has_moe_layers
        self.norm_type = 2
        self.mpu = None
        self.clip_grad = 0.0
        self.verbose = False
        self.overflow = False
        self._global_grad_norm = 0.
        self.loss_scale_config = _LossScale()

    def _update_scale(self, overflow):
        pass


class _LossScale:
    cur_scale = 1.0
    use_grad_scaling = False


@pytest.mark.parametrize("fused_lamb_legacy", [False, True])
def test_expert_gradients_reach_the_clip_norm(fused_lamb_legacy):
    """Four elements of 3.0 and four of 4.0: 6.0 shared alone, 10.0 with the experts."""
    shared_only = math.sqrt(4 * 3.0**2)
    with_experts = math.sqrt(4 * 3.0**2 + 4 * 4.0**2)
    assert (shared_only, with_experts) == (6.0, 10.0)

    optimizer = _Unfused(3.0, 4.0, has_moe_layers=True, fused_lamb_legacy=fused_lamb_legacy)
    optimizer.step()

    assert optimizer._global_grad_norm == pytest.approx(
        with_experts,
        rel=1e-6), ("the expert gradients were left out of the norm the clip coefficient is computed from")


@pytest.mark.parametrize("fused_lamb_legacy", [False, True])
def test_a_model_without_experts_is_unchanged(fused_lamb_legacy):
    """has_moe_layers is off for every non-MoE run, and nothing about those moves."""
    optimizer = _Unfused(3.0, 4.0, has_moe_layers=False, fused_lamb_legacy=fused_lamb_legacy)
    optimizer.step()

    assert optimizer._global_grad_norm == pytest.approx(6.0, rel=1e-6)

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import pytest
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
from unit.common import DistributedTest


class LegacyStepOptimizer(torch.optim.SGD):
    """SGD that also accepts the legacy fused step signature used by step_fused_lamb."""

    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None):
        return super().step(closure)


class TestUnfusedOptimizerGradNorm(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("fused_lamb_legacy", [False, True])
    def test_reported_grad_norm_is_unscaled(self, fused_lamb_legacy):
        # Gradients reaching the optimizer are already multiplied by the loss scale, so a
        # true gradient of 0.25 under a static scale of 128 arrives as 32.0. Over the 16
        # elements below that is a scaled norm of 128.0 and a true norm of 1.0.
        loss_scale = 128.0
        true_grad_value = 0.25
        expected_norm = 1.0
        params = [torch.nn.Parameter(torch.zeros(8, dtype=torch.float16)) for _ in range(2)]
        optimizer = FP16_UnfusedOptimizer(LegacyStepOptimizer(params, lr=0.1),
                                          static_loss_scale=loss_scale,
                                          clip_grad=1.0,
                                          fused_lamb_legacy=fused_lamb_legacy,
                                          verbose=False)
        for p in params:
            p.grad = torch.full_like(p, true_grad_value * loss_scale)

        # Clipping divides by the loss scale itself, so record what it is handed.
        clipped_with = []
        unscale_and_clip_grads = optimizer.unscale_and_clip_grads

        def record(total_norm, apply_scale=True):
            clipped_with.append(total_norm)
            return unscale_and_clip_grads(total_norm, apply_scale=apply_scale)

        optimizer.unscale_and_clip_grads = record
        optimizer.step()

        # The norm is reduced in float32 (eps 1.19e-7) over 16 values, so a handful of eps
        # of relative error is expected and 1e-6 absolute on a norm of 1.0 is well clear of
        # it. Before the fix the reported norm was the scaled one, too large by 128.
        assert optimizer._global_grad_norm == pytest.approx(
            expected_norm,
            abs=1e-6), (f"reported norm {optimizer._global_grad_norm} is not the unscaled norm {expected_norm}, "
                        f"scale factor {optimizer._global_grad_norm / expected_norm}")
        assert clipped_with == [pytest.approx(expected_norm * loss_scale, abs=1e-4)
                                ], (f"clipping must still receive the scaled norm, got {clipped_with}")

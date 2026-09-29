# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import pytest
from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
from unit.common import DistributedTest


class LegacyStepSGD(torch.optim.SGD):
    """SGD that records the scale step_fused_adam hands to the legacy fused step signature."""

    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None):
        self.scale = scale


class TestFusedOptimizerExternalLossScale(DistributedTest):
    world_size = 1

    # override_loss_scale(4) means backward() multiplied the loss by 4, so a true gradient of 0.5
    # arrives as 2.0 over 16 elements: a true norm of 2.0. The config's own scale (128) must not
    # be used to unscale these gradients.
    external_scale = 4.0
    true_grad = 0.5
    true_norm = 2.0

    def _optimizer(self, clip_grad, fused_adam_legacy=False):
        params = [torch.nn.Parameter(torch.zeros(8, dtype=torch.float16)) for _ in range(2)]
        inner = LegacyStepSGD(params, lr=1.0) if fused_adam_legacy else torch.optim.SGD(params, lr=1.0)
        optimizer = FP16_Optimizer(inner,
                                   static_loss_scale=128.0,
                                   clip_grad=clip_grad,
                                   fused_adam_legacy=fused_adam_legacy,
                                   verbose=False)
        optimizer.override_loss_scale(self.external_scale)
        for p in params:
            p.grad = torch.full_like(p, self.true_grad * self.external_scale)
        return optimizer, params

    # clip_grad=0 applies no clipping; clip_grad=1 halves the true norm of 2.
    @pytest.mark.parametrize("clip_grad, expected_grad", [(0.0, 0.5), (1.0, 0.25)])
    def test_step_unscales_by_external_scale(self, clip_grad, expected_grad):
        optimizer, params = self._optimizer(clip_grad)
        optimizer.step()

        # SGD with lr=1 from zero weights leaves each weight at minus the unscaled, clipped grad.
        # Before the fix this divided by 128, a step 32x too small, and clipping never fired.
        for p in params:
            assert torch.allclose(p.float(), torch.full_like(p.float(), -expected_grad), rtol=1e-3)
        assert optimizer._global_grad_norm == pytest.approx(self.true_norm, rel=1e-3)

    @pytest.mark.parametrize("clip_grad, expected_scale", [(0.0, 4.0), (1.0, 8.0)])
    def test_step_fused_adam_uses_external_scale(self, clip_grad, expected_scale):
        optimizer, _ = self._optimizer(clip_grad, fused_adam_legacy=True)
        optimizer.step()

        # The legacy fused kernel divides the gradients by this combined scale itself.
        assert optimizer.optimizer.scale == pytest.approx(expected_scale, rel=1e-3)
        assert optimizer._global_grad_norm == pytest.approx(self.true_norm, rel=1e-3)

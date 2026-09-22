# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math

import torch

from .builder import NPUOpBuilder


class NPUFusedLamb:
    """
    Fused LAMB for Ascend NPU (You et al., 2019).

    Pure-torch implementation of the LAMB update rule:
      g      = grad / combined_scale
      m      = beta1 * m + (1 - beta1) * g
      v      = beta2 * v + (1 - beta2) * g**2
      denom  = sqrt(v + eps)          (eps_mode 0)  or  sqrt(v) + eps  (eps_mode 1)
      update = m / denom + weight_decay * p
      coeff  = clamp(||p|| / ||update||, min_coeff, max_coeff)
      p     -= step_size * coeff * update

    with step_size = lr * sqrt(1 - beta2**step) / (1 - beta1**step) when bias
    correction is on (the reference kernel scales the same two factors into
    step_size). If a native LAMB kernel appears in torch_npu later, the
    internals of this method can be swapped without touching the callers.
    """

    @staticmethod
    def lamb(p, p_copy, exp_avg, exp_avg_sq, grad, lr, beta1, beta2, max_coeff, min_coeff, eps, combined_scale, step,
             eps_mode, bias_correction, weight_decay):
        if bias_correction:
            bc1 = 1.0 - beta1**step
            bc2 = 1.0 - beta2**step
            step_size = lr * math.sqrt(bc2) / bc1
        else:
            step_size = lr

        g = grad.float() / combined_scale

        # Explicit two-step arithmetic instead of in-place ops: in-place
        # tensor ops can contract into a fused multiply-add whose rounding
        # differs between backends, while the explicit form is deterministic
        # everywhere. The state tensors are updated in place via copy_.
        exp_avg_new = beta1 * exp_avg + (1.0 - beta1) * g
        exp_avg_sq_new = beta2 * exp_avg_sq + (1.0 - beta2) * g * g
        exp_avg.data.copy_(exp_avg_new)
        exp_avg_sq.data.copy_(exp_avg_sq_new)

        if eps_mode == 0:
            denom = (exp_avg_sq + eps).sqrt()
        else:
            denom = exp_avg_sq.sqrt() + eps

        update = exp_avg / denom + weight_decay * p.float()

        # trust ratio: clamped ||p|| / ||update||, or 1.0 when a norm vanishes
        p_norm = p.float().norm(2)
        u_norm = update.norm(2)
        if p_norm == 0 or u_norm == 0:
            lamb_coeff = torch.tensor(1.0)
        else:
            lamb_coeff = (p_norm / u_norm).clamp(min_coeff, max_coeff)

        p.data.copy_((p.float() - step_size * lamb_coeff * update).to(p.dtype))
        if p_copy.numel() > 0:
            p_copy.copy_(p.data)

        return lamb_coeff


class FusedLambBuilder(NPUOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_LAMB"
    NAME = "fused_lamb"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.lamb.{self.NAME}_op'

    def sources(self):
        # The optimizer is a pure-torch implementation; there is nothing to compile.
        return []

    def load(self, verbose=True):
        return NPUFusedLamb

    def is_compatible(self, verbose=False):
        return True

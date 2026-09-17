# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import math

from .builder import SUPAOpBuilder

try:
    import torch_supa_ext.deepspeed  # noqa: F401 — registers torch.ops.deepspeed
except Exception:
    pass


class SUPAFusedLamb:
    """
    Fused LAMB optimizer for Biren SUPA GPUs.

    Calls torch.ops.deepspeed.lamb when the compiled kernel is available;
    falls back to a pure-PyTorch loop otherwise.
    """

    @staticmethod
    def lamb(p, p_copy, exp_avg, exp_avg_sq, grad, lr, beta1, beta2, max_coeff, min_coeff, eps, combined_scale, step,
             eps_mode, bias_correction, weight_decay):
        import torch  # ensure torch is available at runtime

        if hasattr(torch.ops, 'deepspeed') and hasattr(torch.ops.deepspeed, 'lamb'):
            return torch.ops.deepspeed.lamb(p, p_copy, exp_avg, exp_avg_sq, grad, lr, beta1, beta2, max_coeff,
                                            min_coeff, eps, combined_scale, step, eps_mode, bias_correction,
                                            weight_decay)

        # Pure-PyTorch fallback
        if bias_correction:
            bc1 = 1.0 - beta1**step
            bc2 = 1.0 - beta2**step
            step_size = lr * math.sqrt(bc2) / bc1
        else:
            step_size = lr

        g = grad.float() / combined_scale

        exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

        if eps_mode == 0:
            denom = (exp_avg_sq + eps).sqrt()
        else:
            denom = exp_avg_sq.sqrt().add_(eps)

        update = exp_avg / denom
        update.add_(p.float(), alpha=weight_decay)

        p_norm = p.float().norm(2)
        u_norm = update.norm(2)
        if p_norm == 0 or u_norm == 0:
            lamb_coeff = torch.tensor(1.0)
        else:
            lamb_coeff = (p_norm / u_norm).clamp(min_coeff, max_coeff)

        p.data.add_(update, alpha=-step_size * lamb_coeff.item())
        if p_copy.numel() > 0:
            p_copy.copy_(p.data)

        return lamb_coeff


class FusedLambBuilder(SUPAOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_LAMB"
    NAME = "fused_lamb"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.lamb.{self.NAME}_op'

    def sources(self):
        return []

    def load(self, verbose=True):
        return SUPAFusedLamb

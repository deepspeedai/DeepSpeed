# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import math

try:
    import torch
    import torch_supa_ext.deepspeed  # noqa: F401 — registers torch.ops.deepspeed
except ImportError:
    pass

try:
    _has_kernel = hasattr(torch.ops, 'deepspeed') and hasattr(torch.ops.deepspeed, 'multi_tensor_adam')
except Exception:
    _has_kernel = False

from .builder import SUPAOpBuilder


class SUPAFusedAdam:
    """
    Fused Adam for Biren SUPA GPUs.

    Calls torch.ops.deepspeed.multi_tensor_adam (registered by torch_supa_ext.deepspeed)
    when the compiled extension is available; falls back to a numerically equivalent
    pure-PyTorch loop for cmodel / functional testing.
    """

    @staticmethod
    def multi_tensor_adam(chunk_size, noop_flag_buffer, tensor_lists, lr, beta1, beta2, epsilon, step, mode,
                          bias_correction, weight_decay):
        import torch  # ensure torch is available at runtime

        # noop_flag guard (kernel also checks internally, but short-circuit here is cheap)
        if noop_flag_buffer.item() == 1:
            return

        if _has_kernel:
            # MR #96 API: four separate Tensor-list arguments (not a nested list)
            grads, params, exp_avgs, exp_avg_sqs = tensor_lists
            torch.ops.deepspeed.multi_tensor_adam(chunk_size, noop_flag_buffer, grads, params, exp_avgs, exp_avg_sqs,
                                                  lr, beta1, beta2, epsilon, step, mode, bias_correction, weight_decay)
            return

        # Pure-PyTorch fallback (cmodel / no compiled backend)
        bias_correction1 = 1.0 - beta1**step if bias_correction else 1.0
        bias_correction2 = 1.0 - beta2**step if bias_correction else 1.0
        for i in range(len(tensor_lists[0])):
            g = tensor_lists[0][i].float()
            p = tensor_lists[1][i]
            m = tensor_lists[2][i]
            v = tensor_lists[3][i]
            if mode == 1:  # AdamW: decoupled weight decay
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(epsilon)
                # Decouple weight decay on the old param before the Adam step so the
                # result matches the kernel's p_old*(1 - lr*wd) - lr*adam_update.
                p.data.add_(p.data, alpha=-lr * weight_decay)
                p.data.addcdiv_(m, denom, value=-(lr / bias_correction1))
            else:  # Adam: L2 regularization
                g_wd = g.add(p.float(), alpha=weight_decay)
                m.mul_(beta1).add_(g_wd, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g_wd, g_wd, value=1.0 - beta2)
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(epsilon)
                p.data.addcdiv_(m, denom, value=-(lr / bias_correction1))


class FusedAdamBuilder(SUPAOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_ADAM"
    NAME = "fused_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return []

    def load(self, verbose=True):
        return SUPAFusedAdam

    def is_compatible(self, verbose=False):
        import torch  # ensure torch is available at runtime
        return hasattr(torch.ops, 'deepspeed') and hasattr(torch.ops.deepspeed, 'multi_tensor_adam')

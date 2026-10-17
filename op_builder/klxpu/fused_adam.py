# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import math

try:
    import torch
except ImportError:
    pass

from .builder import KLXPUOpBuilder


def _has_custom_ops():
    try:
        return hasattr(torch.ops, 'custom_ops') and hasattr(torch.ops.custom_ops, 'optimizer_AdamW')
    except Exception:
        return False


class KLXPUFusedAdam:
    # FusedAdam for KLXPU, compatible with DeepSpeed's multi_tensor_adam API.
    # Uses the KUNLUNXIN kernels (torch.ops.custom_ops) when available, else a
    # numerically equivalent pure-PyTorch fallback for functional / CPU testing.

    @staticmethod
    def multi_tensor_adam(chunk_size, noop_flag_buffer, tensor_lists, lr, beta1, beta2, epsilon, step, adam_w_mode,
                          bias_correction, weight_decay, *args):
        grad_tensor_list, param_tensor_list, mom_tensor_list, var_tensor_list = tensor_lists

        if _has_custom_ops():
            if adam_w_mode:
                lr_tensor = torch.tensor([lr], dtype=torch.float32)
                beta1_pow = torch.tensor([beta1**step], dtype=torch.float32)
                beta2_pow = torch.tensor([beta2**step], dtype=torch.float32)
                for i, param in enumerate(param_tensor_list):
                    grad = grad_tensor_list[i]
                    exp_avg = mom_tensor_list[i]
                    exp_avg_sq = var_tensor_list[i]
                    lr_tensor = lr_tensor.to(grad.device)
                    beta1_pow = beta1_pow.to(grad.device)
                    beta2_pow = beta2_pow.to(grad.device)
                    n = param.numel()
                    torch.ops.custom_ops.optimizer_AdamW(grad, exp_avg, exp_avg_sq, param, beta1_pow, beta2_pow,
                                                         lr_tensor, beta1, beta2, epsilon, weight_decay, n)
            else:
                shape_list = [p.numel() for p in param_tensor_list]
                torch.ops.custom_ops.multi_tensor_adam(grad_tensor_list, param_tensor_list, mom_tensor_list,
                                                       var_tensor_list, shape_list, lr, beta1, beta2, epsilon,
                                                       weight_decay, step, adam_w_mode, bias_correction)
            return

        # Pure-PyTorch fallback (no compiled backend).
        bias_correction1 = 1.0 - beta1**step if bias_correction else 1.0
        bias_correction2 = 1.0 - beta2**step if bias_correction else 1.0
        for i in range(len(param_tensor_list)):
            g = grad_tensor_list[i].float()
            p = param_tensor_list[i]
            m = mom_tensor_list[i]
            v = var_tensor_list[i]
            if adam_w_mode:
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(epsilon)
                p.data.add_(p.data, alpha=-lr * weight_decay)
                p.data.addcdiv_(m, denom, value=-(lr / bias_correction1))
            else:
                g_wd = g.add(p.float(), alpha=weight_decay)
                m.mul_(beta1).add_(g_wd, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g_wd, g_wd, value=1.0 - beta2)
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(epsilon)
                p.data.addcdiv_(m, denom, value=-(lr / bias_correction1))


class FusedAdamBuilder(KLXPUOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_ADAM"
    NAME = "fused_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return []

    def include_paths(self):
        return []

    def load(self, verbose=True):
        # KLXPU provides FusedAdam through torch.ops.custom_ops (with a
        # pure-PyTorch fallback), so there is nothing to compile.
        return KLXPUFusedAdam

    def is_compatible(self, verbose=False):
        return True

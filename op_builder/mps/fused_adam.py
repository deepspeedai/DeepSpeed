# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import MPSOpBuilder

try:
    import torch
except ImportError as e:
    pass


class MPSFusedAdam:
    """Pure-torch replacement for the CUDA multi_tensor_adam kernel, using torch._foreach_* ops on MPS.

    The math mirrors csrc/adam/multi_tensor_adam.cu so checkpoints and numerics stay interchangeable.
    """

    @staticmethod
    def multi_tensor_adam(chunk_size, noop_flag_buffer, tensor_lists, lr, beta1, beta2, epsilon, step, adam_w_mode,
                          bias_correction, weight_decay, *args):
        # The caller passes bf16 params as leaf tensors (not .data), so the in-place update must bypass autograd.
        with torch.no_grad():
            MPSFusedAdam._adam_step(tensor_lists, lr, beta1, beta2, epsilon, step, adam_w_mode, bias_correction,
                                    weight_decay)

    @staticmethod
    def _adam_step(tensor_lists, lr, beta1, beta2, epsilon, step, adam_w_mode, bias_correction, weight_decay):
        grads, params, exp_avgs, exp_avg_sqs = tensor_lists

        bias_correction1 = 1.0
        bias_correction2 = 1.0
        if bias_correction:
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step

        # L2 mode folds weight decay into the gradient; AdamW mode applies it to the parameter directly.
        if weight_decay != 0 and not adam_w_mode:
            grads = torch._foreach_add(grads, params, alpha=weight_decay)

        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

        denom = torch._foreach_div(exp_avg_sqs, bias_correction2)
        torch._foreach_sqrt_(denom)
        torch._foreach_add_(denom, epsilon)

        if weight_decay != 0 and adam_w_mode:
            torch._foreach_mul_(params, 1 - lr * weight_decay)
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-lr / bias_correction1)


class FusedAdamBuilder(MPSOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_ADAM"
    NAME = "fused_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def load(self, verbose=True):
        return MPSFusedAdam

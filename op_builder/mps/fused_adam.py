# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import MPSOpBuilder, MetalOpBuilder

try:
    import torch
except ImportError as e:
    pass


class MPSFusedAdam:
    """Drop-in for the CUDA multi_tensor_adam op: one Metal launch per tensor, math in fp32.

    Falls back to torch._foreach_* ops when torch.mps.compile_shader is unavailable.
    """
    kernels = None
    compile_failed = False

    @staticmethod
    def multi_tensor_adam(chunk_size, noop_flag_buffer, tensor_lists, lr, beta1, beta2, epsilon, step, adam_w_mode,
                          bias_correction, weight_decay, *args):
        bias_correction1 = 1.0
        bias_correction2 = 1.0
        if bias_correction:
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step

        # The caller passes bf16 params as leaf tensors (not .data), so the in-place update must bypass autograd.
        with torch.no_grad():
            if MPSFusedAdam.kernels is None:
                MPSFusedAdam._foreach_adam_step(tensor_lists, lr, beta1, beta2, epsilon, adam_w_mode, bias_correction1,
                                                bias_correction2, weight_decay)
            else:
                MPSFusedAdam._metal_adam_step(tensor_lists, lr, beta1, beta2, epsilon, adam_w_mode, bias_correction1,
                                              bias_correction2, weight_decay)

    @staticmethod
    def _metal_adam_step(tensor_lists, lr, beta1, beta2, epsilon, adam_w_mode, bias_correction1, bias_correction2,
                         weight_decay):
        grads, params, exp_avgs, exp_avg_sqs = tensor_lists
        for grad, param, exp_avg, exp_avg_sq in zip(grads, params, exp_avgs, exp_avg_sqs):
            # The kernel indexes flat storage, so every operand must share the param's contiguous layout.
            if not (param.is_contiguous() and grad.is_contiguous() and exp_avg.is_contiguous()
                    and exp_avg_sq.is_contiguous()):
                MPSFusedAdam._foreach_adam_step([[grad], [param], [exp_avg], [exp_avg_sq]], lr, beta1, beta2, epsilon,
                                                adam_w_mode, bias_correction1, bias_correction2, weight_decay)
                continue
            kernel = MPSFusedAdam.kernels.get(param.dtype)
            if kernel is None:
                MPSFusedAdam._foreach_adam_step([[grad], [param], [exp_avg], [exp_avg_sq]], lr, beta1, beta2, epsilon,
                                                adam_w_mode, bias_correction1, bias_correction2, weight_decay)
                continue
            kernel(param,
                   grad,
                   exp_avg,
                   exp_avg_sq,
                   float(lr),
                   float(beta1),
                   float(beta2),
                   float(epsilon),
                   float(weight_decay),
                   float(bias_correction1),
                   float(bias_correction2),
                   int(adam_w_mode),
                   threads=param.numel())

    @staticmethod
    def _foreach_adam_step(tensor_lists, lr, beta1, beta2, epsilon, adam_w_mode, bias_correction1, bias_correction2,
                           weight_decay):
        grads, params, exp_avgs, exp_avg_sqs = tensor_lists
        if params[0].dtype != torch.float32:
            # Match the kernel contract (fp32 math, storage-dtype results); fp16 intermediates overflow otherwise.
            fp32_lists = [[t.float() for t in tensors] for tensors in tensor_lists]
            MPSFusedAdam._foreach_adam_step(fp32_lists, lr, beta1, beta2, epsilon, adam_w_mode, bias_correction1,
                                            bias_correction2, weight_decay)
            for originals, fp32_tensors in zip((params, exp_avgs, exp_avg_sqs), fp32_lists[1:]):
                torch._foreach_copy_(originals, fp32_tensors)
            return

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


class FusedAdamBuilder(MetalOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_ADAM"
    NAME = "fused_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def metal_sources(self):
        return ['csrc/mps/fused_adam.metal']

    def is_compatible(self, verbose=False):
        # The foreach fallback keeps FusedAdam usable on any MPS build.
        return MPSOpBuilder.is_compatible(self, verbose)

    def load(self, verbose=True):
        if MPSFusedAdam.kernels is None and not MPSFusedAdam.compile_failed and MetalOpBuilder.is_compatible(self):
            try:
                library = self.load_metal_library()
            except RuntimeError as e:
                # Keep the foreach path usable rather than failing the optimizer on a shader compile error.
                MPSFusedAdam.compile_failed = True
                self.warning(f"Metal FusedAdam kernel failed to compile, using torch._foreach fallback: {e}")
                return MPSFusedAdam
            MPSFusedAdam.kernels = {
                torch.float32: library.fused_adam_float,
                torch.float16: library.fused_adam_half,
            }
            # The bfloat kernel is only compiled on Metal 3.1+ (macOS 14); the library raises rather
            # than returning an AttributeError for a missing entry point.
            try:
                MPSFusedAdam.kernels[torch.bfloat16] = library.fused_adam_bfloat
            except RuntimeError:
                pass
        return MPSFusedAdam

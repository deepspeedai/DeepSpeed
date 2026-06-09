# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from .builder import SUPAOpBuilder

try:
    import torch_supa_ext.deepspeed  # noqa: F401 — registers torch.ops.deepspeed
except Exception:
    pass


class SUPAFusedLion:
    """
    Fused Lion for Biren SUPA GPUs.

    Calls torch.ops.deepspeed.multi_tensor_lion when the compiled kernel is
    available; falls back to a pure-PyTorch loop otherwise.

    Lion update rule (Chen et al. 2023):
      c_t  = sign(beta1 * m_{t-1} + (1-beta1) * g_t)
      p_t  = p_{t-1} - lr * (c_t + weight_decay * p_{t-1})
      m_t  = beta2 * m_{t-1} + (1-beta2) * g_t
    """

    @staticmethod
    def multi_tensor_lion(chunk_size, noop_flag_buffer, tensor_lists, lr, beta1, beta2, step, weight_decay):
        import torch  # ensure torch is available at runtime

        if noop_flag_buffer.item() == 1:
            return

        if hasattr(torch.ops, 'deepspeed') and hasattr(torch.ops.deepspeed, 'multi_tensor_lion'):
            grads, params, exp_avgs = tensor_lists
            torch.ops.deepspeed.multi_tensor_lion(chunk_size, noop_flag_buffer, grads, params, exp_avgs, lr, beta1,
                                                  beta2, step, weight_decay)
            return

        # Pure-PyTorch fallback
        for i in range(len(tensor_lists[0])):
            g = tensor_lists[0][i].float()
            p = tensor_lists[1][i]
            m = tensor_lists[2][i]

            update = (beta1 * m.float() + (1.0 - beta1) * g).sign_()
            p.data.add_(update + weight_decay * p.float(), alpha=-lr)
            m.mul_(beta2).add_(g, alpha=1.0 - beta2)


class FusedLionBuilder(SUPAOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_LION"
    NAME = "fused_lion"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.lion.{self.NAME}_op'

    def sources(self):
        return []

    def load(self, verbose=True):
        return SUPAFusedLion

    def is_compatible(self, verbose=False):
        import torch  # ensure torch is available at runtime
        return hasattr(torch.ops, 'deepspeed') and hasattr(torch.ops.deepspeed, 'multi_tensor_lion')

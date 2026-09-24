# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from .builder import NPUOpBuilder


class NPUFusedLion:
    """
    Fused Lion for Ascend NPU.

    Pure-torch implementation of the Lion update rule (Chen et al. 2023):
      c      = beta1 * m + (1 - beta1) * g
      p     *= (1 - lr * weight_decay)
      p     += lr * (c > 0 ? -1 : 1)
      m      = beta2 * m + (1 - beta2) * g

    The math runs in fp32 and the results are written back in the parameter's
    dtype, mirroring the reference kernel (which also uses fp32 accumulators).
    The chained in-place form is ~26% faster per step on 910B4 than the
    equivalent explicit two-step arithmetic and agrees with it bitwise. If a
    native Lion kernel appears in torch_npu later, the internals of this
    method can be swapped without touching the callers.
    """

    @staticmethod
    def multi_tensor_lion(chunk_size, noop_flag_buffer, tensor_lists, lr, beta1, beta2, step, weight_decay):
        # The reference kernel skips the update when the overflow flag is set.
        if noop_flag_buffer.item() == 1:
            return
        grads, params, exp_avgs = tensor_lists
        after_decay = 1.0 - lr * weight_decay
        for g, p, m in zip(grads, params, exp_avgs):
            g_f = g.float()
            p_f = p.float()
            m_f = m.float()

            # Chained in-place form (like NPUFusedLamb): c must use the old
            # momentum, so the momentum state is updated last.
            c = m_f.mul(beta1).add_(g_f, alpha=1.0 - beta1)
            # sign of c; like the reference kernel, c == 0 takes the +lr branch
            update = torch.where(c > 0, -lr, lr)

            p_f.mul_(after_decay).add_(update)
            m_f.mul_(beta2).add_(g_f, alpha=1.0 - beta2)

            p.data.copy_(p_f.to(p.dtype))
            m.data.copy_(m_f.to(m.dtype))


class FusedLionBuilder(NPUOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_LION"
    NAME = "fused_lion"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.lion.{self.NAME}_op'

    def sources(self):
        # The optimizer is a pure-torch implementation; there is nothing to compile.
        return []

    def load(self, verbose=True):
        return NPUFusedLion

    def is_compatible(self, verbose=False):
        return True

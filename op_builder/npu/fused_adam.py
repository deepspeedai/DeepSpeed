# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from .builder import NPUOpBuilder

try:
    import torch_npu  # noqa: F401
except ImportError as e:
    pass


class NPUFusedAdam:

    @staticmethod
    def multi_tensor_adam(chunk_size, noop_flag_buffer, tensor_lists, lr, beta1, beta2, epsilon, step, adam_w_mode,
                          bias_correction, weight_decay, *args):
        # torch._fused_adam(w)_ always applies bias correction.
        # torch_npu's fused adam kernels do not increment step.
        step_tensor = torch.tensor(step, dtype=torch.int64, device=tensor_lists[1][0].device)
        for i in range(len(tensor_lists[0])):
            grad_flat = tensor_lists[0][i]
            param_flat = tensor_lists[1][i]
            m_flat = tensor_lists[2][i]
            v_flat = tensor_lists[3][i]

            if adam_w_mode:
                torch._fused_adamw_([param_flat], [grad_flat], [m_flat], [v_flat], [], [step_tensor],
                                    amsgrad=False,
                                    lr=lr,
                                    beta1=beta1,
                                    beta2=beta2,
                                    weight_decay=weight_decay,
                                    eps=epsilon,
                                    maximize=False)
            else:
                torch._fused_adam_([param_flat], [grad_flat], [m_flat], [v_flat], [], [step_tensor],
                                   amsgrad=False,
                                   lr=lr,
                                   beta1=beta1,
                                   beta2=beta2,
                                   weight_decay=weight_decay,
                                   eps=epsilon,
                                   maximize=False)


class FusedAdamBuilder(NPUOpBuilder):
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
        return NPUFusedAdam

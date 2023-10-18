# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import NPUOpBuilder

try:
    import torch_npu
except ImportError as e:
    pass


class NPUFusedAdam:

    @staticmethod
    def multi_tensor_adam(chunk_size, noop_flag_buffer, tensor_lists, lr, beta1, beta2, epsilon, step, adam_w_mode,
                          bias_correction, weight_decay, *args):
        bias_correction1 = beta1**step
        bias_correction2 = beta2**step

        # iteration group['params']
        for i in range(len(tensor_lists[0])):
            grad_flat = tensor_lists[0][i]
            param_flat = tensor_lists[1][i]
            m_flat = tensor_lists[2][i]
            v_flat = tensor_lists[3][i]

            if adam_w_mode:
                param_flat.data, m_flat, v_flat = torch_npu.npu_apply_adam_w(
                    bias_correction1,
                    bias_correction2,
                    lr,
                    weight_decay,
                    beta1,
                    beta2,
                    epsilon,
                    grad_flat,
                    None,  # max_grad_norm
                    False,  # amsgrad
                    False,  # maximize
                    out=(param_flat.data, m_flat, v_flat))
            else:
                param_flat.data, m_flat, v_flat = torch_npu.npu_apply_adam(
                    bias_correction1,
                    bias_correction2,
                    lr,
                    beta1,
                    beta2,
                    epsilon,
                    grad_flat,
                    False,  # use_locking
                    False,  # use_nesterov
                    out=(param_flat.data, m_flat, v_flat))


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

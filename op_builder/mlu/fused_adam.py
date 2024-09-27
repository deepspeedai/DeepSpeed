# Copyright (c) Microsoft Corporation.
# Copyright (c) 2024 Cambricon Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import MLUOpBuilder

try:
    import torch
except ImportError as e:
    pass


class MLUFusedAdam:

    @staticmethod
    def multi_tensor_adam(chunk_size, noop_flag_buffer, tensor_lists, lr, beta1, beta2, epsilon, step, adam_w_mode,
                          bias_correction, weight_decay, *args):

        torch.ops.torch_mlu.fused_adam(noop_flag_buffer, tensor_lists[0], tensor_lists[1], tensor_lists[2],
                                       tensor_lists[3], lr, beta1, beta2, epsilon, step, adam_w_mode, bias_correction,
                                       weight_decay)


class FusedAdamBuilder(MLUOpBuilder):
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
        return MLUFusedAdam

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# Prefer torch_musa custom op when available; otherwise raise a clear error.

from .builder import MUSAOpBuilder

try:
    import torch
except ImportError:
    torch = None


class MUSAFusedAdam:

    @staticmethod
    def multi_tensor_adam(chunk_size, noop_flag_buffer, tensor_lists, lr, beta1, beta2, epsilon, step, adam_w_mode,
                          bias_correction, weight_decay, *args):
        if torch is None or not hasattr(torch.ops, 'torch_musa'):
            raise RuntimeError(
                "DeepSpeed FusedAdam on MUSA requires torch.ops.torch_musa.fused_adam; "
                "use DeepSpeedCPUAdam / torch optimizers, or install a torch_musa build that exports fused_adam.")

        musa_ops = torch.ops.torch_musa
        if hasattr(musa_ops, 'fused_adam'):
            return musa_ops.fused_adam(noop_flag_buffer, tensor_lists[0], tensor_lists[1], tensor_lists[2],
                                       tensor_lists[3], lr, beta1, beta2, epsilon, step, adam_w_mode, bias_correction,
                                       weight_decay)
        raise RuntimeError(
            "torch.ops.torch_musa has no fused_adam. Available ops may differ by torch_musa version; "
            "use DeepSpeedCPUAdam or another optimizer until fused_adam is provided.")


class FusedAdamBuilder(MUSAOpBuilder):
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
        return MUSAFusedAdam

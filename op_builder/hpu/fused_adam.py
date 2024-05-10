# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

try:
    # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
    # if successful this also means we're doing a local install and not JIT compile path
    from op_builder import __deepspeed__  # noqa: F401 # type: ignore
    from op_builder.builder import OpBuilder
except ImportError:
    from deepspeed.ops.op_builder.builder import OpBuilder

try:
    import torch
    import math
except ImportError as e:
    pass


class HPUFusedAdam:
    htcore = None
    is_lazy_mode = None

    @staticmethod
    def multi_tensor_adam(chunk_size, noop_flag_buffer, tensor_lists, lr, beta1, beta2, epsilon, step, adam_w_mode,
                          bias_correction, weight_decay, *args):

        if HPUFusedAdam.htcore is None:
            from habana_frameworks.torch import core as htcore
            from habana_frameworks.torch.utils.internal import is_lazy
            HPUFusedAdam.htcore = htcore
            HPUFusedAdam.is_lazy_mode = is_lazy()

        htcore = HPUFusedAdam.htcore

        htcore.step_closure._mark_step_if_lazy()
        step_size = lr
        if bias_correction:
            bias_correction1 = 1.0 - pow(beta1, step)
            bias_correction2 = 1.0 - pow(beta2, step)
            step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

        neg_step = -step_size
        neg_step_t = (torch.tensor([neg_step], dtype=torch.float,
                                   requires_grad=False).to(tensor_lists[1][0].dtype).to(tensor_lists[1][0].device,
                                                                                        non_blocking=True))

        weight_decay = weight_decay if adam_w_mode else 0

        # since lr is fed into the kernel as tensor, perform the scalar multiplication of wd here
        # NOTE: TODO if lr is updated every step, then we need to convert it as tensor and
        # perform weight decay unconditonally.
        modified_wd = 1.0 - weight_decay * lr

        if HPUFusedAdam.is_lazy_mode:
            torch.ops.hpu.optimizer_adamw(
                tensor_lists[0],
                tensor_lists[1],
                tensor_lists[2],
                tensor_lists[3],
                neg_step_t,
                beta1,
                beta2,
                epsilon,
                modified_wd,
            )
        else:
            modified_wd_t = (torch.tensor([modified_wd], dtype=torch.float, requires_grad=False).to(
                tensor_lists[1][0].dtype).to(tensor_lists[1][0].device, non_blocking=True))
            torch.ops.hpu.optimizer_adamw(
                tensor_lists[0],
                tensor_lists[1],
                tensor_lists[2],
                tensor_lists[3],
                neg_step_t,
                beta1,
                beta2,
                epsilon,
                modified_wd_t,
                modified_wd != 1.0,
            )

        htcore.step_closure._mark_step_if_lazy()


class FusedAdamBuilder(OpBuilder):
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
        return HPUFusedAdam

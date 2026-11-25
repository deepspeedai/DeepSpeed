# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.utils.torch import restore_optimizer_state_dtype


class DeepSpeedCPUOptimizer(torch.optim.Optimizer):
    """Base class of DeepSpeedCPUAdam, DeepSpeedCPULion and DeepSpeedCPUAdagrad.

    With ``fp32_optimizer_states``, their ``step()`` keeps the states in fp32 for any parameter dtype, and the kernel
    is chosen from the state dtype; ``load_state_dict`` keeps them in fp32 too.
    """

    def __init__(self, params, defaults, fp32_optimizer_states=True):
        super().__init__(params, defaults)
        self.fp32_optimizer_states = fp32_optimizer_states

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # torch.optim.Optimizer.load_state_dict casts the states to the parameter dtype.
        if self.fp32_optimizer_states:
            restore_optimizer_state_dtype(self, state_dict, torch.float)

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn.functional as F
from deepspeed.utils.types import ActivationFuncType
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class GatedActivationOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig = None):
        if config is None:
            config = DeepSpeedInferenceConfig()
        super(GatedActivationOp, self).__init__(config)
        try:
            self.gated_activation_func = self.inference_module.gated_activation
        except AttributeError:
            self.gated_activation_func = self.gated_activation_fallback

    @classmethod
    def gated_activation_fallback(cls, activation, bias, activation_func_type):
        # Expected behavior is that of casting to float32 internally
        # Explicitly using the default GeLU
        activation_func = None
        activations = activation + bias.reshape(1, 1, -1)
        hidden_states, gate = activations.chunk(2, dim=-1)

        if activation_func_type == ActivationFuncType.GATED_SILU:
            activation_func = F.silu
        elif activation_func_type == ActivationFuncType.GATED_GELU:
            activation_func = F.gelu

        return hidden_states * activation_func(gate.to(torch.float32)).to(activations.dtype)

    def forward(self, activation: torch.Tensor, bias: torch.Tensor, activation_func_type: ActivationFuncType):
        return self.gated_activation_func(activation, bias, activation_func_type)

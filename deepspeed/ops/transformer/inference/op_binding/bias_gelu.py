# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn.functional as F
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class BiasGeluOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(BiasGeluOp, self).__init__(config)

        try:
            if self.config.dtype == torch.float16:
                self.bias_gelu_func = self.inference_module.bias_gelu_fp16
            elif self.config.dtype == torch.bfloat16:
                self.bias_gelu_func = self.inference_module.bias_gelu_bf16
            else:
                self.bias_gelu_func = self.inference_module.bias_gelu_fp32
        except AttributeError:
            self.bias_gelu_func = self.bias_gelu_fallback

    @classmethod
    def bias_gelu_fallback(cls, activations, bias):
        # Expected behavior is that of casting to float32 internally and using the tanh approximation
        return F.gelu(activations.to(torch.float32) + bias.to(torch.float32), approximate='tanh').to(activations.dtype)

    def forward(self, activation: torch.Tensor, bias: torch.Tensor):
        return self.bias_gelu_func(activation, bias)

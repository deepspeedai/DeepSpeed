# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn.functional as F
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class BiasReluOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(BiasReluOp, self).__init__(config)

        try:
            if self.config.dtype == torch.float16:
                self.bias_relu_func = self.inference_module.bias_relu_fp16
            elif self.config.dtype == torch.bfloat16:
                self.bias_relu_func = self.inference_module.bias_relu_bf16
            else:
                self.bias_relu_func = self.inference_module.bias_relu_fp32
        except AttributeError:
            self.bias_relu_func = self.bias_relu_fallback

    @classmethod
    def bias_relu_fallback(cls, activations, bias):
        # Expected behavior is that of casting to float32 internally
        return F.relu(activations.to(torch.float32) + bias.to(torch.float32)).to(activations.dtype)

    def forward(self, activation: torch.Tensor, bias: torch.Tensor):
        return self.bias_relu_func(activation, bias)

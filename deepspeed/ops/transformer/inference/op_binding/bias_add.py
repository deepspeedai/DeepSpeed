# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class BiasAddOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(BiasAddOp, self).__init__(config)

        try:
            if self.config.dtype == torch.float16:
                self.bias_add_func = self.inference_module.bias_add_fp16
            elif self.config.dtype == torch.bfloat16:
                self.bias_add_func = self.inference_module.bias_add_bf16
            else:
                self.bias_add_func = self.inference_module.bias_add_fp32
        except AttributeError:
            self.bias_add_func = self.bias_add_fallback

    @classmethod
    def bias_add_fallback(cls, input, bias):
        return torch.add(input, bias)

    def forward(self, activation: torch.Tensor, bias: torch.Tensor):
        return self.bias_add_func(activation, bias)

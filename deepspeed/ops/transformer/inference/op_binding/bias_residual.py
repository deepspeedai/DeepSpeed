# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class BiasResidualOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(BiasResidualOp, self).__init__(config)

        try:
            if self.config.dtype in [torch.float16, torch.int8]:
                self.bias_residual_func = self.inference_module.bias_residual_fp16
            else:
                self.bias_residual_func = self.inference_module.bias_residual_fp32
        except AttributeError:
            self.bias_residual_func = self.bias_residual_fallback

    @classmethod
    def bias_residual_fallback(cls, output, residual, bias):
        raise NotImplementedError("bias residual fallback isn't implemented")

    def forward(self, output, residual, bias):
        return self.bias_residual_func(output, residual, bias)

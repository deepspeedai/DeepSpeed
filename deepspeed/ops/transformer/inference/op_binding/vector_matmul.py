# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class VectorMatMulOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(VectorMatMulOp, self).__init__(config)
        try:
            if self.config.dtype in [torch.float16, torch.int8]:
                self.vector_matmul_func = self.inference_module.vector_matmul_fp16
            elif self.config.dtype == torch.bfloat16:
                self.vector_matmul_func = self.inference_module.vector_matmul_bf16
            else:
                self.vector_matmul_func = self.inference_module.vector_matmul_fp32
        except AttributeError:
            self.vector_matmul_func = self.vector_matmul_fallback

    def vector_matmul_fallback(self, input, weight, async_op, q_scale, q_int8, transpose):
        if os.environ.get('DS_KI_FALLBACK') == 'True' and not transpose:
            return torch.matmul(input, weight)
        else:
            raise NotImplementedError

    def forward(self, input: torch.Tensor, weight: torch.Tensor, async_op: bool = False):
        q_scale = weight.scale if hasattr(weight, 'scale') else torch.empty(1)
        q_int8 = self.config.dtype == torch.int8
        output = self.vector_matmul_func(input, weight, async_op, q_scale, q_int8, self.config.transposed_mode)
        return output

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class GELUGemmOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(GELUGemmOp, self).__init__(config)
        try:
            if self.config.dtype in [torch.float16, torch.int8]:
                self.fused_gemm_gelu = self.inference_module.fused_gemm_gelu_fp16  # type: ignore
            elif self.config.dtype == torch.bfloat16:
                self.fused_gemm_gelu = self.inference_module.fused_gemm_gelu_bf16
            else:
                self.fused_gemm_gelu = self.inference_module.fused_gemm_gelu_fp32  # type: ignore
        except AttributeError:
            self.fused_gemm_gelu = self.gelu_gemm_fallback

    def gelu_gemm_fallback(self, input, weight, scale, bias, out, out_scale, dtype, transpose):
        raise NotImplementedError

    def forward(self, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, weight_out: torch.Tensor):

        output = self.fused_gemm_gelu(
            input,
            weight,
            weight.scale if hasattr(weight, 'scale') else torch.empty(1),  # type: ignore
            bias,
            weight_out,
            weight_out.scale if hasattr(weight_out, 'scale') else torch.empty(1),  # type: ignore
            self.config.dtype == torch.int8,
            self.config.transposed_mode)

        return output

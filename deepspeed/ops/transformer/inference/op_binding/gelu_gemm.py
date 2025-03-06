# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn.functional as F
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp
import deepspeed


class GELUGemmOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(GELUGemmOp, self).__init__(config)
        try:
            if self.config.dtype == torch.int8:
                self.fused_gemm_gelu = self.inference_module.fused_gemm_gelu_int8
            elif self.config.dtype == torch.float16:
                if deepspeed.HAS_TRITON and self.config.use_triton and self.config.dtype == torch.float16:
                    from deepspeed.ops.transformer.inference.triton.ops import fused_gemm_gelu as _triton_fused_gemm_gelu
                    self.fused_gemm_gelu = _triton_fused_gemm_gelu  # type: ignore
                else:
                    self.fused_gemm_gelu = self.inference_module.fused_gemm_gelu_fp16  # type: ignore
            elif self.config.dtype == torch.bfloat16:
                self.fused_gemm_gelu = self.inference_module.fused_gemm_gelu_bf16  # type: ignore
            else:
                self.fused_gemm_gelu = self.inference_module.fused_gemm_gelu_fp32  # type: ignore
        except AttributeError:
            self.fused_gemm_gelu = self.gelu_gemm_fallback

    def gelu_gemm_fallback(self, input, weight, scale, bias, out, out_scale, dtype, transpose):
        tmp = torch.matmul(input, weight)
        tmp = F.gelu(tmp.to(torch.float32) + bias.to(torch.float32), approximate="tanh").to(tmp.dtype)
        output = torch.matmul(tmp, out)

        return output

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

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

import os
import torch
import torch.nn.functional as F
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp
from deepspeed.utils.types import NormType


class MLPGemmOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(MLPGemmOp, self).__init__(config)
        try:
            if self.config.norm_type == NormType.LayerNorm:
                if self.config.dtype in [torch.float16, torch.int8]:
                    self.mlp_gemm_func = self.inference_module.mlp_gemm_fp16  # type: ignore
                elif self.config.dtype == torch.bfloat16:
                    self.mlp_gemm_func = self.inference_module.mlp_gemm_bf16
                else:
                    self.mlp_gemm_func = self.inference_module.mlp_gemm_fp32  # type: ignore
            elif self.config.norm_type == NormType.RMSNorm:
                if self.config.dtype in [torch.float16, torch.int8]:
                    self.mlp_gemm_func = self.inference_module.rms_mlp_gemm_fp16  # type: ignore
                elif self.config.dtype == torch.bfloat16:
                    self.mlp_gemm_func = self.inference_module.rms_mlp_gemm_bf16
                else:
                    self.mlp_gemm_func = self.inference_module.rms_mlp_gemm_fp32  # type: ignore
        except AttributeError:
            if self.config.norm_type == NormType.LayerNorm:
                self.mlp_gemm_func = self.mlp_gemm_fallback
            elif self.config.norm_type == NormType.RMSNorm:
                self.mlp_gemm_func = self.rms_mlp_gemm_fallback

    def mlp_gemm_fallback(self, input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps,
                          pre_layer_norm, mlp_after_attn, interm_scale, out_scale, dtype, mlp_act_func_type,
                          transpose):
        if os.environ.get('DS_KI_FALLBACK') == 'True' and mlp_after_attn and not transpose:
            residual_add = F.layer_norm(input + residual + input_bias, (input.shape[2], ), gamma, beta,
                                        self.config.epsilon)
            tmp = torch.matmul(residual_add, weight_interm)
            tmp = F.gelu(tmp + bias)
            output = torch.matmul(tmp, weight_out)
            return (output, residual_add)
        else:
            raise NotImplementedError

    def rms_mlp_gemm_fallback(self, input, residual, weight_interm, weight_out, gamma, eps, interm_scale, out_scale,
                              dtype, mlp_act_func_type, transpose):
        raise NotImplementedError

    def forward(self,
                input: torch.Tensor,
                residual: torch.Tensor,
                weight_interm: torch.Tensor,
                weight_out: torch.Tensor,
                input_bias: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None,
                gamma: Optional[torch.Tensor] = None,
                beta: Optional[torch.Tensor] = None):
        if self.config.norm_type == NormType.LayerNorm:
            output, residual_add = self.mlp_gemm_func(
                input,
                residual,
                input_bias,
                weight_interm,
                weight_out,
                bias,
                gamma,
                beta,
                self.config.epsilon,
                self.config.pre_layer_norm,
                self.config.mlp_after_attn,
                weight_interm.scale if hasattr(weight_interm, 'scale') else torch.empty(1),  # type: ignore
                weight_out.scale if hasattr(weight_out, 'scale') else torch.empty(1),  # type: ignore
                self.config.dtype == torch.int8,
                self.config.mlp_act_func_type,
                self.config.transposed_mode)
        else:
            output, residual_add = self.mlp_gemm_func(
                input,
                residual,
                weight_interm,
                weight_out,
                gamma,
                self.config.epsilon,
                weight_interm.scale if hasattr(weight_interm, 'scale') else torch.empty(1),  # type: ignore
                weight_out.scale if hasattr(weight_out, 'scale') else torch.empty(1),  # type: ignore
                self.config.dtype == torch.int8,
                self.config.mlp_act_func_type,
                self.config.transposed_mode)
        return output, residual_add

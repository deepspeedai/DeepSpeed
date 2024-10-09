# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn.functional as F
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp
from .rms_norm import RMSNormOp
import deepspeed
from deepspeed.utils.types import NormType


class QKVGemmOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(QKVGemmOp, self).__init__(config)
        try:
            if self.config.norm_type == NormType.LayerNorm:
                if self.config.dtype in [torch.float16, torch.int8]:
                    if deepspeed.HAS_TRITON and self.config.use_triton and self.config.dtype == torch.float16:
                        from deepspeed.ops.transformer.inference.triton.ops import qkv_gemm_func as _triton_qkv_gemm_func
                        self.qkv_gemm_func = _triton_qkv_gemm_func
                        triton_autotune = config.triton_autotune and config.layer_id == 0
                        if triton_autotune:
                            __class__._triton_autotune(2, self.config.max_out_tokens, self.config.hidden_size)
                    else:
                        self.qkv_gemm_func = self.inference_module.qkv_gemm_fp16  # type: ignore
                elif self.config.dtype == torch.bfloat16:
                    self.qkv_gemm_func = self.inference_module.qkv_gemm_bf16
                else:
                    self.qkv_gemm_func = self.inference_module.qkv_gemm_fp32  # type: ignore
            elif self.config.norm_type == NormType.RMSNorm:
                if self.config.dtype in [torch.float16, torch.int8]:
                    self.qkv_gemm_func = self.inference_module.rms_qkv_gemm_fp16  # type: ignore
                elif self.config.dtype == torch.bfloat16:
                    self.qkv_gemm_func = self.inference_module.rms_qkv_gemm_bf16
                else:
                    self.qkv_gemm_func = self.inference_module.rms_qkv_gemm_fp32  # type: ignore
        except AttributeError:
            if self.config.norm_type == NormType.LayerNorm:
                self.qkv_gemm_func = self.qkv_gemm_fallback
            elif self.config.norm_type == NormType.RMSNorm:
                self.qkv_gemm_func = self.rms_qkv_gemm_fallback

    @staticmethod
    def _triton_autotune(min_seqlen, max_seqlen, hidden_size, dtype=torch.float16):
        from deepspeed.ops.transformer.inference.triton.matmul_ext import Fp16Matmul, matmul
        seqlen = [(min_seqlen + i)
                  for i in range(0, max_seqlen - min_seqlen + Fp16Matmul._cache_stride + 1, Fp16Matmul._cache_stride)]
        Fp16Matmul._read_autotune_table()
        for N in seqlen:
            A = torch.randn((N, hidden_size), dtype=dtype, device='cuda')
            B = torch.randn((hidden_size, 3 * hidden_size), dtype=dtype, device='cuda')
            matmul(A, B)
        Fp16Matmul._update_autotune_table()

    @staticmethod
    def qkv_gemm_fallback(input, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
        inp_norm = F.layer_norm(input, (input.shape[2], ), gamma, beta, eps)
        tmp = torch.matmul(inp_norm, weight.t() if transpose else weight)
        if add_bias:
            tmp += bias
        output = [tmp, inp_norm]

        return output

    @staticmethod
    def rms_qkv_gemm_fallback(input, weight, q_scale, gamma, eps, q_int8, transpose):
        inp_norm = RMSNormOp.rms_norm_fallback(input, gamma, eps)
        tmp = torch.matmul(inp_norm, weight.t() if transpose else weight)
        output = [tmp, inp_norm]

        return output

    def forward(self, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, gamma: torch.Tensor,
                beta: torch.Tensor):

        add_bias = bias is not None
        bias = bias if add_bias else torch.empty(1)  # type: ignore
        q_scale = weight.scale if hasattr(weight, 'scale') else torch.empty(1)  # type: ignore
        q_int8 = self.config.dtype == torch.int8

        if self.config.norm_type == NormType.LayerNorm:
            output, norm = self.qkv_gemm_func(input, weight, q_scale, bias, gamma, beta, self.config.epsilon, add_bias,
                                              q_int8, self.config.transposed_mode)
        else:
            output, norm = self.qkv_gemm_func(input, weight, q_scale, gamma, self.config.epsilon, q_int8,
                                              self.config.transposed_mode)
            if add_bias:
                output += bias

        return output, norm

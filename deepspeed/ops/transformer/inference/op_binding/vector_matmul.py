# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp
import deepspeed


class VectorMatMulOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(VectorMatMulOp, self).__init__(config)
        try:
            if self.config.dtype == torch.float16:
                if deepspeed.HAS_TRITON and config.use_triton:
                    from deepspeed.ops.transformer.inference.triton.ops import vector_matmul_func as _triton_vector_matmul_func
                    self.vector_matmul_func = _triton_vector_matmul_func
                    triton_autotune = config.triton_autotune and config.layer_id == 0
                    if triton_autotune:
                        __class__._triton_autotune(2, self.config.max_out_tokens, self.config.hidden_size)
                else:
                    self.vector_matmul_func = self.inference_module.vector_matmul_fp16
            elif self.config.dtype == torch.int8:
                self.vector_matmul_func = self.inference_module.vector_matmul_int8
            elif self.config.dtype == torch.bfloat16:
                self.vector_matmul_func = self.inference_module.vector_matmul_bf16
            else:
                self.vector_matmul_func = self.inference_module.vector_matmul_fp32
        except AttributeError:
            self.vector_matmul_func = self.vector_matmul_fallback

    def vector_matmul_fallback(self, input, weight, async_op, q_scale, q_int8, transpose):
        return torch.matmul(input, weight.t() if transpose else weight)

    def forward(self, input: torch.Tensor, weight: torch.Tensor, async_op: bool = False):
        q_scale = weight.scale if hasattr(weight, 'scale') else torch.empty(1)
        q_int8 = self.config.dtype == torch.int8
        output = self.vector_matmul_func(input, weight, async_op, q_scale, q_int8, self.config.transposed_mode)
        return output

    @staticmethod
    def _triton_autotune(min_seqlen, max_seqlen, hidden_size, dtype=torch.float16):
        from deepspeed.ops.transformer.inference.triton.matmul_ext import Fp16Matmul, matmul
        seqlen = [(min_seqlen + i)
                  for i in range(0, max_seqlen - min_seqlen + Fp16Matmul._cache_stride + 1, Fp16Matmul._cache_stride)]
        Fp16Matmul._read_autotune_table()
        for N in seqlen:
            A = torch.randn((N, hidden_size), dtype=dtype, device='cuda')
            B = torch.randn((hidden_size, hidden_size), dtype=dtype, device='cuda')
            matmul(A, B)
        Fp16Matmul._update_autotune_table()

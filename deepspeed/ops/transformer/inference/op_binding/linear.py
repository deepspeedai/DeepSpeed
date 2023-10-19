# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp
import deepspeed


class LinearOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(LinearOp, self).__init__(config)
        try:
            if self.config.dtype in [torch.float16, torch.int8]:
                if deepspeed.HAS_TRITON and self.config.use_triton and self.config.dtype == torch.float16:
                    from deepspeed.ops.transformer.inference.triton.ops import linear_func as _triton_linear_func
                    self.linear_func = _triton_linear_func
                    triton_autotune = config.triton_autotune and config.layer_id == 0
                    if triton_autotune:
                        __class__._triton_autotune(2, self.config.max_out_tokens, self.config.hidden_size)
                else:
                    self.linear_func = self.inference_module.linear_layer_fp16
                self.linear_func = self.inference_module.linear_layer_fp16
            elif self.config.dtype == torch.bfloat16:
                self.linear_func = self.inference_module.linear_layer_bf16
            else:
                self.linear_func = self.inference_module.linear_layer_fp32
        except AttributeError:
            self.linear_func = self.linear_fallback

    def linear_fallback(self, input, weight, bias, add_bias, do_flash_attn, num_heads, transpose, rope_theta):
        raise NotImplementedError

    def forward(self,
                input: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor,
                add_bias: bool,
                do_flash_attn: bool,
                num_heads: int,
                external_cache: bool = None,
                num_layers: int = None):
        qkv_out = self.linear_func(input, weight, bias, add_bias, do_flash_attn, num_heads,
                                   self.config.transposed_mode, self.config.rope_theta)
        return qkv_out

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

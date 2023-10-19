# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed import comm as dist
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class SoftmaxContextOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(SoftmaxContextOp, self).__init__(config)
        try:
            if self.config.dtype in [torch.float16, torch.int8]:
                self.softmax_context_func = self.inference_module.softmax_context_fp16
            elif self.config.dtype == torch.bfloat16:
                self.softmax_context_func = self.inference_module.softmax_context_bf16
            else:
                self.softmax_context_func = self.inference_module.softmax_context_fp32
        except AttributeError:
            self.softmax_context_func = self.softmax_context_fallback

    def softmax_context_fallback(self, query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two, heads,
                                 num_kv, norm_factor, triangular_masking, local_attention, window_size, no_masking,
                                 layer_id, num_layers, alibi, rope_theta):
        raise NotImplementedError

    def forward(self, query_key_value: torch.Tensor, attn_mask: torch.Tensor, heads: int, num_kv: int,
                norm_factor: float, no_masking: bool, layer_id: int, num_layers: int, alibi: torch.Tensor):

        if alibi is not None:
            batch_heads = query_key_value.shape[0] * heads
            offset = dist.get_rank() * batch_heads if dist.is_initialized() else 0
            alibi = alibi[offset:batch_heads + offset, :, :]
        else:
            alibi = torch.empty(1)

        output = self.softmax_context_func(query_key_value, attn_mask, self.config.rotary_dim, self.config.rotate_half,
                                           self.config.rotate_every_two, heads, num_kv, norm_factor,
                                           self.config.triangular_masking, self.config.local_attention,
                                           self.config.window_size, no_masking, layer_id, num_layers, alibi,
                                           self.config.rope_theta)

        return output

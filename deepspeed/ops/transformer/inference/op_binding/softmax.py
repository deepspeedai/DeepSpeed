# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn.functional as F
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp
from deepspeed.ops.transformer.inference.op_binding.workspace import InferenceContext


class SoftmaxOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(SoftmaxOp, self).__init__(config)
        self.num_attention_heads_per_partition = config.heads // config.mp_size
        try:
            if self.config.dtype in [torch.float16, torch.int8]:
                self.softmax_func = self.inference_module.softmax_fp16
            elif self.config.dtype == torch.bfloat16:
                self.softmax_func = self.inference_module.softmax_bf16
            else:
                self.softmax_func = self.inference_module.softmax_fp32
        except AttributeError:
            self.softmax_func = self.softmax_fallback

    @staticmethod
    def softmax_fallback(attn_scores, attn_mask, alibi, triangular, recompute, local_attention, window_size, async_op,
                         layer_scale, head_offset, mp_size):
        scores_len = len(attn_scores.size())
        heads = 1
        if scores_len > 1:
            heads = attn_scores.size()[1]
        num_attention_heads_per_partition = heads // mp_size

        if alibi is not None:
            if len(alibi.shape) == 1:
                alibi = None
            else:
                alibi = alibi[head_offset:head_offset + num_attention_heads_per_partition]
        if attn_mask is not None and len(attn_mask.shape) == 1:
            attn_mask = None
        input_dtype = attn_scores.dtype
        attn_scores *= layer_scale

        if alibi is not None:
            attn_scores += alibi
        if attn_mask is not None:
            # expand atten_mask from two dim into 4 dim, insert two dims in the middle
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask[:, None, None, :]
            attn_scores += attn_mask
        if triangular:
            if attn_scores.shape[2] == 1:  # query using kv cache
                token_idx = InferenceContext.Instance().current_tokens()
                tri = torch.arange(attn_scores.shape[2], device=attn_scores.device).ge(token_idx)
            else:
                tri = ~torch.tril(torch.ones(attn_scores.size(), device=attn_scores.device)).to(bool)
            attn_scores = torch.masked_fill(attn_scores, tri, float('-inf'))
        output = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(input_dtype)

        return output

    def forward(self, attn_scores: torch.Tensor, attn_mask: torch.Tensor, alibi: torch.Tensor, triangular: bool,
                recompute: bool, local_attention: bool, window_size: int, async_op: bool, layer_scale: float,
                head_offset: int):
        output = self.softmax_func(attn_scores, attn_mask, alibi, triangular, recompute, local_attention, window_size,
                                   async_op, layer_scale, head_offset, self.config.mp_size)

        return output

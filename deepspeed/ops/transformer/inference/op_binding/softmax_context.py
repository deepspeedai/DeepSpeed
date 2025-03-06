# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed import comm as dist
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp
from .softmax import SoftmaxOp
from deepspeed.ops.transformer.inference.op_binding.workspace import InferenceContext


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

    @staticmethod
    def transform4d_0213(x, seq_length):
        assert x.dim() == 3, F"Dim {x.dim()} is not supported"
        batch_size, num_heads, seq_length_head_dim = x.shape
        head_dim = seq_length_head_dim // seq_length
        x = x.view(batch_size, num_heads, seq_length, head_dim)
        x = x.permute(0, 2, 1, 3)

        return x

    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep <= 1 or num_key_value_heads == 1:
            return hidden_states

        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)

        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    @staticmethod
    def bias_add_transform_0213(input, bias, num_heads, trans_count, perform_bias=False):
        assert trans_count == 1 or trans_count == 3, F"Trans count {trans_count} is not supported"
        assert input.dim() == 3, F"Dim {input.dim()} is not supported"
        input_biased = torch.add(input, bias) if perform_bias else input
        batch_size, seq_length, value_size = input_biased.shape
        hid_dim = value_size // trans_count
        head_dim = hid_dim // num_heads

        if trans_count == 1:
            query_layer = input.view(batch_size, seq_length, num_heads, head_dim)
            query_layer = query_layer.permute(0, 2, 1, 3)
            key_layer = torch.zeros_like(query_layer)
            value_layer = torch.zeros_like(query_layer)
            return query_layer, key_layer, value_layer

        qkv_layers = input.view(batch_size, seq_length, 3, num_heads, head_dim)
        query_layer, key_layer, value_layer = qkv_layers[..., 0, :, :], qkv_layers[..., 1, :, :], qkv_layers[...,
                                                                                                             2, :, :]
        query_layer = query_layer.transpose(1, 2)
        key_layer = key_layer.transpose(1, 2)
        value_layer = value_layer.transpose(1, 2)

        return query_layer, key_layer, value_layer

    def softmax_context_fallback(self, query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two, heads,
                                 num_kv, norm_factor, triangular_masking, local_attention, window_size, no_masking,
                                 layer_id, num_layers, alibi, rope_theta, is_prompt, token_idx, position_ids):
        bat_0213_query, bat_0213_key, bat_0213_value = self.bias_add_transform_0213(
            query_key_value, None, heads, 3, False)

        if rotary_dim > 0 and rotate_half:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

            rotary = InferenceContext.Instance().get_rotary(rotary_dim, rope_theta, bat_0213_value.device)
            cos, sin = rotary(bat_0213_value, InferenceContext.Instance().get_max_tokens_num())
            bat_0213_query, bat_0213_key = apply_rotary_pos_emb(bat_0213_query, bat_0213_key, cos, sin, position_ids)

        bat_0213_key, bat_0213_value = InferenceContext.Instance().update_cache(layer_id, token_idx, is_prompt,
                                                                                bat_0213_key, bat_0213_value)

        bat_0213_key = self.repeat_kv(bat_0213_key, num_kv)
        bat_0213_value = self.repeat_kv(bat_0213_value, num_kv)

        bsz = query_key_value.shape[0]
        head_dim = query_key_value.shape[2] // (heads * 3)

        bmm_output = torch.bmm(bat_0213_query.reshape(bsz * heads, bat_0213_query.shape[2], head_dim),
                               bat_0213_key.reshape(bsz * heads, bat_0213_key.shape[2], head_dim).transpose(1, 2))

        layer_scale = 1.0
        if alibi is not None and len(alibi.shape) > 1:
            layer_scale = max(1, layer_id).to(float)

        alpha = norm_factor * norm_factor / layer_scale
        bmm_output *= alpha
        bmm_output_reshape = bmm_output.reshape(bsz, heads, bmm_output.shape[1], bmm_output.shape[2])

        recompute = is_prompt
        if attn_mask is not None and len(attn_mask.shape) > 1 and attn_mask.shape[-1] < bmm_output_reshape.shape[3]:
            attn_mask = torch.nn.functional.pad(attn_mask, (0, bmm_output_reshape.shape[3] - attn_mask.shape[-1]),
                                                value=torch.finfo(attn_mask.dtype).min)
        softmax_output = SoftmaxOp.softmax_fallback(bmm_output_reshape, attn_mask, alibi, triangular_masking,
                                                    recompute, local_attention, window_size, None, layer_scale, 0, 1)

        output = torch.bmm(softmax_output.reshape(bsz * heads, softmax_output.shape[2], softmax_output.shape[3]),
                           bat_0213_value.reshape(bsz * heads, bat_0213_value.shape[2], head_dim))

        output = output.reshape(bsz, heads, output.shape[1], head_dim)
        output = output.reshape(bsz, heads, output.shape[2] * head_dim)
        input_seq_len = query_key_value.shape[1]
        t4d_0123_output = self.transform4d_0213(output, input_seq_len)
        t4d_0123_output = t4d_0123_output.reshape(bsz, t4d_0123_output.shape[1], heads * head_dim)

        if layer_id == num_layers - 1:
            InferenceContext.Instance().advance_tokens()

        return t4d_0123_output, bat_0213_key, bat_0213_value

    def forward(self, query_key_value: torch.Tensor, attn_mask: torch.Tensor, heads: int, num_kv: int,
                norm_factor: float, no_masking: bool, layer_id: int, num_layers: int, alibi: torch.Tensor,
                is_prompt: bool, token_idx: torch.Tensor, position_ids: torch.Tensor):

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
                                           self.config.rope_theta, is_prompt, token_idx, position_ids)

        return output

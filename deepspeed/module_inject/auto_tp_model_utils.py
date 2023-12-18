# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed import comm as dist
import torch
from typing import Optional
from deepspeed.module_inject.tp_shard import get_shard_size, get_shard_size_list


def build_bloom_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    import math
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2**math.floor(math.log2(num_heads))
    base = torch.tensor(2**(-(2**-(math.log2(closest_power_of_2) - 3))),
                        device=attention_mask.device,
                        dtype=torch.float32)
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(2**(-(2**-(math.log2(2 * closest_power_of_2) - 3))),
                                  device=attention_mask.device,
                                  dtype=torch.float32)
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    if dist.is_initialized():
        num_heads_per_rank = get_shard_size(num_heads, dist.get_world_size())
        offset = sum(get_shard_size_list(num_heads, dist.get_world_size())[0:dist.get_rank()])
        alibi = alibi.view(batch_size, num_heads, 1, seq_length)
        alibi = alibi[:, offset:num_heads_per_rank + offset, :, :]
        return alibi.reshape(batch_size * num_heads_per_rank, 1, seq_length).to(dtype)
    else:
        return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


def get_alibi_mask(self, tensor, seq_length_with_past):
    mask = self.get_alibi_mask_orig(tensor, seq_length_with_past)
    if not self.training and dist.is_initialized():
        num_heads_per_rank = get_shard_size(self.n_head, dist.get_world_size())
        offset = sum(get_shard_size_list(self.n_head, dist.get_world_size())[0:dist.get_rank()])
        mask = mask[offset:num_heads_per_rank + offset, :seq_length_with_past, :seq_length_with_past]

    return mask


def build_mpt_atten_bias_tensor(self,
                                device,
                                dtype,
                                attention_mask: Optional[torch.ByteTensor] = None,
                                prefix_mask: Optional[torch.ByteTensor] = None,
                                sequence_id: Optional[torch.LongTensor] = None):
    (attn_bias, attention_mask) = self._attn_bias_orig(device,
                                                       dtype,
                                                       attention_mask=attention_mask,
                                                       prefix_mask=prefix_mask,
                                                       sequence_id=sequence_id)
    if dist.is_initialized():
        num_heads_per_rank = get_shard_size(self.config.n_heads, dist.get_world_size())
        offset = sum(get_shard_size_list(self.config.n_heads, dist.get_world_size())[0:dist.get_rank()])
        attn_bias = attn_bias[:, offset:num_heads_per_rank + offset, :, :]
    return attn_bias, attention_mask


def build_mpt_alibi_tensor(self, num_heads, sequence_length, alibi_bias_max=8, device=None) -> torch.Tensor:
    r"""
    Link to paper: https://arxiv.org/abs/2108.12409 - Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation. This implementation has been copied from
    the alibi implementation of MPT source code that led to slightly different results than the Bloom alibi:
    https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L292
    """
    alibi = self.build_mpt_alibi_tensor_orig(num_heads, sequence_length, alibi_bias_max, device)
    if dist.is_initialized():
        num_heads_per_rank = int(num_heads / dist.get_world_size())
        offset = dist.get_rank() * num_heads_per_rank
        alibi = alibi[offset:num_heads_per_rank + offset, :, :]
    return alibi

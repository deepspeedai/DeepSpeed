# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools

from deepspeed import comm as dist
import torch
from typing import Optional
from deepspeed.module_inject.tp_shard import AutoTPMeta, get_shard_size_list


class _HeadCountProxy:

    def __init__(self, module, total_num_heads):
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "n_head", total_num_heads)

    def __getattr__(self, name):
        return getattr(self._module, name)

    def __setattr__(self, name, value):
        if name == "n_head":
            object.__setattr__(self, name, value)
        else:
            setattr(self._module, name, value)


def get_head_shard_sizes(meta: AutoTPMeta, mp_group=None):
    tp_world_size = dist.get_world_size(group=mp_group)
    return get_shard_size_list(meta.num_attention_heads, tp_world_size, meta)


def install_head_sharded_helper(module, name, wrapper, meta, mp_group=None):
    """Give ``module`` a head-slicing wrapper around one of its own methods.

    The wrapper is bound to this instance instead of installed on its class. A class-wide patch
    outlives the injected model: instances created later never get the ``<name>_orig`` the
    wrapper delegates to, and a subclass inheriting the patched method would record the wrapper
    itself as the original and call it until the interpreter runs out of stack.
    """
    original_name = f"{name}_orig"
    if original_name not in module.__dict__:
        # Wrapping an already wrapped instance would make it delegate to itself.
        setattr(module, original_name, getattr(module, name))
    total_num_heads = meta.num_attention_heads
    shard_sizes = get_head_shard_sizes(meta, mp_group) if total_num_heads is not None else None
    setattr(
        module, name,
        functools.partial(wrapper,
                          module,
                          mp_group=mp_group,
                          head_shard_sizes=shard_sizes,
                          total_num_heads=total_num_heads))


def _head_shard(num_heads, mp_group=None, head_shard_sizes=None, total_num_heads=None):
    """This rank's head count and offset, following the same split AutoTP used for the weights.

    The heads have to be cut the same way as the attention weights, which AutoTP partitions
    across the tensor-parallel group. Falling back to the global rank would select a different
    slice than the weights whenever the group is smaller than the world.
    """
    tp_world_size = dist.get_world_size(group=mp_group)
    tp_index = dist.get_rank(group=mp_group)
    full_num_heads = total_num_heads if total_num_heads is not None else num_heads
    # The fallback split is only reached when no per-head shard sizes were recorded, i.e. when
    # the model is not GQA-aware, so an even split (default meta, no kv-heads) is correct here.
    shard_sizes = head_shard_sizes or get_shard_size_list(full_num_heads, tp_world_size, AutoTPMeta())
    if len(shard_sizes) != tp_world_size or sum(shard_sizes) != full_num_heads:
        raise ValueError(f"Head shard sizes {shard_sizes} do not partition {full_num_heads} heads across "
                         f"{tp_world_size} tensor-parallel ranks.")
    return shard_sizes[tp_index], sum(shard_sizes[0:tp_index])


def build_bloom_alibi_tensor(attention_mask: torch.Tensor,
                             num_heads: int,
                             dtype: torch.dtype,
                             mp_group=None,
                             head_shard_sizes=None,
                             total_num_heads=None) -> torch.Tensor:
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
    num_heads = total_num_heads if total_num_heads is not None else num_heads
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
        num_heads_per_rank, offset = _head_shard(num_heads, mp_group, head_shard_sizes, total_num_heads)
        alibi = alibi.view(batch_size, num_heads, 1, seq_length)
        alibi = alibi[:, offset:num_heads_per_rank + offset, :, :]
        return alibi.reshape(batch_size * num_heads_per_rank, 1, seq_length).to(dtype)
    else:
        return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


def get_alibi_mask(self, tensor, seq_length_with_past, mp_group=None, head_shard_sizes=None, total_num_heads=None):
    original = self.get_alibi_mask_orig
    if total_num_heads is not None and self.n_head != total_num_heads:
        original_function = getattr(original, "__func__", None)
        if original_function is None:
            raise TypeError("Cannot invoke get_alibi_mask with the original total head count.")
        mask = original_function(_HeadCountProxy(self, total_num_heads), tensor, seq_length_with_past)
    else:
        mask = original(tensor, seq_length_with_past)
    if dist.is_initialized():
        num_heads_per_rank, offset = _head_shard(self.n_head, mp_group, head_shard_sizes, total_num_heads)
        mask = mask[offset:num_heads_per_rank + offset, :seq_length_with_past, :seq_length_with_past]

    return mask


def build_mpt_atten_bias_tensor(self,
                                device,
                                dtype,
                                attention_mask: Optional[torch.ByteTensor] = None,
                                prefix_mask: Optional[torch.ByteTensor] = None,
                                sequence_id: Optional[torch.LongTensor] = None,
                                mp_group=None,
                                head_shard_sizes=None,
                                total_num_heads=None):
    (attn_bias, attention_mask) = self._attn_bias_orig(device,
                                                       dtype,
                                                       attention_mask=attention_mask,
                                                       prefix_mask=prefix_mask,
                                                       sequence_id=sequence_id)
    if dist.is_initialized():
        num_heads_per_rank, offset = _head_shard(self.config.n_heads, mp_group, head_shard_sizes, total_num_heads)
        attn_bias = attn_bias[:, offset:num_heads_per_rank + offset, :, :]
    return attn_bias, attention_mask


def build_mpt_alibi_tensor(self,
                           num_heads,
                           sequence_length,
                           alibi_bias_max=8,
                           device=None,
                           mp_group=None,
                           head_shard_sizes=None,
                           total_num_heads=None) -> torch.Tensor:
    r"""
    Link to paper: https://arxiv.org/abs/2108.12409 - Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation. This implementation has been copied from
    the alibi implementation of MPT source code that led to slightly different results than the Bloom alibi:
    https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L292
    """
    full_num_heads = total_num_heads if total_num_heads is not None else num_heads
    alibi = self.build_mpt_alibi_tensor_orig(full_num_heads, sequence_length, alibi_bias_max, device)
    if dist.is_initialized():
        num_heads_per_rank, offset = _head_shard(num_heads, mp_group, head_shard_sizes, total_num_heads)
        alibi = alibi[offset:num_heads_per_rank + offset, :, :]
    return alibi

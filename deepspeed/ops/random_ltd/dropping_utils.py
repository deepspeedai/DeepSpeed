# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.ops.op_builder import RandomLTDBuilder
"""
Returns:
    sampled_indices: [layers, batch_size, reserved_length]
    new_mask: [batch_size, 1, reserved_length, reserved_length]
"""

random_ltd_module = None


def gpt_sample_tokens(reserved_length: int,
                      seq_length: int,
                      batch_size: int,
                      layers: int = 1,
                      device: str = 'cpu',
                      attn_mask: torch.Tensor = None):

    prob_dist = torch.ones((layers * batch_size, seq_length), device=device)
    sampled_indices = torch.multinomial(prob_dist, reserved_length)

    sampled_indices = sampled_indices.reshape(layers, batch_size, reserved_length).to(torch.int32)
    global random_ltd_module
    if random_ltd_module is None:
        random_ltd_module = RandomLTDBuilder().load()
    sampled_indices = random_ltd_module.token_sort_(sampled_indices, seq_length)

    # Not certain the optimized kernel is actually better here, cause it kind of screws
    # with alignment right if the sequence length is not divisible by like 16
    # new_mask = random_ltd_module.mask_gather_gpt(attn_mask, reserved_length)
    if attn_mask is not None:
        new_mask = attn_mask[:, :, :reserved_length, :reserved_length]
    else:
        new_mask = None

    return sampled_indices, new_mask


"""
Returns:
    sampled_indices: [layers, batch_size, reserved_length]
    new_mask: [layers, batch_size, 1, reserved_length, reserved_length]
"""


def bert_sample_tokens(reserved_length: int,
                       seq_length: int,
                       batch_size: int,
                       layers: int = 1,
                       device: str = 'cpu',
                       attn_mask: torch.Tensor = None):
    assert attn_mask is not None
    prob_dist = torch.ones((layers * batch_size, seq_length), device=device)
    sampled_indices = torch.multinomial(prob_dist, reserved_length)

    sampled_indices = sampled_indices.reshape(layers, batch_size, reserved_length).to(torch.int32)
    global random_ltd_module
    if random_ltd_module is None:
        random_ltd_module = RandomLTDBuilder().load()

    sampled_indices = random_ltd_module.token_sort_(sampled_indices, seq_length)
    dtype = sampled_indices.dtype

    sampled_indices = sampled_indices.to(torch.long)
    new_mask = []
    for l in range(layers):
        tmp_mask_list = []
        for i in range(batch_size):
            mask_tmp = attn_mask[i:i + 1, :, sampled_indices[l][i], :]
            tmp_mask_list.append(mask_tmp[:, :, :, sampled_indices[l][i]])
        new_mask.append(torch.cat(tmp_mask_list, dim=0))

    return sampled_indices.to(dtype), new_mask


class GatherTokens(torch.autograd.Function):

    @staticmethod
    def forward(ctx, activations: torch.Tensor, sorted_indices: torch.Tensor, batch_first: bool):
        global random_ltd_module
        if random_ltd_module is None:
            random_ltd_module = RandomLTDBuilder().load()
        ctx.save_for_backward(activations, sorted_indices)
        ctx.batch_first = batch_first
        return activations, random_ltd_module.token_gather(activations, sorted_indices, batch_first)

    @staticmethod
    def backward(ctx, a_gradients: torch.Tensor, g_gradients: torch.Tensor):

        g_gradients = g_gradients.contiguous()
        global random_ltd_module
        if random_ltd_module is None:
            random_ltd_module = RandomLTDBuilder().load()
        activations, sorted_indices = ctx.saved_tensors
        batch_first = ctx.batch_first

        return random_ltd_module.token_scatter_(a_gradients, g_gradients, sorted_indices, batch_first), None, None


class ScatterTokens(torch.autograd.Function):

    @staticmethod
    def forward(ctx, all_activations: torch.Tensor, layer_activations: torch.Tensor, sorted_indices: torch.Tensor,
                batch_first: bool):
        global random_ltd_module
        if random_ltd_module is None:
            random_ltd_module = RandomLTDBuilder().load()
        scatter_results = random_ltd_module.token_scatter_(all_activations.clone(), layer_activations, sorted_indices,
                                                           batch_first)

        ctx.save_for_backward(sorted_indices)
        ctx.batch_first = batch_first
        return scatter_results

    @staticmethod
    def backward(ctx, out_gradients: torch.Tensor):

        out_gradients = out_gradients.contiguous()
        global random_ltd_module
        if random_ltd_module is None:
            random_ltd_module = RandomLTDBuilder().load()
        sorted_indices, = ctx.saved_tensors
        batch_first = ctx.batch_first

        ret_val = random_ltd_module.token_gather(out_gradients, sorted_indices, batch_first)
        return out_gradients, ret_val, None, None

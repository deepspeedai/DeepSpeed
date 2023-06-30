# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import triton
import triton.language as tl
from deepspeed.accelerator import get_accelerator


@triton.jit
def residual_add_bias_kernel(
    hidden_state_ptr,
    residual_ptr,
    attn_output_ptr,
    hidden_state_size,
    attn_bias_ptr,
    final_bias_ptr,
    bias_size,
    output_ptr,
    mp_size: tl.constexpr,
    mlp_after_attn: tl.constexpr,
    pre_attn_norm: tl.constexpr,
    add_attn_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_state_size

    bias_offsets = offsets % bias_size
    bias_mask = bias_offsets < bias_size

    tl_hidden_state = tl.load(hidden_state_ptr + offsets, mask=mask)
    tl_residual = tl.load(residual_ptr + offsets, mask=mask)
    tl_attn_output = tl.load(attn_output_ptr + offsets, mask=mask)
    tl_attn_bias = tl.load(attn_bias_ptr + bias_offsets, mask=bias_mask)
    tl_final_bias = tl.load(final_bias_ptr + bias_offsets, mask=bias_mask)

    if mlp_after_attn:
        if pre_attn_norm:
            output = tl_hidden_state + (tl_residual + tl_final_bias + tl_attn_output + tl_attn_bias) / mp_size
        else:
            output = tl_hidden_state + tl_residual + tl_final_bias
    else:
        output = tl_hidden_state + tl_attn_output + (tl_residual + tl_final_bias) / mp_size
        if add_attn_bias:
            output += tl_attn_bias / mp_size

    tl.store(output_ptr + offsets, output, mask=mask)


def residual_add_bias(hidden_state: torch.Tensor, residual: torch.Tensor, attn_output: torch.Tensor,
                      attn_bias: torch.Tensor, final_bias: torch.Tensor, mp_size: int, mlp_after_attn: bool,
                      add_attn_bias: bool, pre_attn_norm: bool):
    # check that all tensors are on the same device
    assert get_accelerator().on_accelerator(hidden_state) \
        and get_accelerator().on_accelerator(residual) \
        and get_accelerator().on_accelerator(attn_output) \
        and get_accelerator().on_accelerator(attn_bias) \
        and get_accelerator().on_accelerator(final_bias)

    # check that all tensors have the same dtype
    assert hidden_state.dtype == residual.dtype == attn_output.dtype \
        == attn_bias.dtype == final_bias.dtype

    # check that all tensors have the right shape
    assert hidden_state.shape == residual.shape == attn_output.shape
    assert attn_bias.shape == final_bias.shape
    assert attn_bias.shape[0] == hidden_state.shape[2]

    output = torch.empty_like(hidden_state)

    hidden_state_size = output.numel()
    bias_size = attn_bias.numel()

    grid = lambda meta: (triton.cdiv(hidden_state_size, meta['BLOCK_SIZE']), )

    residual_add_bias_kernel[grid](hidden_state, residual, attn_output, hidden_state_size,\
                    attn_bias, final_bias, bias_size, output, mp_size, mlp_after_attn, pre_attn_norm, \
                    add_attn_bias, \
                    BLOCK_SIZE=1024)

    return output

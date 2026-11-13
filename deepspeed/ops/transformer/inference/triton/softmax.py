# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import triton
import triton.language as tl
'''
softmax
modified the triton kernel in
https://github.com/openai/triton/blob/34817ecc954a6f4ca7b4dfb352fdde1f8bd49ca5/python/tutorials/02-fused-softmax.py
'''


@triton.jit
def softmax_kernel(output_ptr, input_ptr, stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf')).to(tl.float32)
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


@triton.jit
def masked_softmax_kernel(output_ptr, input_ptr, stride, mask_ptr, mask_stride0, mask_stride1, mask_stride2, rows1,
                          rows2, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # The input rows form a [rows0, rows1, rows2] grid; the mask is broadcast over it through its strides
    # (a stride is 0 along a broadcast dimension).
    mask_row = (row_idx // (rows1 * rows2)) * mask_stride0 + ((row_idx // rows2) % rows1) * mask_stride1 + \
        (row_idx % rows2) * mask_stride2
    mask_ptrs = mask_ptr + mask_row + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf')).to(tl.float32)
    mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=0).to(tl.float32)
    row_minus_max = row - tl.max(row, axis=0)
    row_minus_max = row_minus_max + mask
    row_minus_max = row_minus_max - tl.max(row_minus_max, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1) -> torch.Tensor:
    assert input.is_contiguous()
    assert (dim == -1) or (dim == len(input.shape) - 1), "Only dim=-1 is supported"

    use_mask = False if mask is None else True
    input_arg = input.view(-1, input.shape[-1])
    n_rows, n_cols = input_arg.shape
    BLOCK_SIZE = max(triton.next_power_of_2(n_cols), 2)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    output = torch.empty_like(input)
    if use_mask:
        # Broadcast the mask against the input like `input + mask`, without copying it: the kernel reads each
        # row's mask through the broadcast strides of up to three leading dimensions.
        mask = mask.expand(input.shape)
        if mask.stride(-1) != 1 or input.dim() > 4:
            mask = mask.contiguous()
        lead_shape = (1, 1, 1, *input.shape[:-1])[-3:]
        lead_strides = (0, 0, 0, *mask.stride()[:-1])[-3:]
        if input.dim() > 4:
            lead_shape, lead_strides = (1, 1, n_rows), (0, 0, n_cols)
        masked_softmax_kernel[(n_rows, )](
            output,
            input,
            input_arg.stride(0),
            mask,
            *lead_strides,
            lead_shape[1],
            lead_shape[2],
            n_cols,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        softmax_kernel[(n_rows, )](
            output,
            input,
            input_arg.stride(0),
            n_cols,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return output

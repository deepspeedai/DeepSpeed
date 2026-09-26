# Copyright (c) DeepSpeed Team.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
#
# Portions of this file are derived from TorchTitan.
# See THIRD_PARTY_NOTICES.md for the BSD-3-Clause notice.

# DeepSpeed Team
"""
Token reordering and permutation utilities for expert parallelism.

Ported from TorchTitan's Triton kernels and alignment
utilities with adaptations for DeepSpeed:
  - Triton import guarded with try/except; pure-PyTorch fallback provided
  - Alignment config exposed as TOKEN_GROUP_ALIGN_SIZE_M

This module is self-contained: no imports from deepspeed.module_inject
or deepspeed.runtime.
"""

import logging
from typing import Callable

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import Triton; fall back gracefully
# ---------------------------------------------------------------------------

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    logger.info("Triton not available; using pure-PyTorch CPU fallback for "
                "permutation index generation.")

# ---------------------------------------------------------------------------
# Alignment constant
# ---------------------------------------------------------------------------

TOKEN_GROUP_ALIGN_SIZE_M = 8
"""Alignment granularity for token groups in grouped GEMM.

 - bf16: 8  (16 bytes / 2 bytes per elem)
 - fp8:  16 (16 bytes / 1 byte per elem)
 - mxfp8: 32 (scaling block size)
"""

# ---------------------------------------------------------------------------
# Utility: round up
# ---------------------------------------------------------------------------


def _round_up(x: int, y: int) -> int:
    """Round *x* up to the nearest multiple of *y*."""
    return ((x + y - 1) // y) * y


# ===================================================================
# Triton kernel for filling permutation indices
# ===================================================================

if _TRITON_AVAILABLE:

    @triton.jit
    def _fill_indices_kernel(
        tokens_per_expert_group_ptr,
        start_index_values_ptr,
        write_offsets_ptr,
        output_ptr,
        experts_per_rank: tl.constexpr,
        num_ranks: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_programs = tl.num_programs(axis=0)

        for expert_id in range(pid, experts_per_rank, num_programs):
            write_offset = tl.load(write_offsets_ptr + expert_id)

            for r in range(num_ranks):
                i = r * experts_per_rank + expert_id
                start_index = tl.load(start_index_values_ptr + i)
                length = tl.load(tokens_per_expert_group_ptr + i)

                offsets = tl.arange(0, BLOCK_SIZE)
                for chunk_start in range(0, length, BLOCK_SIZE):
                    chunk_offsets = chunk_start + offsets
                    mask = chunk_offsets < length
                    values = start_index + chunk_offsets
                    dest_indices = write_offset + chunk_offsets
                    tl.store(output_ptr + dest_indices, values, mask=mask)

                write_offset += length


# ===================================================================
# Triton wrapper
# ===================================================================


def fill_indices_wrapper(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    block_size: int = 128,
    max_blocks: int = 1024,
) -> torch.Tensor:
    """Launch the Triton kernel to fill permutation indices.

    Falls back to :func:`fill_indices_cpu` when Triton is unavailable.
    """
    if not _TRITON_AVAILABLE:
        return fill_indices_cpu(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            max_len,
        )

    permuted_indices = torch.full((max_len, ), -1, dtype=torch.int32, device=tokens_per_expert_group.device)

    num_blocks = min(experts_per_rank, max_blocks)
    grid = (num_blocks, )

    _fill_indices_kernel[grid](
        tokens_per_expert_group,
        start_index_values,
        write_offsets,
        permuted_indices,
        experts_per_rank,
        num_ranks,
        BLOCK_SIZE=block_size,
    )
    return permuted_indices


# ===================================================================
# CPU reference implementation (always available)
# ===================================================================


def fill_indices_cpu(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
) -> torch.Tensor:
    """Pure-PyTorch CPU reference for filling permutation indices."""
    permuted_indices = torch.full(
        (max_len, ),
        -1,
        dtype=torch.int32,
    )
    for e in range(experts_per_rank):
        write_start = write_offsets[e].item()
        for r in range(num_ranks):
            i = r * experts_per_rank + e
            start_index = start_index_values[i].item()
            length = tokens_per_expert_group[i].item()
            if length > 0:
                end_idx = min(write_start + length, max_len)
                permuted_indices[write_start:end_idx] = torch.arange(
                    start_index,
                    start_index + (end_idx - write_start),
                    dtype=torch.int32,
                )
            write_start += length
    return permuted_indices


# ===================================================================
# generate_permute_indices
# ===================================================================


def generate_permute_indices(
    tokens_per_expert_group: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    alignment: int,
    use_cpu: bool = False,
) -> tuple:
    """Prepare permutation indices and aligned token counts per expert.

    Args:
        tokens_per_expert_group: Token counts for each expert from all ranks,
            shape ``(num_ranks * experts_per_rank,)``.
        experts_per_rank: Number of experts per rank.
        num_ranks: Number of ranks.
        max_len: Maximum length of the output index vector.
        alignment: Alignment for ``m_sizes`` and padding minimum.
        use_cpu: Whether to force the CPU implementation.

    Returns:
        Tuple of:
            - permuted_indices: Index mapping from original to expert-grouped order.
            - m_sizes: Aligned token counts per expert.
            - m_offsets: Cumulative sum of m_sizes.
    """
    # Prefix sum for start indices
    start_index_values = (torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group)

    # Total tokens per expert across all ranks
    total_tokens_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)

    # Pad empty experts to alignment minimum
    total_tokens_per_expert = torch.clamp_min(total_tokens_per_expert, alignment)

    # Align chunk sizes (ceiling division * alignment)
    m_sizes = ((total_tokens_per_expert + alignment - 1) // alignment * alignment).to(torch.int32)

    # Write offsets per local expert
    m_offsets = torch.cumsum(m_sizes, 0)
    write_offsets = m_offsets - m_sizes

    if use_cpu:
        permuted_indices = fill_indices_cpu(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            max_len,
        )
    else:
        permuted_indices = fill_indices_wrapper(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            max_len,
        )

    return permuted_indices, m_sizes, m_offsets.to(torch.int32)


# ===================================================================
# Row permutation by gathers
# ===================================================================

if _TRITON_AVAILABLE:

    @triton.jit
    def _gather_rows_kernel(
        src_ptr,
        index_ptr,
        out_ptr,
        num_out_rows,
        num_src_rows,
        hidden,
        ROWS: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        row_mask = rows < num_out_rows
        index = tl.load(index_ptr + rows, mask=row_mask, other=-1).to(tl.int64)
        valid = (index >= 0) & (index < num_src_rows)
        for start in range(0, hidden, BLOCK_H):
            columns = start + tl.arange(0, BLOCK_H)
            mask = row_mask[:, None] & (columns[None, :] < hidden)
            values = tl.load(src_ptr + index[:, None] * hidden + columns[None, :],
                             mask=mask & valid[:, None],
                             other=0.0)
            tl.store(out_ptr + rows[:, None] * hidden + columns[None, :], values, mask=mask)


_GATHER_ROWS = 4


def gather_rows(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``out[r] = src[index[r]]``, or a zero row where ``index[r]`` is outside ``[0, src.shape[0])``.

    ``generate_permute_indices`` marks padding slots with -1, so this reads a
    permuted, alignment-padded buffer in one pass, without first appending a
    zero row to ``src``.
    """
    src = src.contiguous()
    out = torch.empty((index.numel(), src.shape[1]), dtype=src.dtype, device=src.device)
    if out.numel() == 0:
        return out
    hidden = src.shape[1]
    _gather_rows_kernel[(triton.cdiv(index.numel(), _GATHER_ROWS), )](
        src,
        index,
        out,
        index.numel(),
        src.shape[0],
        hidden,
        ROWS=_GATHER_ROWS,
        BLOCK_H=min(1024, triton.next_power_of_2(hidden)),
    )
    return out


def inverse_permutation(permuted_indices: torch.Tensor, num_rows: int) -> torch.Tensor:
    """For each of ``num_rows`` original rows, the padded position ``permuted_indices`` sent it to.

    Every original row appears exactly once among the valid entries. Padding
    entries are scattered to distinct slots past ``num_rows`` and dropped, so
    the scatter never writes one location twice.
    """
    positions = torch.arange(permuted_indices.numel(), device=permuted_indices.device, dtype=torch.int64)
    indices = permuted_indices.to(torch.int64)
    valid = (indices >= 0) & (indices < num_rows)
    targets = torch.where(valid, indices, num_rows + positions)
    inverse = torch.empty(num_rows + permuted_indices.numel(), dtype=torch.int64, device=permuted_indices.device)
    inverse.scatter_(0, targets, positions)
    return inverse[:num_rows]


class _PermuteRows(torch.autograd.Function):
    """Padded expert-major rows from source-major rows; the backward is the inverse gather."""

    @staticmethod
    def forward(ctx, rows, permuted_indices, inverse):
        ctx.save_for_backward(permuted_indices, inverse)
        return gather_rows(rows, permuted_indices)

    @staticmethod
    def backward(ctx, grad_out):
        permuted_indices, inverse = ctx.saved_tensors
        # Each original row went to exactly one padded position, so its gradient is that position's, unsummed.
        return _UnpermuteRows.apply(grad_out, permuted_indices, inverse), None, None


class _UnpermuteRows(torch.autograd.Function):
    """Source-major rows from padded expert-major rows; the backward is the padded gather."""

    @staticmethod
    def forward(ctx, rows, permuted_indices, inverse):
        ctx.save_for_backward(permuted_indices, inverse)
        return rows.index_select(0, inverse)

    @staticmethod
    def backward(ctx, grad_out):
        permuted_indices, inverse = ctx.saved_tensors
        # Padding positions received no original row, so their gradient is zero.
        return _PermuteRows.apply(grad_out, permuted_indices, inverse), None, None


def permute_rows(rows: torch.Tensor, permuted_indices: torch.Tensor) -> torch.Tensor:
    """``rows`` in the padded order of ``permuted_indices``, with zero rows at padding slots.

    Equal, bit for bit and in both directions, to appending a zero row to
    ``rows`` and indexing it with ``permuted_indices``, without copying
    ``rows`` or accumulating its gradient.
    """
    return _PermuteRows.apply(rows, permuted_indices, inverse_permutation(permuted_indices, rows.shape[0]))


def unpermute_rows(rows: torch.Tensor, permuted_indices: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Invert :func:`permute_rows`: the ``num_rows`` original rows back from the padded order."""
    return _UnpermuteRows.apply(rows, permuted_indices, inverse_permutation(permuted_indices, num_rows))


def permute_rows_supported(rows: torch.Tensor) -> bool:
    return _TRITON_AVAILABLE and rows.device.type == "cuda" and rows.dim() == 2


# ===================================================================
# _permute / _unpermute / indices_padding_wrapper
# ===================================================================


def _permute(
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    ep_degree: int,
    num_local_experts: int,
) -> tuple:
    """Permute tokens into expert-grouped order with alignment padding.

    Returns:
        Tuple of (input_shape, permuted_x, permuted_indices, aligned_counts).
    """
    global TOKEN_GROUP_ALIGN_SIZE_M
    x_padded_per_expert = x.shape[0] + num_local_experts * TOKEN_GROUP_ALIGN_SIZE_M
    padded_max_len = _round_up(x_padded_per_expert, TOKEN_GROUP_ALIGN_SIZE_M)

    with torch.no_grad():
        permuted_indices, num_tokens_per_expert, _offsets = generate_permute_indices(
            num_tokens_per_expert,
            num_local_experts,
            ep_degree,
            padded_max_len,
            TOKEN_GROUP_ALIGN_SIZE_M,
        )

    # Append a single zero-row for safe indexing of padding slots
    x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
    input_shape = x.shape
    x = x[permuted_indices, :]

    return input_shape, x, permuted_indices, num_tokens_per_expert


def _unpermute(
    out: torch.Tensor,
    input_shape: torch.Size,
    permuted_indices: torch.Tensor,
) -> torch.Tensor:
    """Reverse the permutation produced by :func:`_permute`."""
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    # Strip the extra zero-row appended during _permute
    out = out_unpermuted[:-1]
    return out


def indices_padding_wrapper(func: Callable) -> Callable:
    """Decorator that pads / aligns token groups for ``torch._grouped_mm``.

    Wraps an expert-computation function so that each expert's token
    count is a multiple of ``TOKEN_GROUP_ALIGN_SIZE_M``.
    """

    def wrapper(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        num_local_experts = w1.shape[0]
        ep_degree = num_tokens_per_expert.shape[0] // num_local_experts

        input_shape, x, permuted_indices, num_tokens_per_expert = _permute(x, num_tokens_per_expert, ep_degree,
                                                                           num_local_experts)

        out = func(w1, w2, w3, x, num_tokens_per_expert)

        out = _unpermute(out, input_shape, permuted_indices)
        return out

    return wrapper

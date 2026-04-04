# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Token reordering and permutation utilities for expert parallelism.

Ported from TorchTitan's TokenReorderer, Triton kernels, and alignment
utilities with adaptations for DeepSpeed:
  - Triton import guarded with try/except; pure-PyTorch fallback provided
  - Alignment config exposed as TOKEN_GROUP_ALIGN_SIZE_M

This module is self-contained: no imports from deepspeed.module_inject
or deepspeed.runtime.
"""

import logging
from typing import Callable

import torch
import torch.nn as nn

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


# ===================================================================
# TokenReorderer
# ===================================================================


class TokenReorderer(nn.Module):
    """Reorder token indices to match expert order for efficient parallel
    processing.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of experts each token is routed to.
    """

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple:
        """
        Args:
            top_scores: Routing scores, shape ``(T, top_k)``.
            selected_experts_indices: Expert indices, shape ``(T, top_k)``.

        Returns:
            Tuple of:
                - top_scores_experts_sorted ``(T * top_k,)``: scores in
                  expert-sorted order.
                - token_indices_experts_sorted ``(T * top_k,)``: flattened
                  token-slot indices sorted by expert.
                - num_tokens_per_expert ``(num_experts,)``: histogram.
        """
        # histc requires float input on CPU, so cast indices
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1).float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        token_indices_experts_sorted = torch.argsort(selected_experts_indices.view(-1), stable=True)

        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        )

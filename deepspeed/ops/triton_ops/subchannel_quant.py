# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Triton kernels for subchannel (block-scaled) FP8/INT8 matmuls.

Two kinds of kernel live here:

``quantize_blocks``
    Quantizes a ``[B, R, K]`` tensor along ``K`` in blocks of ``block_size`` elements, one FP32 scale per block.
    Before taking the block's absolute maximum it can rotate every ``hadamard_size`` chunk of the block by an
    orthonormal Hadamard matrix. The rotation runs as ``log2(hadamard_size)`` butterfly stages, the Kronecker
    factorization ``H_n = H_2 (x) H_2 (x) ... (x) H_2``, so it costs ``O(n log n)`` per chunk and never leaves
    registers. When both operands of a GEMM are rotated along the reduction dimension the rotations cancel
    (``(A H)(B H)^T = A B^T``); what remains is a block whose outliers are spread over all its elements, which
    a single block scale represents with less error. Rounding is round-to-nearest-even or stochastic.

    ``K`` may be split into groups (MoE experts, whose tokens are the reduction dimension of the weight
    gradient). Blocks then restart at every group start so no block, and so no scale, covers two experts.

``block_scaled_grouped_mm`` / ``block_scaled_grouped_wgrad``
    ``C = sum_k (A_q[:, k] * a_scale[:, k]) (B_q[:, k] * b_scale[:, k])^T`` with one FP8/INT8 tensor-core dot
    per quantization block and FP32 accumulation. The first groups rows of ``A`` by expert (forward and input
    gradient); the second groups the reduction dimension by expert (weight gradient).

``quantize_rows`` / ``row_scaled_grouped_mm`` / ``row_scaled_grouped_wgrad``
    The FP8 fast path: one scale for the whole reduction dimension of every operand row (per expert where the
    reduction is grouped). Rescaling between dot products is what holds the block-scaled GEMMs below BF16 speed;
    here the reduction is a plain FP8 dot loop and the two scales multiply each output element once.

The pure-PyTorch reference in ``deepspeed.runtime.quantization.subchannel`` defines the same numerics and is
the oracle these kernels are tested against.
"""

from __future__ import annotations

import torch

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, triton, tl

#: format name -> (torch storage dtype, largest representable magnitude, mantissa bits, smallest normal exponent)
FORMATS = {
    "e4m3": (torch.float8_e4m3fn, 448.0, 3, -6),
    "e5m2": (torch.float8_e5m2, 57344.0, 2, -14),
    "int8": (torch.int8, 127.0, 0, 0),
}
_FORMAT_ID = {"e4m3": 0, "e5m2": 1, "int8": 2}

if _TRITON_AVAILABLE:

    @triton.jit
    def _hadamard_butterfly(x, ROWS: tl.constexpr, WIDTH: tl.constexpr, LOG_H: tl.constexpr):
        """Multiply every contiguous ``2**LOG_H`` chunk of each row of ``x`` by the Sylvester Hadamard matrix.

        Stage ``i`` pairs element ``j`` with ``j + 2**i`` inside each ``2**(i+1)`` window and replaces the pair
        ``(a, b)`` with ``(a + b, a - b)``: one ``H_2`` factor of the Kronecker product. The result is unnormalized.
        """
        for i in tl.static_range(LOG_H):
            x = tl.reshape(x, (ROWS, WIDTH >> (i + 1), 2, 1 << i))
            x = tl.permute(x, (0, 1, 3, 2))
            a, b = tl.split(x)
            x = tl.join(a + b, a - b)
            x = tl.permute(x, (0, 1, 3, 2))
            x = tl.reshape(x, (ROWS, WIDTH))
        return x

    @triton.jit
    def _uniform_hash(index, seed):
        """A uniform value in [0, 1) per element index: an integer hash, far cheaper than a counter-based RNG.

        Rounding dither needs values that are uniform and uncorrelated with the data, not cryptographic quality.
        """
        h = (index.to(tl.uint32) ^ seed) * 0x9E3779B1
        h = (h ^ (h >> 15)) * 0x85EBCA77
        h = h ^ (h >> 13)
        return (h >> 8).to(tl.float32) * (1.0 / 16777216.0)

    @triton.jit
    def _stochastic_round_fp8(y, noise, MANTISSA_BITS: tl.constexpr, MIN_EXP: tl.constexpr):
        """Round the magnitude down to the format's grid, then up one grid step with probability equal to the
        remainder, so the expected result is ``y``. The grid step, ``2**(exponent - MANTISSA_BITS)``, is built from
        the exponent bits of the float rather than with ``log2``/``exp2``."""
        mag = tl.abs(y)
        exponent = tl.maximum((mag.to(tl.int32, bitcast=True) >> 23) - 127, MIN_EXP)
        step = ((127 - MANTISSA_BITS + exponent) << 23).to(tl.float32, bitcast=True)
        inv_step = ((127 + MANTISSA_BITS - exponent) << 23).to(tl.float32, bitcast=True)
        scaled = mag * inv_step
        low = tl.floor(scaled)
        mag = (low + tl.where(noise < scaled - low, 1.0, 0.0)) * step
        return tl.where(y < 0, -mag, mag)

    @triton.jit
    def _quantize_kernel(
        x_ptr,
        q_ptr,
        s_ptr,
        group_end_ptr,  # [G] int32 cumulative group ends along K (GROUPED only)
        block_end_ptr,  # [G] int32 cumulative quantization-block counts (GROUPED only)
        R,
        K,
        NUM_KB,
        NUM_GROUPS,
        stride_xb,
        stride_xr,
        stride_xk,
        seed,
        GROUPED: tl.constexpr,
        GROUPS_POW2: tl.constexpr,
        FORMAT: tl.constexpr,
        QMAX: tl.constexpr,
        MANTISSA_BITS: tl.constexpr,
        MIN_EXP: tl.constexpr,
        STOCHASTIC: tl.constexpr,
        LOG_H: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        kb = tl.program_id(1)
        batch = tl.program_id(2)

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R

        if GROUPED:
            # Find the group that owns output block ``kb``: the number of groups whose blocks all end at or before it.
            offs_g = tl.arange(0, GROUPS_POW2)
            block_ends = tl.load(block_end_ptr + offs_g, mask=offs_g < NUM_GROUPS, other=2147483647)
            group = tl.sum((block_ends <= kb).to(tl.int32), axis=0)
            in_range = group < NUM_GROUPS
            prev_blocks = tl.load(block_end_ptr + group - 1, mask=in_range & (group > 0), other=0)
            group_start = tl.load(group_end_ptr + group - 1, mask=in_range & (group > 0), other=0)
            group_end = tl.load(group_end_ptr + group, mask=in_range, other=0)
            k_start = group_start + (kb - prev_blocks) * BLOCK_K
            k_end = tl.where(in_range, group_end, 0)
        else:
            k_start = kb * BLOCK_K
            k_end = K

        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask = mask_r[:, None] & (offs_k < k_end)[None, :]
        x_ptrs = (x_ptr + batch.to(tl.int64) * stride_xb + offs_r[:, None].to(tl.int64) * stride_xr +
                  offs_k[None, :].to(tl.int64) * stride_xk)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        if LOG_H > 0:
            x = _hadamard_butterfly(x, BLOCK_R, BLOCK_K, LOG_H)
            x = x * (1.0 / tl.sqrt(float(1 << LOG_H)))

        # Reduce the bit patterns: non-negative floats order like them and a NaN sorts above every number, so a NaN
        # in the block reaches the block's scale and through it every product of the block. A float maximum would
        # drop it, and the clamp below would then store it as a finite value.
        amax = tl.max(tl.abs(x).to(tl.int32, bitcast=True), axis=1).to(tl.float32, bitcast=True)
        scale = tl.where(amax == 0, 1.0, amax / QMAX)
        y = x / scale[:, None]

        if STOCHASTIC:
            out_offs = (offs_r[:, None].to(tl.int64) * NUM_KB + kb) * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
            out_offs += batch.to(tl.int64) * R * NUM_KB * BLOCK_K
            noise = _uniform_hash(out_offs, seed)
            if FORMAT == 2:
                y = tl.floor(y + noise)
            else:
                y = _stochastic_round_fp8(y, noise, MANTISSA_BITS, MIN_EXP)
        elif FORMAT == 2:
            y = tl.where(y < 0, -tl.floor(-y + 0.5), tl.floor(y + 0.5))
        y = tl.minimum(tl.maximum(y, -QMAX), QMAX)

        out_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        q_ptrs = q_ptr + (batch.to(tl.int64) * R + offs_r[:, None]) * (NUM_KB * BLOCK_K) + out_k[None, :]
        tl.store(q_ptrs, y.to(q_ptr.dtype.element_ty), mask=mask_r[:, None])
        # Blocks past the last group are never read by the GEMM; a zero scale keeps them inert anyway.
        if GROUPED:
            scale = tl.where(in_range, scale, 0.0)
        tl.store(s_ptr + (batch.to(tl.int64) * R + offs_r) * NUM_KB + kb, scale, mask=mask_r)

    @triton.jit
    def _block_scaled_tile(a_q_ptr, a_s_ptr, b_q_ptr, b_s_ptr, rows, mask_rows, cols, mask_cols, kb_start, kb_end,
                           stride_aq, stride_as, stride_bq, stride_bs, INT8: tl.constexpr, BLOCK_M: tl.constexpr,
                           BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        """sum over quantization blocks ``kb`` of ``(A_q[rows, kb] . B_q[cols, kb]^T) * a_s[rows, kb] * b_s[cols, kb]``."""
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        offs_k = tl.arange(0, BLOCK_K)
        rows64 = rows.to(tl.int64)
        cols64 = cols.to(tl.int64)
        for kb in range(kb_start, kb_end):
            k = kb * BLOCK_K + offs_k
            a = tl.load(a_q_ptr + rows64[:, None] * stride_aq + k[None, :], mask=mask_rows[:, None], other=0.0)
            b = tl.load(b_q_ptr + cols64[None, :] * stride_bq + k[:, None], mask=mask_cols[None, :], other=0.0)
            a_s = tl.load(a_s_ptr + rows64 * stride_as + kb, mask=mask_rows, other=0.0)
            b_s = tl.load(b_s_ptr + cols64 * stride_bs + kb, mask=mask_cols, other=0.0)
            if INT8:
                partial = tl.dot(a, b, out_dtype=tl.int32).to(tl.float32)
            else:
                partial = tl.dot(a, b)
            acc += partial * a_s[:, None] * b_s[None, :]
        return acc

    @triton.jit
    def _row_tile_group(group_end_ptr, pid_m, NUM_GROUPS, GROUPS_POW2: tl.constexpr, BLOCK_M: tl.constexpr):
        """Map row tile ``pid_m`` to ``(group, first row, group's row end)``.

        Tiles are numbered group by group and every group starts a new tile, so no tile spans two experts. A tile
        index past the last group's tiles returns ``group == NUM_GROUPS``.
        """
        offs_g = tl.arange(0, GROUPS_POW2)
        valid_g = offs_g < NUM_GROUPS
        ends = tl.load(group_end_ptr + offs_g, mask=valid_g, other=0)
        starts = tl.load(group_end_ptr + offs_g - 1, mask=valid_g & (offs_g > 0), other=0)
        tiles = tl.where(valid_g, tl.cdiv(ends - starts, BLOCK_M), 0)
        tile_ends = tl.cumsum(tiles, axis=0)
        group = tl.sum((tile_ends <= pid_m).to(tl.int32), axis=0)
        tile_start = tl.sum(tl.where(offs_g == group, tile_ends - tiles, 0), axis=0)
        row_start = tl.sum(tl.where(offs_g == group, starts, 0), axis=0)
        row_end = tl.sum(tl.where(offs_g == group, ends, 0), axis=0)
        return group, row_start + (pid_m - tile_start) * BLOCK_M, row_end

    def _mm_configs():
        return [
            triton.Config({
                "BLOCK_M": bm,
                "BLOCK_N": bn
            }, num_warps=nw, num_stages=ns) for bm, bn, nw, ns in (
                (64, 64, 4, 4),
                (64, 128, 4, 4),
                (128, 128, 8, 3),
                (128, 64, 4, 4),
            )
        ]

    @triton.autotune(configs=_mm_configs(), key=["N", "NUM_KB", "BLOCK_K", "INT8"])
    @triton.jit
    def _grouped_rows_mm_kernel(
        a_q_ptr,  # [M, NUM_KB * BLOCK_K]
        a_s_ptr,  # [M, NUM_KB]
        b_q_ptr,  # [E, N, NUM_KB * BLOCK_K]
        b_s_ptr,  # [E, N, NUM_KB]
        out_ptr,  # [M, N]
        group_end_ptr,  # [E] int32 cumulative row ends
        M,
        N,
        NUM_KB,
        NUM_GROUPS,
        GROUPS_POW2: tl.constexpr,
        INT8: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """out[m] = A[m] B[group(m)]^T, rows of A grouped by expert (forward and input gradient)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        group, first_row, row_end = _row_tile_group(group_end_ptr, pid_m, NUM_GROUPS, GROUPS_POW2, BLOCK_M)
        if group >= NUM_GROUPS:
            return
        rows = first_row + tl.arange(0, BLOCK_M)
        mask_rows = rows < row_end
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_cols = cols < N

        k_padded = NUM_KB * BLOCK_K
        group64 = group.to(tl.int64)
        acc = _block_scaled_tile(a_q_ptr, a_s_ptr, b_q_ptr + group64 * N * k_padded, b_s_ptr + group64 * N * NUM_KB,
                                 rows, mask_rows, cols, mask_cols, 0, NUM_KB, k_padded, NUM_KB, k_padded, NUM_KB, INT8,
                                 BLOCK_M, BLOCK_N, BLOCK_K)
        out_ptrs = out_ptr + rows[:, None].to(tl.int64) * N + cols[None, :]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_rows[:, None] & mask_cols[None, :])

    @triton.autotune(configs=_mm_configs(), key=["M", "N", "BLOCK_K", "INT8"])
    @triton.jit
    def _grouped_reduction_mm_kernel(
        a_q_ptr,  # [M, NUM_KB * BLOCK_K]
        a_s_ptr,  # [M, NUM_KB]
        b_q_ptr,  # [N, NUM_KB * BLOCK_K]
        b_s_ptr,  # [N, NUM_KB]
        out_ptr,  # [E, M, N]
        block_end_ptr,  # [E] int32 cumulative quantization-block ends along the reduction
        M,
        N,
        NUM_KB,
        INT8: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """out[e] = A[:, blocks(e)] B[:, blocks(e)]^T, the reduction grouped by expert (weight gradient)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        group = tl.program_id(2)

        kb_start = tl.load(block_end_ptr + group - 1, mask=group > 0, other=0)
        kb_end = tl.load(block_end_ptr + group)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_rows = rows < M
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_cols = cols < N
        k_padded = NUM_KB * BLOCK_K
        # An expert that received no tokens gets an exact zero gradient.
        acc = _block_scaled_tile(a_q_ptr, a_s_ptr, b_q_ptr, b_s_ptr, rows, mask_rows, cols, mask_cols, kb_start,
                                 kb_end, k_padded, NUM_KB, k_padded, NUM_KB, INT8, BLOCK_M, BLOCK_N, BLOCK_K)
        out_ptrs = out_ptr + group.to(tl.int64) * M * N + rows[:, None].to(tl.int64) * N + cols[None, :]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_rows[:, None] & mask_cols[None, :])

    @triton.jit
    def _reduction_tile(group_end_ptr, pid_k, K, NUM_GROUPS, GROUPED: tl.constexpr, GROUPS_POW2: tl.constexpr,
                        BLOCK_K: tl.constexpr):
        """Map reduction tile ``pid_k`` to ``(group, tile start, first valid element, end of valid elements)``.

        Tiles sit on the global ``BLOCK_K`` grid, which keeps their loads aligned, and are numbered group by group:
        a grid tile that two groups share is visited once per group, each time masked to that group's elements.
        A tile index past the last group's tiles returns an empty span.
        """
        if GROUPED:
            offs_g = tl.arange(0, GROUPS_POW2)
            valid_g = offs_g < NUM_GROUPS
            ends = tl.load(group_end_ptr + offs_g, mask=valid_g, other=0)
            starts = tl.load(group_end_ptr + offs_g - 1, mask=valid_g & (offs_g > 0), other=0)
            tiles = tl.where(valid_g, tl.cdiv(ends, BLOCK_K) - starts // BLOCK_K, 0)
            tile_ends = tl.cumsum(tiles, axis=0)
            group = tl.sum((tile_ends <= pid_k).to(tl.int32), axis=0)
            tile_start = tl.sum(tl.where(offs_g == group, tile_ends - tiles, 0), axis=0)
            k_start = tl.sum(tl.where(offs_g == group, starts, 0), axis=0)
            k_end = tl.sum(tl.where(offs_g == group, ends, 0), axis=0)
            k0 = (k_start // BLOCK_K + pid_k - tile_start) * BLOCK_K
        else:
            group = 0
            k_start = 0
            k_end = K
            k0 = pid_k * BLOCK_K
        return group, k0, k_start, k_end

    @triton.jit
    def _row_amax_kernel(
        x_ptr,  # [B, R, K], any strides
        amax_ptr,  # [B, R, NUM_GROUPS] int32, zero-initialized: the bits of each row's FP32 absolute maximum
        group_end_ptr,  # [NUM_GROUPS] int32 cumulative group ends along K (GROUPED only)
        R,
        TOTAL_ROWS,
        K,
        NUM_GROUPS,
        stride_xb,
        stride_xr,
        stride_xk,
        GROUPED: tl.constexpr,
        GROUPS_POW2: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        group, k0, k_start, k_end = _reduction_tile(group_end_ptr, tl.program_id(1), K, NUM_GROUPS, GROUPED,
                                                    GROUPS_POW2, BLOCK_K)
        rows = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
        mask_r = rows < TOTAL_ROWS
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask = mask_r[:, None] & ((offs_k >= k_start) & (offs_k < k_end))[None, :]
        x_ptrs = (x_ptr + (rows // R)[:, None] * stride_xb + (rows % R)[:, None] * stride_xr +
                  offs_k[None, :].to(tl.int64) * stride_xk)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        # Non-negative floats order like their bit patterns, so an integer max reduces a tile and an integer atomic
        # max merges the tiles of a row. A NaN sorts above every number: it becomes the row's maximum, and so
        # reaches every product of the row through its scale, where a float maximum would drop it.
        amax_bits = tl.max(tl.abs(x).to(tl.int32, bitcast=True), axis=1)
        tl.atomic_max(amax_ptr + rows * NUM_GROUPS + group, amax_bits, mask=mask_r & (amax_bits > 0))

    @triton.jit
    def _quantize_rows_kernel(
        x_ptr,  # [B, R, K], any strides
        q_ptr,  # [B, R, K]
        amax_ptr,  # [B, R, NUM_GROUPS] FP32
        group_end_ptr,  # [NUM_GROUPS] int32 cumulative group ends along K (GROUPED only)
        R,
        TOTAL_ROWS,
        K,
        NUM_GROUPS,
        stride_xb,
        stride_xr,
        stride_xk,
        seed,
        GROUPED: tl.constexpr,
        GROUPS_POW2: tl.constexpr,
        QMAX: tl.constexpr,
        MANTISSA_BITS: tl.constexpr,
        MIN_EXP: tl.constexpr,
        STOCHASTIC: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """q = fp8(x * QMAX / amax[row, group]); the row's scale is ``amax / QMAX``."""
        group, k0, k_start, k_end = _reduction_tile(group_end_ptr, tl.program_id(1), K, NUM_GROUPS, GROUPED,
                                                    GROUPS_POW2, BLOCK_K)
        rows = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
        mask_r = rows < TOTAL_ROWS
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask = mask_r[:, None] & ((offs_k >= k_start) & (offs_k < k_end))[None, :]
        x_ptrs = (x_ptr + (rows // R)[:, None] * stride_xb + (rows % R)[:, None] * stride_xr +
                  offs_k[None, :].to(tl.int64) * stride_xk)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        amax = tl.load(amax_ptr + rows * NUM_GROUPS + group, mask=mask_r & (k_end > k_start), other=0.0)
        y = x * tl.where(amax > 0, QMAX / amax, 0.0)[:, None]
        q_offs = rows[:, None] * K + offs_k[None, :]
        if STOCHASTIC:
            y = _stochastic_round_fp8(y, _uniform_hash(q_offs, seed), MANTISSA_BITS, MIN_EXP)
        y = tl.minimum(tl.maximum(y, -QMAX), QMAX)
        tl.store(q_ptr + q_offs, y.to(q_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _row_scaled_rows_mm_kernel(
        a_q_ptr,  # [M, K]
        a_s_ptr,  # [M] row maxima
        b_q_ptr,  # [E, N, K]
        b_s_ptr,  # [E, N] row maxima
        out_ptr,  # [M, N]
        group_end_ptr,  # [E] int32 cumulative row ends
        N,
        K,
        NUM_GROUPS,
        SCALE_UNIT,  # 1 / QMAX^2: the stored scales are row maxima
        GROUPS_POW2: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """out[m] = A[m] B[group(m)]^T with one scale per operand row, applied once after the reduction."""
        pid_n = tl.program_id(1)
        group, first_row, row_end = _row_tile_group(group_end_ptr, tl.program_id(0), NUM_GROUPS, GROUPS_POW2, BLOCK_M)
        if group >= NUM_GROUPS:
            return
        rows = first_row + tl.arange(0, BLOCK_M)
        mask_rows = rows < row_end
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_cols = cols < N
        b_rows = group.to(tl.int64) * N + cols

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            a = tl.load(a_q_ptr + rows[:, None].to(tl.int64) * K + offs_k[None, :],
                        mask=mask_rows[:, None] & mask_k[None, :],
                        other=0.0)
            b = tl.load(b_q_ptr + b_rows[None, :] * K + offs_k[:, None],
                        mask=mask_cols[None, :] & mask_k[:, None],
                        other=0.0)
            acc = tl.dot(a, b, acc)
        a_s = tl.load(a_s_ptr + rows, mask=mask_rows, other=0.0)
        b_s = tl.load(b_s_ptr + b_rows, mask=mask_cols, other=0.0)
        acc = acc * (a_s * SCALE_UNIT)[:, None] * b_s[None, :]
        out_ptrs = out_ptr + rows[:, None].to(tl.int64) * N + cols[None, :]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_rows[:, None] & mask_cols[None, :])

    @triton.jit
    def _row_scaled_reduction_mm_kernel(
        a_q_ptr,  # [M, K]
        a_s_ptr,  # [M, E] row maxima per expert
        b_q_ptr,  # [N, K]
        b_s_ptr,  # [N, E] row maxima per expert
        out_ptr,  # [E, M, N]
        group_end_ptr,  # [E] int32 cumulative ends along K (GROUPED only)
        M,
        N,
        K,
        NUM_GROUPS,
        SCALE_UNIT,  # 1 / QMAX^2
        GROUPED: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """out[e] = A[:, span(e)] B[:, span(e)]^T with one scale per operand row and expert (weight gradient)."""
        group = tl.program_id(2)
        rows = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        mask_rows = rows < M
        cols = (tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        mask_cols = cols < N
        a_rows = a_q_ptr + rows[:, None] * K
        b_rows = b_q_ptr + cols[None, :] * K
        offs = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if GROUPED:
            # The fast loop needs loads on the BLOCK_K grid and no mask that depends on the loaded span bounds
            # (either costs an order of magnitude), so it covers the grid tiles strictly inside the expert's span
            # and the two boundary tiles are added separately under a mask.
            k_start = tl.load(group_end_ptr + group - 1, mask=group > 0, other=0)
            k_end = tl.load(group_end_ptr + group)
            head = k_start // BLOCK_K
            tail = k_end // BLOCK_K
            for kb in range(head + 1, tail):
                offs_k = kb * BLOCK_K + offs
                a = tl.load(a_rows + offs_k[None, :], mask=mask_rows[:, None], other=0.0)
                b = tl.load(b_rows + offs_k[:, None], mask=mask_cols[None, :], other=0.0)
                acc = tl.dot(a, b, acc)
            for boundary in range(2):
                offs_k = tl.where(boundary == 0, head, tail) * BLOCK_K + offs
                # A span inside one grid tile has head == tail; count it once. An empty span masks everything,
                # which gives an expert without tokens an exact zero gradient.
                mask_k = (offs_k >= k_start) & (offs_k < k_end) & ((boundary == 0) | (tail > head))
                a = tl.load(a_rows + offs_k[None, :], mask=mask_rows[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_rows + offs_k[:, None], mask=mask_cols[None, :] & mask_k[:, None], other=0.0)
                acc = tl.dot(a, b, acc)
        else:
            for k0 in range(0, K, BLOCK_K):
                offs_k = k0 + offs
                mask_k = offs_k < K
                a = tl.load(a_rows + offs_k[None, :], mask=mask_rows[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_rows + offs_k[:, None], mask=mask_cols[None, :] & mask_k[:, None], other=0.0)
                acc = tl.dot(a, b, acc)
        a_s = tl.load(a_s_ptr + rows * NUM_GROUPS + group, mask=mask_rows, other=0.0)
        b_s = tl.load(b_s_ptr + cols * NUM_GROUPS + group, mask=mask_cols, other=0.0)
        acc = acc * (a_s * SCALE_UNIT)[:, None] * b_s[None, :]
        out_ptrs = out_ptr + group.to(tl.int64) * M * N + rows[:, None] * N + cols[None, :]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_rows[:, None] & mask_cols[None, :])


def _log2_exact(value: int) -> int:
    log = value.bit_length() - 1
    if value <= 0 or (1 << log) != value:
        raise ValueError(f"expected a positive power of two, got {value}")
    return log


def quantize_blocks(x: torch.Tensor,
                    *,
                    fmt: str,
                    block_size: int,
                    hadamard_size: int = 0,
                    stochastic: bool = False,
                    group_ends: torch.Tensor | None = None,
                    seed: int = 0):
    """Quantize ``x`` (``[B, R, K]``, any strides) along ``K``; see the module docstring.

    Returns ``(q, scale, block_ends)``: ``q`` is ``[B, R, NUM_KB * block_size]`` in the format's storage dtype,
    ``scale`` is ``[B, R, NUM_KB]`` FP32 and ``block_ends`` the cumulative block count per group (``None`` when
    ungrouped). Elements past ``K`` or past a group's end quantize to zero.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("subchannel quantization kernels require Triton")
    if x.dim() != 3:
        raise ValueError(f"expected a [B, R, K] tensor, got shape {tuple(x.shape)}")
    storage_dtype, qmax, mantissa_bits, min_exp = FORMATS[fmt]
    log_h = _log2_exact(hadamard_size) if hadamard_size else 0
    batch, rows, k = x.shape

    block_ends = None
    num_groups = 1
    if group_ends is None:
        num_kb = triton.cdiv(k, block_size)
        dummy = torch.empty(1, dtype=torch.int32, device=x.device)
        group_ends_arg = block_ends_arg = dummy
    else:
        num_groups = group_ends.shape[0]
        group_ends_arg = group_ends.to(torch.int32).contiguous()
        sizes = torch.diff(group_ends_arg, prepend=group_ends_arg.new_zeros(1))
        block_ends = torch.cumsum(torch.div(sizes + block_size - 1, block_size, rounding_mode="floor"),
                                  dim=0,
                                  dtype=torch.int32)
        block_ends_arg = block_ends
        # Every group can end in a partial block, so this many blocks always fit; the rest stay zero.
        num_kb = triton.cdiv(k, block_size) + num_groups

    q = torch.empty((batch, rows, num_kb * block_size), dtype=storage_dtype, device=x.device)
    scale = torch.empty((batch, rows, num_kb), dtype=torch.float32, device=x.device)
    if q.numel() == 0:
        return q, scale, block_ends
    block_r = max(1, min(64, 4096 // block_size))
    grid = (triton.cdiv(rows, block_r), num_kb, batch)
    _quantize_kernel[grid](
        x,
        q,
        scale,
        group_ends_arg,
        block_ends_arg,
        rows,
        k,
        num_kb,
        num_groups,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        seed,
        GROUPED=group_ends is not None,
        GROUPS_POW2=triton.next_power_of_2(num_groups),
        FORMAT=_FORMAT_ID[fmt],
        QMAX=qmax,
        MANTISSA_BITS=mantissa_bits,
        MIN_EXP=min_exp,
        STOCHASTIC=stochastic,
        LOG_H=log_h,
        BLOCK_R=block_r,
        BLOCK_K=block_size,
    )
    return q, scale, block_ends


def block_scaled_grouped_mm(a_q, a_scale, b_q, b_scale, group_ends, block_size: int, out_dtype: torch.dtype):
    """``out[m] = A[m] B[e]^T`` for rows ``m`` of group ``e``; ``a_*`` are ``[M, ...]``, ``b_*`` ``[E, N, ...]``."""
    m = a_q.shape[0]
    num_groups, n, _ = b_q.shape
    num_kb = a_scale.shape[-1]
    out = torch.empty((m, n), dtype=out_dtype, device=a_q.device)
    if m == 0 or n == 0:
        return out
    group_ends = group_ends.to(torch.int32).contiguous()

    def grid(meta):
        # Every group may end in a partial tile.
        return (triton.cdiv(m, meta["BLOCK_M"]) + num_groups, triton.cdiv(n, meta["BLOCK_N"]))

    _grouped_rows_mm_kernel[grid](a_q,
                                  a_scale,
                                  b_q,
                                  b_scale,
                                  out,
                                  group_ends,
                                  m,
                                  n,
                                  num_kb,
                                  num_groups,
                                  GROUPS_POW2=triton.next_power_of_2(num_groups),
                                  INT8=a_q.dtype == torch.int8,
                                  BLOCK_K=block_size)
    return out


def block_scaled_grouped_wgrad(a_q, a_scale, b_q, b_scale, block_ends, block_size: int, out_dtype: torch.dtype):
    """``out[e] = A[:, blocks(e)] B[:, blocks(e)]^T``; ``a_*`` are ``[M, ...]``, ``b_*`` ``[N, ...]``."""
    m = a_q.shape[0]
    n = b_q.shape[0]
    num_groups = block_ends.shape[0]
    out = torch.empty((num_groups, m, n), dtype=out_dtype, device=a_q.device)
    if out.numel() == 0:
        return out

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]), num_groups)

    _grouped_reduction_mm_kernel[grid](a_q,
                                       a_scale,
                                       b_q,
                                       b_scale,
                                       out,
                                       block_ends,
                                       m,
                                       n,
                                       a_scale.shape[-1],
                                       INT8=a_q.dtype == torch.int8,
                                       BLOCK_K=block_size)
    return out


def quantize_rows(x: torch.Tensor,
                  *,
                  fmt: str,
                  stochastic: bool = False,
                  group_ends: torch.Tensor | None = None,
                  seed: int = 0):
    """Quantize ``x`` (``[B, R, K]``, any strides) to FP8 along ``K`` with one scale per row.

    With ``group_ends`` (int32 cumulative ends of groups along ``K``) every row gets one scale per group.
    Returns ``(q, amax)``: ``q`` is ``[B, R, K]`` in the format's storage dtype and ``amax`` the ``[B, R, G]`` FP32
    absolute maxima; a row's scale is its ``amax`` over the format's largest magnitude. Elements past the last
    group's end are left unwritten.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("subchannel quantization kernels require Triton")
    storage_dtype, qmax, mantissa_bits, min_exp = FORMATS[fmt]
    batch, rows, k = x.shape
    grouped = group_ends is not None
    num_groups = group_ends.shape[0] if grouped else 1
    q = torch.empty((batch, rows, k), dtype=storage_dtype, device=x.device)
    amax_bits = torch.zeros((batch, rows, num_groups), dtype=torch.int32, device=x.device)
    amax = amax_bits.view(torch.float32)
    if q.numel() == 0:
        return q, amax
    total_rows = batch * rows
    # Measured best for row-major and for transposed inputs alike: what matters is that the FP8 rows written
    # are long.
    block_r, block_k = 32, 128
    # A grid tile shared by two groups is visited once for each.
    grid = (triton.cdiv(total_rows, block_r), triton.cdiv(k, block_k) + (num_groups if grouped else 0))
    shared = dict(GROUPED=grouped,
                  GROUPS_POW2=triton.next_power_of_2(num_groups),
                  BLOCK_R=block_r,
                  BLOCK_K=block_k,
                  num_warps=4)
    group_ends_arg = group_ends if grouped else amax_bits
    strides = (x.stride(0), x.stride(1), x.stride(2))
    _row_amax_kernel[grid](x, amax_bits, group_ends_arg, rows, total_rows, k, num_groups, *strides, **shared)
    _quantize_rows_kernel[grid](x,
                                q,
                                amax,
                                group_ends_arg,
                                rows,
                                total_rows,
                                k,
                                num_groups,
                                *strides,
                                seed,
                                QMAX=qmax,
                                MANTISSA_BITS=mantissa_bits,
                                MIN_EXP=min_exp,
                                STOCHASTIC=stochastic,
                                **shared)
    return q, amax


def _row_scaled_tile(m: int, n: int):
    # Wide tiles pay off only when both output dimensions give the grid enough of them.
    return 128, (256 if min(m, n) >= 2048 else 128)


def row_scaled_grouped_mm(a_q, a_amax, b_q, b_amax, group_ends, out_dtype: torch.dtype, *, fmt: str):
    """``out[m] = A[m] B[e]^T`` for rows ``m`` of group ``e``, operands from :func:`quantize_rows`.

    ``a_q`` is ``[M, K]`` with ``a_amax`` ``[M]``; ``b_q`` is ``[E, N, K]`` with ``b_amax`` ``[E, N]``.
    """
    m, k = a_q.shape
    num_groups, n, _ = b_q.shape
    out = torch.empty((m, n), dtype=out_dtype, device=a_q.device)
    if m == 0 or n == 0:
        return out
    block_m, block_n = _row_scaled_tile(m, n)
    # Every group may end in a partial tile.
    grid = (triton.cdiv(m, block_m) + num_groups, triton.cdiv(n, block_n))
    _row_scaled_rows_mm_kernel[grid](a_q,
                                     a_amax,
                                     b_q,
                                     b_amax,
                                     out,
                                     group_ends,
                                     n,
                                     k,
                                     num_groups,
                                     1.0 / FORMATS[fmt][1]**2,
                                     GROUPS_POW2=triton.next_power_of_2(num_groups),
                                     BLOCK_M=block_m,
                                     BLOCK_N=block_n,
                                     BLOCK_K=128,
                                     num_warps=8,
                                     num_stages=3)
    return out


def row_scaled_grouped_wgrad(a_q, a_amax, b_q, b_amax, group_ends, out_dtype: torch.dtype, *, fmt: str):
    """``out[e] = A[:, span(e)] B[:, span(e)]^T``, operands from :func:`quantize_rows` with the same ``group_ends``.

    ``a_q`` is ``[M, K]`` with ``a_amax`` ``[M, E]``; ``b_q`` is ``[N, K]`` with ``b_amax`` ``[N, E]``.
    ``group_ends=None`` is the single-group case.
    """
    m, k = a_q.shape
    n = b_q.shape[0]
    grouped = group_ends is not None
    num_groups = group_ends.shape[0] if grouped else 1
    out = torch.empty((num_groups, m, n), dtype=out_dtype, device=a_q.device)
    if out.numel() == 0:
        return out
    block_m, block_n = _row_scaled_tile(m, n)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), num_groups)
    _row_scaled_reduction_mm_kernel[grid](a_q,
                                          a_amax,
                                          b_q,
                                          b_amax,
                                          out,
                                          group_ends if grouped else a_amax,
                                          m,
                                          n,
                                          k,
                                          num_groups,
                                          1.0 / FORMATS[fmt][1]**2,
                                          GROUPED=grouped,
                                          BLOCK_M=block_m,
                                          BLOCK_N=block_n,
                                          BLOCK_K=128,
                                          num_warps=8,
                                          num_stages=3)
    return out

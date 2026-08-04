# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Triton grouped GEMM as a drop-in replacement for ``torch._grouped_mm``.

Motivation
----------
On NVIDIA Ampere GPUs (sm80 / sm86, e.g. A100 / A6000 / A40 / RTX 30xx) the
native ``torch._grouped_mm`` has no fused CUTLASS/cuBLASLt grouped-GEMM kernel
and silently falls back to a Python-side ``for`` loop that issues one
``at::mm`` per group plus a device->host sync on ``offs`` (see
``aten/src/ATen/native/GroupedMMUtils.h::_grouped_mm_fallback``). That is slow
for MoE expert computation which calls grouped GEMM 3x per layer.

This module implements the same grouped GEMM with Triton so that a single
fused kernel handles all groups with no per-group kernel launch and no
device->host synchronization, giving a real grouped-GEMM path on sm80/sm86.

Supported semantics (mirrors ``torch._grouped_mm`` for the 2D x 3D case used
by DeepSpeed AutoEP experts)
---------------------------------------------------------------------------
    out = grouped_mm(mat_a, mat_b, offs)

    mat_a : ``[M, K]``      (2D, row-major contiguous)
    mat_b : ``[E, K, N]``   (3D; arbitrary strides, so transposed views work)
    offs  : ``[E]`` int32   cumulative row boundaries into ``mat_a`` along M,
                            i.e. group ``g`` owns rows ``offs[g-1] : offs[g]``
                            (``offs[-1] == M``). Empty groups are allowed.
    out   : ``[M, N]``      same floating dtype as inputs.

Only ``float16``, ``bfloat16`` and ``float32`` inputs are supported, and both
operands must share the same dtype. Matmuls accumulate in fp32 and the result
is cast back to the input dtype, matching ``torch._grouped_mm`` numerics.

Autograd is supported (forward + backward for both ``mat_a`` and ``mat_b``).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

__all__ = ["group_gemm_triton", "is_available"]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def is_available() -> bool:
    """Return True if the Triton grouped-GEMM path can be used."""
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:

    def _gmm_configs():
        # Forward / grad_a kernel: output [M, N] tiled by BLOCK_M x BLOCK_N,
        # reduction over K in BLOCK_K steps.
        #   BLOCK_M (M/token tile): 32 / 64 / 128
        #   BLOCK_N (N output tile): 128 / 256
        #   BLOCK_K (K reduction):   32 / 64
        return [
            triton.Config({
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "BLOCK_K": bk
            }, num_warps=nw, num_stages=ns) for bm, bn, bk, nw, ns in (
                (32, 128, 32, 8, 4),
                (64, 128, 32, 8, 4),
                (64, 128, 64, 8, 4),
                (128, 128, 32, 8, 4),
                (64, 256, 32, 8, 3),
                (128, 256, 32, 8, 3),
            )
        ]

    @triton.autotune(configs=_gmm_configs(), key=["KC", "NO", "NUM_GROUPS"])
    @triton.jit
    def _group_gemm_kernel(
        a_ptr,  # [M, KC]
        b_ptr,  # [NUM_GROUPS, KC, NO]
        out_ptr,  # [M, NO]
        group_m_start_ptr,  # [NUM_GROUPS] int32, exclusive prefix start row of each group
        group_m_size_ptr,  # [NUM_GROUPS] int32, rows per group
        M,
        KC,
        NO,
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_om,
        stride_on,
        NUM_GROUPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """out[m, n] = sum_k a[m, k] * b[group(m), k, n], grouped along M."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Map this M-tile to its group. Per-group tile counts are derived here
        # from group sizes and the constexpr BLOCK_M, so the host does not need
        # BLOCK_M-dependent precompute (keeps autotune over BLOCK_M correct).
        # NUM_GROUPS is constexpr -> loop unrolls; tl.where avoids divergence.
        selected = -1
        m_start = 0
        m_size = 0
        local_tile = 0
        prev_prefix = 0
        for g in range(NUM_GROUPS):
            size_g = tl.load(group_m_size_ptr + g)
            tiles_g = tl.cdiv(size_g, BLOCK_M)
            prefix_g = prev_prefix + tiles_g
            is_here = (selected < 0) & (pid_m < prefix_g)
            selected = tl.where(is_here, g, selected)
            m_start = tl.where(is_here, tl.load(group_m_start_ptr + g), m_start)
            m_size = tl.where(is_here, size_g, m_size)
            local_tile = tl.where(is_here, pid_m - prev_prefix, local_tile)
            prev_prefix = prefix_g

        # Programs beyond the real total number of tiles do nothing.
        if selected < 0:
            return

        row0 = m_start + local_tile * BLOCK_M
        offs_m = row0 + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < (m_start + m_size)
        mask_n = offs_n < NO

        b_base = b_ptr + selected * stride_be

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, KC, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < KC
            a_tile = tl.load(
                a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            b_tile = tl.load(
                b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            acc += tl.dot(a_tile, b_tile)  # For FP32 inputs, it use TF32 tensor core for better performance on sm8x

        out = acc.to(out_ptr.dtype.element_ty)
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            out,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    def _dw_configs():
        # Weight-grad kernel: output [E, K, N] tiled by BLOCK_K x BLOCK_N,
        # reduction over M (tokens in the group) in BLOCK_M steps.
        #   BLOCK_K (K output tile): 32 / 64 / 128
        #   BLOCK_N (N output tile): 128 / 256
        #   BLOCK_M (M reduction):   32 / 64
        return [
            triton.Config({
                "BLOCK_K": bk,
                "BLOCK_N": bn,
                "BLOCK_M": bm
            }, num_warps=nw, num_stages=ns) for bk, bn, bm, nw, ns in (
                (32, 128, 32, 8, 4),
                (64, 128, 32, 8, 4),
                (64, 128, 64, 8, 4),
                (128, 128, 32, 8, 4),
                (64, 256, 32, 8, 3),
                (128, 256, 32, 8, 3),
            )
        ]

    @triton.autotune(configs=_dw_configs(), key=["K", "NO", "NUM_GROUPS"])
    @triton.jit
    def _group_gemm_dw_kernel(
        a_ptr,  # [M, K]
        g_ptr,  # [M, NO]
        out_ptr,  # [NUM_GROUPS, K, NO]
        group_m_start_ptr,  # [NUM_GROUPS] int32
        group_m_size_ptr,  # [NUM_GROUPS] int32
        K,
        NO,
        stride_am,
        stride_ak,
        stride_gm,
        stride_gn,
        stride_oe,
        stride_ok,
        stride_on,
        NUM_GROUPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """For each group e, compute out[e] = a_e.T @ g_e by reducing over its M rows.

        a_e.T @ g_e: [K, M_e] @ [M_e, NO] -> [K, NO], so
        out[e, k, n] = sum_{m in group e} a[m, k] * g[m, n].
        """
        pid_e = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_n = tl.program_id(2)

        m_start = tl.load(group_m_start_ptr + pid_e)
        m_size = tl.load(group_m_size_ptr + pid_e)

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_k = offs_k < K
        mask_n = offs_n < NO

        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
        num_steps = tl.cdiv(m_size, BLOCK_M)
        for step in range(0, num_steps):
            row = step * BLOCK_M + tl.arange(0, BLOCK_M)
            mask_m = row < m_size
            offs_m = m_start + row
            a_tile = tl.load(
                a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )  # [BLOCK_M, BLOCK_K]
            g_tile = tl.load(
                g_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )  # [BLOCK_M, BLOCK_N]
            acc += tl.dot(
                tl.trans(a_tile),
                g_tile)  # [BLOCK_K, BLOCK_N]; For FP32 inputs, it use TF32 tensor core for better performance on sm8x

        out = acc.to(out_ptr.dtype.element_ty)
        out_base = out_ptr + pid_e * stride_oe
        tl.store(
            out_base + offs_k[:, None] * stride_ok + offs_n[None, :] * stride_on,
            out,
            mask=mask_k[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _group_meta_kernel(
        offs_ptr,  # [E] int32, cumulative row boundaries (offs[-1] == M)
        m_start_ptr,  # [E] int32 out: exclusive prefix start row of each group
        m_size_ptr,  # [E] int32 out: rows per group
        E,
        BLOCK: tl.constexpr,
    ):
        """m_start[g] = offs[g-1] (0 for g==0); m_size[g] = offs[g] - m_start[g]."""
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        mask = idx < E
        cur = tl.load(offs_ptr + idx, mask=mask, other=0)
        # Previous cumulative boundary; group 0 starts at 0 (masked load -> other=0).
        prev = tl.load(offs_ptr + idx - 1, mask=mask & (idx > 0), other=0)
        tl.store(m_start_ptr + idx, prev, mask=mask)
        tl.store(m_size_ptr + idx, cur - prev, mask=mask)


_GROUP_META_BLOCK = 256


def _group_meta(offs: torch.Tensor):
    """Compute per-group m-start (exclusive prefix) and m-size on device (no D2H sync).

    A single Triton kernel reads ``offs`` and writes both outputs, replacing the
    ``zeros`` + ``cat`` + subtract sequence (which launched several small kernels).
    """
    offs_i = offs.to(torch.int32).contiguous()
    E = offs_i.shape[0]
    m_start = torch.empty_like(offs_i)
    m_size = torch.empty_like(offs_i)
    grid = (triton.cdiv(E, _GROUP_META_BLOCK), )
    _group_meta_kernel[grid](offs_i, m_start, m_size, E, BLOCK=_GROUP_META_BLOCK)
    return m_start, m_size


def _validate(mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor, trans_b: bool) -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("group_gemm_triton requires Triton, which is not available.")
    if mat_a.dim() != 2 or mat_b.dim() != 3:
        raise NotImplementedError(f"Only 2D x 3D grouped GEMM is supported, got mat_a.dim()={mat_a.dim()}, "
                                  f"mat_b.dim()={mat_b.dim()}.")
    if not mat_b.is_contiguous():
        raise ValueError("group_gemm_triton requires a contiguous mat_b. To compute a @ b^T, "
                         "pass the weight in its native [E, N, K] layout with trans_b=True "
                         "instead of a non-contiguous .transpose(-2, -1) view.")
    if mat_a.dtype != mat_b.dtype:
        raise ValueError(f"mat_a and mat_b must share dtype, got {mat_a.dtype} vs {mat_b.dtype}.")
    if mat_a.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype {mat_a.dtype}; supported: {_SUPPORTED_DTYPES}.")
    if offs.dim() != 1 or offs.shape[0] != mat_b.shape[0]:
        raise ValueError(f"offs must be 1D of length E={mat_b.shape[0]}, got shape {tuple(offs.shape)}.")
    # Contraction dim K in mat_b: shape[2] when trans_b (native [E, N, K]),
    # else shape[1] (kernel layout [E, K, N]).
    mat_b_k = mat_b.shape[2] if trans_b else mat_b.shape[1]
    if mat_a.shape[1] != mat_b_k:
        raise ValueError(f"Contraction dim mismatch: mat_a K={mat_a.shape[1]} vs mat_b K={mat_b_k}.")


def _group_gemm(mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Forward-style grouped GEMM: a[M,KC] x b[E,KC,NO] -> out[M,NO], grouped over M."""
    M, KC = mat_a.shape
    E, _, NO = mat_b.shape
    out = torch.empty((M, NO), dtype=mat_a.dtype, device=mat_a.device)
    if M == 0:
        return out

    m_start, m_size = _group_meta(offs)

    def grid(meta):
        block_m = meta["BLOCK_M"]
        # Safe static upper bound on total M-tiles: cdiv(M, BLOCK_M) + E
        # (each group rounds up to at most one extra tile). Excess programs exit.
        max_m_tiles = (M + block_m - 1) // block_m + E
        return (max_m_tiles, triton.cdiv(NO, meta["BLOCK_N"]))

    _group_gemm_kernel[grid](
        mat_a,
        mat_b,
        out,
        m_start,
        m_size,
        M,
        KC,
        NO,
        mat_a.stride(0),
        mat_a.stride(1),
        mat_b.stride(0),
        mat_b.stride(1),
        mat_b.stride(2),
        out.stride(0),
        out.stride(1),
        NUM_GROUPS=E,
    )
    return out


def _grouped_dw(mat_a: torch.Tensor, grad_out: torch.Tensor, offs: torch.Tensor, E: int) -> torch.Tensor:
    """For each group e, compute mat_a_e.T @ grad_out_e.

    ``[K, M_e] @ [M_e, NO] -> [K, NO]``; stacking all groups produces
    ``out[E, K, NO]``. The reduction dimension is the number of rows ``M_e``
    assigned to each group by ``offs``.
    """
    M, K = mat_a.shape
    NO = grad_out.shape[1]
    if M == 0:
        # No tokens routed anywhere -> every weight gradient is zero.
        return torch.zeros((E, K, NO), dtype=mat_a.dtype, device=mat_a.device)

    out = torch.empty((E, K, NO), dtype=mat_a.dtype, device=mat_a.device)

    m_start, m_size = _group_meta(offs)

    def grid(meta):
        return (E, triton.cdiv(K, meta["BLOCK_K"]), triton.cdiv(NO, meta["BLOCK_N"]))

    _group_gemm_dw_kernel[grid](
        mat_a,
        grad_out,
        out,
        m_start,
        m_size,
        K,
        NO,
        mat_a.stride(0),
        mat_a.stride(1),
        grad_out.stride(0),
        grad_out.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        NUM_GROUPS=E,
    )
    return out


class _GroupGemmFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mat_a, mat_b, offs, trans_b):
        # Inputs are already validated in the public group_gemm_triton() wrapper.
        #
        # trans_b=False: mat_b is [E, K, N], out = a @ b  (per group).
        # trans_b=True : mat_b is [E, N, K] (e.g. an expert weight in its native
        #   layout), out = a @ b^T. The transpose is applied here as a strided
        #   VIEW inside the Function, so it never lands on the autograd tape.
        #   Backward then produces grad_b directly in mat_b's [E, N, K] layout,
        #   avoiding the contiguous-materialization copy that an external
        #   ``.transpose(-2, -1)`` would trigger.
        b_kernel = mat_b.transpose(-2, -1) if trans_b else mat_b

        # Per group, the forward equation is
        # out_e = A_e @ B_e.T: [M_e, K] @ [K, N] -> [M_e, N],
        # where B is stored in its native [E, N, K] layout.
        out = _group_gemm(mat_a.contiguous(), b_kernel, offs)
        ctx.save_for_backward(mat_a, mat_b, offs)
        ctx.trans_b = trans_b
        return out

    @staticmethod
    def backward(ctx, grad_out):
        mat_a, mat_b, offs = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        E = mat_b.shape[0]
        trans_b = ctx.trans_b

        grad_a = grad_b = None
        if trans_b:
            # Per group, the forward equation is
            #   out_e = A_e @ B_e.T: [M_e, K] @ [K, N] -> [M_e, N],
            # where B is stored in its native [E, N, K] layout.
            if ctx.needs_input_grad[0]:
                # dA_e = dOut_e @ B_e: [M_e, N] @ [N, K] -> [M_e, K].
                grad_a = _group_gemm(grad_out, mat_b, offs)
            if ctx.needs_input_grad[1]:
                # dB_e = dOut_e.T @ A_e: [N, M_e] @ [M_e, K] -> [N, K].
                # _grouped_dw(lhs, rhs) computes lhs_e.T @ rhs_e, so passing
                # (grad_out, mat_a) directly produces [E, N, K], B's layout.
                grad_b = _grouped_dw(grad_out, mat_a.contiguous(), offs, E)
        else:
            # Per group, the forward equation is
            #   out_e = A_e @ B_e: [M_e, K] @ [K, N] -> [M_e, N],
            # where B is stored in [E, K, N] layout.
            if ctx.needs_input_grad[0]:
                # dA_e = dOut_e @ B_e.T: [M_e, N] @ [N, K] -> [M_e, K].
                grad_a = _group_gemm(grad_out, mat_b.transpose(-2, -1), offs)
            if ctx.needs_input_grad[1]:
                # dB_e = A_e.T @ dOut_e: [K, M_e] @ [M_e, N] -> [K, N].
                grad_b = _grouped_dw(mat_a.contiguous(), grad_out, offs, E)
        return grad_a, grad_b, None, None


def group_gemm_triton(mat_a: torch.Tensor,
                      mat_b: torch.Tensor,
                      offs: torch.Tensor,
                      trans_b: bool = False) -> torch.Tensor:
    """Autograd-aware Triton grouped GEMM (2D x 3D), drop-in for ``torch._grouped_mm``.

    Args:
        mat_a: ``[M, K]`` float16/bfloat16/float32.
        mat_b: contiguous tensor, same dtype as ``mat_a``. Shape depends on ``trans_b``:
            ``[E, K, N]`` when ``trans_b=False`` (``out = a @ b``), or
            ``[E, N, K]`` when ``trans_b=True`` (``out = a @ b^T``).
            A non-contiguous (e.g. ``.transpose(-2, -1)``) ``mat_b`` is rejected --
            use ``trans_b=True`` with the native layout instead.
        offs:  ``[E]`` cumulative row offsets into ``mat_a`` (``offs[-1] == M``).
        trans_b: if True, ``mat_b`` is stored in ``[E, N, K]`` (its native/contiguous
            layout, e.g. an expert weight) and is logically transposed inside the
            kernel. This keeps the ``.transpose`` off the autograd tape so the weight
            gradient is produced directly in ``mat_b``'s layout, avoiding a
            contiguous-materialization copy in backward.

    Returns:
        ``[M, N]`` tensor, same dtype as inputs.
    """
    # All sanity checks live here (not in the autograd Function) so misuse is
    # caught up front on the user-provided tensors. trans_b is passed through so
    # the contraction dim is checked without materializing a transpose view.
    _validate(mat_a, mat_b, offs, trans_b)
    return _GroupGemmFn.apply(mat_a, mat_b, offs, trans_b)

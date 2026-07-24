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

__all__ = ["group_gemm", "grouped_mm_triton", "is_available"]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def is_available() -> bool:
    """Return True if the Triton grouped-GEMM path can be used."""
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:

    def _ab_configs():
        return [
            triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
            for bm, bn, bk, nw, ns in (
                (64, 64, 32, 4, 3),
                (128, 64, 32, 4, 4),
                (64, 128, 32, 4, 4),
                (128, 128, 32, 8, 3),
                (32, 32, 32, 4, 3),
            )
        ]

    @triton.autotune(configs=_ab_configs(), key=["KC", "NO", "NUM_GROUPS"])
    @triton.jit
    def _group_gemm_ab_kernel(
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
        IN_PRECISION: tl.constexpr,
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
            acc += tl.dot(a_tile, b_tile, input_precision=IN_PRECISION)

        out = acc.to(out_ptr.dtype.element_ty)
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            out,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    def _dw_configs():
        return [
            triton.Config({"BLOCK_K": bk, "BLOCK_N": bn, "BLOCK_M": bm}, num_warps=nw, num_stages=ns)
            for bk, bn, bm, nw, ns in (
                (64, 64, 32, 4, 3),
                (64, 128, 32, 4, 4),
                (128, 64, 32, 4, 4),
                (32, 32, 32, 4, 3),
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
        IN_PRECISION: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """out[e, k, n] = sum_{m in group e} a[m, k] * g[m, n] (weight grad)."""
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
            acc += tl.dot(tl.trans(a_tile), g_tile, input_precision=IN_PRECISION)  # [BLOCK_K, BLOCK_N]

        out = acc.to(out_ptr.dtype.element_ty)
        out_base = out_ptr + pid_e * stride_oe
        tl.store(
            out_base + offs_k[:, None] * stride_ok + offs_n[None, :] * stride_on,
            out,
            mask=mask_k[:, None] & mask_n[None, :],
        )


def _validate(mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor) -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("group_gemm_triton requires Triton, which is not available.")
    if mat_a.dim() != 2 or mat_b.dim() != 3:
        raise NotImplementedError(
            f"Only 2D x 3D grouped GEMM is supported, got mat_a.dim()={mat_a.dim()}, "
            f"mat_b.dim()={mat_b.dim()}.")
    if mat_a.dtype != mat_b.dtype:
        raise ValueError(f"mat_a and mat_b must share dtype, got {mat_a.dtype} vs {mat_b.dtype}.")
    if mat_a.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype {mat_a.dtype}; supported: {_SUPPORTED_DTYPES}.")
    if offs.dim() != 1 or offs.shape[0] != mat_b.shape[0]:
        raise ValueError(f"offs must be 1D of length E={mat_b.shape[0]}, got shape {tuple(offs.shape)}.")
    if mat_a.shape[1] != mat_b.shape[1]:
        raise ValueError(f"Contraction dim mismatch: mat_a K={mat_a.shape[1]} vs mat_b K={mat_b.shape[1]}.")


def _group_meta(offs: torch.Tensor):
    """Compute per-group m-start (exclusive prefix) and m-size on device (no D2H sync)."""
    offs_i = offs.to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device=offs_i.device)
    m_start = torch.cat([zero, offs_i[:-1]]).contiguous()
    m_size = (offs_i - m_start).contiguous()
    return m_start, m_size


def _input_precision(dtype: torch.dtype) -> str:
    """Full-precision fp32 matmul, TF32 for the reduced-precision dtypes."""
    return "ieee" if dtype == torch.float32 else "tf32"


def _grouped_ab(mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
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

    _group_gemm_ab_kernel[grid](
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
        IN_PRECISION=_input_precision(mat_a.dtype),
    )
    return out


def _grouped_dw(mat_a: torch.Tensor, grad_out: torch.Tensor, offs: torch.Tensor, E: int) -> torch.Tensor:
    """Weight-grad grouped GEMM: out[e,K,NO] = sum_{m in group e} a[m,K] x g[m,NO]."""
    M, K = mat_a.shape
    NO = grad_out.shape[1]
    if M == 0:
        # No tokens routed anywhere -> every weight gradient is zero.
        return torch.zeros((E, K, NO), dtype=mat_a.dtype, device=mat_a.device)

    # torch.empty (not zeros) is safe: the kernel grid covers every (E, K, NO)
    # output tile and each program stores its result -- empty groups store 0 --
    # so all elements are written. This avoids a large (E*K*NO) memset per call.
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
        IN_PRECISION=_input_precision(mat_a.dtype),
    )
    return out


class _GroupGemmFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mat_a, mat_b, offs, trans_b):
        # trans_b=False: mat_b is [E, K, N], out = a @ b  (per group).
        # trans_b=True : mat_b is [E, N, K] (e.g. an expert weight in its native
        #   layout), out = a @ b^T. The transpose is applied here as a strided
        #   VIEW inside the Function, so it never lands on the autograd tape.
        #   Backward then produces grad_b directly in mat_b's [E, N, K] layout,
        #   avoiding the contiguous-materialization copy that an external
        #   ``.transpose(-2, -1)`` would trigger.
        b_kernel = mat_b.transpose(-2, -1) if trans_b else mat_b
        _validate(mat_a, b_kernel, offs)
        out = _grouped_ab(mat_a.contiguous(), b_kernel, offs)
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
            # out[m,n] = sum_k a[m,k] * b[g,n,k]  with b = mat_b [E, N, K].
            if ctx.needs_input_grad[0]:
                # grad_a[m,k] = sum_n grad_out[m,n] * b[g,n,k] -> ab with b=mat_b (contract N).
                grad_a = _grouped_ab(grad_out, mat_b, offs)
            if ctx.needs_input_grad[1]:
                # grad_b[g,n,k] = sum_{m in g} grad_out[m,n] * a[m,k]
                #   -> dw(grad_out[M,N], a[M,K]) = [E, N, K], i.e. mat_b's own layout (no copy).
                grad_b = _grouped_dw(grad_out, mat_a.contiguous(), offs, E)
        else:
            if ctx.needs_input_grad[0]:
                # grad_a[m] = grad_out[m] @ b[group(m)]^T -> a[M,N] x bT[E,N,K] grouped over M.
                grad_a = _grouped_ab(grad_out, mat_b.transpose(-2, -1), offs)
            if ctx.needs_input_grad[1]:
                # grad_b[e] = a_e^T @ grad_out_e.
                grad_b = _grouped_dw(mat_a.contiguous(), grad_out, offs, E)
        return grad_a, grad_b, None, None


def group_gemm(mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor,
               trans_b: bool = False) -> torch.Tensor:
    """Autograd-aware Triton grouped GEMM (2D x 3D), drop-in for ``torch._grouped_mm``.

    Args:
        mat_a: ``[M, K]`` float16/bfloat16/float32.
        mat_b: same dtype (arbitrary strides allowed). Shape depends on ``trans_b``:
            ``[E, K, N]`` when ``trans_b=False`` (``out = a @ b``), or
            ``[E, N, K]`` when ``trans_b=True`` (``out = a @ b^T``).
        offs:  ``[E]`` cumulative row offsets into ``mat_a`` (``offs[-1] == M``).
        trans_b: if True, ``mat_b`` is stored in ``[E, N, K]`` (its native/contiguous
            layout, e.g. an expert weight) and is logically transposed inside the
            kernel. This keeps the ``.transpose`` off the autograd tape so the weight
            gradient is produced directly in ``mat_b``'s layout, avoiding a
            contiguous-materialization copy in backward.

    Returns:
        ``[M, N]`` tensor, same dtype as inputs.
    """
    return _GroupGemmFn.apply(mat_a, mat_b, offs, trans_b)


def grouped_mm_triton(mat_a: torch.Tensor, mat_b: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
    """Keyword-``offs`` alias matching the ``torch._grouped_mm(a, b, offs=...)`` call style."""
    return group_gemm(mat_a, mat_b, offs)

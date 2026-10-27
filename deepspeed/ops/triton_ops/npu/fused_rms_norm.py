# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Ascend (NPU) variant of the fused RMSNorm kernels.

flash-linear-attention's ``fla/modules/norm/triton_ascend`` tiles norms as
BT-row groups whose weight stays resident per program across a grid-stride
loop over the rows: the Ascend unified buffer is too small to hold a whole
row for wide features, and re-reading the weight for every row is wasteful.
This variant adopts that launch shape with FLA's calibrated budgets (forward
BT=8 at D=1024, backward BT=4) while keeping the portable kernels' arithmetic
unchanged - the FP32 statistics, the cast-before-gamma rounding and the
eager-mirroring backward - so outputs and gradients stay elementwise equal to
the portable kernels. The dweight partial kernels are imported verbatim: their
row-blocked reduction order is part of the eager contract and must not change.
"""

import torch
from torch.autograd.function import once_differentiable

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, current_device, triton, tl
from deepspeed.ops.triton_ops.fused_rms_norm import assert_supported

if _TRITON_AVAILABLE:
    from deepspeed.ops.triton_ops.fused_rms_norm import (_DWEIGHT_BLOCK_M, _DWEIGHT_BLOCK_N,
                                                         _rms_norm_dweight_partial_kernel)
    from deepspeed.ops.triton_ops.npu._ub import (compute_row_tile_block_size, compute_ub_block_size,
                                                  multiprocessor_count)

    # Peak live fp32 tiles relative to [BT, BD], calibrated by FLA on Ascend 910
    # (192 KiB UB, 0.85 margin): fwd BT=8 / bwd BT=4 at D=1024. The backward keeps
    # more live temporaries, so it gets the larger multiplier (smaller tile).
    _BD_MEM_MULT = 6.0
    _FWD_MEM_MULT = 3.0
    _BWD_MEM_MULT = 8.0
    _UB_SAFETY_MARGIN = 0.85
    # Legacy byte cap when the UB capacity cannot be detected (65536 // fp32).
    _FALLBACK_MAX_BD = 65536 // 4
    _MAX_BT = 128

    def _tiles(n_cols: int, *, is_forward: bool) -> tuple[int, int]:
        """UB-safe (BD, BT) for the row-tiled kernels, in FLA's norm budget."""
        bd = compute_ub_block_size(
            dim_size=n_cols,
            memory_multiplier=_BD_MEM_MULT,
            safety_margin=_UB_SAFETY_MARGIN,
            fallback=_FALLBACK_MAX_BD,
            desired=triton.next_power_of_2(n_cols),
        )
        if n_cols > bd:
            raise RuntimeError(f"fused RMSNorm feature dim {n_cols} exceeds the UB-safe block size {bd}")
        # A large synthetic row dim so BT is limited by the UB, not by a host-side row guess.
        bt = compute_row_tile_block_size(
            row_dim=1 << 20,
            fixed_dim=bd,
            memory_multiplier=_FWD_MEM_MULT if is_forward else _BWD_MEM_MULT,
            tiling_row=True,
            safety_margin=_UB_SAFETY_MARGIN,
            fallback=16,
            min_block=1,
            max_block=_MAX_BT,
        )
        return bd, bt

    def _program_count(n_tiles: int, device) -> int:
        """Programs for the grid-stride launch: the processor count, bounded by the tile count."""
        if n_tiles <= 1:
            return n_tiles
        count = multiprocessor_count(device)
        return min(count, n_tiles) if count else n_tiles

    @triton.jit(do_not_specialize=['T', 'NS'])
    def _rms_norm_fwd_kernel(
        hidden_ptr,
        weight_ptr,
        out_ptr,
        rstd_ptr,
        eps,
        T,
        NS,
        n_cols: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BT: tl.constexpr,
    ):
        """Grid-stride forward: each program owns a BT-row tile stream with the weight resident."""
        i_s = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        col_mask = cols < n_cols
        # The weight keeps its element type: eager multiplies the rounded normalized
        # value by the weight in the input dtype, and the portable kernel does too.
        b_w = tl.load(weight_ptr + cols, mask=col_mask, other=0.0)

        NT = tl.cdiv(T, BT)
        for i_t in range(i_s, NT, NS):
            rows = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
            row_mask = rows < T
            mask = row_mask[:, None] & col_mask[None, :]
            row_off = rows[:, None] * n_cols + cols[None, :]

            b_x = tl.load(hidden_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
            variance = tl.sum(b_x * b_x, axis=1) / n_cols
            b_rstd = tl.rsqrt(variance + eps)
            tl.store(rstd_ptr + rows, b_rstd, mask=row_mask)
            # Cast-before-gamma: normalize, round once to the input dtype, then multiply.
            b_norm = (b_x * b_rstd[:, None]).to(out_ptr.dtype.element_ty)
            tl.store(out_ptr + row_off, b_norm * b_w[None, :], mask=mask)

    @triton.jit(do_not_specialize=['T', 'NS'])
    def _rms_norm_dx_kernel(
        grad_out_ptr,
        hidden_ptr,
        weight_ptr,
        rstd_ptr,
        grad_hidden_ptr,
        T,
        NS,
        n_cols: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BT: tl.constexpr,
    ):
        """Grid-stride backward: mirrors the eager autograd graph step by step, as the portable kernel does."""
        i_s = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        col_mask = cols < n_cols
        # The weight keeps its element type: eager rounds the grad-weight product to the
        # input dtype before the remaining math, and the portable kernel does too.
        b_w = tl.load(weight_ptr + cols, mask=col_mask, other=0.0)

        NT = tl.cdiv(T, BT)
        for i_t in range(i_s, NT, NS):
            rows = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
            row_mask = rows < T
            mask = row_mask[:, None] & col_mask[None, :]
            row_off = rows[:, None] * n_cols + cols[None, :]

            b_dy = tl.load(grad_out_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
            b_x = tl.load(hidden_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
            b_rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

            # Mirror the eager autograd graph step by step so values round where the eager backward rounds them.
            grad_norm = (b_dy * b_w[None, :]).to(grad_hidden_ptr.dtype.element_ty).to(tl.float32)
            grad_scaled = grad_norm * b_rstd[:, None]
            grad_rstd = tl.sum(grad_norm * b_x, axis=1)
            grad_variance = (-0.5 * grad_rstd) * (b_rstd * b_rstd * b_rstd)
            grad_hidden = grad_scaled + (grad_variance / n_cols)[:, None] * (2.0 * b_x)
            tl.store(grad_hidden_ptr + row_off, grad_hidden, mask=mask)

    class _FusedRMSNorm(torch.autograd.Function):
        """RMSNorm autograd with FP32 reductions and a rounded pre-gamma value."""

        @staticmethod
        def forward(ctx, hidden, weight, eps):
            n_rows, n_cols = hidden.shape
            out = torch.empty_like(hidden)
            rstd = torch.empty((n_rows, ), dtype=torch.float32, device=hidden.device)

            ctx.save_for_backward(hidden, weight, rstd)
            ctx.n_cols = n_cols

            if n_rows > 0:
                block_n, block_t = _tiles(n_cols, is_forward=True)
                nt = triton.cdiv(n_rows, block_t)
                ns = _program_count(nt, hidden.device)
                # Triton uses the current device and stream, which may differ from the input tensor's device.
                with current_device(hidden):
                    _rms_norm_fwd_kernel[(ns, )](hidden,
                                                 weight,
                                                 out,
                                                 rstd,
                                                 eps,
                                                 n_rows,
                                                 ns,
                                                 n_cols,
                                                 BLOCK_N=block_n,
                                                 BT=block_t)
            return out

        # Autograd cannot see inside the Triton kernels, so differentiating this backward again would silently drop
        # the op's second derivative; once_differentiable makes that raise instead.
        @staticmethod
        @once_differentiable
        def backward(ctx, grad_out):
            hidden, weight, rstd = ctx.saved_tensors
            n_rows, n_cols = hidden.shape
            grad_out = grad_out.contiguous()
            grad_hidden = torch.empty_like(hidden)

            if n_rows == 0:
                return grad_hidden, torch.zeros_like(weight), None

            block_n, block_t = _tiles(n_cols, is_forward=False)
            nt = triton.cdiv(n_rows, block_t)
            ns = _program_count(nt, hidden.device)
            with current_device(hidden):
                _rms_norm_dx_kernel[(ns, )](grad_out,
                                            hidden,
                                            weight,
                                            rstd,
                                            grad_hidden,
                                            n_rows,
                                            ns,
                                            n_cols,
                                            BLOCK_N=block_n,
                                            BT=block_t)

            # The dweight reduction keeps the portable kernel's row-blocked order, which the
            # eager contract depends on, so the partial kernels are imported unchanged.
            dweight_block_n = min(_DWEIGHT_BLOCK_N, block_n)
            n_row_blocks = triton.cdiv(n_rows, _DWEIGHT_BLOCK_M)
            n_col_blocks = triton.cdiv(n_cols, dweight_block_n)
            partial = torch.empty((n_row_blocks, n_cols), dtype=torch.float32, device=hidden.device)
            _rms_norm_dweight_partial_kernel[(n_row_blocks, n_col_blocks)](grad_out,
                                                                           hidden,
                                                                           rstd,
                                                                           partial,
                                                                           n_rows,
                                                                           n_cols,
                                                                           BLOCK_M=_DWEIGHT_BLOCK_M,
                                                                           BLOCK_N=dweight_block_n)
            grad_weight = partial.sum(dim=0).to(weight.dtype)
            return grad_hidden, grad_weight, None


def _run_kernels(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    original_shape = hidden.shape
    if not hidden.is_contiguous():
        hidden = hidden.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    hidden_2d = hidden.reshape(-1, original_shape[-1])
    out = _FusedRMSNorm.apply(hidden_2d, weight, eps)
    return out.reshape(original_shape)


def fused_rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The portable fused RMSNorm with the Ascend launch shape (see the module docstring)."""
    assert_supported(hidden, weight, eps)
    return _run_kernels(hidden, weight, eps)

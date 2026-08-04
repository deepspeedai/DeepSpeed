# Copyright (c) DeepSpeed Team.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
#
# Portions of this file are derived from TorchTitan.
# See THIRD_PARTY_NOTICES.md for the BSD-3-Clause notice.

# DeepSpeed Team
"""
Grouped expert computation for expert parallelism.

Ported from TorchTitan's GroupedExperts with adaptations for DeepSpeed:
  - Replaced hardcoded .bfloat16() with input-dtype-aware casting
  - Fail-fast RuntimeError when use_grouped_mm=True but torch._grouped_mm is unavailable
  - Removed DTensor-specific code paths

This module is self-contained: no imports from deepspeed.module_inject
or deepspeed.runtime.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.logging import warning_once

# ---------------------------------------------------------------------------
# Expert computation: sequential for-loop (reference path)
# ---------------------------------------------------------------------------


def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Compute SwiGLU expert MLP via a sequential for-loop over experts.

    This is the reference implementation that works on all PyTorch versions.

    Args:
        w1: Gate-up weight, shape ``(E, hidden_dim, dim)``.
        w2: Down weight, shape ``(E, dim, hidden_dim)``.
        w3: Up weight, shape ``(E, hidden_dim, dim)``.
        x: Input tokens, shape ``(T, dim)``.
        num_tokens_per_expert: Token counts per expert, shape ``(E,)``.

    Returns:
        Output tensor of shape ``(T, dim)``.
    """
    # NOTE: .tolist() incurs a device-host synchronization
    num_tokens_per_expert_list = num_tokens_per_expert.tolist()

    # Handle padding rows injected by generate_permute_indices
    num_padding = x.shape[0] - sum(num_tokens_per_expert_list)

    x_splits = torch.split(
        x[:sum(num_tokens_per_expert_list)],
        split_size_or_sections=num_tokens_per_expert_list,
        dim=0,
    )

    cast_dtype = x.dtype
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x_splits):
        w1_e = w1[expert_idx].to(cast_dtype).transpose(-2, -1)
        w3_e = w3[expert_idx].to(cast_dtype).transpose(-2, -1)
        w2_e = w2[expert_idx].to(cast_dtype).transpose(-2, -1)
        h = F.silu(torch.matmul(x_expert, w1_e))
        h = h * torch.matmul(x_expert, w3_e)
        h = torch.matmul(h, w2_e)
        out_experts_splits.append(h)

    out = torch.cat(out_experts_splits, dim=0)

    # Re-add padding rows (zeros) so output shape matches input shape
    out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))

    return out


# ---------------------------------------------------------------------------
# Expert computation: grouped GEMM (torch._grouped_mm)
# ---------------------------------------------------------------------------


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Compute SwiGLU expert MLP via torch._grouped_mm (grouped GEMM).

    Uses input dtype for casting instead of hardcoded bfloat16.

    Args:
        w1: Gate-up weight, shape ``(E, hidden_dim, dim)``.
        w2: Down weight, shape ``(E, dim, hidden_dim)``.
        w3: Up weight, shape ``(E, hidden_dim, dim)``.
        x: Input tokens, shape ``(T, dim)``.
        num_tokens_per_expert: Token counts per expert, shape ``(E,)``.

    Returns:
        Output tensor of shape ``(T, dim)``.
    """
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    from deepspeed.ops.triton_ops.swiglu_triton import swiglu

    cast_dtype = x.dtype
    gate = torch._grouped_mm(
        x.to(cast_dtype),
        w1.to(cast_dtype).transpose(-2, -1),
        offs=offsets,
    )
    up = torch._grouped_mm(
        x.to(cast_dtype),
        w3.to(cast_dtype).transpose(-2, -1),
        offs=offsets,
    )
    h = swiglu(gate, up)
    out = torch._grouped_mm(
        h,
        w2.to(cast_dtype).transpose(-2, -1),
        offs=offsets,
    ).type_as(x)

    return out


# ---------------------------------------------------------------------------
# Expert computation: Triton grouped GEMM (sm80 / sm86 fast path)
# ---------------------------------------------------------------------------


def _run_experts_triton_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Compute SwiGLU expert MLP via the Triton grouped GEMM drop-in.

    Numerically and API-compatible with :func:`_run_experts_grouped_mm`, but
    uses ``deepspeed.ops.triton_ops.group_gemm_triton.group_gemm_triton`` instead of
    ``torch._grouped_mm``.

    Args mirror :func:`_run_experts_grouped_mm`.
    """
    from deepspeed.ops.triton_ops.group_gemm_triton import group_gemm_triton
    from deepspeed.ops.triton_ops.swiglu_triton import swiglu

    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    # trans_b=True: pass expert weights in their native [E, hidden, dim] layout
    # (no .transpose on the autograd tape). The kernel applies the transpose via
    # strides, and backward writes the weight gradient directly in that layout,
    # avoiding a contiguous-materialization copy of the transposed grad.

    dtype = x.dtype
    gate = group_gemm_triton(x, w1.to(dtype), offsets, trans_b=True)
    up = group_gemm_triton(x, w3.to(dtype), offsets, trans_b=True)
    h = swiglu(gate, up)
    out = group_gemm_triton(h, w2.to(dtype), offsets, trans_b=True).type_as(x)

    return out


# ---------------------------------------------------------------------------
# GroupedExperts module
# ---------------------------------------------------------------------------


class GroupedExperts(nn.Module):
    """Grouped expert computation for MoE layers.

    Supports three execution paths:
      - **triton_grouped_mm**: Uses a Triton grouped-GEMM kernel
        (``deepspeed.ops.triton_ops.group_gemm_triton``). Auto-selected on sm80/sm86 where
        ``torch._grouped_mm`` would otherwise fall back to a slow per-group loop.
      - **grouped_mm**: Uses ``torch._grouped_mm`` for fused grouped GEMM
        (requires a sufficiently recent PyTorch build).
      - **for-loop**: Sequential per-expert matmuls; always available.

    If ``use_grouped_mm=True`` but neither the Triton path nor
    ``torch._grouped_mm`` is available, the constructor raises ``RuntimeError``.
    Set ``use_grouped_mm=False`` to select the sequential for-loop path.

    Args:
        dim (int): Input / output dimension.
        hidden_dim (int): Hidden dimension of the SwiGLU FFN.
        num_experts (int): Number of experts.
        use_grouped_mm (bool): Whether to attempt using grouped GEMM.
        disable_triton_grouped_mm (bool): Set ``True`` to force the native
            ``torch._grouped_mm`` path even on devices where the Triton
            grouped-GEMM kernel would otherwise be preferred (e.g. sm8x).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool = True,
        disable_triton_grouped_mm: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        # Mark as grouped expert tensors so Muon applies NS per-expert
        self.w1.is_expert_group = True
        self.w2.is_expert_group = True
        self.w3.is_expert_group = True
        self.use_triton_grouped_mm = False
        self.use_grouped_mm = use_grouped_mm

        # Resolve the Triton path. The device-specific decision is delegated to
        # the accelerator backend (e.g. the CUDA backend prefers Triton on
        # sm < 9.0, where torch._grouped_mm falls back to a slow per-group loop).
        # Set disable_triton_grouped_mm=True to force the native path.
        if use_grouped_mm and not disable_triton_grouped_mm:
            self.use_triton_grouped_mm = get_accelerator().prefer_triton_grouped_mm()

        if use_grouped_mm and not hasattr(torch, "_grouped_mm") and not self.use_triton_grouped_mm:
            raise RuntimeError("GroupedExperts was constructed with use_grouped_mm=True but "
                               "torch._grouped_mm is not available in this PyTorch build. "
                               "Upgrade PyTorch to a build that provides torch._grouped_mm, install "
                               "Triton to enable the Triton grouped-GEMM path, or set "
                               "use_grouped_mm=False to use the sequential expert loop.")

        if use_grouped_mm and self.use_triton_grouped_mm:
            warning_once("Triton grouped-GEMM path is selected for grouped_gemm. "
                         "The Triton path is preferred on compute capability smaller than sm90, "
                         "and will be used instead of torch._grouped_mm. Set use_grouped_mm=False or "
                         "disable_triton_grouped_mm=True to avoid this warning.")

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tokens, shape ``(T, dim)``.
            num_tokens_per_expert: Token counts per expert, shape ``(E,)``.

        Returns:
            Output tensor of shape ``(T, dim)``.
        """

        if self.use_triton_grouped_mm:
            return _run_experts_triton_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert)
        elif self.use_grouped_mm:
            return _run_experts_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert)
        else:
            return _run_experts_for_loop(self.w1, self.w2, self.w3, x, num_tokens_per_expert)

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

    cast_dtype = x.dtype
    h = F.silu(torch._grouped_mm(
        x.to(cast_dtype),
        w1.to(cast_dtype).transpose(-2, -1),
        offs=offsets,
    ))
    h = h * torch._grouped_mm(
        x.to(cast_dtype),
        w3.to(cast_dtype).transpose(-2, -1),
        offs=offsets,
    )
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
    uses ``deepspeed.moe.group_gemm_triton.group_gemm`` instead of
    ``torch._grouped_mm``. On sm80/sm86 the native op has no fused grouped-GEMM
    kernel and falls back to a per-group Python loop (plus a device->host sync);
    the Triton path issues a single fused kernel per grouped GEMM with no sync.

    Args mirror :func:`_run_experts_grouped_mm`.
    """
    from deepspeed.moe.group_gemm_triton import group_gemm

    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    cast_dtype = x.dtype
    # trans_b=True: pass expert weights in their native [E, hidden, dim] layout
    # (no .transpose on the autograd tape). The kernel applies the transpose via
    # strides, and backward writes the weight gradient directly in that layout,
    # avoiding a contiguous-materialization copy of the transposed grad.
    h = F.silu(group_gemm(x.to(cast_dtype), w1.to(cast_dtype), offsets, trans_b=True))
    h = h * group_gemm(x.to(cast_dtype), w3.to(cast_dtype), offsets, trans_b=True)
    out = group_gemm(h, w2.to(cast_dtype), offsets, trans_b=True).type_as(x)

    return out


def _prefer_triton_grouped_mm() -> bool:
    """Auto-select Triton grouped GEMM when the native op would fall back to a loop.

    ``torch._grouped_mm`` only has a fused grouped-GEMM kernel on Hopper (sm90)
    and newer; on Ampere/Ada (sm8x) it falls back to a slow per-group loop. So
    we prefer the Triton path when the current CUDA device has compute
    capability major < 9 and Triton is available.

    Environment overrides (mainly for benchmarking / debugging):
      * ``DS_DISABLE_TRITON_GROUPED_MM=1`` forces the native path off.
      * ``DS_FORCE_TRITON_GROUPED_MM=1`` forces the Triton path on.
    """
    import os

    if os.environ.get("DS_DISABLE_TRITON_GROUPED_MM", "0") == "1":
        return False
    try:
        from deepspeed.moe.group_gemm_triton import is_available
    except Exception:
        return False
    if not is_available():
        return False
    if os.environ.get("DS_FORCE_TRITON_GROUPED_MM", "0") == "1":
        return True
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
    except Exception:
        return False
    return major < 9


# ---------------------------------------------------------------------------
# GroupedExperts module
# ---------------------------------------------------------------------------


class GroupedExperts(nn.Module):
    """Grouped expert computation for MoE layers.

    Supports three execution paths:
      - **triton_grouped_mm**: Uses a Triton grouped-GEMM kernel
        (``deepspeed.moe.group_gemm_triton``). Auto-selected on sm80/sm86 where
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
        use_triton_grouped_mm (bool | None): Force (True) / disable (False) the
            Triton grouped-GEMM path. ``None`` (default) auto-selects it on
            devices where ``torch._grouped_mm`` lacks a fused kernel (sm < 9.0).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool = True,
        use_triton_grouped_mm: bool | None = None,
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

        # Resolve the Triton path: explicit override, else auto-detect.
        if not use_grouped_mm:
            self.use_triton_grouped_mm = False
        elif use_triton_grouped_mm is None:
            self.use_triton_grouped_mm = _prefer_triton_grouped_mm()
        else:
            if use_triton_grouped_mm:
                from deepspeed.moe.group_gemm_triton import is_available
                if not is_available():
                    raise RuntimeError("GroupedExperts was constructed with use_triton_grouped_mm=True "
                                       "but Triton is not available in this environment.")
            self.use_triton_grouped_mm = use_triton_grouped_mm

        if use_grouped_mm and not self.use_triton_grouped_mm and not hasattr(torch, "_grouped_mm"):
            raise RuntimeError("GroupedExperts was constructed with use_grouped_mm=True but "
                               "torch._grouped_mm is not available in this PyTorch build. "
                               "Upgrade PyTorch to a build that provides torch._grouped_mm, install "
                               "Triton to enable the Triton grouped-GEMM path, or set "
                               "use_grouped_mm=False to use the sequential expert loop.")
        self.use_grouped_mm = use_grouped_mm

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

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Grouped expert computation for expert parallelism.

Ported from TorchTitan's GroupedExperts with adaptations for DeepSpeed:
  - Replaced hardcoded .bfloat16() with input-dtype-aware casting
  - Runtime check for torch._grouped_mm availability with fallback
  - Removed DTensor-specific code paths
  - CUTLASS backend raises NotImplementedError

This module is self-contained: no imports from deepspeed.module_inject
or deepspeed.runtime.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expert computation: for-loop fallback
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
# GroupedExperts module
# ---------------------------------------------------------------------------


class GroupedExperts(nn.Module):
    """Grouped expert computation for MoE layers.

    Supports two backends:
      - **grouped_mm**: Uses ``torch._grouped_mm`` for fused grouped GEMM
        (requires a sufficiently recent PyTorch build).
      - **for-loop**: Sequential per-expert matmuls; always available.

    If ``use_grouped_mm=True`` but ``torch._grouped_mm`` is not available,
    falls back to the for-loop implementation with a warning.

    Args:
        dim (int): Input / output dimension.
        hidden_dim (int): Hidden dimension of the SwiGLU FFN.
        num_experts (int): Number of experts.
        use_grouped_mm (bool): Whether to attempt using grouped GEMM.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))

        # Check grouped_mm availability at construction time
        self._has_grouped_mm = hasattr(torch, "_grouped_mm")
        if use_grouped_mm and not self._has_grouped_mm:
            logger.warning("torch._grouped_mm not available, falling back to "
                           "for-loop expert computation")
        self.use_grouped_mm = use_grouped_mm and self._has_grouped_mm

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
        if self.use_grouped_mm:
            return _run_experts_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert)
        else:
            return _run_experts_for_loop(self.w1, self.w2, self.w3, x, num_tokens_per_expert)

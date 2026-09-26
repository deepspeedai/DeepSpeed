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

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.logging import warning_once

# ---------------------------------------------------------------------------
# Expert activation registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertActivation:
    """One way an expert MLP turns its gate and up projections into the input of the down projection.

    ``fn(gate, up, alpha, limit)`` computes the form in plain PyTorch; ``fused_fn`` has the same
    signature and runs a fused kernel, when the form has one. ``uses_alpha`` and ``uses_limit`` say
    which of the two scalars the form reads. ``gate_fn`` is the elementwise function the form applies
    to ``gate`` when it is ``gate_fn(gate) * up`` inside the clamp region; AutoEP compares it with the
    ``act_fn`` of the model's experts module to catch a preset that names the wrong form. It is
    ``None`` for forms that are not such a product.
    """
    fn: Callable[[torch.Tensor, torch.Tensor, float, float], torch.Tensor]
    fused_fn: Callable[[torch.Tensor, torch.Tensor, float, float], torch.Tensor] | None = None
    uses_alpha: bool = False
    uses_limit: bool = False
    gate_fn: Callable[[torch.Tensor], torch.Tensor] | None = None


#: The expert activations AutoEP can compute, by name. They are different functions: a model trained
#: with one does not run correctly with another. ``register_expert_activation`` adds a form.
#:   swiglu          silu(gate) * up                    Mixtral, Qwen, DeepSeek-V2/V3, GLM, most others
#:   geglu_tanh      gelu_tanh(gate) * up               Gemma-4, Diffusion-Gemma
#:   swiglu_clamped  silu(clamp(gate)) * clamp(up)      DeepSeek-V4 (limit 10)
#:   swiglu_oai      (clamp(up) + 1) * clamp(gate) * sigmoid(alpha * clamp(gate))
#:                                                      GPT-OSS, MiniMax-M3 (alpha 1.702, limit 7)
#: In the clamped forms ``gate`` is clamped from above only and ``up`` on both sides.
EXPERT_ACTIVATIONS: dict[str, ExpertActivation] = {}


def register_expert_activation(
    name: str,
    fn: Callable[[torch.Tensor, torch.Tensor, float, float], torch.Tensor],
    *,
    fused_fn: Callable[[torch.Tensor, torch.Tensor, float, float], torch.Tensor] | None = None,
    uses_alpha: bool = False,
    uses_limit: bool = False,
    gate_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    """Make ``name`` selectable as a preset's or the config's ``expert_activation``."""
    if name in EXPERT_ACTIVATIONS:
        raise ValueError(f"expert activation {name!r} is already registered")
    EXPERT_ACTIVATIONS[name] = ExpertActivation(fn, fused_fn, uses_alpha, uses_limit, gate_fn)


def get_expert_activation(name: str) -> ExpertActivation:
    entry = EXPERT_ACTIVATIONS.get(name)
    if entry is None:
        raise ValueError(f"unknown expert activation {name!r}; expected one of {tuple(EXPERT_ACTIVATIONS)}")
    return entry


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def _swiglu(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    return F.silu(gate) * up


def _swiglu_fused(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    from deepspeed.ops.triton_ops.swiglu_triton import swiglu
    return swiglu(gate, up)


def _geglu_tanh(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    return _gelu_tanh(gate) * up


def _swiglu_clamped(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    return F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)


def _swiglu_oai(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return (up + 1.0) * (gate * torch.sigmoid(gate * alpha))


def _swiglu_clamped_fused(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    from deepspeed.ops.triton_ops.swiglu_triton import swiglu_clamped
    return swiglu_clamped(gate, up, alpha=alpha, limit=limit, oai=False)


def _swiglu_oai_fused(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    from deepspeed.ops.triton_ops.swiglu_triton import swiglu_clamped
    return swiglu_clamped(gate, up, alpha=alpha, limit=limit, oai=True)


register_expert_activation("swiglu", _swiglu, fused_fn=_swiglu_fused, gate_fn=F.silu)
register_expert_activation("geglu_tanh", _geglu_tanh, gate_fn=_gelu_tanh)
register_expert_activation("swiglu_clamped",
                           _swiglu_clamped,
                           fused_fn=_swiglu_clamped_fused,
                           uses_limit=True,
                           gate_fn=F.silu)
register_expert_activation("swiglu_oai", _swiglu_oai, fused_fn=_swiglu_oai_fused, uses_alpha=True, uses_limit=True)


def apply_expert_activation(gate: torch.Tensor,
                            up: torch.Tensor,
                            activation: str = "swiglu",
                            alpha: float = 1.702,
                            limit: float = 7.0,
                            fused: bool = True) -> torch.Tensor:
    """Combine the gate and up projections of an expert MLP with the named activation.

    ``fused`` selects the form's fused kernel when it has one; ``fused=False`` keeps everything in
    plain PyTorch, which also runs on CPU tensors.
    """
    entry = get_expert_activation(activation)
    if fused and entry.fused_fn is not None:
        return entry.fused_fn(gate, up, alpha, limit)
    return entry.fn(gate, up, alpha, limit)


# ---------------------------------------------------------------------------
# Expert computation: sequential for-loop (reference path)
# ---------------------------------------------------------------------------


def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    activation: str = "swiglu",
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """Compute SwiGLU expert MLP via a sequential for-loop over experts.

    This is the reference implementation that works on all PyTorch versions.

    Args:
        w1: Gate-up weight, shape ``(E, hidden_dim, dim)``.
        w2: Down weight, shape ``(E, dim, hidden_dim)``.
        w3: Up weight, shape ``(E, hidden_dim, dim)``.
        x: Input tokens, shape ``(T, dim)``.
        num_tokens_per_expert: Token counts per expert, shape ``(E,)``.
        activation: Expert activation name from ``EXPERT_ACTIVATIONS``.
        alpha: ``alpha`` for the forms that read it.
        limit: Clamp limit for the forms that read it.

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
        gate = torch.matmul(x_expert, w1_e)
        up = torch.matmul(x_expert, w3_e)
        # fused=False keeps the reference path in plain PyTorch, so it still runs on CPU tensors.
        h = apply_expert_activation(gate, up, activation, alpha, limit, fused=False)
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
    activation: str = "swiglu",
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """Compute SwiGLU expert MLP via torch._grouped_mm (grouped GEMM).

    Uses input dtype for casting instead of hardcoded bfloat16.

    Args:
        w1: Gate-up weight, shape ``(E, hidden_dim, dim)``.
        w2: Down weight, shape ``(E, dim, hidden_dim)``.
        w3: Up weight, shape ``(E, hidden_dim, dim)``.
        x: Input tokens, shape ``(T, dim)``.
        num_tokens_per_expert: Token counts per expert, shape ``(E,)``.
        activation, alpha, limit: As in :func:`_run_experts_for_loop`.

    Returns:
        Output tensor of shape ``(T, dim)``.
    """
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

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
    h = apply_expert_activation(gate, up, activation, alpha, limit)
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
    activation: str = "swiglu",
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """Compute SwiGLU expert MLP via the Triton grouped GEMM drop-in.

    Numerically and API-compatible with :func:`_run_experts_grouped_mm`, but
    uses ``deepspeed.ops.triton_ops.group_gemm_triton.group_gemm_triton`` instead of
    ``torch._grouped_mm``.

    Args mirror :func:`_run_experts_grouped_mm`.
    """
    from deepspeed.ops.triton_ops.group_gemm_triton import group_gemm_triton

    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    # trans_b=True: pass expert weights in their native [E, hidden, dim] layout
    # (no .transpose on the autograd tape). The kernel applies the transpose via
    # strides, and backward writes the weight gradient directly in that layout,
    # avoiding a contiguous-materialization copy of the transposed grad.

    dtype = x.dtype
    gate = group_gemm_triton(x, w1.to(dtype), offsets, trans_b=True)
    up = group_gemm_triton(x, w3.to(dtype), offsets, trans_b=True)
    h = apply_expert_activation(gate, up, activation, alpha, limit)
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
        activation (str): Expert activation name from ``EXPERT_ACTIVATIONS``;
            the default is plain SwiGLU.
        activation_alpha (float): ``alpha`` for the forms that read it (``swiglu_oai``).
        activation_limit (float): Clamp limit for the forms that read it.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool = True,
        disable_triton_grouped_mm: bool = False,
        activation: str = "swiglu",
        activation_alpha: float = 1.702,
        activation_limit: float = 7.0,
    ):
        super().__init__()
        # An unknown name is refused here, at construction, rather than at the first forward step.
        get_expert_activation(activation)
        self.activation = activation
        self.activation_alpha = activation_alpha
        self.activation_limit = activation_limit
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

        act = (self.activation, self.activation_alpha, self.activation_limit)
        if self.use_triton_grouped_mm:
            return _run_experts_triton_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert, *act)
        elif self.use_grouped_mm:
            return _run_experts_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert, *act)
        else:
            return _run_experts_for_loop(self.w1, self.w2, self.w3, x, num_tokens_per_expert, *act)

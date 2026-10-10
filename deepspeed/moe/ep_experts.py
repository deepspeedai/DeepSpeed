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
    signature and runs a fused kernel, when the form has one. ``packed_fused_fn(gate_up, alpha, limit)``
    is a fused kernel that reads ``gate`` and ``up`` packed side by side in one ``[..., 2 * I]`` tensor;
    forms without one run ``fn`` on views of the two halves. ``fused_backward_fn(grad, gate, up, alpha, limit)``
    and ``packed_fused_backward_fn(grad, gate_up, alpha, limit)`` return the gradients the fused kernels' own
    backward computes, for callers that run the forward inside their own autograd function; without them such
    callers recompute the forward to differentiate it. ``uses_alpha`` and ``uses_limit`` say
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
    packed_fused_fn: Callable[[torch.Tensor, float, float], torch.Tensor] | None = None
    fused_backward_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None
    packed_fused_backward_fn: Callable[..., torch.Tensor] | None = None


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
    packed_fused_fn: Callable[[torch.Tensor, float, float], torch.Tensor] | None = None,
    fused_backward_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None,
    packed_fused_backward_fn: Callable[..., torch.Tensor] | None = None,
) -> None:
    """Make ``name`` selectable as a preset's or the config's ``expert_activation``."""
    if name in EXPERT_ACTIVATIONS:
        raise ValueError(f"expert activation {name!r} is already registered")
    EXPERT_ACTIVATIONS[name] = ExpertActivation(fn, fused_fn, uses_alpha, uses_limit, gate_fn, packed_fused_fn,
                                                fused_backward_fn, packed_fused_backward_fn)


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


def _swiglu_packed_fused(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    from deepspeed.ops.triton_ops.swiglu_triton import swiglu_packed
    return swiglu_packed(gate_up)


def _swiglu_fused_backward(grad: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float):
    from deepspeed.ops.triton_ops.swiglu_triton import swiglu_backward
    return swiglu_backward(grad, gate, up)


def _swiglu_packed_fused_backward(grad: torch.Tensor, gate_up: torch.Tensor, alpha: float,
                                  limit: float) -> torch.Tensor:
    from deepspeed.ops.triton_ops.swiglu_triton import swiglu_packed_backward
    return swiglu_packed_backward(grad, gate_up)


def _geglu_tanh(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    return _gelu_tanh(gate) * up


def _swiglu_clamped(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    return F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)


def _swiglu_oai(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return (up + 1.0) * (gate * torch.sigmoid(gate * alpha))


register_expert_activation("swiglu",
                           _swiglu,
                           fused_fn=_swiglu_fused,
                           gate_fn=F.silu,
                           packed_fused_fn=_swiglu_packed_fused,
                           fused_backward_fn=_swiglu_fused_backward,
                           packed_fused_backward_fn=_swiglu_packed_fused_backward)
register_expert_activation("geglu_tanh", _geglu_tanh, gate_fn=_gelu_tanh)
register_expert_activation("swiglu_clamped", _swiglu_clamped, uses_limit=True, gate_fn=F.silu)
register_expert_activation("swiglu_oai", _swiglu_oai, uses_alpha=True, uses_limit=True)


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


def apply_packed_expert_activation(gate_up: torch.Tensor,
                                   activation: str = "swiglu",
                                   alpha: float = 1.702,
                                   limit: float = 7.0,
                                   fused: bool = True) -> torch.Tensor:
    """:func:`apply_expert_activation` for ``gate`` and ``up`` packed side by side in ``[..., 2 * I]``.

    Forms without a packed kernel run on views of the two halves, which their elementwise ops read in
    place.
    """
    entry = get_expert_activation(activation)
    if fused and entry.packed_fused_fn is not None:
        return entry.packed_fused_fn(gate_up, alpha, limit)
    gate, up = gate_up.chunk(2, dim=-1)
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


def _run_experts_grouped_mm_fused_gate_up(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    activation: str = "swiglu",
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """:func:`_run_experts_grouped_mm` with the gate and up projections in one grouped GEMM.

    The parameters keep their separate ``w1``/``w3`` layout; their concatenation is rebuilt on each call.
    One GEMM over ``2 * hidden_dim`` output columns replaces two in the forward pass, and in the backward
    pass a single input-gradient GEMM accumulates both projections in FP32 instead of adding two rounded
    results. The forward values and weight gradients are those of the separate path.

    Args mirror :func:`_run_experts_grouped_mm`.
    """
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    cast_dtype = x.dtype
    w13 = torch.cat([w1.to(cast_dtype), w3.to(cast_dtype)], dim=1)
    gate_up = torch._grouped_mm(
        x.to(cast_dtype),
        w13.transpose(-2, -1),
        offs=offsets,
    )
    h = apply_packed_expert_activation(gate_up, activation, alpha, limit)
    out = torch._grouped_mm(
        h,
        w2.to(cast_dtype).transpose(-2, -1),
        offs=offsets,
    ).type_as(x)

    return out


def _expert_activation_backward(grad: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, activation: str,
                                alpha: float, limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients of :func:`apply_expert_activation` with respect to ``gate`` and ``up``."""
    entry = get_expert_activation(activation)
    if entry.fused_fn is not None and entry.fused_backward_fn is not None:
        return entry.fused_backward_fn(grad, gate, up, alpha, limit)
    forward = entry.fused_fn or entry.fn
    with torch.enable_grad():
        gate = gate.detach().requires_grad_(True)
        up = up.detach().requires_grad_(True)
        return torch.autograd.grad(forward(gate, up, alpha, limit), (gate, up), grad)


def _packed_expert_activation_backward(grad: torch.Tensor, gate_up: torch.Tensor, activation: str, alpha: float,
                                       limit: float) -> torch.Tensor:
    """Gradient of :func:`apply_packed_expert_activation` with respect to ``gate_up``."""
    entry = get_expert_activation(activation)
    if entry.packed_fused_fn is not None and entry.packed_fused_backward_fn is not None:
        return entry.packed_fused_backward_fn(grad, gate_up, alpha, limit)
    with torch.enable_grad():
        gate_up = gate_up.detach().requires_grad_(True)
        return torch.autograd.grad(apply_packed_expert_activation(gate_up, activation, alpha, limit), gate_up, grad)[0]


class ExpertWeightGradSlot:
    """Holds one expert forward's weight-gradient GEMMs until a later backward runs them.

    The expert backward returns its input gradient at once and leaves its weight gradients here, so the backward
    that sends that input gradient onward can start the send first and compute the weight gradients while it is in
    flight.
    """

    def __init__(self):
        self._compute = None

    def defer(self, compute: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
        if self._compute is not None:
            raise RuntimeError("an expert weight gradient was deferred twice into one slot")
        self._compute = compute

    def run(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._compute is None:
            raise RuntimeError("the expert backward did not run before the backward that computes its weight "
                               "gradients")
        compute, self._compute = self._compute, None
        return compute()


class _DeferredWeightGradExperts(torch.autograd.Function):
    """The grouped-GEMM expert MLP, with its weight gradients optionally deferred to an :class:`ExpertWeightGradSlot`.

    Forward runs exactly what :func:`_run_experts_grouped_mm` (or its fused gate+up variant) runs. Backward forms the
    products autograd forms for ``torch._grouped_mm`` (``_grouped_mm_mat1_backward`` and
    ``_grouped_mm_mat2_backward``), so every gradient matches the plain path; only when the weight gradients are
    computed changes.
    """

    @staticmethod
    def forward(ctx, x, num_tokens_per_expert, w1, w2, w3, slot, activation, alpha, limit, fused_gate_up):
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        dtype = x.dtype
        w1c, w2c, w3c = w1.to(dtype), w2.to(dtype), w3.to(dtype)
        if fused_gate_up:
            w13 = torch.cat([w1c, w3c], dim=1)
            gate_up = torch._grouped_mm(x, w13.transpose(-2, -1), offs=offsets)
            h = apply_packed_expert_activation(gate_up, activation, alpha, limit)
            ctx.save_for_backward(x, offsets, w2c, h, w13, gate_up)
        else:
            gate = torch._grouped_mm(x, w1c.transpose(-2, -1), offs=offsets)
            up = torch._grouped_mm(x, w3c.transpose(-2, -1), offs=offsets)
            h = apply_expert_activation(gate, up, activation, alpha, limit)
            ctx.save_for_backward(x, offsets, w2c, h, w1c, w3c, gate, up)
        ctx.slot = slot
        ctx.activation = (activation, alpha, limit)
        ctx.fused_gate_up = fused_gate_up
        ctx.weight_dtypes = (w1.dtype, w2.dtype, w3.dtype)
        return torch._grouped_mm(h, w2c.transpose(-2, -1), offs=offsets).type_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        x, offsets, w2c, h, *projection = ctx.saved_tensors
        grad_h = torch._grouped_mm(grad_out, w2c, offs=offsets)
        if ctx.fused_gate_up:
            w13, gate_up = projection
            grad_gate_up = _packed_expert_activation_backward(grad_h, gate_up, *ctx.activation)
            grad_x = torch._grouped_mm(grad_gate_up, w13, offs=offsets)
        else:
            w1c, w3c, gate, up = projection
            grad_gate, grad_up = _expert_activation_backward(grad_h, gate, up, *ctx.activation)
            grad_x = torch._grouped_mm(grad_gate, w1c, offs=offsets) + torch._grouped_mm(grad_up, w3c, offs=offsets)

        def weight_grads():
            grad_w2 = torch._grouped_mm(grad_out.transpose(-2, -1), h, offs=offsets)
            if ctx.fused_gate_up:
                grad_w13 = torch._grouped_mm(grad_gate_up.transpose(-2, -1), x, offs=offsets)
                inter = grad_w13.shape[1] // 2
                grad_w1, grad_w3 = grad_w13.narrow(1, 0, inter), grad_w13.narrow(1, inter, inter)
            else:
                grad_w1 = torch._grouped_mm(grad_gate.transpose(-2, -1), x, offs=offsets)
                grad_w3 = torch._grouped_mm(grad_up.transpose(-2, -1), x, offs=offsets)
            return tuple(grad.to(dtype) for grad, dtype in zip((grad_w1, grad_w2, grad_w3), ctx.weight_dtypes))

        if ctx.slot is None:
            return (grad_x, None, *weight_grads(), None, None, None, None, None)
        ctx.slot.defer(weight_grads)
        return (grad_x, ) + (None, ) * 9


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
        gate_up_impl (str): ``"separate"`` runs the gate and up projections as two grouped GEMMs;
            ``"fused"`` runs them as one over the concatenated weights. ``"fused"`` needs the
            ``torch._grouped_mm`` path.
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
        gate_up_impl: str = "separate",
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

        if gate_up_impl not in ("separate", "fused"):
            raise ValueError(f'gate_up_impl must be "separate" or "fused", got {gate_up_impl!r}')
        self.gate_up_impl = gate_up_impl
        if gate_up_impl == "fused" and (not use_grouped_mm or self.use_triton_grouped_mm):
            selected = "the Triton grouped GEMM" if use_grouped_mm else "the sequential expert loop"
            raise ValueError('gate_up_impl="fused" runs the gate and up projections as one torch._grouped_mm, '
                             f"but {selected} was selected. Set use_grouped_mm=true and, on devices that prefer "
                             'the Triton kernel, disable_triton_grouped_mm=true, or leave gate_up_impl unset.')

        if use_grouped_mm and self.use_triton_grouped_mm:
            warning_once("Triton grouped-GEMM path is selected for grouped_gemm. "
                         "The Triton path is preferred on compute capability smaller than sm90, "
                         "and will be used instead of torch._grouped_mm. Set use_grouped_mm=False or "
                         "disable_triton_grouped_mm=True to avoid this warning.")

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        weight_grad_slot: ExpertWeightGradSlot | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tokens, shape ``(T, dim)``.
            num_tokens_per_expert: Token counts per expert, shape ``(E,)``.
            weight_grad_slot: Where the backward leaves the weight-gradient GEMMs for a later backward to run;
                the expert weights' gradients then reach them through that later backward. Needs the
                ``torch._grouped_mm`` path.

        Returns:
            Output tensor of shape ``(T, dim)``.
        """

        act = (self.activation, self.activation_alpha, self.activation_limit)
        if weight_grad_slot is not None:
            if self.use_triton_grouped_mm or not self.use_grouped_mm:
                raise RuntimeError("deferring expert weight gradients needs the torch._grouped_mm path")
            return _DeferredWeightGradExperts.apply(x, num_tokens_per_expert, self.w1, self.w2, self.w3,
                                                    weight_grad_slot, *act, self.gate_up_impl == "fused")
        if self.use_triton_grouped_mm:
            return _run_experts_triton_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert, *act)
        elif self.use_grouped_mm:
            if self.gate_up_impl == "fused":
                return _run_experts_grouped_mm_fused_gate_up(self.w1, self.w2, self.w3, x, num_tokens_per_expert, *act)
            return _run_experts_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert, *act)
        else:
            return _run_experts_for_loop(self.w1, self.w2, self.w3, x, num_tokens_per_expert, *act)

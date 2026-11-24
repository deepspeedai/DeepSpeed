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

import math
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


def _swiglu_clamped_fused(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    from deepspeed.ops.triton_ops.swiglu_triton import swiglu_clamped
    return swiglu_clamped(gate, up, alpha=alpha, limit=limit, oai=False)


def _swiglu_oai_fused(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    from deepspeed.ops.triton_ops.swiglu_triton import swiglu_clamped
    return swiglu_clamped(gate, up, alpha=alpha, limit=limit, oai=True)


def _swiglu_clamped_packed_fused(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    from deepspeed.ops.triton_ops.swiglu_triton import swiglu_clamped_packed
    return swiglu_clamped_packed(gate_up, alpha=alpha, limit=limit, oai=False)


def _swiglu_oai_packed_fused(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    from deepspeed.ops.triton_ops.swiglu_triton import swiglu_clamped_packed
    return swiglu_clamped_packed(gate_up, alpha=alpha, limit=limit, oai=True)


register_expert_activation("swiglu",
                           _swiglu,
                           fused_fn=_swiglu_fused,
                           gate_fn=F.silu,
                           packed_fused_fn=_swiglu_packed_fused,
                           fused_backward_fn=_swiglu_fused_backward,
                           packed_fused_backward_fn=_swiglu_packed_fused_backward)
register_expert_activation("geglu_tanh", _geglu_tanh, gate_fn=_gelu_tanh)
register_expert_activation("swiglu_clamped",
                           _swiglu_clamped,
                           fused_fn=_swiglu_clamped_fused,
                           uses_limit=True,
                           gate_fn=F.silu,
                           packed_fused_fn=_swiglu_clamped_packed_fused)
register_expert_activation("swiglu_oai",
                           _swiglu_oai,
                           fused_fn=_swiglu_oai_fused,
                           uses_alpha=True,
                           uses_limit=True,
                           packed_fused_fn=_swiglu_oai_packed_fused)


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


class _ZeroEmptyExpertGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, num_tokens_per_expert):
        ctx.save_for_backward(num_tokens_per_expert)
        return weight

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return None, None
        (num_tokens_per_expert, ) = ctx.saved_tensors
        inactive = num_tokens_per_expert == 0
        # Every expert is live on the common path. masked_fill would still copy each
        # weight gradient; return the buffer the grouped GEMM already produced.
        if not bool(inactive.any()):
            return grad, None
        # Zero only the empty-expert rows. A non-contiguous grad cannot be written in place.
        if not grad.is_contiguous():
            grad = grad.contiguous()
        grad[inactive] = 0
        return grad, None


def _zero_empty_expert_grad(weight: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
    # Some grouped-GEMM backends leave the weight gradient for a zero-row group undefined.
    if not weight.requires_grad:
        return weight
    return _ZeroEmptyExpertGrad.apply(weight, num_tokens_per_expert)


def _grouped_mm_weight(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``weight`` in ``dtype``, at an address ``torch._grouped_mm`` accepts.

    The CUDA grouped GEMM needs 16-byte aligned operands. A tensor's own allocation is, but ZeRO-3 gathers a
    parameter into a slice of a shared flat buffer, which starts wherever the parameters before it end.
    """
    weight = weight.to(dtype)
    if weight.data_ptr() % 16 != 0:
        weight = weight.clone()
    return weight


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
    w1 = _zero_empty_expert_grad(w1, num_tokens_per_expert)
    w2 = _zero_empty_expert_grad(w2, num_tokens_per_expert)
    w3 = _zero_empty_expert_grad(w3, num_tokens_per_expert)
    gate = torch._grouped_mm(
        x.to(cast_dtype),
        _grouped_mm_weight(w1, cast_dtype).transpose(-2, -1),
        offs=offsets,
    )
    up = torch._grouped_mm(
        x.to(cast_dtype),
        _grouped_mm_weight(w3, cast_dtype).transpose(-2, -1),
        offs=offsets,
    )
    h = apply_expert_activation(gate, up, activation, alpha, limit)
    out = torch._grouped_mm(
        h,
        _grouped_mm_weight(w2, cast_dtype).transpose(-2, -1),
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
    results. The forward expression and parameter-gradient semantics are unchanged; exact numerical equality
    across GEMM shapes and implementations is not guaranteed.

    Args mirror :func:`_run_experts_grouped_mm`.
    """
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    cast_dtype = x.dtype
    w1 = _zero_empty_expert_grad(w1, num_tokens_per_expert)
    w2 = _zero_empty_expert_grad(w2, num_tokens_per_expert)
    w3 = _zero_empty_expert_grad(w3, num_tokens_per_expert)
    w13 = torch.cat([w1.to(cast_dtype), w3.to(cast_dtype)], dim=1)
    gate_up = torch._grouped_mm(
        x.to(cast_dtype),
        w13.transpose(-2, -1),
        offs=offsets,
    )
    h = apply_packed_expert_activation(gate_up, activation, alpha, limit)
    out = torch._grouped_mm(
        h,
        _grouped_mm_weight(w2, cast_dtype).transpose(-2, -1),
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

    def defer(self, compute: Callable[[], tuple[torch.Tensor | None, torch.Tensor | None,
                                                torch.Tensor | None]]) -> None:
        if self._compute is not None:
            raise RuntimeError("an expert weight gradient was deferred twice into one slot")
        self._compute = compute

    def run(self) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self._compute is None:
            raise RuntimeError("the expert backward did not run before the backward that computes its weight "
                               "gradients")
        compute, self._compute = self._compute, None
        return compute()


def _grouped_mlp_weight_grads(grad_out, x, h, offsets, projection_grads, fused_gate_up, needs_w1, needs_w2, needs_w3):
    """The weight gradients of the grouped-GEMM expert MLP in ``x.dtype``, or ``None`` where not needed.

    The products are the ones autograd forms for ``torch._grouped_mm`` (``_grouped_mm_mat2_backward``).
    """
    grad_w1 = grad_w2 = grad_w3 = None
    if needs_w2:
        grad_w2 = torch._grouped_mm(grad_out.transpose(-2, -1), h, offs=offsets)
    if fused_gate_up:
        if needs_w1 or needs_w3:
            (grad_gate_up, ) = projection_grads
            grad_w13 = torch._grouped_mm(grad_gate_up.transpose(-2, -1), x, offs=offsets)
            inter = grad_w13.shape[1] // 2
            if needs_w1:
                grad_w1 = grad_w13.narrow(1, 0, inter)
            if needs_w3:
                grad_w3 = grad_w13.narrow(1, inter, inter)
    else:
        grad_gate, grad_up = projection_grads
        if needs_w1:
            grad_w1 = torch._grouped_mm(grad_gate.transpose(-2, -1), x, offs=offsets)
        if needs_w3:
            grad_w3 = torch._grouped_mm(grad_up.transpose(-2, -1), x, offs=offsets)
    return grad_w1, grad_w2, grad_w3


def _add_weight_grads(first, second):
    """Sum two ``(grad_w1, grad_w2, grad_w3)`` tuples, either of which may be ``None`` or hold ``None``."""
    if first is None:
        return second
    if second is None:
        return first
    total = []
    for a, b in zip(first, second):
        if a is None:
            total.append(b)
        elif b is None:
            total.append(a)
        else:
            total.append(a + b)
    return tuple(total)


class MoeDeferredDwStaging:
    """Stages the rows an expert MLP's weight-gradient GEMMs read, per expert, across backward passes.

    With hundreds of experts or small microbatches each expert receives few rows per backward, and the
    weight-gradient GEMMs write three full-size gradients for a handful of rows: they are bound by memory
    bandwidth, not arithmetic. A backward that stages appends each expert's rows behind that expert's cursor and
    returns no weight gradient. One backward in every ``flush_period`` runs the GEMMs once over everything staged
    and returns the result, which is the sum the staged backwards would have accumulated one at a time. So does
    every backward while ``flush_on_stage`` is set. The engine sets it for the backward that closes a gradient
    accumulation window, so no staged row outlives the optimizer step it belongs to.

    Which backward returns a weight gradient therefore depends on the count of backward passes alone, never on the
    rows. Ranks that hold replicas of the same experts route different rows, and ZeRO reduces a gradient across
    them in the backward that returns it, so they must all return it in the same one. When an expert's cursor
    would pass ``rows_per_expert`` the GEMMs do run early to empty the buffer, but their result is held and
    returned with the next of those backwards.

    Args:
        num_experts: Experts the staged MLP holds.
        flush_period: Backward passes between two periodic flushes.
        rows_per_expert: Rows each expert can stage. ``0`` sizes it on the first backward, to ``flush_period``
            times the larger of that backward's busiest expert and twice its mean rows per expert.
    """

    def __init__(self, num_experts: int, flush_period: int, rows_per_expert: int = 0):
        self.num_experts = num_experts
        self.flush_period = flush_period
        self.rows_per_expert = rows_per_expert
        self.flush_on_stage = False
        #: Flushes run so far, and how many of them an expert's cursor forced.
        self.num_flushes = 0
        self.num_overflow_flushes = 0
        self._buffers = None
        self._cursor = None
        self._staged_rows = 0
        self._held_grads = None
        self._backwards_in_period = 0
        self._flush_args = None

    @property
    def has_staged_rows(self) -> bool:
        """Whether a later backward still owes the weight gradients of rows staged so far."""
        return self._staged_rows > 0 or self._held_grads is not None

    def stage(self, rows, group_counts, valid_counts, flush_args, compute_now):
        """Stage one backward's rows, and return the weight gradients of any flush this backward runs.

        Args:
            rows: ``(x, h, grad_out, *projection_grads)``, each with ``group_counts[e]`` consecutive rows per
                expert.
            group_counts: Rows per expert in ``rows``, shape ``(E,)``.
            valid_counts: How many leading rows of each expert's group are real; the rest are alignment padding
                that contributes nothing and is not staged. ``None`` stages every row.
            flush_args: ``(fused_gate_up, needs_w1, needs_w2, needs_w3, weight_dtypes)`` of this backward.
            compute_now: Runs this backward's own weight-gradient GEMMs, for rows no cursor can hold.

        Returns:
            ``(grad_w1, grad_w2, grad_w3)``; all ``None`` unless this backward closes a period.
        """
        has_padding = valid_counts is not None
        if valid_counts is None:
            valid_counts = group_counts
        valid_counts = valid_counts.to(torch.long)
        if self._cursor is None or self._cursor.device != valid_counts.device:
            self._require_empty("the expert rows moved to another device")
            self._cursor = torch.zeros_like(valid_counts)
        if self._flush_args is not None and self._flush_args != flush_args:
            self._require_empty("requires_grad or the dtype of an expert weight changed")
        self._flush_args = flush_args

        # Whether the rows fit has to be decided on the host; one synchronization reads all it needs.
        stats = torch.stack(((self._cursor + valid_counts).max(), valid_counts.max(), valid_counts.sum()))
        fullest_cursor, busiest_expert, num_rows = stats.tolist()
        if self.rows_per_expert == 0 and num_rows > 0:
            mean_rows = math.ceil(num_rows / self.num_experts)
            self.rows_per_expert = self.flush_period * max(2 * mean_rows, busiest_expert)

        # An early flush below does not move the period, and a closed accumulation window restarts it.
        closes_period = self.flush_on_stage or self._backwards_in_period + 1 >= self.flush_period
        self._backwards_in_period = 0 if closes_period else self._backwards_in_period + 1

        if fullest_cursor > self.rows_per_expert:
            self.num_overflow_flushes += 1
            if self._staged_rows > 0:
                self._held_grads = _add_weight_grads(self._held_grads, self._flush())
        if busiest_expert > self.rows_per_expert:
            # Not even the emptied buffer holds this backward.
            self._held_grads = _add_weight_grads(self._held_grads, compute_now())
        elif num_rows > 0:
            self._append(rows, group_counts if has_padding else None, valid_counts, num_rows)

        if not closes_period:
            return None, None, None
        grads = self._held_grads
        self._held_grads = None
        if self._staged_rows > 0:
            grads = _add_weight_grads(grads, self._flush())
        if grads is None:
            # No expert received a row in this period. The other ranks still return a gradient now, so return
            # this backward's own, which is zero.
            grads = compute_now()
        return grads

    @property
    def _zero_row(self) -> int:
        # One row past every expert's slice stays zero; a flush reads it for experts that staged nothing.
        return self.num_experts * self.rows_per_expert

    def _require_empty(self, what_changed: str, buffers_only: bool = False) -> None:
        # Gradients held from an early flush do not depend on the layout of the buffers.
        has_staged_rows = self._staged_rows > 0 if buffers_only else self.has_staged_rows
        if has_staged_rows:
            raise RuntimeError(f"{what_changed} while expert weight gradients were staged; it may only change "
                               "after the backward that flushes them")

    def _append(self, rows, group_counts, valid_counts, num_rows):
        reusable = self._buffers is not None and all(
            buffer.dtype == row.dtype and buffer.device == row.device and buffer.shape[1] == row.shape[1]
            and buffer.shape[0] == self._zero_row + 1 for buffer, row in zip(self._buffers, rows))
        if not reusable:
            self._require_empty("the dtype, device or width of the expert rows changed", buffers_only=True)
            self._buffers = [row.new_empty((self._zero_row + 1, row.shape[1])) for row in rows]
            for buffer in self._buffers:
                buffer[self._zero_row] = 0

        # Staged row r of this backward is the within-th valid row of its expert.
        row_ids = torch.arange(num_rows, device=valid_counts.device)
        valid_ends = torch.cumsum(valid_counts, dim=0)
        expert = torch.searchsorted(valid_ends, row_ids, right=True)
        within = row_ids - (valid_ends - valid_counts)[expert]
        destination = expert * self.rows_per_expert + self._cursor[expert] + within
        source = None
        if group_counts is not None:
            # Each expert's group ends in alignment padding, so its valid rows are not next to the following group's.
            group_ends = torch.cumsum(group_counts.to(torch.long), dim=0)
            source = (group_ends - group_counts)[expert] + within
        for buffer, row in zip(self._buffers, rows):
            staged = row[:num_rows] if source is None else row.index_select(0, source)
            buffer.index_copy_(0, destination, staged)
        self._cursor += valid_counts
        self._staged_rows += num_rows

    def _flush(self):
        fused_gate_up, needs_w1, needs_w2, needs_w3, weight_dtypes = self._flush_args
        cursor = self._cursor
        # torch._grouped_mm does not return for a group without rows, so an expert that staged nothing gets one
        # zero row, which also makes its gradient exactly zero.
        group_sizes = cursor.clamp(min=1)
        num_rows = self._staged_rows + int((cursor == 0).sum())
        # Gather the rows of every expert next to each other, the layout the grouped GEMM reads.
        row_ids = torch.arange(num_rows, device=cursor.device)
        ends = torch.cumsum(group_sizes, dim=0)
        expert = torch.searchsorted(ends, row_ids, right=True)
        staged_index = expert * self.rows_per_expert + row_ids - (ends - group_sizes)[expert]
        index = torch.where(cursor[expert] > 0, staged_index, self._zero_row)
        x, h, grad_out, *projection_grads = [buffer.index_select(0, index) for buffer in self._buffers]
        grads = _grouped_mlp_weight_grads(grad_out, x, h, ends.to(torch.int32), projection_grads, fused_gate_up,
                                          needs_w1, needs_w2, needs_w3)
        self._cursor = torch.zeros_like(cursor)
        self._staged_rows = 0
        self.num_flushes += 1
        return tuple(grad.to(dtype) if grad is not None else None for grad, dtype in zip(grads, weight_dtypes))


class _DeferredWeightGradExperts(torch.autograd.Function):
    """The grouped-GEMM expert MLP, with its weight gradients optionally deferred to an :class:`ExpertWeightGradSlot`
    or staged across backward passes in a :class:`MoeDeferredDwStaging`.

    Forward runs exactly what :func:`_run_experts_grouped_mm` (or its fused gate+up variant) runs. Backward forms the
    products autograd forms for ``torch._grouped_mm`` (``_grouped_mm_mat1_backward`` and
    ``_grouped_mm_mat2_backward``), so every gradient matches the plain path; only when the weight gradients are
    computed changes.
    """

    @staticmethod
    def forward(ctx,
                x,
                num_tokens_per_expert,
                w1,
                w2,
                w3,
                slot,
                activation,
                alpha,
                limit,
                fused_gate_up,
                dw_staging=None,
                num_valid_tokens_per_expert=None):
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        dtype = x.dtype
        w1c, w2c, w3c = (_grouped_mm_weight(weight.detach(), dtype) for weight in (w1, w2, w3))
        # The parameters are saved, not the tensors computed with: ZeRO-3 releases a parameter's gathered data once
        # the module's forward returns and gathers it again for the backward, and a saved alias or copy would keep
        # every layer's expert weights allocated in between. The packed gate+up weight is this forward's own copy
        # and is kept, since packing it again in the backward costs more than the staging saves.
        if fused_gate_up:
            w13 = torch.cat([w1c, w3c], dim=1)
            gate_up = torch._grouped_mm(x, w13.transpose(-2, -1), offs=offsets)
            h = apply_packed_expert_activation(gate_up, activation, alpha, limit)
            ctx.save_for_backward(x, offsets, w2, h, w13, gate_up)
        else:
            gate = torch._grouped_mm(x, w1c.transpose(-2, -1), offs=offsets)
            up = torch._grouped_mm(x, w3c.transpose(-2, -1), offs=offsets)
            h = apply_expert_activation(gate, up, activation, alpha, limit)
            ctx.save_for_backward(x, offsets, w2, h, w1, w3, gate, up)
        ctx.slot = slot
        ctx.dw_staging = dw_staging
        ctx.num_valid_tokens_per_expert = num_valid_tokens_per_expert
        ctx.activation = (activation, alpha, limit)
        ctx.fused_gate_up = fused_gate_up
        ctx.weight_dtypes = (w1.dtype, w2.dtype, w3.dtype)
        return torch._grouped_mm(h, w2c.transpose(-2, -1), offs=offsets).type_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        if torch.is_grad_enabled():
            raise RuntimeError("Deferred expert gradients have no second derivative; "
                               "backward with create_graph=True is not supported")
        x, offsets, w2, h, *projection = ctx.saved_tensors
        w2c = _grouped_mm_weight(w2.detach(), x.dtype)
        grad_out = grad_out.contiguous()
        grad_h = torch._grouped_mm(grad_out, w2c, offs=offsets)
        if ctx.fused_gate_up:
            w13, gate_up = projection
            grad_gate_up = _packed_expert_activation_backward(grad_h, gate_up, *ctx.activation)
            grad_x = torch._grouped_mm(grad_gate_up, w13, offs=offsets)
        else:
            w1, w3, gate, up = projection
            w1c, w3c = _grouped_mm_weight(w1.detach(), x.dtype), _grouped_mm_weight(w3.detach(), x.dtype)
            grad_gate, grad_up = _expert_activation_backward(grad_h, gate, up, *ctx.activation)
            grad_x = torch._grouped_mm(grad_gate, w1c, offs=offsets) + torch._grouped_mm(grad_up, w3c, offs=offsets)

        needs_w1, needs_w2, needs_w3 = ctx.needs_input_grad[2:5]
        projection_grads = (grad_gate_up, ) if ctx.fused_gate_up else (grad_gate, grad_up)

        def weight_grads():
            grads = _grouped_mlp_weight_grads(grad_out, x, h, offsets, projection_grads, ctx.fused_gate_up, needs_w1,
                                              needs_w2, needs_w3)
            return tuple(grad.to(dtype) if grad is not None else None for grad, dtype in zip(grads, ctx.weight_dtypes))

        no_grads = (None, ) * 7
        if ctx.dw_staging is not None and (needs_w1 or needs_w2 or needs_w3):
            group_counts = torch.diff(offsets, prepend=offsets.new_zeros(1))
            flush_args = (ctx.fused_gate_up, needs_w1, needs_w2, needs_w3, ctx.weight_dtypes)
            staged_grads = ctx.dw_staging.stage((x, h, grad_out, *projection_grads), group_counts,
                                                ctx.num_valid_tokens_per_expert, flush_args, weight_grads)
            return (grad_x, None, *staged_grads, *no_grads)
        if ctx.slot is None:
            return (grad_x, None, *weight_grads(), *no_grads)
        ctx.slot.defer(weight_grads)
        return (grad_x, None, None, None, None, *no_grads)


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
        dw_staging: MoeDeferredDwStaging | None = None,
        num_valid_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tokens, shape ``(T, dim)``.
            num_tokens_per_expert: Token counts per expert, shape ``(E,)``.
            weight_grad_slot: Where the backward leaves the weight-gradient GEMMs for a later backward to run;
                the expert weights' gradients then reach them through that later backward. Needs the
                ``torch._grouped_mm`` path.
            dw_staging: Where the backward stages the rows of the weight-gradient GEMMs, to run them once over
                several backward passes. Needs the ``torch._grouped_mm`` path.
            num_valid_tokens_per_expert: With ``dw_staging``, how many leading rows of each expert's group are
                real when ``num_tokens_per_expert`` counts alignment padding too, shape ``(E,)``.

        Returns:
            Output tensor of shape ``(T, dim)``.
        """

        act = (self.activation, self.activation_alpha, self.activation_limit)
        if weight_grad_slot is not None or dw_staging is not None:
            if self.use_triton_grouped_mm or not self.use_grouped_mm:
                raise RuntimeError("deferring expert weight gradients needs the torch._grouped_mm path")
            if weight_grad_slot is not None and dw_staging is not None:
                raise RuntimeError("expert weight gradients are either deferred to a slot or staged, not both")
            return _DeferredWeightGradExperts.apply(x, num_tokens_per_expert, self.w1, self.w2, self.w3,
                                                    weight_grad_slot, *act, self.gate_up_impl == "fused", dw_staging,
                                                    num_valid_tokens_per_expert)
        if self.use_triton_grouped_mm:
            return _run_experts_triton_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert, *act)
        elif self.use_grouped_mm:
            if self.gate_up_impl == "fused":
                return _run_experts_grouped_mm_fused_gate_up(self.w1, self.w2, self.w3, x, num_tokens_per_expert, *act)
            return _run_experts_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert, *act)
        else:
            return _run_experts_for_loop(self.w1, self.w2, self.w3, x, num_tokens_per_expert, *act)

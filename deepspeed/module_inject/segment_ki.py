# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Segment-style kernel injection (KI) prototype.

Architecture contract (see experiments/segment-ki-proto.md):

1. The sharding plane (AutoTP) is the sole owner of tensor-parallel structure.
   This module never shards, gathers, or re-partitions weights; it only
   consumes what AutoTP already produced.
2. Kernel replacement happens strictly inside comm-free segments: a segment is
   a contiguous run of submodules whose forwards execute no collectives. Any
   module whose forward carries a collective (e.g. ``*Allreduce`` layers) is a
   hard segment boundary and is left untouched.
3. Weight-layout transforms (e.g. fusing gate/up shards) happen per shard and
   are legal only because both projections are column-parallel with the same
   input: concatenating along the output dim inside one shard never crosses a
   shard boundary.
4. Module protocols (return types, HF Cache updates) stay native: this module
   swaps computation, not the surrounding HF module contracts.

This prototype implements one kernel: ``fused_glu`` (gate+up GEMM merge with
SiLU-mul), applied to structurally-detected gated MLPs. Detection is by
structure (attribute names + collective markers), not by HF class names, so
the same pattern serves every family that lays out its MLP as
gate_proj/up_proj/down_proj.
"""

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

# AutoTP layer classes whose forward performs an output all-reduce. These are
# the canonical segment boundaries. Importing layers lazily keeps this module
# importable in builds without AutoTP materialized.
from deepspeed.module_inject.layers import LinearAllreduce, LmHeadLinearAllreduce, SubParamLinearAllreduce

_ALLREDUCE_LAYERS = (LinearAllreduce, LmHeadLinearAllreduce, SubParamLinearAllreduce)


def carries_collective(module: torch.nn.Module) -> bool:
    """True if the module's own forward executes a collective.

    Conservative by design: anything that looks like a communication-bearing
    layer is treated as a boundary even if a particular runtime path (e.g.
    inference_mode collectives) would make it a no-op. A production version
    should replace this check with an explicit marker exported by the sharding
    plane, which knows authoritatively where collectives live.
    """
    return isinstance(module, _ALLREDUCE_LAYERS)


@dataclass
class GLUSegment:
    """A gated-MLP segment: gate/up are comm-free projections, down is the
    collective-bearing boundary that must be delegated to untouched."""
    parent: torch.nn.Module
    gate: torch.nn.Module
    up: torch.nn.Module
    down: torch.nn.Module


def find_glu_segments(root: torch.nn.Module) -> List[GLUSegment]:
    """Detect gated MLPs by structure anywhere under ``root``.

    The pattern is attribute-based (gate_proj/up_proj/down_proj present), not
    class-name based, so new HF families with the same layout are picked up
    without code changes here.
    """
    segments = []
    for module in root.modules():
        gate = getattr(module, "gate_proj", None)
        up = getattr(module, "up_proj", None)
        down = getattr(module, "down_proj", None)
        if not all(isinstance(m, torch.nn.Module) for m in (gate, up, down)):
            continue
        if gate.bias is not None or up.bias is not None:
            # Fusing bias-carrying projections changes the epilogue; out of
            # scope for the prototype, keep the native forward.
            continue
        if carries_collective(gate) or carries_collective(up):
            # Both projections must be comm-free for the fusion to stay inside
            # one segment; otherwise this MLP is not a candidate.
            continue
        segments.append(GLUSegment(parent=module, gate=gate, up=up, down=down))
    return segments


def _fused_glu_forward(self, input):
    """Replacement forward: one GEMM over the fused gate|up weight, then the
    fused SiLU-mul activation (native CUDA op when installed, torch composite
    otherwise), then delegate to the untouched down projection (which keeps
    its own collective). The parent's return contract is unchanged."""
    hidden = torch.matmul(input, self._ki_fused_glu_weight.transpose(-1, -2))
    if getattr(self, "_ki_fused_glu_op", None) is not None:
        return self.down_proj(self._ki_fused_glu_op.fused_silu_mul_halves(hidden))
    gate_out, up_out = hidden.chunk(2, dim=-1)
    return self.down_proj(F.silu(gate_out) * up_out)


def apply_segment_ki(model: torch.nn.Module, kernel: str = "fused_glu", backend: str = "auto") -> dict:
    """Install segment kernels on ``model`` (post-AutoTP, pre-generate).

    ``backend`` selects the implementation: "auto" uses the native CUDA op
    when a non-cpu accelerator backend won (falls back to the torch composite
    oracle otherwise); "composite" forces the oracle path.

    Returns a small report so callers (tests, journals) can assert what was
    found and replaced without introspecting the module tree again.
    """
    if kernel != "fused_glu":
        raise ValueError(f"Unknown segment kernel {kernel!r}; only 'fused_glu' is implemented")

    cuda_op = None
    if backend in ("auto", "cuda"):
        from deepspeed.accelerator import get_accelerator
        if get_accelerator().device_name() != "cpu" or backend == "cuda":
            from deepspeed.ops.module_inject import get_fused_glu_op
            cuda_op = get_fused_glu_op()

    segments = find_glu_segments(model)
    replaced = 0
    for seg in segments:
        # Per-shard layout transform: gate and up are column-parallel shards
        # sharing one input, so concatenating along dim 0 never crosses a
        # shard boundary. A production version would fold this into the
        # partitioning step instead of materializing a copy here.
        fused_weight = torch.cat([seg.gate.weight.data, seg.up.weight.data], dim=0)
        seg.parent._ki_fused_glu_weight = fused_weight
        seg.parent._ki_fused_glu_op = cuda_op
        seg.parent.forward = _fused_glu_forward.__get__(seg.parent, type(seg.parent))
        replaced += 1

    return {
        "kernel": kernel,
        "backend": "cuda" if cuda_op is not None else "composite",
        "segments_found": len(segments),
        "segments_replaced": replaced,
        "boundaries_delegated": replaced,  # each replacement delegates one collective module
    }

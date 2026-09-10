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


@dataclass
class GDNSegment:
    """A GatedDeltaNet middle segment for hybrid families (e.g. Qwen3.5).

    The four input projections share one input and are comm-free under TP, so
    they collapse into a single GEMM. Everything from conv1d through the FLA
    scan kernel to out_proj is delegated untouched: the ecosystem kernels are
    already state of the art and the trailing out_proj is the collective
    boundary."""
    parent: torch.nn.Module
    in_proj_qkv: torch.nn.Module
    in_proj_z: torch.nn.Module
    in_proj_b: torch.nn.Module
    in_proj_a: torch.nn.Module


def find_gdn_segments(root: torch.nn.Module) -> List[GDNSegment]:
    """Detect GatedDeltaNet blocks by structure (projection attribute set)."""
    segments = []
    for module in root.modules():
        parts = [getattr(module, name, None) for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]
        if not all(isinstance(m, torch.nn.Module) for m in parts):
            continue
        if any(p.bias is not None for p in parts):
            continue
        if not all(isinstance(p, torch.nn.Linear) for p in parts):
            # Under AutoTP the projections become sharded wrapper layers while
            # the grouped conv1d keeps its full-channel layout; fusing across
            # that mismatch needs shard-aware slicing (documented limitation,
            # native single-device layout only for now).
            continue
        if any(carries_collective(p) for p in parts):
            # Projections must stay inside one comm-free segment.
            continue
        segments.append(GDNSegment(module, *parts))
    return segments


def _fused_gdn_forward(self, hidden_states, *args, **kwargs):
    """GDN block forward with the four input projections fused into one GEMM.

    Mirrors the native flow (transformers Qwen3_5GatedDeltaNet.forward) while
    delegating conv1d, the delta-rule scan, the gated norm, and the
    collective-bearing out_proj to the untouched original submodules."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    # Consume the layer-level kwargs so they cannot leak into the scan call
    # (the native forward's named parameters absorb them; ours must too).
    attention_mask = kwargs.pop("attention_mask", None)
    cache_params = kwargs.pop("cache_params", None) or (args[0] if args else None)
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

    batch_size, seq_len, _ = hidden_states.shape
    use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

    # Fused [qkv | z | b | a] GEMM replacing the four separate projections.
    fused = torch.matmul(hidden_states, self._ki_gdn_fused_weight.transpose(-1, -2))
    key_dim, value_dim = self.key_dim, self.value_dim
    qkv_end = key_dim * 2 + value_dim
    z_end = qkv_end + value_dim
    b_end = z_end + self.num_v_heads
    # One explicit repack of the qkv slice into the [b, conv_dim, s] layout
    # conv1d consumes; handing FLA a non-contiguous view triggers a slower
    # multi-op internal path (measured as ~40 extra launches per layer).
    mixed_qkv = fused[..., :qkv_end].transpose(1, 2).contiguous()
    z = fused[..., qkv_end:z_end].reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = fused[..., z_end:b_end]
    a = fused[..., b_end:]

    if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
        conv_state = cache_params.layers[self.layer_idx].conv_states[0]
        mixed_qkv = self._ki_gdn_conv_update(mixed_qkv, conv_state)
    else:
        if cache_params is not None:
            mixed_qkv = cache_params.update_conv_state(mixed_qkv,
                                                       self.layer_idx,
                                                       conv_kernel_size=self.conv_kernel_size)
        mixed_qkv = self._ki_gdn_conv_fn(mixed_qkv, **kwargs)
        if cache_params is not None:
            mixed_qkv = mixed_qkv[:, :, -seq_len:]

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    gdn_op = getattr(self, "_ki_gdn_op", None)
    from deepspeed.accelerator import get_accelerator
    if gdn_op is not None and get_accelerator().on_accelerator(a):
        a_log_f = self.A_log.detach().float().contiguous()
        dt_f = self.dt_bias.detach().float().contiguous()
        # gdn_gates accepts row-strided a/b, so the fused-output slices go in
        # without contiguous copies.
        beta, g = gdn_op.gdn_gates(a, b, a_log_f, dt_f)
    else:
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
    # Whitelist the scan inputs: transformers threads unrelated layer kwargs
    # (use_cache, cache_position, ...) down to every block and the fused
    # kernel signatures reject unknown names.
    scan_kwargs = dict(g=g,
                       beta=beta,
                       initial_state=recurrent_state,
                       output_final_state=cache_params is not None,
                       use_qk_l2norm_in_kernel=True)
    cu_seqlens = kwargs.get("cu_seq_lens_q") or kwargs.get("cu_seqlens")
    if cu_seqlens is not None:
        scan_kwargs["cu_seqlens"] = cu_seqlens
    core_attn_out, last_recurrent_state = self._ki_gdn_scan(query, key, value, **scan_kwargs)
    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


def _install_gdn_segment(seg: GDNSegment, backend: str) -> bool:
    """Wire the fused forward onto a GDN block, binding its original helpers.

    The conv/scan callables are captured from the transformers module so the
    replacement keeps using the exact kernels (FLA/causal-conv1d paths and
    their cache contracts) that the native forward would have selected."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as m5

    conv_update = getattr(m5, "causal_conv1d_update", None)
    conv_fn = getattr(m5, "causal_conv1d_fn", None)
    if conv_update is None or conv_fn is None:
        return False

    parent = seg.parent
    fused_weight = torch.cat(
        [seg.in_proj_qkv.weight.data, seg.in_proj_z.weight.data, seg.in_proj_b.weight.data, seg.in_proj_a.weight.data],
        dim=0)

    def conv_update_with_weights(mixed_qkv, conv_state):
        return conv_update(mixed_qkv, conv_state, parent.conv1d.weight.squeeze(1), parent.conv1d.bias,
                           parent.activation)

    def conv_fn_with_weights(mixed_qkv, **kw):
        return conv_fn(mixed_qkv,
                       parent.conv1d.weight.squeeze(1),
                       parent.conv1d.bias,
                       activation=parent.activation,
                       **kw)

    def scan(query, key, value, **kw):
        # Route through the block's own instance-bound kernels: transformers
        # binds the FLA fused implementations onto the module when available
        # (recurrent_gated_delta_rule/chunk_gated_delta_rule) and only falls
        # back to the pure-torch references otherwise. Calling the module-level
        # torch_* functions directly forces the slow fallback on every layer.
        single = kw.get("initial_state") is not None and query.shape[1] == 1
        kw.pop("cu_seqlens", None)
        kw.pop("cu_seq_lens_q", None)
        fn = parent.recurrent_gated_delta_rule if single else parent.chunk_gated_delta_rule
        return fn(query, key, value, **kw)

    cuda_op = None
    if backend in ("auto", "cuda"):
        try:
            from deepspeed.accelerator import get_accelerator
            if get_accelerator().device_name() != "cpu" or backend == "cuda":
                from deepspeed.ops.module_inject import get_fused_glu_op
                cuda_op = get_fused_glu_op()
        except Exception:
            cuda_op = None
    parent._ki_gdn_fused_weight = fused_weight
    parent._ki_gdn_op = cuda_op
    parent._ki_gdn_conv_update = conv_update_with_weights
    parent._ki_gdn_conv_fn = conv_fn_with_weights
    parent._ki_gdn_scan = scan
    parent.forward = _fused_gdn_forward.__get__(parent, type(parent))
    return True


def apply_segment_ki(model: torch.nn.Module, kernel: str = "all", backend: str = "auto") -> dict:
    """Install segment kernels on ``model`` (post-AutoTP, pre-generate).

    ``kernel`` selects the pattern set: "fused_glu" (gated MLP), "fused_gdn"
    (GatedDeltaNet input projections), or "all". ``backend`` selects the
    implementation: "auto" uses the native CUDA op when a non-cpu accelerator
    backend won (falls back to the torch composite oracle otherwise);
    "composite" forces the oracle path.

    Returns a small report so callers (tests, journals) can assert what was
    found and replaced without introspecting the module tree again.
    """
    if kernel not in ("fused_glu", "fused_gdn", "all"):
        raise ValueError(f"Unknown segment kernel {kernel!r}; choose from fused_glu/fused_gdn/all")

    cuda_op = None
    if backend in ("auto", "cuda") and kernel in ("fused_glu", "all"):
        from deepspeed.accelerator import get_accelerator
        if get_accelerator().device_name() != "cpu" or backend == "cuda":
            from deepspeed.ops.module_inject import get_fused_glu_op
            cuda_op = get_fused_glu_op()

    report = {"kernel": kernel, "backend": "cuda" if cuda_op is not None else "composite"}

    if kernel in ("fused_glu", "all"):
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
        report["fused_glu"] = {"segments_found": len(segments), "segments_replaced": replaced}

    if kernel in ("fused_gdn", "all"):
        gdn_segments = find_gdn_segments(model)
        gdn_replaced = sum(1 for seg in gdn_segments if _install_gdn_segment(seg, backend))
        report["fused_gdn"] = {"segments_found": len(gdn_segments), "segments_replaced": gdn_replaced}

    return report

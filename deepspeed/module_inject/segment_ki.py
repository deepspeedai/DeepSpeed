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
from deepspeed.module_inject.layers import LinearAllreduce, SubParamLinearAllreduce

# Only list base classes here — isinstance() covers all subclasses
# automatically (e.g., Yuan_LinearAllreduce and Conv_LinearALlreduce are
# caught via LinearAllreduce). Do not add subclasses to this tuple.
_ALLREDUCE_LAYERS = (LinearAllreduce, SubParamLinearAllreduce)


def carries_collective(module: torch.nn.Module) -> bool:
    """True if the module's own forward executes a collective."""
    return isinstance(module, _ALLREDUCE_LAYERS)


@dataclass
class GLUSegment:
    """A gated-MLP segment: gate/up are comm-free projections, down is the
    collective-bearing boundary that must be delegated to untouched."""
    parent: torch.nn.Module
    gate: torch.nn.Module
    up: torch.nn.Module
    down: torch.nn.Module


def _find_fusable_projections(root: torch.nn.Module, attr_names: tuple) -> dict:
    """Yield (module, projections_dict) for each module under ``root`` whose
    named projections are bias-free, comm-free linear-like layers with a
    2-D weight.

    Duck-typed (weight + bias attrs): AutoTP replaces nn.Linear with a
    LinearLayer wrapper that still exposes both. Shard-aware fusion is
    unnecessary — the kernels derive per-rank shapes from weight.shape,
    so sharded weights work unchanged at any TP degree.
    """
    for module in root.modules():
        projections = {name: getattr(module, name, None) for name in attr_names}
        if not all(p is not None and hasattr(p, "weight") and hasattr(p, "bias") for p in projections.values()):
            continue
        if any(p.weight is None or p.weight.dim() != 2 for p in projections.values()):
            continue
        if any(p.bias is not None for p in projections.values()):
            continue
        if any(carries_collective(p) for p in projections.values()):
            continue
        yield module, projections


def _is_silu_activation(act) -> bool:
    """True when ``act`` is SiLU as a torch module, the plain function, or
    transformers' SiLUActivation wrapper (what Qwen3.5 MLPs carry)."""
    if isinstance(act, torch.nn.SiLU):
        return True
    if act is torch.nn.functional.silu:
        return True
    return type(act).__name__ == "SiLUActivation"


def find_glu_segments(root: torch.nn.Module) -> List[GLUSegment]:
    """Detect gated MLPs by structure (gate_proj/up_proj/down_proj attribute set)."""
    segments = []
    # Only gate/up are fused (read directly by the kernel); down_proj is
    # delegated to its own forward, so it may carry a collective — under
    # AutoTP it is a LinearAllreduce and must stay excluded from the
    # comm-free check above.
    for module, p in _find_fusable_projections(root, ("gate_proj", "up_proj")):
        down = getattr(module, "down_proj", None)
        if down is None or not hasattr(down, "weight"):
            continue
        # The fused replacement hardcodes SiLU, while the name-based selector
        # also matches GELU-gated families (e.g. Gemma). Only fuse when the
        # module's own activation is positively identified as SiLU.
        activations = [getattr(module, name, None) for name in ("act_fn", "activation_fn", "activation_func")]
        if not any(_is_silu_activation(act) for act in activations if act is not None):
            continue
        segments.append(GLUSegment(parent=module, gate=p["gate_proj"], up=p["up_proj"], down=down))
    return segments


def _native_kernel_ready(kernel_op, hidden, *weights) -> bool:
    """The CUDA kernels read raw BFloat16 pointers only, so anything else
    (fp16 weights, non-contiguous views) must take the composite reference
    path instead of raising inside the binding."""
    if kernel_op is None:
        return False
    if not hidden.is_contiguous() or hidden.dtype is not torch.bfloat16:
        return False
    from deepspeed.accelerator import get_accelerator
    if not get_accelerator().on_accelerator(hidden):
        return False
    return all(w.is_contiguous() and w.dtype is torch.bfloat16 for w in weights)


class DualWeightGluGEMV(torch.autograd.Function):
    """silu(hidden @ gate_w.T) * (hidden @ up_w.T) reading weights directly.

    Forward (b=1): fused kernel (one warp per output feature, coalesced
    reads from both weight matrices, fp32 accumulation).
    Forward (b>1): two GEMMs + elementwise activation.
    Backward: standard PyTorch ops — gradients flow to the
    original gate.weight / up.weight Parameters.
    """

    @staticmethod
    def forward(ctx, hidden, gate_w, up_w, kernel_op):
        ctx.save_for_backward(hidden, gate_w, up_w)
        if _native_kernel_ready(kernel_op, hidden, gate_w, up_w):
            if hidden.dim() == 1:
                out = torch.empty(gate_w.shape[0], dtype=hidden.dtype, device=hidden.device)
                kernel_op.dual_gemv_silu_mul(hidden, gate_w, up_w, out)
                return out
            if hidden.dim() == 2:
                # Route through grouped_linear: the backend dispatcher owns
                # the WMMA-vs-composite crossover (arch-adaptive since loop 2)
                return kernel_op.grouped_linear(hidden, [gate_w, up_w], True)[0]
        from deepspeed.module_inject.kernel_reference import dual_gemv_silu_mul
        return dual_gemv_silu_mul(hidden, gate_w, up_w)

    @staticmethod
    def backward(ctx, grad_out):
        hidden, gate_w, up_w = ctx.saved_tensors
        gate_out = torch.matmul(hidden, gate_w.t())
        up_out = torch.matmul(hidden, up_w.t())
        sig = torch.sigmoid(gate_out)
        silu_prime = sig * (1.0 + gate_out * (1.0 - sig))
        grad_gate_coef = silu_prime * up_out * grad_out
        grad_up_coef = (gate_out * sig) * grad_out
        if hidden.dim() == 1:
            grad_gate_w = torch.outer(grad_gate_coef, hidden).view_as(gate_w)
            grad_up_w = torch.outer(grad_up_coef, hidden).view_as(up_w)
            grad_hidden = torch.matmul(grad_gate_coef, gate_w) + torch.matmul(grad_up_coef, up_w)
        else:
            # Training activations are [batch, seq, hidden]; .t() only works
            # on 2-D, so flatten the leading dims as the GDN backward does.
            flat_hidden = hidden.reshape(-1, hidden.shape[-1])
            grad_gate_w = torch.matmul(grad_gate_coef.reshape(-1, grad_gate_coef.shape[-1]).t(),
                                       flat_hidden).view_as(gate_w)
            grad_up_w = torch.matmul(grad_up_coef.reshape(-1, grad_up_coef.shape[-1]).t(), flat_hidden).view_as(up_w)
            grad_hidden = torch.matmul(grad_gate_coef, gate_w) + torch.matmul(grad_up_coef, up_w)
        return grad_hidden, grad_gate_w, grad_up_w, None


class GDNInputProj(torch.autograd.Function):
    """GDN input projections (qkv|z|b|a) reading the four original weights.

    Forward (b=1): quad_gemv kernel — one warp per output row across all
    four weight matrices, no concatenation.
    Forward (b>1): four GEMMs concatenated on the fly.
    Backward: standard PyTorch ops — gradients flow to the original
    in_proj_qkv/z/b/a Parameters.
    """

    @staticmethod
    def forward(ctx, hidden, w_qkv, w_z, w_b, w_a, kernel_op):
        ctx.save_for_backward(hidden, w_qkv, w_z, w_b, w_a)
        if _native_kernel_ready(kernel_op, hidden, w_qkv, w_z, w_b, w_a):
            if hidden.dim() == 1:
                total = w_qkv.shape[0] + w_z.shape[0] + w_b.shape[0] + w_a.shape[0]
                out = torch.empty(total, dtype=hidden.dtype, device=hidden.device)
                kernel_op.quad_gemv(hidden, w_qkv, w_z, w_b, w_a, out)
                return out
            if hidden.dim() == 3 and hidden.shape[1] == 1 and 1 < hidden.shape[0] <= 32:
                # decode activations arrive [B,1,H]; flatten for the grouped kernel
                out = kernel_op.quad_gemm(hidden.reshape(hidden.shape[0], -1), w_qkv, w_z, w_b, w_a)
                return out.view(hidden.shape[0], 1, -1)
            if hidden.dim() == 2 and 1 < hidden.shape[0] <= 32:
                return kernel_op.quad_gemm(hidden, w_qkv, w_z, w_b, w_a)
        from deepspeed.module_inject.kernel_reference import gdn_input_proj
        return gdn_input_proj(hidden, w_qkv, w_z, w_b, w_a)

    @classmethod
    def apply6(cls, x, q_w, k_w, v_w, z_w, b_w, a_w, kernel_op):
        """TP block-sliced GDN in_proj through the same grouped_linear
        interface as the fused 4-mat form (path consistency TP=1/TP=2).
        Callers that pre-concatenated pass the same tensor for q/k/v and
        for z/b/a: identical weights collapse to a 2-mat group."""
        if q_w is k_w and k_w is v_w and z_w is b_w and b_w is a_w:
            ws = [q_w, z_w]
        else:
            ws = [q_w, k_w, v_w, z_w, b_w, a_w]
        if _native_kernel_ready(kernel_op, x, *ws):
            if hasattr(kernel_op, "grouped_linear"):
                out = kernel_op.grouped_linear(x, ws, False)
                if len(out) == 2:
                    return torch.cat(out, -1)
                return out[0]
        return torch.cat([torch.nn.functional.linear(x, w) for w in ws], dim=-1)

    @staticmethod
    def backward(ctx, grad_out):
        hidden, w_qkv, w_z, w_b, w_a = ctx.saved_tensors
        rows = (w_qkv.shape[0], w_z.shape[0], w_b.shape[0], w_a.shape[0])
        # Slice grad_out back per weight, then standard GEMM backward.
        g0 = grad_out[..., :rows[0]]
        g1 = grad_out[..., rows[0]:rows[0] + rows[1]]
        g2 = grad_out[..., rows[0] + rows[1]:rows[0] + rows[1] + rows[2]]
        g3 = grad_out[..., rows[0] + rows[1] + rows[2]:]
        if hidden.dim() == 1:
            grads_w = tuple(
                torch.outer(g, hidden).view_as(w) for g, w in ((g0, w_qkv), (g1, w_z), (g2, w_b), (g3, w_a)))
            grad_hidden = (torch.matmul(g0, w_qkv) + torch.matmul(g1, w_z) + torch.matmul(g2, w_b) +
                           torch.matmul(g3, w_a))
        else:
            flat = hidden.reshape(-1, hidden.shape[-1])
            grads_w = tuple(
                torch.matmul(g.reshape(-1, g.shape[-1]).t(), flat).view_as(w)
                for g, w in ((g0, w_qkv), (g1, w_z), (g2, w_b), (g3, w_a)))
            grad_hidden = (torch.matmul(g0, w_qkv) + torch.matmul(g1, w_z) + torch.matmul(g2, w_b) +
                           torch.matmul(g3, w_a))
        return (grad_hidden, ) + grads_w + (None, )


def _dual_weight_glu_forward(self, input):
    """Replacement forward: reads gate_proj.weight and up_proj.weight
    directly (zero copies).  b=1 uses the fused dual-weight GEMV kernel;
    b>1 falls through to the original projections.  Gradients flow to the
    original Parameters through autograd.Function — train and generate share
    the same path with no forward switching."""
    # One hidden vector in any [*, 1, hidden] / [1, hidden] / [hidden]
    # shape — HF decode activations arrive as [1, 1, hidden], so keying on
    # dim()==2 would exclude the exact greedy-decode case this fusion
    # targets. The kernel op handles a flat vector; reshape on the way out.
    m = input.numel() // input.shape[-1]
    if input.numel() == input.shape[-1] and getattr(self, "_ki_dual_op", None) is not None:
        out = DualWeightGluGEMV.apply(input.reshape(-1), self.gate_proj.weight, self.up_proj.weight, self._ki_dual_op)
        return self.down_proj(out.reshape(*input.shape[:-1], -1))
    if 1 < m <= 32 and getattr(self, "_ki_dual_op", None) is not None:
        out = DualWeightGluGEMV.apply(input.reshape(m, -1), self.gate_proj.weight, self.up_proj.weight,
                                      self._ki_dual_op)
        return self.down_proj(out.reshape(*input.shape[:-1], -1))
    # other batches or no kernel: the composite reference (gradients also
    # correct there); measured faster than the batched GEMV kernels
    from deepspeed.module_inject.kernel_reference import dual_gemv_silu_mul
    return self.down_proj(dual_gemv_silu_mul(input, self.gate_proj.weight, self.up_proj.weight))


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
    """Detect GatedDeltaNet blocks by structure (in_proj attribute set)."""
    segments = []
    names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    for module, p in _find_fusable_projections(root, names):
        segments.append(GDNSegment(parent=module, **{n: p[n] for n in names}))
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
    # Shard-aware dims: under TP the in_proj weights are head-sharded, so all
    # split boundaries and head counts derive from the per-rank shapes (the
    # config values are the global ones). The colwise/rowwise GDN partition
    # keeps the whole block rank-local (per-head math); out_proj's AllReduce
    # is the layer's only collective.
    import deepspeed.comm as dist
    tp_size = dist.get_world_size() if dist.is_initialized() else 1
    r = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    key_dim = self.key_dim // tp_size
    value_dim = self.value_dim // tp_size
    num_v_heads = self.num_v_heads // tp_size
    num_k_heads = self.num_k_heads // tp_size

    # Fused [qkv | z | b | a] projection reading the original weights (b=1
    # quad_gemv kernel, b>1 on-the-fly concat) — no weight copies.
    gdn_kernel_op = getattr(self, "_ki_gdn_op", None)
    if tp_size > 1:
        # The fused [q|k|v] weight cannot be column-sharded head-aligned
        # (a contiguous row split lands mid-block), so it stays replicated
        # and each rank slices its own head ranges per block — zero-copy
        # views, no gather. z/b/a are single-block weights and shard
        # head-aligned under AutoTP.
        nq = self.key_dim // tp_size
        nv = value_dim
        qkv_w = self.in_proj_qkv.weight
        q_w = qkv_w[r * nq:(r + 1) * nq]
        k_w = qkv_w[self.key_dim + r * nq:self.key_dim + (r + 1) * nq]
        v_w = qkv_w[2 * self.key_dim + r * nv:2 * self.key_dim + (r + 1) * nv]
        x2 = hidden_states.reshape(-1, hidden_states.shape[-1])
        # Cache the per-rank concatenated weights (static during rollout):
        # one wide GEMM per group lets the backend's cuBLAS tile better
        # than six skinny row-slice GEMMs. Still routed through
        # grouped_linear — the backend owns the kernel choice.
        cw = getattr(self, "_ki_tp_cat_w", None)
        if cw is None or cw[0] is not self.in_proj_qkv.weight:
            zba_w = torch.cat([self.in_proj_z.weight, self.in_proj_b.weight, self.in_proj_a.weight])
            cw = (self.in_proj_qkv.weight, torch.cat([q_w, k_w, v_w]), zba_w)
            self._ki_tp_cat_w = cw
        fused = GDNInputProj.apply6(x2, cw[1], cw[1], cw[1], cw[2], cw[2], cw[2], gdn_kernel_op).view(
            batch_size, seq_len, -1)
    elif batch_size == 1 and seq_len == 1:
        fused = GDNInputProj.apply(
            hidden_states.squeeze(0).squeeze(0), self.in_proj_qkv.weight, self.in_proj_z.weight, self.in_proj_b.weight,
            self.in_proj_a.weight, gdn_kernel_op).view(1, 1, -1)
    else:
        fused = GDNInputProj.apply(hidden_states, self.in_proj_qkv.weight, self.in_proj_z.weight,
                                   self.in_proj_b.weight, self.in_proj_a.weight, gdn_kernel_op)
    qkv_end = key_dim * 2 + value_dim
    z_end = qkv_end + value_dim
    b_end = z_end + num_v_heads
    # One explicit repack of the qkv slice into the [b, conv_dim, s] layout
    # conv1d consumes; handing FLA a non-contiguous view triggers a slower
    # multi-op internal path (measured as ~40 extra launches per layer).
    # varlen keeps the transposed view: the conv package only accepts
    # seq_idx with the native memory format.
    qkv_view = fused[..., :qkv_end].transpose(1, 2)
    mixed_qkv = qkv_view if kwargs.get("seq_idx") is not None else qkv_view.contiguous()
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
    # The native gdn_gates op has no autograd registration, so a training
    # step through this branch would silently cut gradients to the a/b
    # projections, A_log, and dt_bias. The kernel's win is decode launch
    # overhead (a no-grad path); training runs the composite ops instead.
    # It also reads raw BFloat16 pointers only — fp16 models must take the
    # composite path rather than raise inside the binding.
    if (gdn_op is not None and get_accelerator().on_accelerator(a) and not torch.is_grad_enabled()
            and a.dtype is torch.bfloat16):
        # A_log/dt_bias are static across the rollout — cache the fp32
        # rank slice once instead of re-casting every decode step
        hv = self.num_v_heads
        cached = getattr(self, "_ki_gates_cache", None)
        if cached is None or cached[0] is not self.A_log:
            sl = (lambda t: t.detach().reshape(-1)[r * num_v_heads:(r + 1) * num_v_heads]) if hv > num_v_heads \
                else (lambda t: t.detach())
            cached = (self.A_log, sl(self.A_log).float().contiguous(), sl(self.dt_bias).float().contiguous())
            self._ki_gates_cache = cached
        a_log_f, dt_f = cached[1], cached[2]
        # gdn_gates accepts row-strided a/b, so the fused-output slices go in
        # without contiguous copies.
        beta, g = gdn_op.gdn_gates(a, b, a_log_f, dt_f)
    else:
        beta = b.sigmoid()
        hv = self.num_v_heads
        a_log_l = self.A_log.reshape(-1)[r * num_v_heads:(r + 1) * num_v_heads] if hv > num_v_heads else self.A_log
        dt_l = self.dt_bias.reshape(-1)[r * num_v_heads:(r + 1) * num_v_heads] if hv > num_v_heads else self.dt_bias
        g = -a_log_l.float().exp() * F.softplus(a.float() + dt_l)
    if num_v_heads // num_k_heads > 1:
        query = query.repeat_interleave(num_v_heads // num_k_heads, dim=2)
        key = key.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
    # Whitelist the scan inputs: transformers threads unrelated layer kwargs
    # (use_cache, cache_position, ...) down to every block and the fused
    # kernel signatures reject unknown names.
    scan_kwargs = dict(g=g,
                       beta=beta,
                       initial_state=recurrent_state,
                       output_final_state=cache_params is not None,
                       use_qk_l2norm_in_kernel=True)
    # The dict.get default avoids truth-value evaluation: `or` on a
    # multi-element cu-seqlens tensor raises "ambiguous boolean value".
    cu_seqlens = kwargs.get("cu_seq_lens_q", kwargs.get("cu_seqlens"))
    if cu_seqlens is not None:
        scan_kwargs["cu_seqlens"] = cu_seqlens
    core_attn_out, last_recurrent_state = self._ki_gdn_scan(query, key, value, **scan_kwargs)
    if cache_params is not None and last_recurrent_state is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    gdn2_op = getattr(self, "_ki_gdn_op", None)
    if (gdn2_op is not None and hasattr(gdn2_op, "gated_norm") and not torch.is_grad_enabled()
            and core_attn_out.is_contiguous() and z.is_contiguous() and core_attn_out.dtype is torch.bfloat16):
        normed = torch.empty_like(core_attn_out)
        norm_eps = getattr(self.norm, "variance_epsilon", None) or self.norm.eps
        gdn2_op.gated_norm(core_attn_out, z, self.norm.weight, normed, norm_eps)
        core_attn_out = normed
    else:
        core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


def _install_gdn_segment(seg: GDNSegment) -> bool:
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

    def _local_conv_wb():
        # The conv weight channels follow the fused [q|k|v] block order; a
        # rank's local channels are one range per block, so the local weight
        # is the cat of the three rank-indexed block rows (cached — tiny).
        import deepspeed.comm as dist
        tp_sz = dist.get_world_size() if dist.is_initialized() else 1
        if tp_sz == 1:
            return parent.conv1d.weight.squeeze(1), parent.conv1d.bias
        rk = torch.distributed.get_rank()
        nq = parent.key_dim // tp_sz
        nv = parent.value_dim // tp_sz
        w = parent.conv1d.weight.squeeze(1)
        cached = getattr(parent, "_ki_conv_w_local", None)
        if cached is not None and cached[2] == (rk, tp_sz):
            return cached[0], cached[1]
        key_dim = parent.key_dim
        wl = torch.cat([
            w[rk * nq:(rk + 1) * nq], w[key_dim + rk * nq:key_dim + (rk + 1) * nq],
            w[2 * key_dim + rk * nv:2 * key_dim + (rk + 1) * nv]
        ])
        bl = None
        if parent.conv1d.bias is not None:
            b = parent.conv1d.bias
            bl = torch.cat([
                b[rk * nq:(rk + 1) * nq], b[key_dim + rk * nq:key_dim + (rk + 1) * nq],
                b[2 * key_dim + rk * nv:2 * key_dim + (rk + 1) * nv]
            ])
        parent._ki_conv_w_local = (wl.contiguous(), bl, (rk, tp_sz))
        return wl.contiguous(), bl

    def conv_update_with_weights(mixed_qkv, conv_state):
        w, bias = _local_conv_wb()
        return conv_update(mixed_qkv, conv_state, w, bias, parent.activation)

    def conv_fn_with_weights(mixed_qkv, **kw):
        # seq_idx carries the packed-sequence boundaries; the conv takes
        # nothing else from the layer kwargs (matching the native call).
        w, bias = _local_conv_wb()
        return conv_fn(mixed_qkv, w, bias, activation=parent.activation, seq_idx=kw.get("seq_idx"))

    def scan(query, key, value, **kw):
        init_state = kw.get("initial_state")
        # decode (single token) with a live state: the in-place recurrent
        # kernel updates the cache buffer directly — no FLA output tensor,
        # no HF copy-back, half the state traffic
        step_op = getattr(parent, "_ki_gdn_op", None)
        # TP block-sliced q/k/v arrive as strided views; materializing them
        # lets our in-place kernel run, but it loses to FLA at TP head counts
        # on A100 (loop 4: -24% at bs=32) — keep the strided-view rejection
        # so TP decodes route to FLA until the kernel is retuned for H<=16
        if (init_state is not None and query.shape[1] == 1 and step_op is not None
                and hasattr(step_op, "gdn_recurrent_step") and not torch.is_grad_enabled()
                and query.shape[0] >= 16  # below that our block count starves vs FLA's tiling
                and query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
                and init_state.is_contiguous() and init_state.dtype is torch.float32):
            b_sz, _, h, dk = query.shape
            dv = value.shape[-1]
            out = torch.empty(b_sz, 1, h, dv, dtype=query.dtype, device=query.device)
            step_op.gdn_recurrent_step(query.squeeze(1), key.squeeze(1), value.squeeze(1), kw["g"], kw["beta"],
                                       init_state, out.squeeze(1))
            return out, None
        # Route through the block's own instance-bound kernels: transformers
        # binds the FLA fused implementations onto the module when available
        # (recurrent_gated_delta_rule/chunk_gated_delta_rule) and only falls
        # back to the pure-torch references otherwise. Newer transformers
        # (5.19+) no longer bind them per-layer; fall back to the module-level
        # torch_* implementations there.
        single = kw.get("initial_state") is not None and query.shape[1] == 1
        fn_name = "recurrent_gated_delta_rule" if single else "chunk_gated_delta_rule"
        fn = getattr(parent, fn_name, None)
        if fn is None:
            import sys
            fn = getattr(sys.modules[type(parent).__module__], f"torch_{fn_name}", None)
        if fn is None:
            raise AttributeError(f"GDN scan unavailable: neither {fn_name} nor torch_{fn_name}")
        return fn(query, key, value, **kw)

    parent._ki_gdn_op = _load_native_op()
    parent._ki_gdn_conv_update = conv_update_with_weights
    parent._ki_gdn_conv_fn = conv_fn_with_weights
    parent._ki_gdn_scan = scan
    parent.forward = _fused_gdn_forward.__get__(parent, type(parent))
    return True


_NATIVE_OP_LOAD_WHY = None


def _load_native_op():
    """Best-effort load of the segment-KI native op; None → composite.

    Ask the accelerator whether it registers this builder — the kernels are
    CUDA source today, so other backends answer NotImplemented and skip the
    futile JIT attempt, while a backend that implements the op later is
    picked up automatically. A registered builder whose JIT toolchain is
    unusable on this host falls back instead of raising through rollout
    construction. The fallback reason is logged once so users can tell
    which path is running."""
    global _NATIVE_OP_LOAD_WHY
    from deepspeed.accelerator import get_accelerator
    import logging
    logger = logging.getLogger(__name__)
    try:
        builder = get_accelerator().get_op_builder("FusedGLUBuilder")
    except Exception as e:
        _NATIVE_OP_LOAD_WHY = f"builder lookup failed: {e}"
        logger.warning("segment-KI native op disabled (%s); using composite fallback", _NATIVE_OP_LOAD_WHY)
        return None
    if builder is None or builder.__name__ == "NotImplementedBuilder":
        _NATIVE_OP_LOAD_WHY = f"accelerator registers no FusedGLUBuilder ({builder.__name__ if builder else 'None'})"
        logger.warning("segment-KI native op disabled (%s); using composite fallback", _NATIVE_OP_LOAD_WHY)
        return None
    try:
        return builder().load()
    except Exception as e:
        _NATIVE_OP_LOAD_WHY = f"JIT load failed: {e}"
        logger.warning("segment-KI native op disabled (%s); using composite fallback", _NATIVE_OP_LOAD_WHY)
        return None


def apply_segment_ki(model: torch.nn.Module) -> dict:
    """Install segment kernels on ``model`` (post-AutoTP, pre-generate).

    Uses the native CUDA op when a non-cpu accelerator backend is active;
    falls back to the torch composite reference otherwise.

    Returns a small report so callers (tests, journals) can assert what was
    found and replaced without introspecting the module tree again.
    """
    kernel_op = _load_native_op()

    report = {"backend": "cuda" if kernel_op is not None else "composite"}

    segments = find_glu_segments(model)
    replaced = 0
    for seg in segments:
        seg.parent._ki_dual_op = kernel_op if kernel_op is not None and hasattr(kernel_op,
                                                                                "dual_gemv_silu_mul") else None
        seg.parent.forward = _dual_weight_glu_forward.__get__(seg.parent, type(seg.parent))
        replaced += 1
    report["fused_glu"] = {"segments_found": len(segments), "segments_replaced": replaced}

    gdn_segments = find_gdn_segments(model)
    gdn_replaced = sum(1 for seg in gdn_segments if _install_gdn_segment(seg))
    report["fused_gdn"] = {"segments_found": len(gdn_segments), "segments_replaced": gdn_replaced}

    return report


# ─── Custom b=1 decode attention: replace SDPA in full-attention layers ───


def _decode_attn_forward(self,
                         hidden_states,
                         position_embeddings=None,
                         attention_mask=None,
                         past_key_values=None,
                         **kwargs):
    """Replacement forward for full-attention layers: run the custom
    decode_attn kernel at b=1, seq_len=1 instead of SDPA-with-mask (which
    falls into the slow mem_efficient backend against the full-width static
    cache buffer).

    Everything except the attention core is a verbatim replay of the HF
    forward (q_proj 2x-wide gate split, per-head q/k norms, RoPE, KV cache
    update, gate multiply, o_proj), so numerics match the original path.
    The kernel reads the valid KV length from the same GPU-resident
    write_pos tensor the cache update writes through, keeping the whole
    step CUDA-graph replayable."""
    if (hidden_states.shape[1] > 1 or hidden_states.dtype is not torch.bfloat16 or not hidden_states.is_contiguous()
            or past_key_values is None or getattr(self, "_ki_attn_op", None) is None
            or getattr(self, "_ki_prompt_padded", False)
            or getattr(past_key_values, "_write_position", None) is not self._ki_write_pos):
        # Prefill, a non-graph cache (e.g. DynamicCache
        # inside module.generate), or a padded prompt: the original forward
        # is the correct path. The kernel attends every cached slot up to
        # write_pos unconditionally, so padded prompt positions would leak
        # into the softmax; SDPA masks them natively via attention_mask.
        return self._ki_orig_attn_forward(hidden_states, position_embeddings, attention_mask, past_key_values,
                                          **kwargs)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # b=1: triple_gemv single-vector kernel; 1 < b <= 32: triple_gemm
    # grouped kernel (same weights, one launch); larger batches use the
    # projections as-is.
    batch = hidden_states.shape[0]
    qkv_op = getattr(self, "_ki_qkv_op", None)
    # every branch yields the raw (pre-norm, pre-rope) flat K/V so the
    # fused-K path can hand them to the kernel
    if qkv_op is not None and hasattr(qkv_op, "grouped_linear"):
        # single dispatcher entry; the backend routes by batch internally
        q_out, k_flat, v_flat = qkv_op.grouped_linear(hidden_states.reshape(-1, hidden_states.shape[-1]),
                                                      [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
                                                      False)
        query_states, gate = torch.chunk(q_out.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    elif qkv_op is not None and batch == 1:
        # One GEMV launch over q/k/v weights directly — replaces the
        # three separate F.linear launches. q_proj keeps its 2x-wide
        # query|gate layout, so the gate split below is unchanged from HF.
        q_out, k_flat, v_flat = qkv_op.triple_gemv(hidden_states.view(-1), self.q_proj.weight, self.k_proj.weight,
                                                   self.v_proj.weight)
        query_states, gate = torch.chunk(q_out.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    elif qkv_op is not None:
        q_out, k_flat, v_flat = qkv_op.triple_gemm(hidden_states.reshape(-1, hidden_states.shape[-1]),
                                                   self.q_proj.weight, self.k_proj.weight, self.v_proj.weight)
        query_states, gate = torch.chunk(q_out.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    else:
        query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
                                         2,
                                         dim=-1)
        k_flat = self.k_proj(hidden_states).reshape(batch, -1)
        v_flat = self.v_proj(hidden_states).reshape(batch, -1)

    num_q_heads, num_kv_heads, head_dim = self._ki_num_q_heads, self._ki_num_kv_heads, self.head_dim
    batch = hidden_states.shape[0]
    out = torch.empty(batch, num_q_heads, head_dim, dtype=hidden_states.dtype, device=hidden_states.device)
    fuse_q = getattr(self, "_ki_q_norm_w", None) is not None
    fuse_k = fuse_q and getattr(self, "_ki_k_norm_w", None) is not None
    if fuse_k:
        # The kernel norms + ropes K and writes both K/V cache slots itself;
        # skip the eager k_norm, rope helper, and cache update. Cache tensors
        # are taken directly (update() would only copy what the kernel writes).
        slot = past_key_values._layers[self.layer_idx]
        key_states = slot.keys
        value_states = slot.values
    else:
        key_states = self.k_norm(k_flat.view(hidden_shape)).transpose(1, 2)
        value_states = v_flat.view(hidden_shape).transpose(1, 2)
        # K keeps the eager norm + RoPE (the cache stores post-rope keys from
        # prefill, so the kernel cannot re-derive them); the q side's norm,
        # partial RoPE, and output gate are fused into the decode_attn kernel.
        # The model uses partial RoPE (rotary_dim < head_dim), so K rides the
        # upstream helper verbatim — only its output is kept.
        cos, sin = position_embeddings
        _, key_states = self._ki_rope(query_states, key_states, cos, sin)
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    if fuse_q:
        # chunk() leaves query strided ([q|gate] interleaved layout); the
        # kernel indexes packed memory, so materialize it explicitly
        query_states = query_states.reshape(batch, num_q_heads, head_dim)
        gate = gate.reshape(batch, num_q_heads, head_dim)
        self._ki_attn_op.decode_attn(
            query_states,
            key_states,
            value_states,
            self._ki_write_pos,
            out,
            num_q_heads,
            num_kv_heads,
            head_dim,
            key_states.shape[2],
            q_norm_w=self._ki_q_norm_w,
            cos_t=self._ki_cos_t,
            sin_t=self._ki_sin_t,
            gate=gate,
            k_raw=(k_flat.reshape(batch, num_kv_heads, head_dim) if fuse_k else key_states.new_empty(0)),
            v_raw=(v_flat.reshape(batch, num_kv_heads, head_dim) if fuse_k else key_states.new_empty(0)),
            k_norm_w=(self._ki_k_norm_w if fuse_k else key_states.new_empty(0)),
            eps=self.q_norm.eps,
            rope_dim=self._ki_cos_t.shape[-1])
    else:
        self._ki_attn_op.decode_attn(query_states.reshape(batch, num_q_heads, head_dim), key_states, value_states,
                                     self._ki_write_pos, out, num_q_heads, num_kv_heads, head_dim, key_states.shape[2])

    attn_output = out.view(*input_shape, -1)
    return self.o_proj(attn_output), None


def install_lm_head(model: torch.nn.Module, kernel_op) -> int:
    """Route lm_head decode batches of 16..32 through the grouped kernel.

    cuBLAS reaches only ~490 GB/s on the M=32, N=vocab lm_head shape; the
    grouped kernel's weight-streaming tiles do better. Weights are read
    in place (no copies); the grad path keeps the original Linear.
    """
    lm = getattr(model, "lm_head", None)
    if lm is None or not hasattr(lm, "weight"):
        return 0
    if kernel_op is None or not hasattr(kernel_op, "single_gemm"):
        return 0
    if lm.weight.dtype is not torch.bfloat16 or lm.bias is not None:
        return 0
    # vocab-parallel lm_head wrappers carry gather/allreduce semantics in
    # their forward; a plain-GEMM patch would silently drop the collective
    if carries_collective(lm):
        return 0

    def _lm_forward(self, hidden_states):
        b = hidden_states.shape[0]
        if hidden_states.dim() == 3 and hidden_states.shape[1] == 1:
            # Vocab-parallel decode: each rank GEMMs its contiguous half of
            # the (replicated) weight — half the weight traffic — and the
            # rank-ordered all-gather reassembles full logits for argmax
            # and sampling alike; downstream sees identical semantics.
            import deepspeed.comm as dist
            tp = dist.get_world_size() if dist.is_initialized() else 1
            x2 = hidden_states.reshape(b, -1)
            if tp > 1 and self.weight.shape[0] % tp == 0:
                half = self.weight.shape[0] // tp
                w_loc = self.weight[dist.get_rank() * half:(dist.get_rank() + 1) * half]
                if not torch.is_grad_enabled() and 16 <= b <= 32 and x2.is_contiguous():
                    loc = self._ki_gemm_op.single_gemm(x2, w_loc)
                else:
                    loc = torch.nn.functional.linear(x2, w_loc)
                parts = [torch.empty_like(loc) for _ in range(tp)]
                dist.all_gather(parts, loc.contiguous())
                return torch.cat(parts, dim=-1).view(b, 1, -1)
            if (not torch.is_grad_enabled() and 16 <= b <= 32 and hidden_states.is_contiguous()):
                out = self._ki_gemm_op.single_gemm(x2, self.weight)
                return out.view(b, 1, -1)
        return torch.nn.functional.linear(hidden_states, self.weight, self.bias)

    lm._ki_gemm_op = kernel_op
    lm.forward = _lm_forward.__get__(lm, type(lm))
    return 1


def install_decode_attention(model: torch.nn.Module,
                             write_pos: torch.Tensor,
                             kernel_op,
                             cos_table: torch.Tensor = None,
                             sin_table: torch.Tensor = None) -> int:
    """Patch full-attention layers to use the custom decode_attn kernel.

    Detection mirrors find_glu_segments: structural (q/k/v/o projections
    plus per-head q_norm/k_norm present, q_proj carrying the 2x-wide
    query|gate layout), cross-checked against the config's layer_types so
    linear-attention blocks are never touched. Returns the patched count;
    layers whose GQA ratio the kernel cannot serve are left native."""
    if kernel_op is None or not hasattr(kernel_op, "decode_attn"):
        return 0
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    except ImportError:
        return 0

    patched = 0
    for module in model.modules():
        projections = [getattr(module, name, None) for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
        # duck-typed projection check: AutoTP replaces nn.Linear with a
        # LinearLayer wrapper that still exposes .weight
        if not all(hasattr(p, "weight") and p.weight is not None for p in projections):
            continue
        if not all(isinstance(getattr(module, name, None), torch.nn.Module) for name in ("q_norm", "k_norm")):
            continue
        head_dim = module.head_dim
        # per-rank head counts from the actual weight shapes (AutoTP-sharded
        # weights carry the per-rank output dimension in weight.shape[0])
        q_out = projections[0].weight.shape[0]
        kv_out = projections[1].weight.shape[0]
        num_q_heads = q_out // (2 * head_dim)
        num_kv_heads = kv_out // head_dim
        if num_q_heads * head_dim * 2 != q_out:
            continue  # not the 2x-wide query|gate layout this forward replays
        layer_types = getattr(getattr(module, "config", None), "layer_types", None)
        if layer_types is not None and layer_types[module.layer_idx] != "full_attention":
            continue
        # Kernel contract: supported head dims and a GQA ratio that divides
        # its 16 warps-per-block KV split.
        if head_dim not in (64, 128, 256) or 16 % (num_q_heads // num_kv_heads) != 0:
            continue
        if getattr(module, "_ki_orig_attn_forward", None) is None:
            module._ki_orig_attn_forward = module.forward
        # Fused q-side preprocessing inputs: the (1+w) norm scale and the
        # full-length RoPE tables (indexable by the GPU-resident position,
        # so replays stay graph-safe). Missing tables disable the fusion.
        module._ki_q_norm_w = (1.0 +
                               module.q_norm.weight.detach().float()).contiguous() if cos_table is not None else None
        module._ki_k_norm_w = (1.0 +
                               module.k_norm.weight.detach().float()).contiguous() if cos_table is not None else None
        module._ki_cos_t = cos_table
        module._ki_sin_t = sin_table
        module._ki_attn_op = kernel_op
        # The fused QKV kernel consumes weights only; biased projections would lose their bias.
        unbiased_qkv = all(p.bias is None for p in projections[:3])
        module._ki_qkv_op = kernel_op if (hasattr(kernel_op, "triple_gemv") and unbiased_qkv) else None
        module._ki_write_pos = write_pos
        module._ki_rope = apply_rotary_pos_emb
        module._ki_num_q_heads = num_q_heads
        module._ki_num_kv_heads = num_kv_heads
        module.forward = _decode_attn_forward.__get__(module, type(module))
        patched += 1
    return patched


# ─── Fused residual-add + RMSNorm in the decoder-layer forward ───


def _fused_norm_layer_forward(self,
                              hidden_states,
                              position_embeddings,
                              attention_mask=None,
                              position_ids=None,
                              past_key_values=None,
                              **kwargs):
    """Replacement forward for decoder layers at b=1, seq_len=1: the
    post-attention residual-add + RMSNorm pair runs through the
    fused_add_norm kernel (one launch updates the residual stream in-place
    and produces the normalized MLP input) instead of two launches. The
    pre-attention norm stays native — the HF layer normifies the raw hidden
    state there, with no add to fuse — and the final post-MLP add stays a
    plain add for the same reason. Attention and MLP are called unchanged,
    so attention-level patches (decode_attn / triple_gemv) still apply."""
    if (hidden_states.shape[1] > 1 or hidden_states.dtype is not torch.bfloat16 or not hidden_states.is_contiguous()
            or getattr(self, "_ki_norm_op", None) is None or torch.is_grad_enabled()):
        # Prefill, a non-graph path, or any grad-enabled
        # forward: the original forward is the correct path. fused_add_norm
        # has no autograd registration and mutates the residual stream
        # in-place, so a training step through it would silently lose the
        # residual/norm gradients (the kernel's win is decode-only).
        return self._ki_orig_layer_forward(hidden_states, position_embeddings, attention_mask, position_ids,
                                           past_key_values, **kwargs)

    op = self._ki_norm_op
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    if self.block_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
    else:
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    # fused_add_norm mutates its first argument in-place into the new
    # residual stream and returns the normalized input to the MLP — keep
    # both handles, the final add must go through the residual stream.
    # The 1+w scale is recomputed from the live Parameter on every call:
    # HF's norm multiplies by 1+weight, the kernel by its weight argument
    # directly. During capture the add becomes part of the graph, so replay
    # reads the parameter's memory and reflects post-training values.
    new_residual = hidden_states
    postattn_scale = torch.add(1.0, self.post_attention_layernorm.weight).contiguous()
    mlp_input = op.fused_add_norm(new_residual, residual, postattn_scale, self.post_attention_layernorm.eps)
    hidden_states = self.mlp(mlp_input)
    hidden_states = hidden_states + new_residual

    return hidden_states


def install_fused_norm(model: torch.nn.Module, kernel_op) -> int:
    """Patch decoder layers to use fused_add_norm at b=1 decode.

    Detection is structural (block_type + mlp + both layernorms), so it
    covers full-attention and linear-attention layers alike. Returns the
    patched count."""
    if kernel_op is None or not hasattr(kernel_op, "fused_add_norm"):
        return 0
    patched = 0
    for module in model.modules():
        norms = [getattr(module, name, None) for name in ("input_layernorm", "post_attention_layernorm")]
        if not all(isinstance(n, torch.nn.Module) and hasattr(n, "weight") and hasattr(n, "eps") for n in norms):
            continue
        if getattr(module, "block_type", None) not in ("linear_attention", "full_attention"):
            continue
        if getattr(module, "_ki_orig_layer_forward", None) is None:
            module._ki_orig_layer_forward = module.forward
        module._ki_norm_op = kernel_op
        module.forward = _fused_norm_layer_forward.__get__(module, type(module))
        patched += 1
    return patched

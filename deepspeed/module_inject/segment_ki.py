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

import os
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
    named projections are all plain, bias-free, comm-free nn.Linear.

    Rejecting non-Linear types guards against AutoTP sharded wrappers:
    fusing across a shard/full-layout mismatch needs shard-aware slicing,
    which is out of scope.
    """
    for module in root.modules():
        projections = {name: getattr(module, name, None) for name in attr_names}
        if not all(isinstance(p, torch.nn.Linear) for p in projections.values()):
            continue
        if any(p.bias is not None for p in projections.values()):
            continue
        if any(carries_collective(p) for p in projections.values()):
            continue
        yield module, projections


def find_glu_segments(root: torch.nn.Module) -> List[GLUSegment]:
    """Detect gated MLPs by structure (gate_proj/up_proj/down_proj attribute set)."""
    segments = []
    for module, p in _find_fusable_projections(root, ("gate_proj", "up_proj", "down_proj")):
        segments.append(GLUSegment(parent=module, gate=p["gate_proj"], up=p["up_proj"], down=p["down_proj"]))
    return segments



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
        if hidden.dim() == 1 and kernel_op is not None:
            out = torch.empty(gate_w.shape[0], dtype=hidden.dtype, device=hidden.device)
            kernel_op.dual_gemv_silu_mul(hidden, gate_w, up_w, out)
            return out
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
            grad_gate_w = torch.matmul(grad_gate_coef.t(), hidden)
            grad_up_w = torch.matmul(grad_up_coef.t(), hidden)
            grad_hidden = torch.matmul(grad_gate_coef, gate_w) + torch.matmul(grad_up_coef, up_w)
        return grad_hidden, grad_gate_w, grad_up_w, None


def _dual_weight_glu_forward(self, input):
    """Replacement forward: reads gate_proj.weight and up_proj.weight
    directly (zero copies).  b=1 uses the fused dual-weight GEMV kernel;
    b>1 falls through to the original projections.  Gradients flow to the
    original Parameters through autograd.Function — train and generate share
    the same path with no forward switching."""
    if input.shape[0] == 1 and input.dim() == 2 and getattr(self, "_ki_dual_op", None) is not None:
        out = DualWeightGluGEMV.apply(input.squeeze(0), self.gate_proj.weight, self.up_proj.weight, self._ki_dual_op)
        return self.down_proj(out.unsqueeze(0))
    # b>1 or no kernel: original forward (gradients also correct here)
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
    fused_weight = torch.cat(
        [seg.in_proj_qkv.weight.data, seg.in_proj_z.weight.data, seg.in_proj_b.weight.data, seg.in_proj_a.weight.data],
        dim=0)

    def conv_update_with_weights(mixed_qkv, conv_state):
        return conv_update(mixed_qkv, conv_state, parent.conv1d.weight.squeeze(1), parent.conv1d.bias,
                           parent.activation)

    def conv_fn_with_weights(mixed_qkv, **kw):
        # causal_conv1d_fn does not accept HF generation kwargs; filter them
        # the same way the scan closure does.
        kw.pop("use_cache", None)
        kw.pop("cu_seqlens", None)
        kw.pop("cu_seq_lens_q", None)
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

    kernel_op = None
    try:
        from deepspeed.accelerator import get_accelerator
        if get_accelerator().device_name() != "cpu":
            from deepspeed.ops.module_inject import get_fused_glu_op
            kernel_op = get_fused_glu_op()
    except Exception:
        kernel_op = None
    parent._ki_gdn_fused_weight = fused_weight
    parent._ki_gdn_op = kernel_op
    parent._ki_gdn_conv_update = conv_update_with_weights
    parent._ki_gdn_conv_fn = conv_fn_with_weights
    parent._ki_gdn_scan = scan
    parent.forward = _fused_gdn_forward.__get__(parent, type(parent))
    return True


def apply_segment_ki(model: torch.nn.Module) -> dict:
    """Install segment kernels on ``model`` (post-AutoTP, pre-generate).

    Uses the native CUDA op when a non-cpu accelerator backend is active;
    falls back to the torch composite reference otherwise.

    Returns a small report so callers (tests, journals) can assert what was
    found and replaced without introspecting the module tree again.
    """
    kernel_op = None
    from deepspeed.accelerator import get_accelerator
    if get_accelerator().device_name() != "cpu":
        from deepspeed.ops.module_inject import get_fused_glu_op
        kernel_op = get_fused_glu_op()

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


def _decode_attn_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                         past_key_values=None, **kwargs):
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
    if (hidden_states.shape[0] != 1 or hidden_states.shape[1] > 1 or past_key_values is None
            or getattr(self, "_ki_attn_op", None) is None
            or getattr(past_key_values, "_write_position", None) is not self._ki_write_pos):
        # Prefill, batched decode, or a non-graph cache (e.g. DynamicCache
        # inside module.generate): the original forward is the correct path.
        return self._ki_orig_attn_forward(hidden_states, position_embeddings, attention_mask, past_key_values,
                                          **kwargs)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    qkv_op = getattr(self, "_ki_qkv_op", None)
    if qkv_op is not None:
        # One GEMV launch over q/k/v weights directly — replaces the
        # three separate F.linear launches. q_proj keeps its 2x-wide
        # query|gate layout, so the gate split below is unchanged from HF.
        q_out, k_out, v_out = qkv_op.triple_gemv(hidden_states.view(-1), self.q_proj.weight, self.k_proj.weight,
                                                 self.v_proj.weight)
        query_states, gate = torch.chunk(q_out.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(k_out.view(hidden_shape)).transpose(1, 2)
        value_states = v_out.view(hidden_shape).transpose(1, 2)
    else:
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = self._ki_rope(query_states, key_states, cos, sin)

    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    num_q_heads, num_kv_heads, head_dim = self._ki_num_q_heads, self._ki_num_kv_heads, self.head_dim
    out = torch.empty(num_q_heads, head_dim, dtype=hidden_states.dtype, device=hidden_states.device)
    self._ki_attn_op.decode_attn(query_states.reshape(num_q_heads, head_dim), key_states[0], value_states[0],
                                 self._ki_write_pos, out, num_q_heads, num_kv_heads, head_dim, key_states.shape[2])

    attn_output = out.view(*input_shape, -1) * torch.sigmoid(gate)
    return self.o_proj(attn_output), None


def install_decode_attention(model: torch.nn.Module, write_pos: torch.Tensor, kernel_op) -> int:
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
        if not all(isinstance(p, torch.nn.Linear) for p in projections):
            continue
        if not all(isinstance(getattr(module, name, None), torch.nn.Module) for name in ("q_norm", "k_norm")):
            continue
        head_dim = module.head_dim
        num_q_heads = projections[0].out_features // (2 * head_dim)
        num_kv_heads = projections[1].out_features // head_dim
        if num_q_heads * head_dim * 2 != projections[0].out_features:
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
        module._ki_attn_op = kernel_op
        module._ki_qkv_op = kernel_op if (hasattr(kernel_op, "triple_gemv")
                                        and os.environ.get("DS_TIER2", "1") == "1") else None
        module._ki_write_pos = write_pos
        module._ki_rope = apply_rotary_pos_emb
        module._ki_num_q_heads = num_q_heads
        module._ki_num_kv_heads = num_kv_heads
        module.forward = _decode_attn_forward.__get__(module, type(module))
        patched += 1
    return patched


# ─── Fused residual-add + RMSNorm in the decoder-layer forward ───


def _fused_norm_layer_forward(self, hidden_states, position_embeddings, attention_mask=None,
                          position_ids=None, past_key_values=None, **kwargs):
    """Replacement forward for decoder layers at b=1, seq_len=1: the
    post-attention residual-add + RMSNorm pair runs through the
    fused_add_norm kernel (one launch updates the residual stream in-place
    and produces the normalized MLP input) instead of two launches. The
    pre-attention norm stays native — the HF layer normifies the raw hidden
    state there, with no add to fuse — and the final post-MLP add stays a
    plain add for the same reason. Attention and MLP are called unchanged,
    so attention-level patches (decode_attn / triple_gemv) still apply."""
    if (hidden_states.shape[0] != 1 or hidden_states.shape[1] > 1
            or hidden_states.dtype is not torch.bfloat16 or not hidden_states.is_contiguous()
            or getattr(self, "_ki_norm_op", None) is None):
        # Prefill, batched decode, or a non-graph path: the original forward
        # is the correct path.
        return self._ki_orig_layer_forward(hidden_states, position_embeddings, attention_mask,
                                           position_ids, past_key_values, **kwargs)

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
    new_residual = hidden_states
    mlp_input = op.fused_add_norm(new_residual, residual, self._ki_postattn_scale,
                                  self.post_attention_layernorm.eps)
    hidden_states = self.mlp(mlp_input)
    hidden_states = hidden_states + new_residual

    return hidden_states


def install_fused_norm(model: torch.nn.Module, kernel_op) -> int:
    """Patch decoder layers to use fused_add_norm at b=1 decode.

    Detection is structural (block_type + mlp + both layernorms), so it
    covers full-attention and linear-attention layers alike. Pre-computes
    the 1+w RMSNorm scale into a persistent contiguous buffer (the HF norm
    multiplies by 1+weight; the kernel multiplies by its weight argument
    directly) so the captured graph never rebuilds it. Returns the patched
    count."""
    if kernel_op is None or not hasattr(kernel_op, "fused_add_norm") or os.environ.get("DS_TIER1", "1") != "1":
        return 0
    patched = 0
    for module in model.modules():
        norms = [getattr(module, name, None) for name in ("input_layernorm", "post_attention_layernorm")]
        if not all(isinstance(n, torch.nn.Module) and hasattr(n, "weight") and hasattr(n, "eps")
                   for n in norms):
            continue
        if getattr(module, "block_type", None) not in ("linear_attention", "full_attention"):
            continue
        if getattr(module, "_ki_orig_layer_forward", None) is None:
            module._ki_orig_layer_forward = module.forward
        module._ki_norm_op = kernel_op
        module._ki_postattn_scale = torch.add(1.0, norms[1].weight.detach()).contiguous()
        module.forward = _fused_norm_layer_forward.__get__(module, type(module))
        patched += 1
    return patched



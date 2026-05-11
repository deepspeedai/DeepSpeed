# Copyright (c) The DeepSpeed Contributors
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Sequence-parallel-aware forward for Qwen3.5 ``Qwen3_5GatedDeltaNet`` (GDN).

Qwen3.5's GDN layers are linear recurrent; under sequence parallelism, ranks > 0
silently diverge from rank 0 unless the recurrent state is communicated across
ranks at each layer. This module installs a drop-in replacement
``Qwen3_5GatedDeltaNet.forward`` that uses FLA's native context-parallel API
(``fla.ops.cp``) for the recurrence.

The patch is installed automatically by
``UlyssesSPAttentionHF.register_with_transformers`` when it detects a Qwen3.5
GDN model. Users should not need to import this module directly.

Requirements:

* ``fla-core >= 0.5.0`` (for ``fla.ops.cp.FLACPContext``,
  ``build_cp_context``, ``conv_cp_send_recv_fwd``, ``conv_cp_send_recv_bwd``).
* ``transformers`` with ``transformers.models.qwen3_5.modeling_qwen3_5``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist

from deepspeed.utils.logging import logger


# ----------------------------------------------------------------------------
# Module state
# ----------------------------------------------------------------------------

_ORIGINAL_GDN_FORWARD = None
_INSTALLED = False


# ----------------------------------------------------------------------------
# SP group helpers
# ----------------------------------------------------------------------------

def _get_sp_group_info():
    """Return ``(sp_group, sp_world_size, sp_rank)`` or ``(None, 1, 0)`` if SP inactive."""
    try:
        import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu
    except Exception:
        return None, 1, 0
    if getattr(mpu, "_SEQUENCE_PARALLEL_GROUP", None) is None:
        return None, 1, 0
    sp_group = mpu.get_sequence_parallel_group()
    return sp_group, mpu.get_sequence_parallel_world_size(), mpu.get_sequence_parallel_rank()


# ----------------------------------------------------------------------------
# FLA CP availability probe
# ----------------------------------------------------------------------------

def _fla_supports_cp() -> bool:
    """Detect whether the installed FLA exposes the operator-level CP API
    (``FLACPContext`` + ``build_cp_context`` + ``conv_cp_send_recv_fwd``).

    Both landed together in fla 0.5.0. Older FLAs raise ImportError on
    ``from fla.ops.cp import ...``. We probe at module import and cache.
    """
    try:
        from fla.ops.cp import (  # noqa: F401
            FLACPContext, build_cp_context, conv_cp_send_recv_fwd, conv_cp_send_recv_bwd)
        return True
    except Exception:
        return False


_FLA_HAS_CP = _fla_supports_cp()


# ----------------------------------------------------------------------------
# Conv-boundary ring exchange (autograd-aware wrapper around FLA's CP primitive)
# ----------------------------------------------------------------------------

class _ConvCPBoundary(torch.autograd.Function):
    """Autograd wrapper around FLA's conv-boundary ring exchange.

    Fwd sends each rank's last ``K-1`` tokens to the next rank; bwd routes
    ``d_heads`` back to the previous rank as ``d_tails``. FLA's primitives use
    bare ``all_gather_into_tensor`` with no autograd, so without this wrapper
    each rank silently drops its right neighbor's boundary grads.
    """

    @staticmethod
    def forward(ctx, tails, group):
        from fla.ops.cp import conv_cp_send_recv_fwd
        heads = conv_cp_send_recv_fwd(tails, group=group)
        ctx.group = group
        return heads

    @staticmethod
    def backward(ctx, d_heads):
        from fla.ops.cp import conv_cp_send_recv_bwd
        d_tails = conv_cp_send_recv_bwd(d_heads.contiguous(), group=ctx.group)
        return d_tails, None


# ----------------------------------------------------------------------------
# QKV-to-conv layout transform
# ----------------------------------------------------------------------------
# Pure-PyTorch transpose + cat. Autograd is handled by stock torch ops, so no
# custom Function is needed.


def _qkv_to_conv_layout(mixed_qkv, heads, K_minus_1):
    """Build the conv-ready, boundary-prepended layout from local ``mixed_qkv``.

    Forward:  ``(mixed_qkv [B, Sp, qkv_ch], heads [B, qkv_ch, K-1], K_minus_1)
               -> qkv_padded [B, qkv_ch, Sp + K-1]``
    """
    qkv_for_conv = mixed_qkv.transpose(1, 2).contiguous()  # [B, qkv_ch, Sp]
    if K_minus_1 > 0:
        return torch.cat([heads, qkv_for_conv], dim=-1).contiguous()
    return qkv_for_conv


# ----------------------------------------------------------------------------
# Per-forward CP-context cache (shared across GDN layers in one fwd pass)
# ----------------------------------------------------------------------------
# All GDN layers see the same ``position_ids``, so we cache the all-gather +
# build_cp_context result. Cache key includes shape and data_ptr to guard
# against ``id`` reuse; bounded LRU-style to drop stale prior iterations.
_CP_CTX_CACHE: "dict[tuple, tuple]" = {}
_CP_CTX_CACHE_MAX = 4


def _pid_cache_key(position_ids, P):
    return (id(position_ids), P, tuple(position_ids.shape), position_ids.data_ptr())


def _get_cp_context(
    position_ids: Optional[torch.LongTensor],
    P: int,
    sp_rank: int,
    sp_group,
    conv_kernel_size: int,
    Sp: int,
    device: torch.device,
):
    """Build ``(cp_ctx, local_seq_idx)``, cached across layers in a forward pass.

    Single-sequence: ``cu_seqlens = [0, P*Sp]``, no all-gather. Packed
    multi-seq: all-gather ``position_ids``; FLA partitions cu_seqlens per-rank.
    """
    from fla.ops.cp import build_cp_context  # local import: gated on _FLA_HAS_CP
    from transformers.models.qwen3_5.modeling_qwen3_5 import prepare_fla_varlen_kwargs_from_position_ids

    if position_ids is None:
        # No position_ids -> assume single global sequence of length P*Sp.
        S_global = P * Sp
        cu_seqlens_cpu = torch.tensor([0, S_global], dtype=torch.int32)
        cu_seqlens_full = cu_seqlens_cpu.to(device=device, non_blocking=True)
        cp_ctx = build_cp_context(
            cu_seqlens=cu_seqlens_full,
            cu_seqlens_cpu=cu_seqlens_cpu,
            group=sp_group,
            conv1d_kernel_size=conv_kernel_size,
        )
        return cp_ctx, None

    cache_key = _pid_cache_key(position_ids, P)
    cached = _CP_CTX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # All-gather position_ids -> global cu_seqlens.
    pid_list = [torch.empty_like(position_ids) for _ in range(P)]
    dist.all_gather(pid_list, position_ids.contiguous(), group=sp_group)
    full_position_ids = torch.cat(pid_list, dim=1)
    cu_seqlens_full, full_seq_idx = prepare_fla_varlen_kwargs_from_position_ids(full_position_ids)

    if cu_seqlens_full is None:
        # Single global sequence (no resets). Synthesize cu_seqlens=[0, P*Sp].
        S_global = P * Sp
        cu_seqlens_cpu = torch.tensor([0, S_global], dtype=torch.int32)
        cu_seqlens_full = cu_seqlens_cpu.to(device=device, non_blocking=True)
        local_seq_idx = None
    else:
        cu_seqlens_cpu = cu_seqlens_full.cpu()
        local_seq_idx = (full_seq_idx[:, sp_rank * Sp:(sp_rank + 1) * Sp].contiguous()
                         if full_seq_idx is not None else None)

    cp_ctx = build_cp_context(
        cu_seqlens=cu_seqlens_full,
        cu_seqlens_cpu=cu_seqlens_cpu,
        group=sp_group,
        conv1d_kernel_size=conv_kernel_size,
    )

    if len(_CP_CTX_CACHE) >= _CP_CTX_CACHE_MAX:
        _CP_CTX_CACHE.pop(next(iter(_CP_CTX_CACHE)))
    _CP_CTX_CACHE[cache_key] = (cp_ctx, local_seq_idx)
    return cp_ctx, local_seq_idx


# ----------------------------------------------------------------------------
# SP-aware GDN forward
# ----------------------------------------------------------------------------

def _sp_aware_gdn_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
):
    """SP-aware replacement for ``Qwen3_5GatedDeltaNet.forward``.

    Each rank keeps its ``[B, S/P, H]`` shard end-to-end. Cross-rank
    communication is limited to:

    * a one-time ``all_gather`` of ``position_ids`` for packed multi-sequence
      batches (cached across layers via ``_CP_CTX_CACHE``);
    * a ring exchange of the last ``K-1`` conv tokens per layer
      (``_ConvCPBoundary``);
    * FLA's internal state-scan collectives inside ``chunk_gated_delta_rule``.

    Delegates to the original GDN forward when SP is inactive or when in
    incremental-decode mode. Raises ``RuntimeError`` if ``cache_params`` is
    provided outside the decode path (SP is training-only).
    """
    sp_group, sp_world_size, sp_rank = _get_sp_group_info()

    if sp_world_size == 1:
        return _ORIGINAL_GDN_FORWARD(
            self,
            hidden_states,
            cache_params=cache_params,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    # Incremental decode (single token, has previous state): delegate to original.
    if hidden_states.shape[1] == 1 and cache_params is not None \
            and cache_params.has_previous_state(self.layer_idx):
        return _ORIGINAL_GDN_FORWARD(
            self,
            hidden_states,
            cache_params=cache_params,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    if cache_params is not None:
        raise RuntimeError(
            "DeepSpeed SP-aware Qwen3.5 GDN forward is training-only; "
            "got cache_params != None outside the incremental-decode path. "
            "Either disable SP for this forward or pass cache_params=None.")

    P = sp_world_size
    nv = self.num_v_heads
    nk = self.num_k_heads
    dk = self.head_k_dim
    dv = self.head_v_dim
    K = self.conv_kernel_size

    # Skip apply_mask_to_padding_states when no padding (saves a memset on
    # pad-free batches).
    if attention_mask is not None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

    B, Sp, _ = hidden_states.shape
    device = hidden_states.device

    mixed_qkv = self.in_proj_qkv(hidden_states)  # [B, Sp, qkv_ch]
    z_local = self.in_proj_z(hidden_states)
    b_local = self.in_proj_b(hidden_states)
    a_local = self.in_proj_a(hidden_states)

    cp_ctx, _local_seq_idx = _get_cp_context(
        position_ids,
        P,
        sp_rank,
        sp_group,
        K,
        Sp,
        device,
    )

    # Ring-exchange last K-1 conv tokens to the next rank for a gap-free causal conv1d.
    K_minus_1 = K - 1
    if K_minus_1 > 0:
        tails = mixed_qkv[:, -K_minus_1:, :].transpose(1, 2).contiguous()
        heads = _ConvCPBoundary.apply(tails, sp_group)  # [B, qkv_ch, K-1]
    else:
        # Defensive: K=1 (no boundary). FLA's conv_cp_send_recv_fwd expects
        # K-1>=1 in practice; this branch is just to keep the math sane.
        heads = torch.empty(
            B,
            mixed_qkv.shape[-1],
            0,
            device=mixed_qkv.device,
            dtype=mixed_qkv.dtype,
        )

    # Build the conv-ready layout: [B, Sp, qkv_ch] -> [B, qkv_ch, Sp + K-1]
    # with the received ``heads`` prepended.
    qkv_padded = _qkv_to_conv_layout(mixed_qkv, heads, K_minus_1)

    conv_w_full = self.conv1d.weight.squeeze(1).contiguous()  # [qkv_ch, K]
    conv_b_full = self.conv1d.bias

    if self.causal_conv1d_fn is not None:
        qkv_out = self.causal_conv1d_fn(
            x=qkv_padded,
            weight=conv_w_full,
            bias=conv_b_full,
            activation=self.activation,
            seq_idx=None,
        )
    else:
        ch = qkv_padded.shape[1]
        qkv_out = F.conv1d(
            qkv_padded,
            conv_w_full.unsqueeze(1),
            bias=conv_b_full,
            padding=K - 1,
            groups=ch,
        )[:, :, :Sp + K - 1]
        qkv_out = F.silu(qkv_out)

    qkv_out = qkv_out[..., (K - 1):].transpose(1, 2)

    key_dim = nk * dk
    qk2 = 2 * key_dim
    q = qkv_out[..., :key_dim].view(B, Sp, nk, dk)
    k = qkv_out[..., key_dim:qk2].view(B, Sp, nk, dk)
    v = qkv_out[..., qk2:].view(B, Sp, nv, dv)

    # GQA repeat: FLA's chunk_gated_delta_rule needs one shared head count, but
    # Qwen3.5 GDN has nv > nk (e.g. 32/16). Broadcast Q/K up to nv before the
    # kernel call; skipping it silently produces NaN.
    v_per_k = nv // nk
    if v_per_k > 1:
        q = q.repeat_interleave(v_per_k, dim=2)  # [B, Sp, nv, dk]
        k = k.repeat_interleave(v_per_k, dim=2)

    # Precompute log-space gate ourselves: stock FLA's chunk_gated_delta_rule
    # silently ignores A_log / dt_bias kwargs (only present in some forks), so
    # passing the raw ``a`` would integrate ``g=a`` instead and produce NaN.
    A_log = self.A_log
    dt_bias = self.dt_bias
    beta = b_local.sigmoid()
    g = -A_log.float().exp() * F.softplus(a_local.float() + dt_bias)

    core_attn_out, _last = self.chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        cp_context=cp_ctx,
        use_qk_l2norm_in_kernel=True,
        transpose_state_layout=True,
    )

    core_attn_out = core_attn_out.reshape(-1, dv)
    z_for_norm = z_local.reshape(-1, dv)
    core_attn_out = self.norm(core_attn_out, z_for_norm)
    core_attn_out = core_attn_out.view(B, Sp, nv * dv)
    return self.out_proj(core_attn_out)


# ----------------------------------------------------------------------------
# Auto-installer (called by UlyssesSPAttentionHF.register_with_transformers)
# ----------------------------------------------------------------------------

def _install_sp_aware_gdn_forward():
    """Monkey-patch ``Qwen3_5GatedDeltaNet.forward`` with the SP-aware version.

    Idempotent. Called automatically by ``UlyssesSPAttentionHF.register_with_transformers``
    on Qwen3.5 GDN models. Raises ``ImportError`` if ``fla.ops.cp``
    (``fla-core >= 0.5.0``) or the Qwen3.5 modeling module is unavailable.
    """
    global _ORIGINAL_GDN_FORWARD, _INSTALLED
    if _INSTALLED:
        return
    if not _FLA_HAS_CP:
        raise ImportError("SP-aware Qwen3.5 GDN forward requires fla.ops.cp (fla-core >= 0.5.0). "
                          "Install with: pip install 'fla-core>=0.5.0'")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    _ORIGINAL_GDN_FORWARD = Qwen3_5GatedDeltaNet.forward
    Qwen3_5GatedDeltaNet.forward = _sp_aware_gdn_forward
    _INSTALLED = True
    logger.info("[deepspeed.gdn_sp] installed SP-aware Qwen3_5GatedDeltaNet.forward")


__all__ = ["_install_sp_aware_gdn_forward"]

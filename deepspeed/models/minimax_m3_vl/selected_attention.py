# Copyright (c) DeepSpeed Team.
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li.
# SPDX-License-Identifier: Apache-2.0 AND MIT
#
# Portions of this file are derived from flash-linear-attention (fla).
# See THIRD_PARTY_NOTICES.md for the MIT notice.

# DeepSpeed Team
"""Block-sparse ("selected") attention in Triton for MiniMax-M3: forward, dq and dk/dv kernels.

Each query token attends, per key-value head, to the keys of the key blocks selected for it, and only to keys at or
before its own position. The 16 (or any power of two >= 16) query heads that share a key-value head share the
selection and are computed together, so a tile is [query heads of the group, block_size keys].

Adapted from https://github.com/fla-org/flash-linear-attention, ``fla/ops/nsa/parallel.py`` and
``fla/ops/utils/csr.py`` at commit 1fe52b5d05c53601885610fab3d39157647d836d (MIT). Changes:

* The queries may be the last ``TQ`` positions of ``TK`` keys in the backward too (fla allows it in the forward
  only). This is the sequence-split (context parallel) layout: a rank holds queries ``[r * L, (r + 1) * L)`` and
  passes the keys ``[0, (r + 1) * L)``.
* The softmax scale multiplies the float32 scores, instead of rounding ``q * scale`` to bf16 first.
* Only the fixed-count, equal-length-batch path (no ``cu_seqlens``, no ``block_counts`` tensor, value dim in one
  tile).
* The dk/dv pass gives each key block's list of selecting queries to several programs, ``SEG`` queries each, which
  add their float32 partial sums atomically: a block chosen by every query (block 0) no longer sets the time.
"""

import torch

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, tl, triton

if _TRITON_AVAILABLE:

    @triton.autotune(
        configs=[triton.Config({}, num_warps=w, num_stages=s) for w in [1, 2, 4, 8] for s in [1, 2, 3]],
        key=['BS', 'K', 'V', 'G'],
    )
    @triton.jit
    def _fwd_kernel(q, k, v, o, lse, scale, block_indices, TQ, TK, H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
                    K: tl.constexpr, V: tl.constexpr, S: tl.constexpr, BS: tl.constexpr):
        # q [B, TQ, HQ, K], k/v [B, TK, H, K/V], block_indices [B, TQ, H, S], o [B, TQ, HQ, V], lse [B, TQ, HQ]
        i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
        i_b, i_h = i_bh // H, i_bh % H
        bos_q, bos_k = i_b * TQ, i_b * TK
        pos = TK - TQ + i_t  # position of this query among the keys

        k += (bos_k * H + i_h) * K
        v += (bos_k * H + i_h) * V
        block_indices += (bos_q + i_t) * H * S + i_h * S

        o_g = i_h * G + tl.arange(0, G)
        o_d = tl.arange(0, K)
        o_v = tl.arange(0, V)
        b_q = tl.load(q + (bos_q + i_t) * HQ * K + o_g[:, None] * K + o_d[None, :])  # [G, K]

        b_o = tl.zeros([G, V], dtype=tl.float32)
        b_m = tl.full([G], float('-inf'), dtype=tl.float32)
        b_acc = tl.zeros([G], dtype=tl.float32)
        for i in range(S):
            i_s = tl.load(block_indices + i).to(tl.int64) * BS  # first key of the selected block
            if i_s <= pos and i_s >= 0:
                o_s = i_s + tl.arange(0, BS)
                m_s = o_s < TK
                b_k = tl.load(k + o_d[:, None] + o_s[None, :] * (H * K), mask=m_s[None, :], other=0.0)  # [K, BS]
                b_v = tl.load(v + o_s[:, None] * (H * V) + o_v[None, :], mask=m_s[:, None], other=0.0)  # [BS, V]
                b_s = tl.dot(b_q, b_k) * scale  # [G, BS], float32
                b_s = tl.where((pos >= o_s)[None, :], b_s, float('-inf'))
                b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
                b_r = tl.exp(b_mp - b_m)
                b_p = tl.exp(b_s - b_m[:, None])
                b_acc = b_acc * b_r + tl.sum(b_p, 1)
                b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

        b_o = b_o / b_acc[:, None]
        b_m += tl.log(b_acc)
        tl.store(o + (bos_q + i_t) * HQ * V + o_g[:, None] * V + o_v[None, :], b_o.to(o.dtype.element_ty))
        tl.store(lse + (bos_q + i_t) * HQ + o_g, b_m)

    @triton.autotune(
        configs=[triton.Config({}, num_warps=w, num_stages=s) for w in [1, 2, 4, 8] for s in [1, 2, 3]],
        key=['BS', 'K', 'V', 'G'],
    )
    @triton.jit
    def _bwd_dq_kernel(q, k, v, lse, delta, do, dq, scale, block_indices, TQ, TK, H: tl.constexpr, HQ: tl.constexpr,
                       G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, S: tl.constexpr, BS: tl.constexpr):
        i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
        i_b, i_h = i_bh // H, i_bh % H
        bos_q, bos_k = i_b * TQ, i_b * TK
        pos = TK - TQ + i_t

        k += (bos_k * H + i_h) * K
        v += (bos_k * H + i_h) * V
        block_indices += (bos_q + i_t) * H * S + i_h * S

        o_g = i_h * G + tl.arange(0, G)
        o_d = tl.arange(0, K)
        o_v = tl.arange(0, V)
        row = bos_q + i_t
        b_q = tl.load(q + row * HQ * K + o_g[:, None] * K + o_d[None, :])  # [G, K]
        b_do = tl.load(do + row * HQ * V + o_g[:, None] * V + o_v[None, :])  # [G, V]
        b_lse = tl.load(lse + row * HQ + o_g)
        b_delta = tl.load(delta + row * HQ + o_g)

        b_dq = tl.zeros([G, K], dtype=tl.float32)
        for i in range(S):
            i_s = tl.load(block_indices + i).to(tl.int64) * BS
            if i_s <= pos and i_s >= 0:
                o_s = i_s + tl.arange(0, BS)
                m_s = o_s < TK
                b_k = tl.load(k + o_d[:, None] + o_s[None, :] * (H * K), mask=m_s[None, :], other=0.0)  # [K, BS]
                b_v = tl.load(v + o_v[:, None] + o_s[None, :] * (H * V), mask=m_s[None, :], other=0.0)  # [V, BS]
                b_s = tl.dot(b_q, b_k) * scale
                b_p = tl.exp(b_s - b_lse[:, None])
                b_p = tl.where((pos >= o_s)[None, :], b_p, 0.0)
                b_dp = tl.dot(b_do, b_v)
                b_ds = b_p * (b_dp - b_delta[:, None])
                b_dq = tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k), b_dq)
        b_dq *= scale
        tl.store(dq + row * HQ * K + o_g[:, None] * K + o_d[None, :], b_dq.to(dq.dtype.element_ty))

    @triton.jit
    def _csr_kernel(block_indices, cursor, csr_indices, csr_offsets, TQ, TK, H: tl.constexpr, S: tl.constexpr,
                    SP: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, TC: tl.constexpr, COUNT_ONLY: tl.constexpr):
        # Counting sort of (query, selected block) pairs by block: COUNT_ONLY tallies, the second pass scatters.
        # SP is S rounded up to a power of 2, which tl.arange needs; the extra columns are masked out.
        i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
        i_b, i_h = i_bh // H, i_bh % H
        o_t = i_t * BT + tl.arange(0, BT)
        o_s = tl.arange(0, SP)
        m_t = o_t < TQ
        i_qh = (i_b * TQ + o_t) * H + i_h
        b_i = tl.load(block_indices + i_qh[:, None] * S + o_s[None, :],
                      mask=m_t[:, None] & (o_s[None, :] < S),
                      other=-1).to(tl.int64)
        pos = TK - TQ + o_t
        b_m = m_t[:, None] & (b_i >= 0) & (b_i < TC) & (b_i * BS <= pos[:, None])
        block_id = (i_b * H + i_h) * TC + b_i
        if COUNT_ONLY:
            tl.atomic_add(csr_offsets + block_id + 1, 1, mask=b_m)
        else:
            dst = tl.load(csr_offsets + block_id, mask=b_m, other=0).to(tl.int64) + tl.atomic_add(
                cursor + block_id, 1, mask=b_m)
            b_q = tl.broadcast_to((i_b * TQ + o_t)[:, None], (BT, SP))
            tl.store(csr_indices + dst, b_q.to(csr_indices.dtype.element_ty), mask=b_m)

    def _prune_dkv(configs, nargs, **kwargs):
        G = {**(nargs or {}), **kwargs}.get('G', 1)
        return [c for c in configs if c.kwargs['BQ'] * G <= 256]

    @triton.autotune(
        configs=[
            triton.Config({'BQ': bq}, num_warps=w, num_stages=s) for bq in [1, 2, 4, 8] for w in [4, 8]
            for s in [1, 2]
        ],
        key=['BS', 'K', 'V', 'G', 'SEG'],
        prune_configs_by={'early_config_prune': _prune_dkv},
        reset_to_zero=['dk', 'dv'],
    )
    @triton.jit
    def _bwd_dkv_kernel(q, k, v, lse, delta, do, dk, dv, scale, csr_indices, csr_offsets, TQ, TK, H: tl.constexpr,
                        HQ: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, TC: tl.constexpr,
                        BS: tl.constexpr, SEG: tl.constexpr, BQ: tl.constexpr):
        # Program (key block, segment): the segment-th run of SEG queries in the block's list of selecting queries.
        i_blk, i_seg = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
        i_q0 = tl.load(csr_offsets + i_blk).to(tl.int64) + i_seg * SEG
        i_q1 = tl.minimum(tl.load(csr_offsets + i_blk + 1).to(tl.int64), i_q0 + SEG)
        NQ = i_q1 - i_q0
        if NQ <= 0:
            return
        i_s = i_blk % TC
        i_b, i_h = (i_blk // TC) // H, (i_blk // TC) % H
        o_t = i_s * BS + tl.arange(0, BS)  # key positions of this block
        m_t = o_t < TK
        o_d = tl.arange(0, K)
        o_v = tl.arange(0, V)
        o_h = i_h * G + tl.arange(0, BQ * G) % G

        p_k = k + ((i_b * TK + o_t) * H + i_h)[:, None] * K + o_d[None, :]
        p_v = v + ((i_b * TK + o_t) * H + i_h)[:, None] * V + o_v[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None], other=0.0)  # [BS, K]
        b_v = tl.load(p_v, mask=m_t[:, None], other=0.0)  # [BS, V]
        b_dk = tl.zeros([BS, K], dtype=tl.float32)
        b_dv = tl.zeros([BS, V], dtype=tl.float32)

        for i in range(tl.cdiv(NQ, BQ)):
            # BQ queries x G heads as [BQ * G] rows; row r is query slot i * BQ + r // G, head o_h[r]
            o_slot = i * BQ + tl.arange(0, BQ * G) // G
            m_q = o_slot < NQ
            row = tl.load(csr_indices + i_q0 + o_slot, mask=m_q, other=0).to(tl.int64)  # i_b * TQ + t
            q_pos = TK - TQ + (row - i_b * TQ)
            b_q = tl.load(q + row[:, None] * HQ * K + o_h[:, None] * K + o_d[None, :], mask=m_q[:, None], other=0.0)
            b_do = tl.load(do + row[:, None] * HQ * V + o_h[:, None] * V + o_v[None, :], mask=m_q[:, None], other=0.0)
            b_lse = tl.load(lse + row * HQ + o_h, mask=m_q, other=0.0)
            b_delta = tl.load(delta + row * HQ + o_h, mask=m_q, other=0.0)

            b_s = tl.dot(b_k, tl.trans(b_q)) * scale  # [BS, BQ * G]
            b_p = tl.exp(b_s - b_lse[None, :])
            b_p = tl.where((q_pos[None, :] >= o_t[:, None]) & m_q[None, :], b_p, 0.0)
            b_dv = tl.dot(b_p.to(b_do.dtype), b_do, b_dv)
            b_dp = tl.dot(b_v, tl.trans(b_do))
            b_ds = b_p * (b_dp - b_delta[None, :])
            b_dk = tl.dot(b_ds.to(b_q.dtype), b_q, b_dk)

        b_dk *= scale
        o_kv = (i_b * TK + o_t) * H + i_h
        tl.atomic_add(dk + o_kv[:, None] * K + o_d[None, :], b_dk, mask=m_t[:, None])
        tl.atomic_add(dv + o_kv[:, None] * V + o_v[None, :], b_dv, mask=m_t[:, None])


def _csr(block_indices, TK, block_size):
    B, TQ, H, S = block_indices.shape
    TC = triton.cdiv(TK, block_size)
    NB = B * H * TC
    cursor = block_indices.new_zeros(NB, dtype=torch.int32)
    csr_offsets = block_indices.new_zeros(NB + 1, dtype=torch.int32)
    csr_indices = block_indices.new_empty(B * TQ * H * S, dtype=torch.int32)
    SP = triton.next_power_of_2(S)
    BT = max(1, min(128, triton.next_power_of_2(max(1, 2048 // SP))))
    grid = (triton.cdiv(TQ, BT), B * H)
    kw = dict(block_indices=block_indices,
              cursor=cursor,
              csr_indices=csr_indices,
              csr_offsets=csr_offsets,
              TQ=TQ,
              TK=TK,
              H=H,
              S=S,
              SP=SP,
              BT=BT,
              BS=block_size,
              TC=TC)
    _csr_kernel[grid](**kw, COUNT_ONLY=True)
    csr_offsets.cumsum_(0)
    _csr_kernel[grid](**kw, COUNT_ONLY=False)
    return csr_indices, csr_offsets, TC


class _SelectedAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, block_indices, block_size, scale, seg):
        B, TQ, HQ, K = q.shape
        TK, H, V = k.shape[1], k.shape[2], v.shape[-1]
        S = block_indices.shape[-1]
        o = torch.empty(B, TQ, HQ, V, dtype=q.dtype, device=q.device)
        lse = torch.empty(B, TQ, HQ, dtype=torch.float32, device=q.device)
        # Triton uses the current device, which may differ from the inputs' device.
        with torch.cuda.device(q.device):  #ignore-cuda
            _fwd_kernel[(TQ, B * H)](q,
                                     k,
                                     v,
                                     o,
                                     lse,
                                     scale,
                                     block_indices,
                                     TQ,
                                     TK,
                                     H=H,
                                     HQ=HQ,
                                     G=HQ // H,
                                     K=K,
                                     V=V,
                                     S=S,
                                     BS=block_size)
        ctx.save_for_backward(q, k, v, o, lse, block_indices)
        ctx.block_size, ctx.scale, ctx.seg = block_size, scale, seg
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, block_indices = ctx.saved_tensors
        block_size, scale, seg = ctx.block_size, ctx.scale, ctx.seg
        do = do.contiguous()
        B, TQ, HQ, K = q.shape
        TK, H, V = k.shape[1], k.shape[2], v.shape[-1]
        S, G = block_indices.shape[-1], HQ // H
        delta = (o.float() * do.float()).sum(-1)  # [B, TQ, HQ]
        dq = torch.empty_like(q)
        # dk and dv are float32: one key block's gradient sums a term from every query that selected it.
        dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
        dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
        with torch.cuda.device(q.device):  #ignore-cuda
            _bwd_dq_kernel[(TQ, B * H)](q,
                                        k,
                                        v,
                                        lse,
                                        delta,
                                        do,
                                        dq,
                                        scale,
                                        block_indices,
                                        TQ,
                                        TK,
                                        H=H,
                                        HQ=HQ,
                                        G=G,
                                        K=K,
                                        V=V,
                                        S=S,
                                        BS=block_size)
            csr_indices, csr_offsets, TC = _csr(block_indices, TK, block_size)
            grid = (B * H * TC, triton.cdiv(TQ, seg))
            _bwd_dkv_kernel[grid](q,
                                  k,
                                  v,
                                  lse,
                                  delta,
                                  do,
                                  dk,
                                  dv,
                                  scale,
                                  csr_indices,
                                  csr_offsets,
                                  TQ,
                                  TK,
                                  H=H,
                                  HQ=HQ,
                                  G=G,
                                  K=K,
                                  V=V,
                                  TC=TC,
                                  BS=block_size,
                                  SEG=seg)
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def _power_of_two(n, minimum):
    return n >= minimum and n & (n - 1) == 0


def selected_attention_supported(heads, kv_heads, head_dim, block_size):
    """Whether :func:`selected_attention` takes these shapes: query heads per key-value head, head dim and block size
    each a power of two of at least 16 (they are Triton tile dimensions)."""
    return (heads % kv_heads == 0 and _power_of_two(heads // kv_heads, 16) and _power_of_two(head_dim, 16)
            and _power_of_two(block_size, 16))


def selected_attention(q, k, v, block_indices, block_size, scale, seg=1024):
    """Attention of every query over the keys of its selected key blocks, causal by position.

    Args:
        q: ``[batch, TQ, heads, head_dim]``, the queries at positions ``TK - TQ .. TK - 1``.
        k, v: ``[batch, TK, kv_heads, head_dim]``; ``heads / kv_heads`` a power of two of at least 16.
        block_indices: ``[batch, TQ, kv_heads, topk]`` selected key-block ids, ``-1`` for an unused slot; the ids of
            one query must be distinct and must include a block with a key at or before the query.
        block_size: keys per block (a power of two, at least 16).
        scale: softmax scale applied to ``q . k``.
        seg: queries per program in the dk/dv pass (a key block chosen by more queries is split).

    Returns:
        ``[batch, TQ, heads, head_dim]``.
    """
    if not selected_attention_supported(q.shape[2], k.shape[2], q.shape[3], block_size):
        raise ValueError(
            f"query heads per key-value head ({q.shape[2]}/{k.shape[2]}), head dim ({q.shape[3]}) and block "
            f"size ({block_size}) must each be a power of two of at least 16")
    if not _TRITON_AVAILABLE:
        raise RuntimeError("selected_attention needs Triton; use block_sparse_attention without it")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    block_indices = block_indices.to(torch.int32).contiguous()
    return _SelectedAttention.apply(q, k, v, block_indices, block_size, scale, seg)

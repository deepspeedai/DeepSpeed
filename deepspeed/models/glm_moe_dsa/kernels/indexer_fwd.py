# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Lightning indexer forward kernel.

Adapted from https://github.com/tile-ai/tilelang/blob/4956b5835fa554af6c03d4a6289cad44bf310869/examples/deepseek_v32/fp8_lighting_indexer.py
through the GLM-5 plugin of https://github.com/THUDM/slime. For a block of query rows it computes
``sum_h weights[q, h] * relu(index_q[q, h] . index_k[k])`` against every key ``k`` in
``[min ks, max ke)`` of the block and writes float32 logits; ``clean_logits`` then sets every entry
outside a row's own ``[ks, ke)`` window to ``-inf``.
"""

import tilelang
import torch
from tilelang import language as T


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}, )
def tl_indexer_fwd_impl(
    heads,
    index_dim,
    block_N=256,
    num_stages=3,
    threads=512,
    block_Q=None,
):
    if block_Q is None:
        block_Q = 128 // heads
    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    logits_shape = [seq_len, seq_len_kv]

    @T.prim_func
    def tl_indexer_fwd_kernel(
            IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
            IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
            Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
            Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
            CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
            CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, block_Q, heads))
            logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s_min = T.alloc_var(index_dtype)
            cu_k_e_max = T.alloc_var(index_dtype)

            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv))
            for bq_i in T.serial(block_Q):
                cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv))

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for nbn_i in T.Pipelined(T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages):
                T.copy(IndexK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    Logits[seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i] = logits[bn_i, bq_i]

    return tl_indexer_fwd_kernel


@tilelang.jit
def clean_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_logits_kernel(
            Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
            CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
            CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx < cu_k_s or idx >= cu_k_e:
                        Logits[bx, idx] = -T.infinity(dtype)

    return clean_logits_kernel


def indexer_logits(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke):
    """Dense float32 indexer logits ``[rows, tokens_kv]`` for a block of query rows.

    ``index_q`` is ``[rows, heads, index_dim]`` bf16, ``index_k`` is ``[tokens_kv, index_dim]`` bf16,
    ``weights`` is ``[rows, heads]`` float32 (any per-head scale already folded in), and
    ``cu_seqlen_ks`` / ``cu_seqlen_ke`` are int32 ``[rows]`` bounds of the keys a row may see. Entries
    outside a row's window are ``-inf``.
    """
    seq_len, heads, index_dim = index_q.shape
    seq_len_kv = index_k.shape[0]

    clean_logits_kernel = clean_logits_()
    tl_indexer_fwd_kernel = tl_indexer_fwd_impl(heads=heads, index_dim=index_dim)

    logits = torch.empty([seq_len, seq_len_kv], device=index_q.device, dtype=torch.float32)
    tl_indexer_fwd_kernel(
        index_q.view(seq_len * heads, index_dim),
        index_k,
        logits,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    clean_logits_kernel(logits, cu_seqlen_ks, cu_seqlen_ke)
    return logits

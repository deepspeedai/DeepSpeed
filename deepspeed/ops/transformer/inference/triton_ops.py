# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Inspired by original Triton implementation:
https://github.com/openai/triton/blob/release/2.1.x/python/tutorials/06-fused-attention.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(base=Q + qvk_offset,
                                    shape=(N_CTX, BLOCK_DMODEL),
                                    strides=(stride_qm, stride_qk),
                                    offsets=(start_m * BLOCK_M, 0),
                                    block_shape=(BLOCK_M, BLOCK_DMODEL),
                                    order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + qvk_offset,
                                    shape=(BLOCK_DMODEL, N_CTX),
                                    strides=(stride_kk, stride_kn),
                                    offsets=(0, 0),
                                    block_shape=(BLOCK_DMODEL, BLOCK_N),
                                    order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + qvk_offset,
                                    shape=(N_CTX, BLOCK_DMODEL),
                                    strides=(stride_vk, stride_vn),
                                    offsets=(0, 0),
                                    block_shape=(BLOCK_N, BLOCK_DMODEL),
                                    order=(1, 0))
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)
    # loop over k, v and update accumulator
    lo = 0
    #hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    hi = N_CTX
    #hi = (start_m + 1) * BLOCK_M
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        #if IS_CAUSAL:
        #qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    acc = acc / l_i[:, None]
    #l_ptrs = L + off_hz * N_CTX + offs_m
    #tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    O_block_ptr = tl.make_block_ptr(base=Out + qvk_offset,
                                    shape=(N_CTX, BLOCK_DMODEL),
                                    strides=(stride_om, stride_on),
                                    offsets=(start_m * BLOCK_M, 0),
                                    block_shape=(BLOCK_M, BLOCK_DMODEL),
                                    order=(1, 0))
    tl.store(O_block_ptr, acc.to(tl.float16))


class triton_flash_attn(torch.nn.Module):

    def __init__(self, ):
        super(triton_flash_attn, self).__init__()

    def forward(self, q, k, v, sm_scale, block_128=True):
        BLOCK = 128 if block_128 else 64
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1])
        num_warps = 4 if Lk <= 64 else 8

        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            k.shape[0],
            k.shape[1],
            k.shape[2],
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            BLOCK_DMODEL=Lk,
            num_warps=num_warps,
            num_stages=1,
        )
        return o

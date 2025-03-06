# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
from deepspeed.accelerator import get_accelerator
from deepspeed import comm as dist
from deepspeed.ops.transformer.inference.op_binding import LinearOp, VectorMatMulOp, SoftmaxContextOp, QKVGemmOp
from deepspeed.ops.transformer.inference.triton import (
    softmax,
    score_4d_matmul,
    context_4d_matmul,
)

minus_inf = -10000.0


class TritonSelfAttention(nn.Module):
    num_layers = 0

    def __init__(self, config, mp_group=None, q_scales=None, q_groups=1, merge_count=1, qkv_merging=False):
        super(TritonSelfAttention, self).__init__()
        self.config = config
        data_type = self.config.dtype
        data_type_fp = torch.half if self.config.dtype == torch.int8 else self.config.dtype
        assert data_type_fp == torch.half, "triton supports fp16 data_type_fp"

        self.config.layer_id = TritonSelfAttention.num_layers
        TritonSelfAttention.num_layers = TritonSelfAttention.num_layers + 1
        device = get_accelerator().current_device_name()  #if config.bigscience_bloom else 'cpu'

        assert config.mp_size == 1, "mp_size has to be 1 with triton attention yet"
        if self.config.set_empty_params:
            self.attn_qw = None
            self.attn_qb = None
            self.attn_kw = None
            self.attn_kb = None
            self.attn_vw = None
            self.attn_vb = None
            self.attn_qkvw = None
            self.attn_qkvb = None
            self.attn_ow = None
            self.attn_ob = None
        else:
            qkv_size_per_partition = (self.config.hidden_size // self.config.mp_size) * 3
            self.attn_qkvw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                      qkv_size_per_partition,
                                                      dtype=data_type,
                                                      device=device),
                                          requires_grad=False)
            self.attn_qkvb = nn.Parameter(torch.empty(qkv_size_per_partition, dtype=data_type_fp, device=device),
                                          requires_grad=False)
            # self-ouput weights
            out_size_per_partition = self.config.hidden_size // self.config.mp_size
            self.attn_ow = nn.Parameter(torch.empty(out_size_per_partition,
                                                    self.config.hidden_size,
                                                    dtype=data_type,
                                                    device=device),
                                        requires_grad=False)

            self.attn_ob = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device),
                                        requires_grad=False)

        self.num_attention_heads_per_partition = self.config.heads // self.config.mp_size
        self.hidden_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.hidden_size_per_attention_head = self.config.hidden_size // self.config.heads

        self.mp_group = mp_group
        self.use_flash = False
        # triton flash attention is enabled when the compute capability >= 8.0
        if get_accelerator().is_triton_supported():
            self.use_flash = True

        # used for quantization
        self.q_scales = q_scales
        self.q_groups = q_groups
        self.merge_count = int(math.log2(merge_count))

        self.norm_factor = math.sqrt(self.config.hidden_size // self.config.heads)
        if not config.use_mup:
            self.norm_factor = math.sqrt(self.norm_factor)

        if self.config.scale_attn_by_inverse_layer_idx is True:
            self.norm_factor *= math.sqrt(self.config.layer_id + 1)
            # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/gpt2/modeling_gpt2.py#L191

        triton_autotune = self.config.triton_autotune and self.config.layer_id == 0
        self.qkv_func = QKVGemmOp(config)
        self.score_context_func = SoftmaxContextOp(config)
        self.linear_func = LinearOp(config)
        self.vector_matmul_func = VectorMatMulOp(config)

        self.hidden_size = config.hidden_size
        self.head_size = config.hidden_size // config.heads
        self.scale = (1 / self.norm_factor / self.norm_factor if self.config.scale_attention else 1.0
                      )  # making it back to 1/sqrt(head_size)
        self.triangular_masking = self.config.triangular_masking

        # triton autotune table update for score/context matmul
        if triton_autotune:
            print(f"running triton autotune for regular attention kernel")
            __class__._triton_autotune(2, self.config.max_out_tokens, self.head_size, self.config.hidden_size,
                                       self.triangular_masking, self.scale)

    @staticmethod
    def _triton_autotune(min_seqlen,
                         max_seqlen,
                         head_size,
                         hidden_size,
                         triangular_masking,
                         scale,
                         dtype=torch.float16):
        from deepspeed.ops.transformer.inference.triton.matmul_ext import Fp16Matmul, score_4d_matmul, context_4d_matmul
        seqlen = [(min_seqlen + i)
                  for i in range(0, max_seqlen - min_seqlen + Fp16Matmul._cache_stride + 1, Fp16Matmul._cache_stride)]
        Fp16Matmul._read_autotune_table()
        for N in seqlen:
            qkv = torch.randn((1, N, 3 * hidden_size), dtype=dtype, device='cuda')
            output = score_4d_matmul(qkv, head_size, triangular_masking, scale)
            context_4d_matmul(output, qkv, head_size)
        Fp16Matmul._update_autotune_table()

    def ds_compute_attention(self, qkv_out, input_mask, layer_past, alibi, is_prompt, token_idx, position_ids):
        if isinstance(qkv_out, list):
            qkv_out = qkv_out[0]

        no_masking = input_mask is None

        if no_masking:
            input_mask = torch.empty(1)

        attn_key_value = self.score_context_func(
            query_key_value=qkv_out,
            attn_mask=((1 - input_mask).to(qkv_out.dtype) *
                       minus_inf) if input_mask.dtype == torch.int64 else input_mask,
            heads=self.num_attention_heads_per_partition,
            norm_factor=(1 / self.norm_factor if self.config.scale_attention else 1.0),
            no_masking=no_masking,
            layer_id=self.config.layer_id,
            num_layers=TritonSelfAttention.num_layers,
            alibi=alibi,
            is_prompt=is_prompt,
            token_idx=token_idx,
            position_ids=position_ids)

        context_layer, key_layer, value_layer = attn_key_value
        return context_layer, key_layer, value_layer

    def forward(
            self,
            input,
            input_mask,
            head_mask=None,
            layer_past=None,
            get_present=False,  # not used
            encoder_hidden_states=None,  # not used
            encoder_attention_mask=None,  # not used
            triangularutput_attentions=False,  # not used
            norm_w=None,
            norm_b=None,
            alibi=None,
            use_triton_attention=True,
            **kwargs):

        if not self.config.pre_layer_norm:
            qkv_out = self.linear_func(input=input,
                                       weight=self.attn_qkvw,
                                       bias=self.attn_qkvb,
                                       add_bias=self.attn_qkvb is not None,
                                       do_flash_attn=False,
                                       num_heads=self.num_attention_heads_per_partition,
                                       num_layers=TritonSelfAttention.num_layers)
            qkv = qkv_out
        else:
            qkv_out = self.qkv_func(input=input,
                                    weight=self.attn_qkvw,
                                    bias=(self.attn_qkvb if self.attn_qkvb is not None else norm_b),
                                    gamma=norm_w,
                                    beta=norm_b)
            qkv = qkv_out[0]

        if use_triton_attention and (alibi is None):
            context_layer = _triton_attention(qkv=qkv,
                                              input_mask=input_mask,
                                              scale=self.scale,
                                              layer_past=layer_past,
                                              alibi=alibi,
                                              head_size=self.head_size,
                                              use_triton_flash=self.use_flash,
                                              use_cuda_flash=False,
                                              triangular=self.triangular_masking)
            key_layer, value_layer = qkv[:, :, self.hidden_size:2 * self.hidden_size], qkv[:, :, 2 * self.hidden_size:]
        else:
            is_prompt = kwargs.get("first_token", qkv_out[0].shape[1] > 1)
            token_idx = kwargs.get("token_idx", None)
            position_ids = kwargs.get("position_ids", None)
            context_layer, key_layer, value_layer = self.ds_compute_attention(qkv_out=qkv_out,
                                                                              input_mask=input_mask,
                                                                              layer_past=layer_past,
                                                                              alibi=alibi,
                                                                              is_prompt=is_prompt,
                                                                              toke_idx=token_idx,
                                                                              position_ids=position_ids)
        output = self.vector_matmul_func(input=context_layer, weight=self.attn_ow)

        inp_norm = qkv_out[-1]

        if self.config.mlp_after_attn and self.mp_group is not None and dist.get_world_size(group=self.mp_group) > 1:
            dist.all_reduce(output, group=self.mp_group)

        return (output, key_layer, value_layer, context_layer, inp_norm)


global inference_module


def _triton_attention(qkv,
                      input_mask,
                      layer_past,
                      alibi,
                      scale,
                      head_size,
                      triangular=False,
                      use_cuda_flash=False,
                      use_triton_flash=False,
                      use_ds_attention=False):
    if isinstance(qkv, list):
        qkv = qkv[0]

    assert alibi is None, "layer_past not supported in alibi yet"

    if use_triton_flash:
        output = _triton_packed_flash(qkv,
                                      head_size,
                                      input_mask,
                                      scale,
                                      causal=triangular,
                                      add_mask=(not triangular and input_mask is not None))
    else:
        output = score_4d_matmul(qkv, head_size, triangular, scale)
        if triangular:
            output = softmax(output)
        else:
            output = softmax(output, input_mask)
        output = context_4d_matmul(output, qkv, head_size)

    return output


'''
flash attention 2
modified the triton kernel in
https://github.com/openai/triton/blob/08c16589573621fcb8cd5a9c3b8a0537077f876d/python/tutorials/06-fused-attention.py
'''


@triton.jit
def _flash_packed_kernel(
    QKV,
    mask,
    ADD_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    sm_scale,
    Out,
    stride_qz,
    stride_qn,
    stride_qm,
    stride_mz,
    stride_oz,
    stride_on,
    Z,
    H,
    N_CTX,
    P_SEQ,
    hidden_size,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch = off_hz // H
    head = off_hz % H

    q_offset = batch * stride_qz + head * BLOCK_DMODEL
    k_offset = q_offset + hidden_size
    v_offset = k_offset + hidden_size

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = QKV + q_offset + offs_m[:, None] * stride_qn + offs_d[None, :]
    k_ptrs = QKV + hidden_size + q_offset + offs_n[:, None] * stride_qn + offs_d[None, :]
    v_ptrs = QKV + 2 * hidden_size + q_offset + offs_n[:, None] * stride_qn + offs_d[None, :]

    # mask
    off_mask = batch * stride_mz + offs_n[None, :]
    mask_ptrs = mask + off_mask

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    q = (q * qk_scale).to(tl.float16)
    # loop over k, v and update accumulator
    lo = 0
    hi = P_SEQ + (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX + P_SEQ
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(k_ptrs + start_n * stride_qn, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_qn, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)

        if ADD_MASK:
            mask_val = tl.load(mask_ptrs)
            mask_ptrs += BLOCK_N
            qk = qk + mask_val.to(tl.float32)

        if IS_CAUSAL:
            qk = tl.where(P_SEQ + offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        qk += tl.dot(q, tl.trans(k), out_dtype=tl.float16)
        qk += tl.where((start_n + offs_n)[None, :] < N_CTX, 0, minus_inf)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16))
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back l and m
    acc = acc / l_i[:, None]
    o_offset = batch * stride_oz + head * BLOCK_DMODEL
    out_ptrs = Out + o_offset + (offs_m[:, None] * stride_on + offs_d[None, :])
    tl.store(out_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)


def _triton_packed_flash(qkv, head_size, mask, sm_scale, causal=False, add_mask=True):
    heads = qkv.shape[-1] // 3 // head_size
    hidden_size = qkv.shape[-1] // 3

    BLOCK_M = 128
    BLOCK_N = 64 if head_size <= 64 else 32

    o = torch.empty((qkv.shape[0], qkv.shape[1], hidden_size), device=qkv.device, dtype=torch.half)
    if mask is None:
        mask = torch.empty(0)
        add_mask = False

    grid = (triton.cdiv(qkv.shape[1], BLOCK_M), qkv.shape[0] * heads, 1)
    num_stages = 4 if head_size <= 64 else 3
    num_warps = 4
    P_SEQ = 0

    _flash_packed_kernel[grid](qkv,
                               mask,
                               add_mask,
                               causal,
                               sm_scale,
                               o,
                               qkv.stride(0),
                               qkv.stride(1),
                               qkv.stride(2),
                               mask.stride(1) if add_mask else 0,
                               o.stride(0),
                               o.stride(1),
                               qkv.shape[0],
                               heads,
                               qkv.shape[1],
                               P_SEQ,
                               hidden_size,
                               BLOCK_M=BLOCK_M,
                               BLOCK_N=BLOCK_N,
                               BLOCK_DMODEL=head_size,
                               num_warps=num_warps,
                               num_stages=num_stages)

    return o

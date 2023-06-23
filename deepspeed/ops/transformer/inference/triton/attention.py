# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
import torch
import torch.nn as nn
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
            print(f"running triton autotune for attention")
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

    def ds_compute_attention(self, qkv_out, input_mask, layer_past, alibi):
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
            alibi=alibi)

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
            use_triton_attention=True):

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
            context_layer = compute_attention(qkv=qkv,
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
            context_layer, key_layer, value_layer = self.ds_compute_attention(qkv_out=qkv_out,
                                                                              input_mask=input_mask,
                                                                              layer_past=layer_past,
                                                                              alibi=alibi)
        output = self.vector_matmul_func(input=context_layer, weight=self.attn_ow)

        inp_norm = qkv_out[-1]

        if self.config.mlp_after_attn and self.mp_group is not None and dist.get_world_size(group=self.mp_group) > 1:
            dist.all_reduce(output, group=self.mp_group)

        return (output, key_layer, value_layer, context_layer, inp_norm)


global inference_module


def compute_attention(qkv,
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

    #assert layer_past is None, "layer_past not supported in triton yet"
    assert alibi is None, "layer_past not supported in alibi yet"
    output = score_4d_matmul(qkv, head_size, triangular, scale)
    if triangular:
        output = softmax(output)
    else:
        output = softmax(output, input_mask)
    output = context_4d_matmul(output, qkv, head_size)

    return output

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
import torch
import torch.nn as nn
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from .op_binding import LinearOp, VectorMatMulOp, SoftmaxContextOp, QKVGemmOp, SoftmaxOp

minus_inf = -10000.0


class DeepSpeedSelfAttention(nn.Module):
    num_layers = 0
    _qkv_buffers = []

    def __init__(self, config, mp_group=None, q_scales=None, q_groups=1, merge_count=1):
        super(DeepSpeedSelfAttention, self).__init__()
        self.config = config
        data_type = self.config.dtype
        data_type_fp = torch.half if self.config.dtype == torch.int8 else self.config.dtype
        self.config.layer_id = DeepSpeedSelfAttention.num_layers
        DeepSpeedSelfAttention.num_layers = DeepSpeedSelfAttention.num_layers + 1
        device = get_accelerator().current_device_name()  #if config.bigscience_bloom else 'cpu'
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
            qkv_size_per_partition = (self.config.hidden_size // self.config.mp_size) * 3 if config.num_kv < 0 else \
                                     ((self.config.heads + self.config.num_kv * 2) // self.config.mp_size) * (self.config.hidden_size // self.config.heads)
            self.attn_qkvw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                      qkv_size_per_partition,
                                                      dtype=data_type,
                                                      device=device),
                                          requires_grad=False)
            self.attn_qkvb = nn.Parameter(torch.empty(qkv_size_per_partition, dtype=data_type_fp, device=device),
                                          requires_grad=False)
            out_size_per_partition = self.config.hidden_size // self.config.mp_size
            self.attn_ow = nn.Parameter(torch.empty(out_size_per_partition,
                                                    self.config.hidden_size,
                                                    dtype=data_type,
                                                    device=device),
                                        requires_grad=False)

            self.attn_ob = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device),
                                        requires_grad=False)

        self.num_attention_heads_per_partition = self.config.heads // self.config.mp_size
        self.num_kv_partition = self.config.num_kv // self.config.mp_size
        self.hidden_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.hidden_size_per_attention_head = self.config.hidden_size // self.config.heads

        self.mp_group = mp_group

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

        self.qkv_func = QKVGemmOp(config)
        self.score_context_func = SoftmaxContextOp(config)
        self.linear_func = LinearOp(config)
        self.vector_matmul_func = VectorMatMulOp(config)
        if len(DeepSpeedSelfAttention._qkv_buffers) == 0:
            DeepSpeedSelfAttention._qkv_buffers = [
                torch.empty(self.hidden_size_per_partition * 3,
                            self.config.hidden_size,
                            dtype=data_type_fp,
                            device=device),
                torch.empty(self.hidden_size_per_partition * 3, dtype=data_type_fp, device=device)
            ]

    def compute_attention(self, qkv_out, input_mask, layer_past, alibi):
        if isinstance(qkv_out, list) or isinstance(qkv_out, tuple):
            qkv_out = qkv_out[0]

        no_masking = input_mask is None

        if no_masking:
            input_mask = torch.empty(1)

        attn_key_value = self.score_context_func(
            query_key_value=qkv_out,
            attn_mask=((1 - input_mask).to(qkv_out.dtype) *
                       minus_inf) if input_mask.dtype == torch.int64 else input_mask,
            heads=self.num_attention_heads_per_partition,
            num_kv=self.num_kv_partition,
            norm_factor=(1 / self.norm_factor if self.config.scale_attention else 1.0),
            no_masking=no_masking,
            layer_id=self.config.layer_id,
            num_layers=DeepSpeedSelfAttention.num_layers,
            alibi=alibi)

        context_layer, key_layer, value_layer = attn_key_value
        return context_layer, key_layer, value_layer

    def _merge_qkv(self):
        qvkw = DeepSpeedSelfAttention._qkv_buffers[0]
        qvkw[:self.hidden_size_per_partition, :] = self.attn_qw  # type: ignore
        qvkw[self.hidden_size_per_partition:2 * self.hidden_size_per_partition, :] = self.attn_kw  # type: ignore
        qvkw[2 * self.hidden_size_per_partition:, :] = self.attn_vw  # type: ignore
        if self.attn_qb is not None:
            qvkb = DeepSpeedSelfAttention._qkv_buffers[1]
            qvkb[:self.hidden_size_per_partition] = self.attn_qb
            qvkb[self.hidden_size_per_partition:2 * self.hidden_size_per_partition] = self.attn_kb  # type: ignore
            qvkb[2 * self.hidden_size_per_partition:] = self.attn_vb  # type: ignore
        return DeepSpeedSelfAttention._qkv_buffers

    def forward(self,
                input,
                input_mask,
                head_mask=None,
                layer_past=None,
                get_present=False,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False,
                norm_w=None,
                norm_b=None,
                alibi=None):
        if self.attn_qkvw is None:
            self._attn_qkvw, self._attn_qkvb = self._merge_qkv()
        else:
            self._attn_qkvw = self.attn_qkvw
            self._attn_qkvb = self.attn_qkvb
        if not self.config.pre_layer_norm:
            qkv_out = self.linear_func(input=input,
                                       weight=self._attn_qkvw,
                                       bias=self._attn_qkvb,
                                       add_bias=self.attn_qkvb is not None,
                                       do_flash_attn=False,
                                       num_heads=self.num_attention_heads_per_partition,
                                       num_layers=DeepSpeedSelfAttention.num_layers)
        else:
            qkv_out = self.qkv_func(input=input,
                                    weight=self._attn_qkvw,
                                    bias=self._attn_qkvb,
                                    gamma=norm_w,
                                    beta=norm_b)

        context_layer, key_layer, value_layer = self.compute_attention(qkv_out=qkv_out,
                                                                       input_mask=input_mask,
                                                                       layer_past=layer_past,
                                                                       alibi=alibi)

        output = self.vector_matmul_func(input=context_layer, weight=self.attn_ow)
        inp_norm = qkv_out[-1]

        if self.config.mlp_after_attn and self.mp_group is not None and dist.get_world_size(group=self.mp_group) > 1:
            dist.all_reduce(output, group=self.mp_group)
        return (output, key_layer, value_layer, context_layer, inp_norm)


class BloomSelfAttention(DeepSpeedSelfAttention):

    def __init__(self, *args, **kwargs):
        super(BloomSelfAttention, self).__init__(*args, **kwargs)
        self.softmax_func = SoftmaxOp(self.config)

    ########### This part is taken/modified form the HF modeling_bloom.py ################
    # Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py

    def _transpose_for_context(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_layer_shape = x.size()[:-2] + \
                                    (self.hidden_size_per_partition,)
        return x.view(*new_x_layer_shape).contiguous()

    def _split_tensor_along_last_dim(self, tensor, num_partitions, contiguous_split_chunks=True):
        """Split a tensor along its last dimension.

        Args:
            tensor: ([`torch.tensor`], *required*):
                input tensor to split
            num_partitions ([`int`], *required*):
                number of partitions to split the tensor
            contiguous_split_chunks ([`bool`], *optional*, default=`False`)::
                If True, make each chunk contiguous in memory.
        """
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        numerator, denominator = tensor.size()[last_dim], num_partitions
        if not (numerator % denominator == 0):
            raise ValueError(f"{numerator} is not divisible by {denominator}")
        last_dim_size = numerator // denominator
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list

    def compute_attention(self, qkv_out, input_mask, layer_past, alibi):
        if isinstance(qkv_out, list) or isinstance(qkv_out, tuple):
            qkv_out = qkv_out[0]

        no_masking = input_mask is None

        if no_masking:
            input_mask = torch.empty(1)

        mixed_x_layer = qkv_out
        alibi = alibi.to(get_accelerator().current_device_name())
        head_dim = self.hidden_size_per_partition // self.num_attention_heads_per_partition
        new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads_per_partition, 3 * head_dim)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        query_layer, key_layer, value_layer = self._split_tensor_along_last_dim(mixed_x_layer, 3)

        # [batch_size, head_dim, q_length, k_length]
        output_size = (query_layer.size(0), query_layer.size(2), query_layer.size(1), key_layer.size(1))
        # [batch_size, q_length, num_heads, head_dim] -> [q_length, batch_size * num_heads, head_dim]
        query_layer = query_layer.transpose(1, 2).reshape(output_size[0] * output_size[1], output_size[2], -1)
        # [batch_size, k_length, num_heads, head_dim] -> [k_length, batch_size * num_heads, head_dim]
        key_layer = key_layer.transpose(1, 2).reshape(output_size[0] * output_size[1], output_size[3],
                                                      -1).transpose(-1, -2)
        value_layer = value_layer.transpose(1, 2).reshape(output_size[0] * output_size[1], output_size[3], -1)
        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension -> [batch_size, qk_length, num_heads, head_dim]
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=-1)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=-2)

        presents = (key_layer, value_layer)
        # Raw attention scores. [batch_size * num_heads, q_length, k_length]
        matmul_result = torch.matmul(query_layer, key_layer)
        # change view to [batch_size, num_heads, q_length, k_length]
        attention_scores = matmul_result.view(output_size[0], output_size[1], output_size[2], -1)

        offset = dist.get_rank() * self.num_attention_heads_per_partition if dist.is_initialized() else 0
        target_dtype = torch.float16 if self.config.dtype == torch.int8 else self.config.dtype

        # When using the hybrid engine with BLOOM, input_mask needs to be converted from torch.bool -> torch.int64
        if input_mask.dtype == torch.bool:
            input_mask = input_mask.long()

        # Invert input_mask per transformer implementation (eg, in BLOOM, it's already inverted)
        if self.config.invert_mask:
            input_mask = 1 - input_mask

        attention_probs = self.softmax_func(attn_scores=attention_scores,
                                            attn_mask=input_mask.to(target_dtype) * minus_inf,
                                            alibi=alibi,
                                            triangular=(self.config.triangular_masking
                                                        and (attention_scores.shape[-2] > 1)),
                                            recompute=False,
                                            local_attention=False,
                                            window_size=1,
                                            async_op=False,
                                            layer_scale=1 / (self.norm_factor * self.norm_factor),
                                            head_offset=offset)

        # change view [batch_size x num_heads, q_length, k_length]
        attention_probs_reshaped = attention_probs.view(*matmul_result.shape)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = context_layer.view(
            context_layer.size(0) // self.num_attention_heads_per_partition, self.num_attention_heads_per_partition,
            context_layer.size(1), context_layer.shape[-1])

        context_layer = self._transpose_for_context(context_layer)
        key_layer = presents[0]
        value_layer = presents[1]

        return context_layer, key_layer, value_layer

    ###################### End of HF modeling_bloom addition ########################

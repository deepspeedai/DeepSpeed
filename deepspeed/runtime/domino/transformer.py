# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn.functional as F
import enum
import deepspeed.comm as dist

from .async_linear import DominoAsyncColumnParallelLinear, RowParallelLinearNoComm


class LayerType(enum.Enum):
    encoder = 1
    decoder = 2


class AttnType(enum.Enum):
    self_attn = 1
    cross_attn = 2


class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2


class ModelType(enum.Enum):
    encoder_or_decoder = 1
    encoder_and_decoder = 2


class DominoUtil:

    BATCH_0 = "BATCH0"

    BATCH_1 = "BATCH1"

    HANDLE_DIC = {"BATCH0": None, "BATCH1": None}


class DominoModule(torch.nn.Module):
    """extensions of torch Module."""

    def __init__(self, ):
        super(DominoModule, self).__init__()


def _Wait_bwd_comm(input_, dic_, h_id):
    return NoOper.apply(input_, dic_, h_id)


class NoOper(torch.autograd.Function):

    @staticmethod
    def symbolic(graph, input_, handle_dic, h_id):
        return input_

    @staticmethod
    def forward(ctx, input_, handle_dic, h_id):
        ctx.handle_dic = handle_dic
        ctx.h_id = h_id
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        handle = ctx.handle_dic[ctx.h_id]
        handle.wait()
        return grad_output, None, None


class CoreAttention(DominoModule):

    def __init__(self, config, tp_world_size, attn_mask_type=AttnMaskType.causal):
        super(CoreAttention, self).__init__()

        self.attn_mask_type = attn_mask_type

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        assert projection_size % tp_world_size == 0, f"projection size {projection_size} should be multiple of TP world size {tp_world_size}"
        self.hidden_size_per_partition = projection_size // tp_world_size
        self.attention_dropout_rate = config.attention_dropout

    def forward(self, query_layer, key_layer, value_layer, attention_mask):

        context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer,
                                                                         key_layer,
                                                                         value_layer,
                                                                         attn_mask=None,
                                                                         dropout_p=self.attention_dropout_rate,
                                                                         is_causal=True,
                                                                         scale=None)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class ShardedAttention(DominoModule):
    """Sharded self-attention layer class.
    Only support self attention and causal attention mask for now.
    """

    def __init__(self,
                 config,
                 mpu,
                 apply_rotary_pos_emb,
                 layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.causal):
        super(ShardedAttention, self).__init__()

        assert attention_type == AttnType.self_attn, "Only support self_attn for now!"

        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = config.params_dtype
        self.apply_rotary_pos_emb = apply_rotary_pos_emb

        query_projection_size = config.kv_channels * config.num_attention_heads
        kv_projection_size = config.kv_channels * config.num_attention_heads

        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = query_projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads // tp_world_size

        qkv_projection_per_partition = (query_projection_size + 2 * kv_projection_size) // tp_world_size

        self.query_key_value = DominoAsyncColumnParallelLinear(config.hidden_size,
                                                               qkv_projection_per_partition,
                                                               mpu.get_tensor_model_parallel_group(),
                                                               config=config,
                                                               init_method=config.init_method,
                                                               bias=config.add_bias_linear)

        self.core_attention = CoreAttention(config, tp_world_size, self.attn_mask_type)

        query_projection_size_per_partition = query_projection_size // tp_world_size

        # Output.
        self.dense = RowParallelLinearNoComm(query_projection_size_per_partition,
                                             config.hidden_size,
                                             config=config,
                                             init_method=config.output_layer_init_method,
                                             bias=config.add_bias_linear,
                                             skip_bias_add=True)

    def forward(self, hidden_states, attention_mask, micro_batch_num, rotary_pos_emb=None):
        # hidden_states: [sq, b, h]

        mixed_x_layer, _ = self.query_key_value(hidden_states, DominoUtil.HANDLE_DIC, micro_batch_num)

        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )

        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        mixed_x_layer = mixed_x_layer.permute(1, 2, 0, 3).contiguous()

        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, [
            self.hidden_size_per_attention_head, self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head
        ],
                                                            dim=3)

        query_layer = query_layer.view(query_layer.size(0), query_layer.size(1), -1,
                                       self.hidden_size_per_attention_head)

        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = ((rotary_pos_emb, ) * 2)
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query_layer = self.apply_rotary_pos_emb(query_layer, q_pos_emb)
            key_layer = self.apply_rotary_pos_emb(key_layer, k_pos_emb)

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        output, bias = self.dense(context_layer)
        return output, bias

    def domino_core_attention_forward(self, mixed_x_layer, attention_mask, rotary_pos_emb=None):
        # hidden_states: [sq, b, h]

        # To illustrate the difference between intra-layer overlap and inter-layer overlap
        # mixed_x_layer, _ = self.query_key_value(hidden_states, handle_dic, micro_batch_num)

        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )

        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        mixed_x_layer = mixed_x_layer.permute(1, 2, 0, 3).contiguous()

        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, [
            self.hidden_size_per_attention_head, self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head
        ],
                                                            dim=3)

        query_layer = query_layer.view(query_layer.size(0), query_layer.size(1), -1,
                                       self.hidden_size_per_attention_head)

        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = ((rotary_pos_emb, ) * 2)
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query_layer = self.apply_rotary_pos_emb(query_layer, q_pos_emb)
            key_layer = self.apply_rotary_pos_emb(key_layer, k_pos_emb)

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # output, bias = self.dense(context_layer)
        # return output, bias

        return context_layer


class bias_dropout_add(torch.nn.Module):

    def __init__(self, prob: float):
        super(bias_dropout_add, self).__init__()
        self.dropout = torch.nn.Dropout(prob)

    def forward(self, x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if bias is not None:
            x = x + bias
        out = self.dropout(x)
        out = out + residual
        return out


class DominoTransformerLayer(DominoModule):
    """A domino single transformer layer.
    [s, b, h] -> [s, b, h]
    """

    def __init__(self,
                 config,
                 mpu,
                 apply_rotary_pos_emb,
                 layer_number,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.causal,
                 drop_path_rate=0.):

        super(DominoTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = config.apply_residual_connection_post_layernorm

        self.llama_model = False

        self.input_layernorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)

        # Self attention.
        self.self_attention = ShardedAttention(config,
                                               mpu,
                                               apply_rotary_pos_emb,
                                               layer_number,
                                               attention_type=AttnType.self_attn,
                                               attn_mask_type=self_attn_mask_type)

        self.hidden_dropout = config.hidden_dropout

        self.post_attention_layernorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)

        # MLP
        ffn_hidden_size = config.ffn_hidden_size
        if config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.output_size_c = config.ffn_hidden_size
        self.input_size_c = config.hidden_size
        self.input_size_r = config.ffn_hidden_size
        self.output_size_r = self.input_size_c

        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        self.TP_group = mpu.get_tensor_model_parallel_group()
        self.output_size_per_partition = self.output_size_c // tp_world_size
        self.input_size_per_partition = self.input_size_r // tp_world_size

        self.linear_fc1 = DominoAsyncColumnParallelLinear(self.input_size_c,
                                                          self.output_size_per_partition,
                                                          mpu.get_tensor_model_parallel_group(),
                                                          config=config,
                                                          init_method=config.init_method,
                                                          bias=config.add_bias_linear)

        self.mlp_activation_func = F.gelu

        self.linear_fc2 = RowParallelLinearNoComm(self.input_size_per_partition,
                                                  self.output_size_r,
                                                  config=config,
                                                  init_method=config.output_layer_init_method,
                                                  bias=config.add_bias_linear,
                                                  skip_bias_add=True)

        self.bias_dropout_add_func = bias_dropout_add(self.hidden_dropout)

    def forward(self, hidden_states, attention_mask, rotary_pos_emb=None):

        hidden_states0, hidden_states1 = hidden_states

        layernorm_output0 = self.input_layernorm(hidden_states0)
        layernorm_output0 = _Wait_bwd_comm(layernorm_output0, DominoUtil.HANDLE_DIC, DominoUtil.BATCH_0)

        # Micro batch 0: attention
        attention_output0, attention_bias0 = self.self_attention(layernorm_output0,
                                                                 attention_mask,
                                                                 DominoUtil.BATCH_0,
                                                                 rotary_pos_emb=rotary_pos_emb)

        fwd_handle0 = dist.all_reduce(attention_output0, group=self.TP_group, async_op=True)
        # End of Micro batch 0: attention

        # Micro batch 1: attention
        layernorm_output1 = self.input_layernorm(hidden_states1)
        layernorm_output1 = _Wait_bwd_comm(layernorm_output1, DominoUtil.HANDLE_DIC, DominoUtil.BATCH_1)

        attention_output1, attention_bias1 = self.self_attention(layernorm_output1,
                                                                 attention_mask,
                                                                 DominoUtil.BATCH_1,
                                                                 rotary_pos_emb=rotary_pos_emb)
        fwd_handle1 = dist.all_reduce(attention_output1, group=self.TP_group, async_op=True)

        # Micro batch 0: Residual connection.
        fwd_handle0.wait()
        if self.apply_residual_connection_post_layernorm:
            residual0 = layernorm_output0
        else:
            residual0 = hidden_states0

        layernorm_input0 = self.bias_dropout_add_func(attention_output0, attention_bias0, residual0)

        layernorm_output0 = self.post_attention_layernorm(layernorm_input0)
        layernorm_output0 = _Wait_bwd_comm(layernorm_output0, DominoUtil.HANDLE_DIC, DominoUtil.BATCH_0)

        if self.apply_residual_connection_post_layernorm:
            residual0 = layernorm_output0
        else:
            residual0 = layernorm_input0
        # End of Micro batch 0: Residual connection.

        # ------------ MLP ------------
        # Micro batch 0: MLP
        output0, _ = self.linear_fc1(layernorm_output0, DominoUtil.HANDLE_DIC, DominoUtil.BATCH_0)
        output0 = self.mlp_activation_func(output0)

        # Micro batch 1: Residual connection.
        fwd_handle1.wait()
        if self.apply_residual_connection_post_layernorm:
            residual1 = layernorm_output1
        else:
            residual1 = hidden_states1

        layernorm_input1 = self.bias_dropout_add_func(attention_output1, attention_bias1, residual1)

        layernorm_output1 = self.post_attention_layernorm(layernorm_input1)
        layernorm_output1 = _Wait_bwd_comm(layernorm_output1, DominoUtil.HANDLE_DIC, DominoUtil.BATCH_1)

        if self.apply_residual_connection_post_layernorm:
            residual1 = layernorm_output1
        else:
            residual1 = layernorm_input1
        # End of Micro batch 1: Residual connection.

        hidden_states0, last_mlp_bias = self.linear_fc2(output0)
        fwd_handle0 = dist.all_reduce(hidden_states0, group=self.TP_group, async_op=True)
        # End of Micro batch 0: MLP

        # Micro batch 1: MLP
        output1, _ = self.linear_fc1(layernorm_output1, DominoUtil.HANDLE_DIC, DominoUtil.BATCH_1)
        output1 = self.mlp_activation_func(output1)

        hidden_states1, last_mlp_bias = self.linear_fc2(output1)

        fwd_handle1 = dist.all_reduce(hidden_states1, group=self.TP_group, async_op=True)
        # End of Micro batch 1: MLP

        # ------------  End of MLP ------------

        fwd_handle0.wait()
        hidden_states0 = self.bias_dropout_add_func(hidden_states0, last_mlp_bias, residual0)

        fwd_handle1.wait()
        hidden_states1 = self.bias_dropout_add_func(hidden_states1, last_mlp_bias, residual1)

        return hidden_states0, hidden_states1


class DominoTransformer(DominoModule):
    """Transformer class."""

    def __init__(self,
                 config,
                 mpu,
                 apply_rotary_pos_emb,
                 model_type,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.causal,
                 post_layer_norm=True,
                 pre_process=True,
                 post_process=True,
                 drop_path_rate=0.0):
        super(DominoTransformer, self).__init__()

        self.layer_type = layer_type
        self.model_type = model_type
        self.post_layer_norm = post_layer_norm
        self.post_process = post_process
        self.input_tensor = None
        self.drop_path_rate = drop_path_rate
        self.TP_group = mpu.get_tensor_model_parallel_group()

        if not dist.is_initialized():
            dist.init_distributed()
            assert dist.is_initialized(), "deepspeed.comm failed to initialize!"

        self.num_layers = config.num_layers

        self.drop_path_rates = [rate.item() for rate in torch.linspace(0, self.drop_path_rate, config.num_layers)]

        def build_layer(layer_number):

            current_layer_type = layer_type
            return DominoTransformerLayer(config,
                                          mpu,
                                          apply_rotary_pos_emb,
                                          layer_number,
                                          layer_type=current_layer_type,
                                          self_attn_mask_type=self_attn_mask_type,
                                          drop_path_rate=self.drop_path_rates[layer_number - 1])

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_process and self.post_layer_norm:
            self.final_layernorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)

        self._forward_impl = self.inter_layer_overlap_forward
        if config.domino_intra_layer_overlap:
            self._forward_impl = self.intra_layer_overlap_forward

    def forward(self, hidden_states, attention_mask, rotary_pos_emb=None):

        return self._forward_impl(hidden_states, attention_mask, rotary_pos_emb)

    def inter_layer_overlap_forward(self, hidden_states, attention_mask, rotary_pos_emb=None):
        # hidden_states: [s, b, h]

        hidden_states0, hidden_states1 = torch.chunk(hidden_states, chunks=2, dim=1)

        last_mlp_bias = None
        fwd_handle0, fwd_handle1 = None, None
        residual0, residual1 = None, None

        layernorm_output0 = self.layers[0].input_layernorm(hidden_states0)
        layernorm_output0 = _Wait_bwd_comm(layernorm_output0, DominoUtil.HANDLE_DIC, DominoUtil.BATCH_0)

        for index in range(self.num_layers):

            # Micro batch 0: attention
            attention_output0, _ = self.layers[index].self_attention.query_key_value(
                layernorm_output0, DominoUtil.HANDLE_DIC, DominoUtil.BATCH_0)
            attention_output0 = self.layers[index].self_attention.domino_core_attention_forward(
                attention_output0, attention_mask, rotary_pos_emb=rotary_pos_emb)

            # Micro batch 1: Residual connection
            if index > 0:
                fwd_handle1.wait()
                hidden_states1 = self.layers[index - 1].bias_dropout_add_func(hidden_states1, last_mlp_bias, residual1)

            layernorm_output1 = self.layers[index].input_layernorm(hidden_states1)
            layernorm_output1 = _Wait_bwd_comm(layernorm_output1, DominoUtil.HANDLE_DIC, DominoUtil.BATCH_1)
            # End of Micro batch 1: Residual connection

            attention_output0, attention_bias0 = self.layers[index].self_attention.dense(attention_output0)

            fwd_handle0 = dist.all_reduce(attention_output0, group=self.TP_group, async_op=True)
            # End of Micro batch 0: attention

            # Micro batch 1: attention
            attention_output1, _ = self.layers[index].self_attention.query_key_value(
                layernorm_output1, DominoUtil.HANDLE_DIC, DominoUtil.BATCH_1)
            attention_output1 = self.layers[index].self_attention.domino_core_attention_forward(
                attention_output1, attention_mask, rotary_pos_emb=rotary_pos_emb)

            # Micro batch 0: Residual connection.
            fwd_handle0.wait()
            if self.layers[index].apply_residual_connection_post_layernorm:
                residual0 = layernorm_output0
            else:
                residual0 = hidden_states0

            layernorm_input0 = self.layers[index].bias_dropout_add_func(attention_output0, attention_bias0, residual0)

            layernorm_output0 = self.layers[index].post_attention_layernorm(layernorm_input0)
            layernorm_output0 = _Wait_bwd_comm(layernorm_output0, DominoUtil.HANDLE_DIC, DominoUtil.BATCH_0)

            if self.layers[index].apply_residual_connection_post_layernorm:
                residual0 = layernorm_output0
            else:
                residual0 = layernorm_input0
            # End of Micro batch 0: Residual connection.

            attention_output1, attention_bias1 = self.layers[index].self_attention.dense(attention_output1)
            fwd_handle1 = dist.all_reduce(attention_output1, group=self.TP_group, async_op=True)
            #  End of Micro batch 1: attention

            # ------------ MLP ------------
            # Micro batch 0: MLP
            output0, _ = self.layers[index].linear_fc1(layernorm_output0, DominoUtil.HANDLE_DIC, DominoUtil.BATCH_0)
            output0 = self.layers[index].mlp_activation_func(output0)

            # Micro batch 1: Residual connection.
            fwd_handle1.wait()
            if self.layers[index].apply_residual_connection_post_layernorm:
                residual1 = layernorm_output1
            else:
                residual1 = hidden_states1

            layernorm_input1 = self.layers[index].bias_dropout_add_func(attention_output1, attention_bias1, residual1)

            layernorm_output1 = self.layers[index].post_attention_layernorm(layernorm_input1)
            layernorm_output1 = _Wait_bwd_comm(layernorm_output1, DominoUtil.HANDLE_DIC, DominoUtil.BATCH_1)

            if self.layers[index].apply_residual_connection_post_layernorm:
                residual1 = layernorm_output1
            else:
                residual1 = layernorm_input1
            # End of Micro batch 1: Residual connection.

            hidden_states0, last_mlp_bias = self.layers[index].linear_fc2(output0)
            fwd_handle0 = dist.all_reduce(hidden_states0, group=self.TP_group, async_op=True)
            # End of Micro batch 0: MLP

            # Micro batch 1: MLP
            output1, _ = self.layers[index].linear_fc1(layernorm_output1, DominoUtil.HANDLE_DIC, DominoUtil.BATCH_1)
            output1 = self.layers[index].mlp_activation_func(output1)

            # Micro batch 0: Residual connection.
            fwd_handle0.wait()
            hidden_states0 = self.layers[index].bias_dropout_add_func(hidden_states0, last_mlp_bias, residual0)

            if index < self.num_layers - 1:
                layernorm_output0 = self.layers[index + 1].input_layernorm(hidden_states0)
                layernorm_output0 = _Wait_bwd_comm(layernorm_output0, DominoUtil.HANDLE_DIC, DominoUtil.BATCH_0)
            # End of Micro batch 0: Residual connection.

            hidden_states1, last_mlp_bias = self.layers[index].linear_fc2(output1)

            fwd_handle1 = dist.all_reduce(hidden_states1, group=self.TP_group, async_op=True)
            # End of Micro batch 1: MLP

            # ------------  End of MLP ------------

        if self.post_process and self.post_layer_norm:
            hidden_states0 = self.final_layernorm(hidden_states0)

        index = self.num_layers - 1

        fwd_handle1.wait()
        hidden_states1 = self.layers[index].bias_dropout_add_func(hidden_states1, last_mlp_bias, residual1)

        if self.post_process and self.post_layer_norm:
            hidden_states1 = self.final_layernorm(hidden_states1)

        hidden_states = torch.cat([hidden_states0, hidden_states1], dim=1)

        return hidden_states

    def intra_layer_overlap_forward(self, hidden_states, attention_mask, rotary_pos_emb=None):

        hidden_states = torch.chunk(hidden_states, chunks=2, dim=1)

        for index in range(self.num_layers):
            layer = self.layers[index]
            hidden_states = layer(hidden_states, attention_mask, rotary_pos_emb)

        hidden_states0, hidden_states1 = hidden_states
        if self.post_process and self.post_layer_norm:
            hidden_states0 = self.final_layernorm(hidden_states0)
            hidden_states1 = self.final_layernorm(hidden_states1)

        hidden_states = torch.cat([hidden_states0, hidden_states1], dim=1)
        return hidden_states

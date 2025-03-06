# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator


def is_rank_0():
    if dist.get_rank() == 0:
        return True


class DominoModule(torch.nn.Module):
    """extensions of torch Module."""

    def __init__(self, ):
        super(DominoModule, self).__init__()


import enum


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


handle_dic = {}


def no_oper(input_, dic_, h_id):
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


def copy_to_tensor_model_parallel_region_a(mpu, input_, dic_, h_id):
    return _CopyToModelParallelRegionA.apply(mpu, input_, dic_, h_id)


class _CopyToModelParallelRegionA(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, mpu, input_, handle_dic, h_id):
        return input_

    @staticmethod
    def forward(ctx, mpu, input_, handle_dic, h_id):
        ctx.mpu = mpu
        ctx.handle_dic = handle_dic
        ctx.h_id = h_id
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # Bypass the function if we are using only 1 GPU.
        if ctx.mpu.get_tensor_model_parallel_world_size() == 1:
            return grad_output

        # Async All-reduce.
        handle = dist.all_reduce(grad_output, group=ctx.mpu.get_tensor_model_parallel_group(), async_op=True)
        ctx.handle_dic[ctx.h_id] = handle
        return None, grad_output, None, None


class CoreAttention(DominoModule):

    def __init__(self, config, layer_number, mpu, attn_mask_type=AttnMaskType.causal):
        super(CoreAttention, self).__init__()

        self.layer_number = max(1, layer_number)
        self.att_dropout_p = config.attention_dropout
        self.is_causal = True
        projection_size = config.kv_channels * config.num_attention_heads
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = projection_size // world_size

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # attn_mask is None when is_causal=True
        context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer,
                                                                         key_layer,
                                                                         value_layer,
                                                                         attn_mask=None,
                                                                         dropout_p=self.att_dropout_p,
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
    Only support self attention and causal attention mask
    """

    def __init__(self,
                 config,
                 layer_number,
                 mpu,
                 ColumnParallelLinear,
                 RowParallelLinearNoComm,
                 apply_rotary_pos_emb,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.causal):
        super(ShardedAttention, self).__init__()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = config.params_dtype
        self.apply_rotary_pos_emb = apply_rotary_pos_emb

        query_projection_size = config.kv_channels * config.num_attention_heads
        kv_projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = query_projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads // world_size

        self.query_key_value = ColumnParallelLinear(config.hidden_size,
                                                    query_projection_size + 2 * kv_projection_size,
                                                    config=config,
                                                    init_method=config.init_method,
                                                    bias=config.add_bias_linear,
                                                    gather_output=False)

        self.core_attention = CoreAttention(config, self.layer_number, mpu, self.attn_mask_type)

        self.dense = RowParallelLinearNoComm(query_projection_size,
                                             config.hidden_size,
                                             config=config,
                                             init_method=config.output_layer_init_method,
                                             bias=config.add_bias_linear,
                                             input_is_parallel=True,
                                             skip_bias_add=True)

    def forward(self, hidden_states, attention_mask, rotary_pos_emb=None):
        # hidden_states: [s, b, h]

        # Query, Key, and Value
        # Attention heads [s, b, h] --> [s, b, np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [s, b, np * 3 * hn] --> [s, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [s, b, np, 3 * hn] -> [b, np, s, 3*hn]
        mixed_x_layer = mixed_x_layer.permute(1, 2, 0, 3).contiguous()

        # [s, b, np, 3 * hn] --> [s, b, np, hn], [s, b, np, hn], [s, b, np, hn]
        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, [
            self.hidden_size_per_attention_head, self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head
        ],
                                                            dim=3)
        # [s, b, np, np * hn] -> [s, b, np, hn]
        query_layer = query_layer.view(query_layer.size(0), query_layer.size(1), -1,
                                       self.hidden_size_per_attention_head)

        # apply rotary embedding
        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = ((rotary_pos_emb, ) * 2)
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query_layer = self.apply_rotary_pos_emb(query_layer, q_pos_emb)
            key_layer = self.apply_rotary_pos_emb(key_layer, k_pos_emb)

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # Output. [s, b, h]
        output, bias = self.dense(context_layer)

        return output, bias


class DominoTransformerLayer(DominoModule):
    """A domino single transformer layer.
    [s, b, h] -> [s, b, h]
    """

    def __init__(self,
                 config,
                 layer_number,
                 mpu,
                 fused_layer_norm,
                 _initialize_affine_weight_gpu,
                 ColumnParallelLinear,
                 RowParallelLinearNoComm,
                 apply_rotary_pos_emb,
                 bias_dropout_add_fused_train,
                 bias_dropout_add_fused_inference,
                 skip_bias_add=True,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.causal,
                 drop_path_rate=0.,
                 output_bias=None):
        super(DominoTransformerLayer, self).__init__()

        if not dist.is_initialized():
            dist.init_distributed()
            assert dist.is_initialized(), "deepspeed.comm is not initialized!"

        self.llama_model = config.llama_model
        self.layer_number = layer_number
        self.layer_type = layer_type
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.bias_dropout_add_fused_train = bias_dropout_add_fused_train
        self.bias_dropout_add_fused_inference = bias_dropout_add_fused_inference
        self.mpu = mpu
        self.output_bias = output_bias

        # Layernorm on the input data.
        self.input_layernorm = fused_layer_norm(config.hidden_size,
                                                eps=config.layernorm_epsilon,
                                                no_persist_layer_norm=config.no_persist_layer_norm)

        # Self attention.
        self.self_attention = ShardedAttention(config,
                                               layer_number,
                                               mpu,
                                               ColumnParallelLinear,
                                               RowParallelLinearNoComm,
                                               apply_rotary_pos_emb,
                                               attention_type=AttnType.self_attn,
                                               attn_mask_type=self_attn_mask_type)

        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = fused_layer_norm(config.hidden_size,
                                                         eps=config.layernorm_epsilon,
                                                         no_persist_layer_norm=config.no_persist_layer_norm)

        # ------------ init mlp start ------------
        init_method = config.init_method
        output_layer_init_method = config.output_layer_init_method
        self.add_bias = config.add_bias_linear
        self.skip_bias_add = skip_bias_add

        ffn_hidden_size = config.ffn_hidden_size
        if config.gated_linear_unit:
            ffn_hidden_size *= 2
        self.output_size_c = config.ffn_hidden_size
        self.input_size_c = config.hidden_size
        self.input_size_r = config.ffn_hidden_size
        self.output_size_r = self.input_size_c

        world_size = mpu.get_tensor_model_parallel_world_size()
        self.output_size_per_partition = self.output_size_c // world_size
        self.input_size_per_partition = self.input_size_r // world_size

        # Initialize weight.
        self.weight_c = Parameter(
            torch.empty(self.output_size_per_partition,
                        self.input_size_c,
                        device=get_accelerator().current_device_name(),
                        dtype=config.params_dtype))
        self.weight_r = Parameter(
            torch.empty(self.output_size_r,
                        self.input_size_per_partition,
                        device=get_accelerator().current_device_name(),
                        dtype=config.params_dtype))

        if config.perform_initialization:
            _initialize_affine_weight_gpu(self.weight_c, init_method, partition_dim=0, stride=1)

            _initialize_affine_weight_gpu(self.weight_r, output_layer_init_method, partition_dim=1, stride=1)

        if self.add_bias:
            self.bias_c = Parameter(
                torch.empty(self.output_size_per_partition,
                            device=get_accelerator().current_device_name(),
                            dtype=config.params_dtype))
            self.bias_r = Parameter(
                torch.empty(self.output_size_r,
                            device=get_accelerator().current_device_name(),
                            dtype=config.params_dtype))
            if config.perform_initialization:
                with torch.no_grad():
                    self.bias_c.zero_()
                    self.bias_r.zero_()
        else:
            self.register_parameter('bias_c', None)
            self.register_parameter('bias_r', None)

        if config.swiglu:

            def swiglu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]

            self.mlp_activation_func = swiglu
        else:
            self.mlp_activation_func = F.gelu
        # ------------ init mlp end ------------

    def forward(self, hidden_states, attention_mask, rotary_pos_emb=None):
        # hidden_states: [s, b, h]
        hidden_states0, hidden_states1 = hidden_states

        layernorm_output0 = self.input_layernorm(hidden_states0)
        layernorm_output1 = self.input_layernorm(hidden_states1)

        if not self.llama_model:
            rotary_pos_emb = None

        attention_output0, attention_bias0  = \
            self.self_attention(
                layernorm_output0,
                attention_mask,
                rotary_pos_emb=rotary_pos_emb)
        handle0 = dist.all_reduce(attention_output0, group=self.mpu.get_tensor_model_parallel_group(), async_op=True)

        attention_output1, attention_bias1 = \
            self.self_attention(
            layernorm_output1,
            attention_mask,
            rotary_pos_emb=rotary_pos_emb)
        handle1 = dist.all_reduce(attention_output1, group=self.mpu.get_tensor_model_parallel_group(), async_op=True)
        handle0.wait()

        # Residual0 connection.
        if self.apply_residual_connection_post_layernorm:
            residual0 = layernorm_output0
        else:
            residual0 = hidden_states0

        if self.training:
            bias_dropout_add_func = self.bias_dropout_add_fused_train
        else:
            bias_dropout_add_func = self.bias_dropout_add_fused_inference
        if attention_bias0 is not None:
            attention_bias0 = attention_bias0.expand_as(residual0)
        layernorm_input0 = bias_dropout_add_func(attention_output0, attention_bias0, residual0, self.hidden_dropout)

        layernorm_output0 = self.post_attention_layernorm(layernorm_input0)
        layernorm_output0 = no_oper(layernorm_output0, handle_dic, f'{self.layer_number}_0')

        # Residual1 connection.
        if self.apply_residual_connection_post_layernorm:
            residual1 = layernorm_output1
        else:
            residual1 = hidden_states1

        if attention_bias1 is not None:
            attention_bias1 = attention_bias1.expand_as(residual1)
        layernorm_input1 = bias_dropout_add_func(attention_output1, attention_bias1, residual1, self.hidden_dropout)

        layernorm_output1 = self.post_attention_layernorm(layernorm_input1)
        layernorm_output1 = no_oper(layernorm_output1, handle_dic, f'{self.layer_number}_1')

        # ------------ explicit mlp start ------------
        bias_c = self.bias_c if not self.skip_bias_add else None

        input0 = copy_to_tensor_model_parallel_region_a(self.mpu, layernorm_output0, handle_dic,
                                                        f'{self.layer_number}_0')
        # Batch0 Matrix multiply.
        output0 = torch.matmul(input0, self.weight_c.t())
        if bias_c is not None:
            output0 = output0 + bias_c
        output0 = self.mlp_activation_func(output0)
        output0 = torch.matmul(output0, self.weight_r.t())
        handle2 = dist.all_reduce(output0, group=self.mpu.get_tensor_model_parallel_group(), async_op=True)

        handle1.wait()

        input1 = copy_to_tensor_model_parallel_region_a(self.mpu, layernorm_output1, handle_dic,
                                                        f'{self.layer_number}_1')
        # Batch1 Matrix multiply.
        output1 = torch.matmul(input1, self.weight_c.t())
        output1 = self.mlp_activation_func(output1)
        if bias_c is not None:
            output1 = output1 + bias_c
        output1 = torch.matmul(output1, self.weight_r.t())
        dist.all_reduce(output1, group=self.mpu.get_tensor_model_parallel_group())

        handle2.wait()

        output0 = output0 + self.bias_r if self.bias_r is not None else output0
        output1 = output1 + self.bias_r if self.bias_r is not None else output1
        output_bias = self.output_bias
        mlp_output0, mlp_output1, mlp_bias0, mlp_bias1 = output0, output1, output_bias, output_bias
        # ------------ explicit mlp end ------------

        if self.apply_residual_connection_post_layernorm:
            residual0 = layernorm_output0
            residual1 = layernorm_output1
        else:
            residual0 = layernorm_input0
            residual1 = layernorm_input1

        if mlp_bias0 is not None:
            mlp_bias0 = mlp_bias0.expand_as(residual0)
            mlp_bias1 = mlp_bias1.expand_as(residual1)
        output0 = bias_dropout_add_func(mlp_output0, mlp_bias0, residual0, self.hidden_dropout)
        output1 = bias_dropout_add_func(mlp_output1, mlp_bias1, residual1, self.hidden_dropout)

        return output0, output1


class DominoTransformer(DominoModule):
    """Transformer class."""

    def __init__(self,
                 config,
                 model_type,
                 mpu,
                 fused_layer_norm,
                 _initialize_affine_weight_gpu,
                 ColumnParallelLinear,
                 RowParallelLinearNoComm,
                 apply_rotary_pos_emb,
                 bias_dropout_add_fused_train,
                 bias_dropout_add_fused_inference,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.causal,
                 pre_process=True,
                 post_process=True,
                 post_layer_norm=True,
                 drop_path_rate=0.0):
        super(DominoTransformer, self).__init__()

        self.layer_type = layer_type
        self.model_type = model_type
        self.post_process = post_process
        self.post_layer_norm = post_layer_norm
        self.num_layers = config.num_layers
        self.drop_path_rate = drop_path_rate
        self.drop_path_rates = [rate.item() for rate in torch.linspace(0, self.drop_path_rate, config.num_layers)]

        def build_layer(layer_number):
            return DominoTransformerLayer(config,
                                          layer_number,
                                          mpu,
                                          fused_layer_norm,
                                          _initialize_affine_weight_gpu,
                                          ColumnParallelLinear,
                                          RowParallelLinearNoComm,
                                          apply_rotary_pos_emb,
                                          bias_dropout_add_fused_train,
                                          bias_dropout_add_fused_inference,
                                          layer_type=layer_type,
                                          self_attn_mask_type=self_attn_mask_type,
                                          drop_path_rate=self.drop_path_rates[layer_number - 1])

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_process and self.post_layer_norm:
            self.final_layernorm = fused_layer_norm(config.hidden_size,
                                                    eps=config.layernorm_epsilon,
                                                    no_persist_layer_norm=config.no_persist_layer_norm)

    def forward(self, hidden_states, attention_mask, rotary_pos_emb=None):
        # hidden_states: [s, b, h]

        for index in range(self.num_layers):
            layer = self.layers[index]
            hidden_states = layer(hidden_states, attention_mask, rotary_pos_emb)

        hidden_states0, hidden_states1 = hidden_states
        if self.post_process and self.post_layer_norm:
            hidden_states0 = self.final_layernorm(hidden_states0)
            hidden_states1 = self.final_layernorm(hidden_states1)

        return hidden_states0, hidden_states1

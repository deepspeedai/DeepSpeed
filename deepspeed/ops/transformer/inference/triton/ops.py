# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
from deepspeed.ops.op_builder import InferenceBuilder
import deepspeed.ops.transformer.inference.triton.matmul_ext as matmul_ext
from deepspeed.ops.transformer.inference.triton.layer_norm import layer_norm, layer_norm_residual

inference_module = None


def vector_matmul_func(input, weight, async_op, q_scale, q_int8, transposed_mode):
    assert not transposed_mode and not async_op and not q_int8
    return matmul_ext.matmul(input, weight, bias=None, activation="", use_triton=True)


def fused_gemm_gelu(input,
                    weight,
                    weight_scale,
                    bias,
                    weight_out,
                    weight_out_scale,
                    epsilon,
                    pre_layer_norm,
                    q_int8,
                    async_op,
                    transposed_mode,
                    use_triton_ln=True):
    assert not transposed_mode

    # activation
    activation = "gelu"

    # intermediate fc in FF
    intm_out = matmul_ext.matmul(input, weight, bias=bias, activation=activation, use_triton=True)

    # output fc in FF
    ff_out = matmul_ext.matmul(
        intm_out,
        weight_out,
        bias=None,
        activation="",  # bias added layer with residual_add + bias + layerNorm layer
        use_triton=True)
    return ff_out


def linear_func(input, weight, bias, add_bias, do_flash_attn, num_heads, transposed_mode=False):
    assert not transposed_mode and not do_flash_attn
    qkv_out = matmul_ext.matmul(input, weight, bias=(bias if add_bias else None), activation="", use_triton=True)

    return qkv_out


def mlp_gemm_func(input,
                  residual,
                  input_bias,
                  weight_interm,
                  weight_out,
                  bias,
                  gamma,
                  beta,
                  epsilon,
                  pre_layer_norm,
                  mlp_after_attn,
                  weight_interm_scale,
                  weight_out_scale,
                  q_int8,
                  mlp_act_func_type,
                  transposed_mode,
                  use_triton_ln=True):
    assert not transposed_mode

    # residual add and layerNorm after attention
    if use_triton_ln:
        mlp_input = layer_norm_residual(input, input_bias, residual, gamma, beta, epsilon)
    else:
        global inference_module
        if inference_module is None:
            inference_module = InferenceBuilder().load()
        mlp_input = inference_module._layer_norm_residual(input, input_bias, residual, gamma, beta, epsilon)

    # activation
    if deepspeed.utils.types.ActivationFuncType(mlp_act_func_type) == deepspeed.utils.types.ActivationFuncType.GELU:
        activation = "gelu"
    elif deepspeed.utils.types.ActivationFuncType(mlp_act_func_type) == deepspeed.utils.types.ActivationFuncType.ReLU:
        activation = "relu"
    else:
        activation = ""

    # intermediate fc in FF
    intm_out = matmul_ext.matmul(mlp_input, weight_interm, bias=bias, activation=activation, use_triton=True)
    # output fc in FF
    ff_out = matmul_ext.matmul(
        intm_out,
        weight_out,
        bias=None,
        activation="",  # bias added layer with residual_add + bias + layerNorm layer
        use_triton=True)

    return ff_out, mlp_input


def qkv_gemm_func(
    input,
    weight,
    q_scale,
    bias,
    gamma,
    beta,
    epsilon,
    add_bias,
    q_int8,
    transposed_mode=False,
    use_triton_ln=True,
):

    assert not transposed_mode
    # residual add and layerNorm after attention
    if use_triton_ln:
        qkv_input = layer_norm(input, gamma, beta, epsilon)
    else:
        global inference_module
        if inference_module is None:
            inference_module = InferenceBuilder().load()
        qkv_input = inference_module.layer_norm(input, gamma, beta, epsilon)

    qkv_out = matmul_ext.matmul(qkv_input, weight, bias=(bias if add_bias else None), activation="", use_triton=True)

    return qkv_out, qkv_input

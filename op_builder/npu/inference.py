# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from enum import IntEnum
from .builder import NPUOpBuilder

try:
    import torch
    import torch_npu
except ImportError as e:
    pass


class ActivationFuncType(IntEnum):
    UNKNOWN = 0
    GELU = 1
    ReLU = 2
    GATED_GELU = 3
    GATED_SILU = 4


class InferenceContext:
    _workspace = None

    _seed = 42
    _curr_offset = 0
    _stream = 0
    _free_memory_size = 0
    _num_tokens = 1
    _attention_unfused_workspace_offset = 0
    _workSpaceSize = 0

    workSpaceSize = 0
    kv_caches = None

    @staticmethod
    def reset_tokens(initial_tokens=1):
        InferenceContext._num_tokens = initial_tokens

    @staticmethod
    def current_tokens():
        return InferenceContext._num_tokens

    @staticmethod
    def GetWorkSpace():
        return InferenceContext._workspace


class NPUInference:

    @staticmethod
    def layer_norm(inputs, gamma, beta, epsilon):
        return torch.nn.functional.layer_norm(inputs, [inputs.shape[-1]], gamma, beta, eps=epsilon)

    @staticmethod
    def _qkv_gemm(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
        inp_norm = torch.nn.functional.layer_norm(inputs, (inputs.shape[2], ), gamma, beta, eps)
        weight = weight.t() if transpose else weight
        tmp = torch.matmul(inp_norm, weight)
        if add_bias:
            tmp += bias
        output = [tmp, inp_norm]
        return output

    @staticmethod
    def qkv_gemm_fp16(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
        return NPUInference._qkv_gemm(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose)

    @staticmethod
    def qkv_gemm_bf16(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
        return NPUInference._qkv_gemm(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose)

    @staticmethod
    def qkv_gemm_fp32(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
        return NPUInference._qkv_gemm(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose)

    @staticmethod
    def _bias_add_transform_0213(vals, bias, hidden_dim, seq_length, seq_offset, heads, num_kv, rotary_dim,
                                 rotate_half, rotate_every_two, rope_theta):
        bsz, _, _ = vals.shape
        q = vals[..., :hidden_dim].reshape(bsz, seq_length, heads, -1)
        k = vals[..., hidden_dim:hidden_dim + num_kv * (hidden_dim // heads)].reshape(bsz, seq_length, num_kv, -1)
        v = vals[..., hidden_dim + num_kv * (hidden_dim // heads):]

        if rotary_dim > 0 and rotate_every_two:
            # sin, cos may use cache
            seq_id = torch.arange(0, seq_length).to("npu")
            inv_freq = torch.arange(0, rotary_dim, 2) / rotary_dim
            inv_freq = inv_freq.to("npu")
            inv_freq = 1.0 / torch.pow(rope_theta, inv_freq)
            inv_freq = torch.outer(seq_id, inv_freq)
            sin = inv_freq.sin()
            cos = inv_freq.cos()
            # shape: [bsz=1, seq_len, heads=1, rotary_dim]
            sin = sin.view(-1, seq_length, 1, rotary_dim // 2).repeat_interleave(2, dim=-1)
            cos = cos.view(-1, seq_length, 1, rotary_dim // 2).repeat_interleave(2, dim=-1)

            q_pos, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_pos, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

            q_pos = torch_npu.npu_rotary_mul(q_pos, cos, sin)
            q = torch.cat([q_pos, q_pass], dim=-1)
            k_pos = torch_npu.npu_rotary_mul(k_pos, cos, sin)
            k = torch.cat([k_pos, k_pass], dim=-1)

        output = q.reshape(bsz, seq_length, -1).contiguous()  # [b, s, H]
        k_cache = k.reshape(bsz, seq_length, heads, -1).transpose(1, 2).contiguous()  # [b, n, s, d]
        v_cache = v.reshape(bsz, seq_length, heads, -1).transpose(1, 2).contiguous()  # [b, n, s, d]
        return output, k_cache, v_cache

    @staticmethod
    def _softmax_context(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two, heads, num_kv,
                         norm_factor, triangular_masking, local_attention, window_size, no_masking, layer_id,
                         num_layers, alibi, rope_theta):
        bsz, seq_len, k = query_key_value.size()
        k = k // (heads + 2 * (num_kv if num_kv > 0 else heads))
        hidden_dim = heads * k

        is_promt = seq_len > 1
        if not InferenceContext.kv_caches:
            InferenceContext.kv_caches = [[None, None] for _ in range(num_layers)]
        if is_promt:
            InferenceContext.reset_tokens(seq_len)
            InferenceContext.kv_caches[layer_id] = [None, None]

        soft_len = InferenceContext.current_tokens()
        workspace = InferenceContext.GetWorkSpace()
        seq_offset = 0 if is_promt else soft_len - 1

        q, k, v = NPUInference._bias_add_transform_0213(vals=query_key_value,
                                                        bias=None,
                                                        hidden_dim=hidden_dim,
                                                        seq_length=seq_len,
                                                        seq_offset=seq_offset,
                                                        heads=heads,
                                                        num_kv=num_kv if num_kv > 0 else heads,
                                                        rotary_dim=rotary_dim,
                                                        rotate_half=rotate_half,
                                                        rotate_every_two=rotate_every_two,
                                                        rope_theta=rope_theta)

        if not is_promt:
            k_cache, v_cache = InferenceContext.kv_caches[layer_id]
            if k_cache is not None:
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)
        InferenceContext.kv_caches[layer_id] = [k, v]
        seq_len = k.shape[2]

        layer_scale = max(1, layer_id) if len(alibi.size()) > 1 else 1.0
        alpha = norm_factor * norm_factor / layer_scale

        output = torch_npu.npu_fusion_attention(q,
                                                k.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous(),
                                                v.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous(),
                                                heads,
                                                "BSH",
                                                pse=None,
                                                padding_mask=None,
                                                atten_mask=attn_mask.bool(),
                                                scale=alpha,
                                                pre_tockens=65536,
                                                next_tockens=65536,
                                                keep_prob=1,
                                                inner_precise=0)[0]

        return output, k, v

    @staticmethod
    def softmax_context_fp16(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two, heads, num_kv,
                             norm_factor, triangular_masking, local_attention, window_size, no_masking, layer_id,
                             num_layers, alibi, rope_theta):
        return NPUInference._softmax_context(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two,
                                             heads, num_kv, norm_factor, triangular_masking, local_attention,
                                             window_size, no_masking, layer_id, num_layers, alibi, rope_theta)

    @staticmethod
    def softmax_context_bf16(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two, heads, num_kv,
                             norm_factor, triangular_masking, local_attention, window_size, no_masking, layer_id,
                             num_layers, alibi, rope_theta):
        return NPUInference._softmax_context(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two,
                                             heads, num_kv, norm_factor, triangular_masking, local_attention,
                                             window_size, no_masking, layer_id, num_layers, alibi, rope_theta)

    @staticmethod
    def softmax_context_fp32(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two, heads, num_kv,
                             norm_factor, triangular_masking, local_attention, window_size, no_masking, layer_id,
                             num_layers, alibi, rope_theta):
        return NPUInference._softmax_context(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two,
                                             heads, num_kv, norm_factor, triangular_masking, local_attention,
                                             window_size, no_masking, layer_id, num_layers, alibi, rope_theta)

    @staticmethod
    def _vector_matmul(input, weight, async_op, q_scale, q_int8, transposed_mode):
        if transposed_mode:
            return torch.matmul(input, weight.t())
        return torch.matmul(input, weight)

    @staticmethod
    def vector_matmul_fp16(input, weight, async_op, q_scale, q_int8, transposed_mode):
        return NPUInference._vector_matmul(input, weight, async_op, q_scale, q_int8, transposed_mode)

    @staticmethod
    def vector_matmul_bf16(input, weight, async_op, q_scale, q_int8, transposed_mode):
        return NPUInference._vector_matmul(input, weight, async_op, q_scale, q_int8, transposed_mode)

    @staticmethod
    def vector_matmul_fp32(input, weight, async_op, q_scale, q_int8, transposed_mode):
        return NPUInference._vector_matmul(input, weight, async_op, q_scale, q_int8, transposed_mode)

    @staticmethod
    def _mlp_gemm(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps, pre_layer_norm,
                  mlp_after_attn, interm_scale, out_scale, dtype, mlp_act_func_type, transpose):
        if mlp_after_attn:
            residual_add = torch.nn.functional.layer_norm(input + residual + input_bias, (input.shape[-1], ), gamma,
                                                          beta, eps)
        else:
            residual_add = torch.nn.functional.layer_norm(input, (input.shape[-1], ), gamma, beta, eps)

        weight_interm = weight_interm.t() if transpose else weight_interm
        tmp = torch.matmul(residual_add, weight_interm)
        if mlp_act_func_type == ActivationFuncType.GELU:
            tmp = torch.nn.functional.gelu(tmp + bias)
        elif mlp_act_func_type == ActivationFuncType.ReLU:
            tmp = torch.nn.functional.relu(tmp + bias)
        else:
            raise Exception('Unsupported ActivationFuncType {}'.format(mlp_act_func_type))
        output = torch.matmul(tmp, weight_out.t())
        return output, residual_add

    @staticmethod
    def mlp_gemm_fp16(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps, pre_layer_norm,
                      mlp_after_attn, interm_scale, out_scale, dtype, mlp_act_func_type, transpose):
        return NPUInference._mlp_gemm(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps,
                                      pre_layer_norm, mlp_after_attn, interm_scale, out_scale, dtype,
                                      mlp_act_func_type, transpose)

    @staticmethod
    def mlp_gemm_bf16(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps, pre_layer_norm,
                      mlp_after_attn, interm_scale, out_scale, dtype, mlp_act_func_type, transpose):
        return NPUInference._mlp_gemm(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps,
                                      pre_layer_norm, mlp_after_attn, interm_scale, out_scale, dtype,
                                      mlp_act_func_type, transpose)

    @staticmethod
    def mlp_gemm_fp32(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps, pre_layer_norm,
                      mlp_after_attn, interm_scale, out_scale, dtype, mlp_act_func_type, transpose):
        return NPUInference._mlp_gemm(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps,
                                      pre_layer_norm, mlp_after_attn, interm_scale, out_scale, dtype,
                                      mlp_act_func_type, transpose)

    @staticmethod
    def _residual_add_bias(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size,
                           mlp_after_attn, add_bias, pre_layer_norm):
        if mlp_after_attn:
            if pre_layer_norm:
                tmp = (residual.float() + attention_output.float() + attention_bias.float() +
                       final_bias.float()) / mp_size + hidden_state.float()
            else:
                tmp = residual.float() + hidden_state.float() + final_bias.float()
        else:
            if add_bias:
                residual += attention_bias.float()
            tmp = hidden_state.float() + attention_output.float() + (residual.float() + final_bias.float()) / mp_size

        input_dtype = hidden_state.dtype
        residual.set_(tmp.to(input_dtype))

    @staticmethod
    def residual_add_bias_fp16(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size,
                               mlp_after_attn, add_bias, pre_layer_norm):
        return NPUInference._residual_add_bias(hidden_state, residual, attention_output, attention_bias, final_bias,
                                               mp_size, mlp_after_attn, add_bias, pre_layer_norm)

    @staticmethod
    def residual_add_bias_bf16(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size,
                               mlp_after_attn, add_bias, pre_layer_norm):
        return NPUInference._residual_add_bias(hidden_state, residual, attention_output, attention_bias, final_bias,
                                               mp_size, mlp_after_attn, add_bias, pre_layer_norm)

    @staticmethod
    def residual_add_bias_fp32(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size,
                               mlp_after_attn, add_bias, pre_layer_norm):
        return NPUInference._residual_add_bias(hidden_state, residual, attention_output, attention_bias, final_bias,
                                               mp_size, mlp_after_attn, add_bias, pre_layer_norm)


class InferenceBuilder(NPUOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER_INFERENCE"
    NAME = "transformer_inference"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.inference.{self.NAME}_op'

    def sources(self):
        return []

    def include_paths(self):
        return []

    def load(self, verbose=True):
        return NPUInference

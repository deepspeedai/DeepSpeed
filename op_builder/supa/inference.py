# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
SUPA Transformer Inference op builder.
"""

try:
    import torch
    import torch_supa_ext.deepspeed  # noqa: F401 — registers torch.ops.deepspeed
except ImportError:
    pass

from .builder import SUPAOpBuilder


class SUPAInference:
    """Python wrapper around the SUPA compiled inference kernels.

    Each static method delegates to the corresponding torch.ops.deepspeed
    function, mirroring the interface that DeepSpeed's op_binding layer
    expects from inference_module.
    """

    @staticmethod
    def _op(name):
        """Return torch.ops.deepspeed.<name>, raising clearly if not registered."""
        import torch  # ensure torch is available at runtime
        if not hasattr(torch.ops, 'deepspeed') or not hasattr(torch.ops.deepspeed, name):
            raise RuntimeError(f"torch.ops.deepspeed.{name} is not available. "
                               "Ensure torch_supa_ext is built with the transformer inference extension.")
        return getattr(torch.ops.deepspeed, name)

    # ── workspace management ────────────────────────────────────────────────

    @staticmethod
    def allocate_workspace_fp16(hidden_dim, heads, sequence_length, num_layers, batch_size, mp_size, bigscience_bloom,
                                seed, max_out_tokens, min_out_tokens):
        return SUPAInference._op('allocate_workspace_fp16')(hidden_dim, heads, sequence_length, num_layers, batch_size,
                                                            mp_size, bigscience_bloom, seed, max_out_tokens,
                                                            min_out_tokens)

    @staticmethod
    def allocate_workspace_bf16(hidden_dim, heads, sequence_length, num_layers, batch_size, mp_size, bigscience_bloom,
                                seed, max_out_tokens, min_out_tokens):
        return SUPAInference._op('allocate_workspace_bf16')(hidden_dim, heads, sequence_length, num_layers, batch_size,
                                                            mp_size, bigscience_bloom, seed, max_out_tokens,
                                                            min_out_tokens)

    @staticmethod
    def allocate_workspace_fp32(hidden_dim, heads, sequence_length, num_layers, batch_size, mp_size, bigscience_bloom,
                                seed, max_out_tokens, min_out_tokens):
        return SUPAInference._op('allocate_workspace_fp32')(hidden_dim, heads, sequence_length, num_layers, batch_size,
                                                            mp_size, bigscience_bloom, seed, max_out_tokens,
                                                            min_out_tokens)

    @staticmethod
    def release_workspace():
        return SUPAInference._op('release_workspace')()

    @staticmethod
    def retake_workspace():
        return SUPAInference._op('retake_workspace')()

    @staticmethod
    def reset_cache():
        return SUPAInference._op('reset_cache')()

    # ── normalisation ────────────────────────────────────────────────────────

    @staticmethod
    def layer_norm(inputs, gamma, beta, epsilon):
        return SUPAInference._op('layer_norm')(inputs, gamma, beta, epsilon)

    @staticmethod
    def rms_norm(inputs, gamma, epsilon):
        return SUPAInference._op('rms_norm')(inputs, gamma, epsilon)

    @staticmethod
    def pre_rms_norm(inputs, residual, gamma, epsilon):
        return SUPAInference._op('pre_rms_norm')(inputs, residual, gamma, epsilon)

    # ── softmax ──────────────────────────────────────────────────────────────

    @staticmethod
    def softmax_fp16(scores, mask, alibi, triangular, recompute, local_attention, window_size, async_op, layer_scale,
                     head_offset, mp_size):
        return SUPAInference._op('softmax_fp16')(scores, mask, alibi, triangular, recompute, local_attention,
                                                 window_size, async_op, layer_scale, head_offset, mp_size)

    @staticmethod
    def softmax_bf16(scores, mask, alibi, triangular, recompute, local_attention, window_size, async_op, layer_scale,
                     head_offset, mp_size):
        return SUPAInference._op('softmax_bf16')(scores, mask, alibi, triangular, recompute, local_attention,
                                                 window_size, async_op, layer_scale, head_offset, mp_size)

    @staticmethod
    def softmax_fp32(scores, mask, alibi, triangular, recompute, local_attention, window_size, async_op, layer_scale,
                     head_offset, mp_size):
        return SUPAInference._op('softmax_fp32')(scores, mask, alibi, triangular, recompute, local_attention,
                                                 window_size, async_op, layer_scale, head_offset, mp_size)

    @staticmethod
    def softmax_context_fp16(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two, heads, num_kv,
                             norm_factor, triangular_masking, local_attention, window_size, no_masking, layer_id,
                             num_layers, alibi, rope_theta, is_prompt, token_idx, position_ids):
        return SUPAInference._op('softmax_context_fp16')(query_key_value, attn_mask, rotary_dim, rotate_half,
                                                         rotate_every_two, heads, num_kv, norm_factor,
                                                         triangular_masking, local_attention, window_size, no_masking,
                                                         layer_id, num_layers, alibi, rope_theta, is_prompt, token_idx,
                                                         position_ids)

    @staticmethod
    def softmax_context_bf16(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two, heads, num_kv,
                             norm_factor, triangular_masking, local_attention, window_size, no_masking, layer_id,
                             num_layers, alibi, rope_theta, is_prompt, token_idx, position_ids):
        return SUPAInference._op('softmax_context_bf16')(query_key_value, attn_mask, rotary_dim, rotate_half,
                                                         rotate_every_two, heads, num_kv, norm_factor,
                                                         triangular_masking, local_attention, window_size, no_masking,
                                                         layer_id, num_layers, alibi, rope_theta, is_prompt, token_idx,
                                                         position_ids)

    @staticmethod
    def softmax_context_fp32(query_key_value, attn_mask, rotary_dim, rotate_half, rotate_every_two, heads, num_kv,
                             norm_factor, triangular_masking, local_attention, window_size, no_masking, layer_id,
                             num_layers, alibi, rope_theta, is_prompt, token_idx, position_ids):
        return SUPAInference._op('softmax_context_fp32')(query_key_value, attn_mask, rotary_dim, rotate_half,
                                                         rotate_every_two, heads, num_kv, norm_factor,
                                                         triangular_masking, local_attention, window_size, no_masking,
                                                         layer_id, num_layers, alibi, rope_theta, is_prompt, token_idx,
                                                         position_ids)

    # ── bias ops ─────────────────────────────────────────────────────────────

    @staticmethod
    def bias_add_fp16(input, bias):
        return SUPAInference._op('bias_add_fp16')(input, bias)

    @staticmethod
    def bias_add_bf16(input, bias):
        return SUPAInference._op('bias_add_bf16')(input, bias)

    @staticmethod
    def bias_add_fp32(input, bias):
        return SUPAInference._op('bias_add_fp32')(input, bias)

    @staticmethod
    def bias_gelu_fp16(input, bias):
        return SUPAInference._op('bias_gelu_fp16')(input, bias)

    @staticmethod
    def bias_gelu_bf16(input, bias):
        return SUPAInference._op('bias_gelu_bf16')(input, bias)

    @staticmethod
    def bias_gelu_fp32(input, bias):
        return SUPAInference._op('bias_gelu_fp32')(input, bias)

    @staticmethod
    def bias_relu_fp16(input, bias):
        return SUPAInference._op('bias_relu_fp16')(input, bias)

    @staticmethod
    def bias_relu_bf16(input, bias):
        return SUPAInference._op('bias_relu_bf16')(input, bias)

    @staticmethod
    def bias_relu_fp32(input, bias):
        return SUPAInference._op('bias_relu_fp32')(input, bias)

    @staticmethod
    def bias_residual_fp16(input, residual, bias):
        return SUPAInference._op('bias_residual_fp16')(input, residual, bias)

    @staticmethod
    def bias_residual_fp32(input, residual, bias):
        return SUPAInference._op('bias_residual_fp32')(input, residual, bias)

    @staticmethod
    def residual_add_bias_fp16(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size,
                               mlp_after_attn, add_bias, pre_layer_norm):
        return SUPAInference._op('residual_add_bias_fp16')(hidden_state, residual, attention_output, attention_bias,
                                                           final_bias, mp_size, mlp_after_attn, add_bias,
                                                           pre_layer_norm)

    @staticmethod
    def residual_add_bias_bf16(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size,
                               mlp_after_attn, add_bias, pre_layer_norm):
        return SUPAInference._op('residual_add_bias_bf16')(hidden_state, residual, attention_output, attention_bias,
                                                           final_bias, mp_size, mlp_after_attn, add_bias,
                                                           pre_layer_norm)

    @staticmethod
    def residual_add_bias_fp32(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size,
                               mlp_after_attn, add_bias, pre_layer_norm):
        return SUPAInference._op('residual_add_bias_fp32')(hidden_state, residual, attention_output, attention_bias,
                                                           final_bias, mp_size, mlp_after_attn, add_bias,
                                                           pre_layer_norm)

    # ── QKV GEMM ─────────────────────────────────────────────────────────────

    @staticmethod
    def qkv_gemm_fp16(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
        return SUPAInference._op('qkv_gemm_fp16')(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8,
                                                  transpose)

    @staticmethod
    def qkv_gemm_bf16(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
        return SUPAInference._op('qkv_gemm_bf16')(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8,
                                                  transpose)

    @staticmethod
    def qkv_gemm_fp32(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
        return SUPAInference._op('qkv_gemm_fp32')(inputs, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8,
                                                  transpose)

    @staticmethod
    def rms_qkv_gemm_fp16(inputs, weight, q_scale, gamma, eps, q_int8, transpose):
        return SUPAInference._op('rms_qkv_gemm_fp16')(inputs, weight, q_scale, gamma, eps, q_int8, transpose)

    @staticmethod
    def rms_qkv_gemm_bf16(inputs, weight, q_scale, gamma, eps, q_int8, transpose):
        return SUPAInference._op('rms_qkv_gemm_bf16')(inputs, weight, q_scale, gamma, eps, q_int8, transpose)

    # ── MLP GEMM ─────────────────────────────────────────────────────────────

    @staticmethod
    def mlp_gemm_fp16(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps, pre_layer_norm,
                      mlp_after_attn, interm_scale, out_scale, q_int8, act_func_type, transpose):
        return SUPAInference._op('mlp_gemm_fp16')(input, residual, input_bias, weight_interm, weight_out, bias, gamma,
                                                  beta, eps, pre_layer_norm, mlp_after_attn, interm_scale, out_scale,
                                                  q_int8, act_func_type, transpose)

    @staticmethod
    def mlp_gemm_bf16(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps, pre_layer_norm,
                      mlp_after_attn, interm_scale, out_scale, q_int8, act_func_type, transpose):
        return SUPAInference._op('mlp_gemm_bf16')(input, residual, input_bias, weight_interm, weight_out, bias, gamma,
                                                  beta, eps, pre_layer_norm, mlp_after_attn, interm_scale, out_scale,
                                                  q_int8, act_func_type, transpose)

    @staticmethod
    def mlp_gemm_fp32(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps, pre_layer_norm,
                      mlp_after_attn, interm_scale, out_scale, q_int8, act_func_type, transpose):
        return SUPAInference._op('mlp_gemm_fp32')(input, residual, input_bias, weight_interm, weight_out, bias, gamma,
                                                  beta, eps, pre_layer_norm, mlp_after_attn, interm_scale, out_scale,
                                                  q_int8, act_func_type, transpose)

    @staticmethod
    def rms_mlp_gemm_fp16(input, residual, weight_interm, weight_out, gamma, eps, interm_scale, out_scale, q_int8,
                          act_func_type, transpose):
        return SUPAInference._op('rms_mlp_gemm_fp16')(input, residual, weight_interm, weight_out, gamma, eps,
                                                      interm_scale, out_scale, q_int8, act_func_type, transpose)

    @staticmethod
    def rms_mlp_gemm_bf16(input, residual, weight_interm, weight_out, gamma, eps, interm_scale, out_scale, q_int8,
                          act_func_type, transpose):
        return SUPAInference._op('rms_mlp_gemm_bf16')(input, residual, weight_interm, weight_out, gamma, eps,
                                                      interm_scale, out_scale, q_int8, act_func_type, transpose)

    # ── vector / linear ops ──────────────────────────────────────────────────

    @staticmethod
    def vector_matmul_fp16(input, weight, async_op, q_scale, q_int8, transposed_mode):
        return SUPAInference._op('vector_matmul_fp16')(input, weight, async_op, q_scale, q_int8, transposed_mode)

    @staticmethod
    def vector_matmul_bf16(input, weight, async_op, q_scale, q_int8, transposed_mode):
        return SUPAInference._op('vector_matmul_bf16')(input, weight, async_op, q_scale, q_int8, transposed_mode)

    @staticmethod
    def vector_matmul_fp32(input, weight, async_op, q_scale, q_int8, transposed_mode):
        return SUPAInference._op('vector_matmul_fp32')(input, weight, async_op, q_scale, q_int8, transposed_mode)

    @staticmethod
    def linear_layer_fp16(input, weight, bias, add_bias, do_flash_attn, num_heads, transposed_mode, rope_theta):
        return SUPAInference._op('linear_layer_fp16')(input, weight, bias, add_bias, do_flash_attn, num_heads,
                                                      transposed_mode, rope_theta)

    @staticmethod
    def linear_layer_bf16(input, weight, bias, add_bias, do_flash_attn, num_heads, transposed_mode, rope_theta):
        return SUPAInference._op('linear_layer_bf16')(input, weight, bias, add_bias, do_flash_attn, num_heads,
                                                      transposed_mode, rope_theta)

    @staticmethod
    def linear_layer_fp32(input, weight, bias, add_bias, do_flash_attn, num_heads, transposed_mode, rope_theta):
        return SUPAInference._op('linear_layer_fp32')(input, weight, bias, add_bias, do_flash_attn, num_heads,
                                                      transposed_mode, rope_theta)

    @staticmethod
    def _vector_add(a, b, gamma):
        return SUPAInference._op('_vector_add')(a, b, gamma)

    # ── transform / rotary ops ───────────────────────────────────────────────

    @staticmethod
    def pad_transform_fp16(query, key, value, heads, add_padding):
        return SUPAInference._op('pad_transform_fp16')(query, key, value, heads, add_padding)

    @staticmethod
    def apply_rotary_pos_emb(mixed_query, key_layer, rotary_dim, offset, num_heads, rotate_half, rope_theta):
        return SUPAInference._op('apply_rotary_pos_emb')(mixed_query, key_layer, rotary_dim, offset, num_heads,
                                                         rotate_half, rope_theta)

    # ── einsum / MoE ─────────────────────────────────────────────────────────

    @staticmethod
    def einsum_sec_sm_ecm_fp16(Q, W):
        return SUPAInference._op('einsum_sec_sm_ecm_fp16')(Q, W)

    @staticmethod
    def einsum_sec_sm_ecm_fp32(Q, W):
        return SUPAInference._op('einsum_sec_sm_ecm_fp32')(Q, W)

    @staticmethod
    def moe_res_matmul(moe_res, coef, output):
        return SUPAInference._op('moe_res_matmul')(moe_res, coef, output)

    # ── activation ops ───────────────────────────────────────────────────────

    @staticmethod
    def gated_activation(activation_func_type, vals, bias):
        return SUPAInference._op('gated_activation')(activation_func_type, vals, bias)


class InferenceBuilder(SUPAOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER_INFERENCE"
    NAME = "transformer_inference"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.inference.{self.NAME}_op'

    def sources(self):
        return []

    def load(self, verbose=True):
        return SUPAInference

    def is_compatible(self, verbose=False):
        return hasattr(torch.ops, 'deepspeed') and hasattr(torch.ops.deepspeed, 'allocate_workspace_fp16')

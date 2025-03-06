# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Create a container object to save model-specific tensors using the policy file above.

from deepspeed.inference.v2.model_implementations.common_parameters import *
from deepspeed.inference.v2.model_implementations.layer_container_base import LayerContainer


class MixtralTransformerContainer(LayerContainer):

    qkv_w: UnfusedQKVParameter
    attn_out_w: AttentionOutputParameter
    moe_gate: MoEGatingWeightParameter
    moe_mlp_1: UnfusedMoEGatedMLPParameter
    moe_mlp_2: UnfusedMoEMLP2Parameter
    attn_norm_gamma: NormParameter
    mlp_norm_gamma: NormParameter

    PARAM_MAPPING = {
        "input_layernorm.weight": "attn_norm_gamma.params",
        "post_attention_layernorm.weight": "mlp_norm_gamma.params",
        "self_attn.q_proj.weight": "qkv_w.q_params",
        "self_attn.k_proj.weight": "qkv_w.k_params",
        "self_attn.v_proj.weight": "qkv_w.v_params",
        "self_attn.o_proj.weight": "attn_out_w.params",
        "block_sparse_moe.gate.weight": "moe_gate.params",
        "block_sparse_moe.experts.*.w1.weight": "moe_mlp_1.gating_experts",
        "block_sparse_moe.experts.*.w3.weight": "moe_mlp_1.up_experts",
        "block_sparse_moe.experts.*.w2.weight": "moe_mlp_2.experts",
    }


class MixtralNonTransformerContainer(LayerContainer):

    word_emb: EmbeddingParameter
    word_unembed: UnembedParameter
    final_norm: NormParameter

    PARAM_MAPPING = {
        "model.embed_tokens.weight": "word_emb.params",
        "lm_head.weight": "word_unembed.params",
        "model.norm.weight": "final_norm.params",
    }

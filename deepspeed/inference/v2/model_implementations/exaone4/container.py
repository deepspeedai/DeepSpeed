# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from deepspeed.inference.v2.model_implementations.common_parameters import *
from deepspeed.inference.v2.model_implementations.layer_container_base import LayerContainer


class Exaone4TransformerContainer(LayerContainer):
    """
        Transformer layer container for the EXAONE 4.0 model.
    """
    qkv_w: UnfusedQKVParameter
    attn_out_w: AttentionOutputParameter
    mlp_1_w: GatedMLPParameter
    mlp_2_w: MLP2Parameter
    q_norm_gamma: NormParameter
    k_norm_gamma: NormParameter
    post_attn_norm_gamma: NormParameter
    post_ff_norm_gamma: NormParameter

    PARAM_MAPPING = {
        "self_attn.q_proj.weight": "qkv_w.q_params",
        "self_attn.k_proj.weight": "qkv_w.k_params",
        "self_attn.v_proj.weight": "qkv_w.v_params",
        "self_attn.o_proj.weight": "attn_out_w.params",
        "mlp.gate_proj.weight": "mlp_1_w.gate_params",
        "mlp.up_proj.weight": "mlp_1_w.up_params",
        "mlp.down_proj.weight": "mlp_2_w.params",
        "self_attn.q_norm.weight": "q_norm_gamma.params",
        "self_attn.k_norm.weight": "k_norm_gamma.params",
        "post_attention_layernorm.weight": "post_attn_norm_gamma.params",
        "post_feedforward_layernorm.weight": "post_ff_norm_gamma.params",
    }


class Exaone4NonTransformerContainer(LayerContainer):
    """
        Non-Transformer layer container for the EXAONE 4.0 model.
    """
    word_emb: EmbeddingParameter
    word_unembed: UnembedParameter
    final_norm: NormParameter

    PARAM_MAPPING = {
        "model.embed_tokens.weight": "word_emb.params",
        "model.norm.weight": "final_norm.params",
        "lm_head.weight": "word_unembed.params",
    }

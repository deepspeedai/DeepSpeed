# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Create a container object to save model-specific tensors using the policy file above.

from ..common_parameters import *
from ..layer_container_base import LayerContainer
'''
 # HF OPT model looks like this:

OPTForCausalLM(
  (model): OPTModel(
    (decoder): OPTDecoder(
      (embed_tokens): Embedding(50272, 768, padding_idx=1)
      (embed_positions): OPTLearnedPositionalEmbedding(2050, 768)
      (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (layers): ModuleList(
        (0-11): 12 x OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=768, out_features=3072, bias=True)
          (fc2): Linear(in_features=3072, out_features=768, bias=True)
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
  )
  (lm_head): Linear(in_features=768, out_features=50272, bias=False)
)

'''


class OPTTransformerContainer(LayerContainer):
    """
        Transformer layer container for the OPT model.
    """
    qkv_w: UnfusedQKVParameter
    qkv_b: UnfusedQKVParameter
    attn_out_w: AttentionOutputParameter
    attn_out_b: AttentionOutputParameter
    mlp_1_w: MLP1Parameter
    mlp_1_b: MLP1Parameter
    mlp_2_w: MLP2Parameter
    mlp_2_b: MLP2Parameter
    attn_norm_beta: NormParameter
    attn_norm_gamma: NormParameter
    mlp_norm_beta: NormParameter
    mlp_norm_gamma: NormParameter

    PARAM_MAPPING = {
        "self_attn.q_proj.weight": "qkv_w.q_params",
        "self_attn.q_proj.bias": "qkv_b.q_params",
        "self_attn.k_proj.weight": "qkv_w.k_params",
        "self_attn.k_proj.bias": "qkv_b.k_params",
        "self_attn.v_proj.weight": "qkv_w.v_params",
        "self_attn.v_proj.bias": "qkv_b.v_params",
        "self_attn.out_proj.weight": "attn_out_w.params",
        "self_attn.out_proj.bias": "attn_out_b.params",
        "fc1.weight": "mlp_1_w.params",
        "fc1.bias": "mlp_1_b.params",
        "fc2.weight": "mlp_2_w.params",
        "fc2.bias": "mlp_2_b.params",
        "self_attn_layer_norm.weight": "attn_norm_gamma.params",
        "self_attn_layer_norm.bias": "attn_norm_beta.params",
        "final_layer_norm.weight": "mlp_norm_gamma.params",
        "final_layer_norm.bias": "mlp_norm_beta.params",
    }


class OPTNonTransformerContainer(LayerContainer):
    """
        Non-Transformer layer container for the OPT model.
    """
    word_emb: EmbeddingParameter
    word_emb_pos: EmbeddingParameter
    word_unembed: UnembedParameter
    final_norm_w: NormParameter
    final_norm_b: NormParameter

    PARAM_MAPPING = {
        "*decoder.embed_tokens.weight": ["word_emb.params", "word_unembed.params"],
        "*decoder.embed_positions.weight": "word_emb_pos.params",
        "*decoder.final_layer_norm.weight": "final_norm_w.params",
        "*decoder.final_layer_norm.bias": "final_norm_b.params",
    }

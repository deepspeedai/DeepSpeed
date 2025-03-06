# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Create a container object to save model-specific tensors using the policy file above.

from ..common_parameters import *
from ..layer_container_base import LayerContainer
'''
 # HF Falcon 7b model looks like this:

FalconForCausalLM(
  (transformer): FalconModel(
    (word_embeddings): Embedding(65024, 4544)
    (h): ModuleList(
      (0-31): 32 x FalconDecoderLayer(
        (self_attention): FalconAttention(
          (maybe_rotary): FalconRotaryEmbedding()
          (query_key_value): FalconLinear(in_features=4544, out_features=4672, bias=False)
          (dense): FalconLinear(in_features=4544, out_features=4544, bias=False)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (mlp): FalconMLP(
          (dense_h_to_4h): FalconLinear(in_features=4544, out_features=18176, bias=False)
          (act): GELU(approximate='none')
          (dense_4h_to_h): FalconLinear(in_features=18176, out_features=4544, bias=False)
        )
        (input_layernorm): LayerNorm((4544,), eps=1e-05, elementwise_affine=True)
      )
    )
    (ln_f): LayerNorm((4544,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=4544, out_features=65024, bias=False)
)
'''


class FalconTransformerContainer(LayerContainer):
    """
        Transformer layer container for the Falcon model.
    """
    qkv_w: FusedQKVParameter
    attn_out_w: AttentionOutputParameter
    mlp_1_w: MLP1Parameter
    mlp_2_w: MLP2Parameter
    ln_attn_gamma: NormParameter
    ln_attn_beta: NormParameter

    PARAM_MAPPING = {
        "self_attention.query_key_value.weight": "qkv_w.params",
        "self_attention.dense.weight": "attn_out_w.params",
        "mlp.dense_h_to_4h.weight": "mlp_1_w.params",
        "mlp.dense_4h_to_h.weight": "mlp_2_w.params",
        "input_layernorm.weight": "ln_attn_gamma.params",
        "input_layernorm.bias": "ln_attn_beta.params",
    }


class FalconNonTransformerContainer(LayerContainer):
    """
        Non-Transformer layer container for the Falcon model.
    """
    word_emb: EmbeddingParameter
    word_unembed: UnembedParameter
    final_norm_gamma: NormParameter
    final_norm_beta: NormParameter

    PARAM_MAPPING = {
        "transformer.word_embeddings.weight": "word_emb.params",
        "transformer.ln_f.weight": "final_norm_gamma.params",
        "transformer.ln_f.bias": "final_norm_beta.params",
        "lm_head.weight": "word_unembed.params",
    }


'''
 # HF Falcon 40b model looks like this:

 FalconForCausalLM(
  (transformer): FalconModel(
    (word_embeddings): Embedding(65024, 8192)
    (h): ModuleList(
      (0-59): 60 x FalconDecoderLayer(
        (self_attention): FalconAttention(
          (maybe_rotary): FalconRotaryEmbedding()
          (query_key_value): FalconLinear(in_features=8192, out_features=9216, bias=False)
          (dense): FalconLinear(in_features=8192, out_features=8192, bias=False)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (mlp): FalconMLP(
          (dense_h_to_4h): FalconLinear(in_features=8192, out_features=32768, bias=False)
          (act): GELU(approximate='none')
          (dense_4h_to_h): FalconLinear(in_features=32768, out_features=8192, bias=False)
        )
        (ln_attn): LayerNorm((8192,), eps=1e-05, elementwise_affine=True)
        (ln_mlp): LayerNorm((8192,), eps=1e-05, elementwise_affine=True)
      )
    )
    (ln_f): LayerNorm((8192,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=8192, out_features=65024, bias=False)
)
'''


class FalconNewArchTransformerContainer(LayerContainer):
    """
        Transformer layer container for the Falcon model.
    """
    qkv_w: GQAMegatronQKVParameter
    attn_out_w: AttentionOutputParameter
    mlp_1_w: MLP1Parameter
    mlp_2_w: MLP2Parameter
    ln_attn_gamma: NormParameter
    ln_attn_beta: NormParameter
    ln_mlp_gamma: NormParameter
    ln_mlp_beta: NormParameter

    PARAM_MAPPING = {
        "self_attention.query_key_value.weight": "qkv_w.params",
        "self_attention.dense.weight": "attn_out_w.params",
        "mlp.dense_h_to_4h.weight": "mlp_1_w.params",
        "mlp.dense_4h_to_h.weight": "mlp_2_w.params",
        "ln_attn.weight": "ln_attn_gamma.params",
        "ln_attn.bias": "ln_attn_beta.params",
        "ln_mlp.weight": "ln_mlp_gamma.params",
        "ln_mlp.bias": "ln_mlp_beta.params",
    }

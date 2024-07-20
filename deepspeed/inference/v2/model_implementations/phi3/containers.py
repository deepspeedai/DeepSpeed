# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Create a container object to save model-specific tensors using the policy file above.

from ..common_parameters import *
from ..layer_container_base import LayerContainer
'''
 # HF Phi-3 model looks like this:

Phi3ForCausalLM(
  (model): Phi3Model(
    (embed_tokens): Embedding(32064, 3072)
    (embed_dropout): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-31): 32 x Phi3DecoderLayer(
        (self_attn): Phi3Attention(
          (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
          (qkv_proj): Linear(in_features=3072, out_features=9216, bias=False)
          (rotary_emb): Phi3RotaryEmbedding()
        )
        (mlp): PhiMLP(
          (gate_up_proj): Linear(in_features=3072, out_features=16384, bias=False)
          (down_proj): Linear(in_features=16384, out_features=3072, bias=False)
          (activation_fn): SiLU()
        )
        (input_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
        (resid_attn_dropout): Dropout(p=0.0)
        (resid_mlp_dropout): Dropout(p=0.0)
        (post_attention_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
      )
    )
    (final_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
  )
  (lm_head): Linear(in_features=3072, out_features=32064, bias=False)
)
'''


class Phi3TransformerContainer(LayerContainer):
    """
        Transformer layer container for the Phi model.
    """
    qkv_w: FusedQKVParameter
    attn_out_w: AttentionOutputParameter
    mlp_1_w: FusedGatedMLPParameter
    mlp_2_w: MLP2Parameter
    attn_norm_gamma: NormParameter
    mlp_norm_gamma: NormParameter

    PARAM_MAPPING = {
        "self_attn.qkv_proj.weight": "qkv_w.params",
        "self_attn.o_proj.weight": "attn_out_w.params",
        "mlp.gate_up_proj.weight": "mlp_1_w.params",
        "mlp.down_proj.weight": "mlp_2_w.params",
        "input_layernorm.weight": "attn_norm_gamma.params",
        "post_attention_layernorm.weight": "mlp_norm_gamma.params",
    }


class Phi3NonTransformerContainer(LayerContainer):
    """
        Non-Transformer layer container for the Phi model.
    """
    word_emb: EmbeddingParameter
    word_unembed_w: UnembedParameter
    final_norm_gamma: NormParameter

    PARAM_MAPPING = {
        "model.embed_tokens.weight": "word_emb.params",
        "model.norm.weight": "final_norm_gamma.params",
        "lm_head.weight": "word_unembed_w.params",
    }

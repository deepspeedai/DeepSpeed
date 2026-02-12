# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Create a container object to save model-specific tensors for EXAONE 4.0

from ..common_parameters import *
from ..layer_container_base import LayerContainer
"""
HF EXAONE 4.0 model structure:

Exaone4ForCausalLM(
  (model): Exaone4Model(
    (embed_tokens): Embedding(102400, 5120)
    (layers): ModuleList(
      (0-63): 64 x Exaone4DecoderLayer(
        (self_attn): Exaone4Attention(
          (q_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (k_proj): Linear(in_features=5120, out_features=1024, bias=False)
          (v_proj): Linear(in_features=5120, out_features=1024, bias=False)
          (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (rotary_emb): Exaone4RotaryEmbedding()
        )
        (mlp): Exaone4MLP(
          (gate_proj): Linear(in_features=5120, out_features=27392, bias=False)
          (up_proj): Linear(in_features=5120, out_features=27392, bias=False)
          (down_proj): Linear(in_features=27392, out_features=5120, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Exaone4RMSNorm()
        (post_attention_layernorm): Exaone4RMSNorm()
      )
    )
    (norm): Exaone4RMSNorm()
  )
  (lm_head): Linear(in_features=5120, out_features=102400, bias=False)
)

Key EXAONE 4.0 features:
- Hybrid attention: sliding_attention (local) vs full_attention (global) layers
- Grouped Query Attention: 40 query heads, 8 key-value heads
- QK-Reorder-Norm: RMSNorm applied after Q/K projections
- SiLU activation in MLP
"""


class ExaoneTransformerContainer(LayerContainer):
    """
    Transformer layer container for the EXAONE 4.0 model.
    Handles both sliding_attention and full_attention layer types.
    """
    qkv_w: UnfusedQKVParameter
    attn_out_w: AttentionOutputParameter
    mlp_1_w: GatedMLPParameter
    mlp_2_w: MLP2Parameter
    attn_norm_gamma: NormParameter
    mlp_norm_gamma: NormParameter

    PARAM_MAPPING = {
        # Attention parameters - Q, K, V projections
        "self_attn.q_proj.weight": "qkv_w.q_params",
        "self_attn.k_proj.weight": "qkv_w.k_params",
        "self_attn.v_proj.weight": "qkv_w.v_params",
        "self_attn.o_proj.weight": "attn_out_w.params",

        # MLP parameters - gate, up, down projections
        "mlp.gate_proj.weight": "mlp_1_w.gate_params",
        "mlp.up_proj.weight": "mlp_1_w.up_params",
        "mlp.down_proj.weight": "mlp_2_w.params",

        # Normalization parameters
        "input_layernorm.weight": "attn_norm_gamma.params",
        "post_attention_layernorm.weight": "mlp_norm_gamma.params",
    }


class ExaoneNonTransformerContainer(LayerContainer):
    """
    Non-Transformer layer container for the EXAONE 4.0 model.
    Contains embedding, final normalization, and output projection parameters.
    """
    word_emb: EmbeddingParameter
    word_unembed: UnembedParameter
    final_norm: NormParameter

    PARAM_MAPPING = {
        # Embedding and output parameters
        "model.embed_tokens.weight": "word_emb.params",
        "model.norm.weight": "final_norm.params",
        "lm_head.weight": "word_unembed.params",
    }

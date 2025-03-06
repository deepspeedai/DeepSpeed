# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Create a container object to save model-specific tensors using the policy file above.

from ..common_parameters import *
from ..layer_container_base import LayerContainer
'''
 # HF Qwen model looks like this:

QWenLMHeadModel(
  (transformer): QWenModel(
    (wte): Embedding(151936, 4096)
    (drop): Dropout(p=0.0, inplace=False)
    (rotary_emb): RotaryEmbedding()
    (h): ModuleList(
      (0-31): 32 x QWenBlock(
        (ln_1): RMSNorm()
        (attn): QWenAttention(
          (c_attn): Linear(in_features=4096, out_features=12288, bias=True)
          (c_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (attn_dropout): Dropout(p=0.0, inplace=False)
        )
        (ln_2): RMSNorm()
        (mlp): QWenMLP(
          (w1): Linear(in_features=4096, out_features=11008, bias=False)
          (w2): Linear(in_features=4096, out_features=11008, bias=False)
          (c_proj): Linear(in_features=11008, out_features=4096, bias=False)
        )
      )
    )
    (ln_f): RMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=151936, bias=False)
)
'''


class QwenTransformerContainer(LayerContainer):
    """
        Transformer layer container for the Qwen model.
    """
    qkv_w: FusedQKVParameter
    qkv_b: FusedQKVParameter
    attn_out_w: AttentionOutputParameter
    mlp_1_w: GatedMLPParameter
    mlp_2_w: MLP2Parameter
    attn_norm_gamma: NormParameter
    mlp_norm_gamma: NormParameter

    PARAM_MAPPING = {
        "attn.c_attn.weight": "qkv_w.params",
        "attn.c_attn.bias": "qkv_b.params",
        "attn.c_proj.weight": "attn_out_w.params",
        "mlp.w1.weight": "mlp_1_w.up_params",
        "mlp.w2.weight": "mlp_1_w.gate_params",
        "mlp.c_proj.weight": "mlp_2_w.params",
        "ln_1.weight": "attn_norm_gamma.params",
        "ln_2.weight": "mlp_norm_gamma.params",
    }


class QwenNonTransformerContainer(LayerContainer):
    """
        Non-Transformer layer container for the Qwen model.
    """
    word_emb: EmbeddingParameter
    word_unembed: UnembedParameter
    final_norm: NormParameter

    PARAM_MAPPING = {
        "transformer.wte.weight": "word_emb.params",
        "transformer.ln_f.weight": "final_norm.params",
        "lm_head.weight": "word_unembed.params",
    }

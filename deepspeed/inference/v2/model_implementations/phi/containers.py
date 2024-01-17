# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Create a container object to save model-specific tensors using the policy file above.

from ..common_parameters import *
from ..layer_container_base import LayerContainer
'''
 # HF Phi-2 model looks like this:

PhiForCausalLM(
  (model): PhiModel(
    (embed_tokens): Embedding(51200, 2560)
    (embed_dropout): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-31): 32 x PhiDecoderLayer(
        (self_attn): PhiAttention(
          (q_proj): Linear(in_features=2560, out_features=2560, bias=True)
          (k_proj): Linear(in_features=2560, out_features=2560, bias=True)
          (v_proj): Linear(in_features=2560, out_features=2560, bias=True)
          (dense): Linear(in_features=2560, out_features=2560, bias=True)
          (rotary_emb): PhiRotaryEmbedding()
        )
        (mlp): PhiMLP(
          (activation_fn): NewGELUActivation()
          (fc1): Linear(in_features=2560, out_features=10240, bias=True)
          (fc2): Linear(in_features=10240, out_features=2560, bias=True)
        )
        (input_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (final_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=2560, out_features=51200, bias=True)
)
'''


class PhiTransformerContainer(LayerContainer):
    """
        Transformer layer container for the Phi model.
    """
    qkv_w: UnfusedQKVParameter
    qkv_b: UnfusedQKVParameter
    attn_out_w: AttentionOutputParameter
    attn_out_b: AttentionOutputParameter
    mlp_1_w: MLP1Parameter
    mlp_1_b: MLP1Parameter
    mlp_2_w: MLP2Parameter
    mlp_2_b: MLP2Parameter
    ln_gamma: NormParameter
    ln_beta: NormParameter

    PARAM_MAPPING = {
        "self_attn.q_proj.weight": "qkv_w.q_params",
        "self_attn.k_proj.weight": "qkv_w.k_params",
        "self_attn.v_proj.weight": "qkv_w.v_params",
        "self_attn.q_proj.bias": "qkv_b.q_params",
        "self_attn.k_proj.bias": "qkv_b.k_params",
        "self_attn.v_proj.bias": "qkv_b.v_params",
        "self_attn.dense.weight": "attn_out_w.params",
        "self_attn.dense.bias": "attn_out_b.params",
        "mlp.fc1.weight": "mlp_1_w.params",
        "mlp.fc1.bias": "mlp_1_b.params",
        "mlp.fc2.weight": "mlp_2_w.params",
        "mlp.fc2.bias": "mlp_2_b.params",
        "input_layernorm.weight": "ln_gamma.params",
        "input_layernorm.bias": "ln_beta.params",
    }


class PhiNonTransformerContainer(LayerContainer):
    """
        Non-Transformer layer container for the Phi model.
    """
    word_emb: EmbeddingParameter
    word_unembed_w: UnembedParameter
    word_unembed_b: UnembedParameter
    final_norm_gamma: NormParameter
    final_norm_beta: NormParameter

    PARAM_MAPPING = {
        "model.embed_tokens.weight": "word_emb.params",
        "model.final_layernorm.weight": "final_norm_gamma.params",
        "model.final_layernorm.bias": "final_norm_beta.params",
        "lm_head.weight": "word_unembed_w.params",
        "lm_head.bias": "word_unembed_b.params",
    }

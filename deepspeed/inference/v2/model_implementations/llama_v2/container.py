# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Create a container object to save model-specific tensors using the policy file above.

from ..common_parameters import *
from ..layer_container_base import LayerContainer
'''
 # HF Llama model looks like this:

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
'''


class Llama2TransformerContainer(LayerContainer):
    """
        Transformer layer container for the Llama-2 model.
    """
    qkv_w: UnfusedQKVParameter
    attn_out_w: AttentionOutputParameter
    mlp_1_w: GatedMLPParameter
    mlp_2_w: MLP2Parameter
    attn_norm_gamma: NormParameter
    mlp_norm_gamma: NormParameter

    PARAM_MAPPING = {
        "self_attn.q_proj.weight": "qkv_w.q_params",
        "self_attn.k_proj.weight": "qkv_w.k_params",
        "self_attn.v_proj.weight": "qkv_w.v_params",
        "self_attn.o_proj.weight": "attn_out_w.params",
        "mlp.gate_proj.weight": "mlp_1_w.gate_params",
        "mlp.up_proj.weight": "mlp_1_w.up_params",
        "mlp.down_proj.weight": "mlp_2_w.params",
        "input_layernorm.weight": "attn_norm_gamma.params",
        "post_attention_layernorm.weight": "mlp_norm_gamma.params",
    }


class Llama2NonTransformerContainer(LayerContainer):
    """
        Non-Transformer layer container for the Llama-2 model.
    """
    word_emb: EmbeddingParameter
    word_unembed: UnembedParameter
    final_norm: NormParameter

    PARAM_MAPPING = {
        "model.embed_tokens.weight": "word_emb.params",
        "model.norm.weight": "final_norm.params",
        "lm_head.weight": "word_unembed.params",
    }

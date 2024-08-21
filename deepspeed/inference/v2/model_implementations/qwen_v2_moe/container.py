# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Create a container object to save model-specific tensors using the policy file above.

from ..common_parameters import *
from ..layer_container_base import LayerContainer
'''
 # HF Qwen2-57B-A14B model looks like this:

Qwen2MoeForCausalLM(
  (model): Qwen2MoeModel(
    (embed_tokens): Embedding(151936, 3584)
    (layers): ModuleList(
      (0-27): 28 x Qwen2MoeDecoderLayer(
        (self_attn): Qwen2MoeSdpaAttention(
          (q_proj): Linear(in_features=3584, out_features=3584, bias=True)
          (k_proj): Linear(in_features=3584, out_features=512, bias=True)
          (v_proj): Linear(in_features=3584, out_features=512, bias=True)
          (o_proj): Linear(in_features=3584, out_features=3584, bias=False)
          (rotary_emb): Qwen2MoeRotaryEmbedding()
        )
        (mlp): Qwen2MoeSparseMoeBlock(
          (gate): Linear(in_features=3584, out_features=64, bias=False)
          (experts): ModuleList(
            (0-63): 64 x Qwen2MoeMLP(
              (gate_proj): Linear(in_features=3584, out_features=2560, bias=False)
              (up_proj): Linear(in_features=3584, out_features=2560, bias=False)
              (down_proj): Linear(in_features=2560, out_features=3584, bias=False)
              (act_fn): SiLU()
            )
          )
          (shared_expert): Qwen2MoeMLP(
            (gate_proj): Linear(in_features=3584, out_features=20480, bias=False)
            (up_proj): Linear(in_features=3584, out_features=20480, bias=False)
            (down_proj): Linear(in_features=20480, out_features=3584, bias=False)
            (act_fn): SiLU()
          )
          (shared_expert_gate): Linear(in_features=3584, out_features=1, bias=False)
        )
        (input_layernorm): Qwen2MoeRMSNorm((3584,), eps=1e-06)
        (post_attention_layernorm): Qwen2MoeRMSNorm((3584,), eps=1e-06)
      )
    )
    (norm): Qwen2MoeRMSNorm((3584,), eps=1e-06)
  )
  (lm_head): Linear(in_features=3584, out_features=151936, bias=False)
)
'''


class Qwen2MoeTransformerContainer(LayerContainer):
    """
        Transformer layer container for the Qwen2Moe model.
    """
    qkv_w: UnfusedQKVParameter
    qkv_b: UnfusedQKVParameter
    attn_out_w: AttentionOutputParameter
    moe_gate: MoEGatingWeightParameter
    moe_mlp_1: UnfusedMoEGatedMLPParameter
    moe_mlp_2: UnfusedMoEMLP2Parameter
    shared_moe_mlp_1: GatedMLPParameter
    shared_moe_mlp_2: MLP2Parameter
    shared_moe_gate: MoEGatingWeightParameter
    attn_norm_gamma: NormParameter
    mlp_norm_gamma: NormParameter

    PARAM_MAPPING = {
        "self_attn.q_proj.weight": "qkv_w.q_params",
        "self_attn.k_proj.weight": "qkv_w.k_params",
        "self_attn.v_proj.weight": "qkv_w.v_params",
        "self_attn.q_proj.bias": "qkv_b.q_params",
        "self_attn.k_proj.bias": "qkv_b.k_params",
        "self_attn.v_proj.bias": "qkv_b.v_params",
        "self_attn.o_proj.weight": "attn_out_w.params",
        "mlp.gate.weight": "moe_gate.params",
        "mlp.experts.*.gate_proj.weight": "moe_mlp_1.gating_experts",
        "mlp.experts.*.up_proj.weight": "moe_mlp_1.up_experts",
        "mlp.experts.*.down_proj.weight": "moe_mlp_2.experts",
        "mlp.shared_expert.gate_proj.weight": "shared_moe_mlp_1.gate_params",
        "mlp.shared_expert.up_proj.weight": "shared_moe_mlp_1.up_params",
        "mlp.shared_expert.down_proj.weight": "shared_moe_mlp_2.params",
        "mlp.shared_expert_gate.weight": "shared_moe_gate.params",
        "input_layernorm.weight": "attn_norm_gamma.params",
        "post_attention_layernorm.weight": "mlp_norm_gamma.params",
    }


class Qwen2MoeNonTransformerContainer(LayerContainer):
    """
        Non-Transformer layer container for the Qwen2Moe model.
    """
    word_emb: EmbeddingParameter
    word_unembed: UnembedParameter
    final_norm: NormParameter

    PARAM_MAPPING = {
        "model.embed_tokens.weight": "word_emb.params",
        "model.norm.weight": "final_norm.params",
        "lm_head.weight": "word_unembed.params",
    }

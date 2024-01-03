# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Create a container object to save model-specific tensors using the policy file above.

from ..common_parameters import *
from ..layer_container_base import LayerContainer
'''
 # HF Phi-2 model looks like this:

PhiForCausalLM(
  (transformer): PhiModel(
    (embd): Embedding(
      (wte): Embedding(51200, 2560)
      (drop): Dropout(p=0.0, inplace=False)
    )
    (h): ModuleList(
      (0-31): 32 x ParallelBlock(
        (ln): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (resid_dropout): Dropout(p=0.1, inplace=False)
        (mixer): MHA(
          (rotary_emb): RotaryEmbedding()
          (Wqkv): Linear(in_features=2560, out_features=7680, bias=True)
          (out_proj): Linear(in_features=2560, out_features=2560, bias=True)
          (inner_attn): SelfAttention(
            (drop): Dropout(p=0.0, inplace=False)
          )
          (inner_cross_attn): CrossAttention(
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (mlp): MLP(
          (fc1): Linear(in_features=2560, out_features=10240, bias=True)
          (fc2): Linear(in_features=10240, out_features=2560, bias=True)
          (act): NewGELUActivation()
        )
      )
    )
  )
  (lm_head): CausalLMHead(
    (ln): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
    (linear): Linear(in_features=2560, out_features=51200, bias=True)
  )
  (loss): CausalLMLoss(
    (loss_fct): CrossEntropyLoss()
  )
)
'''


class PhiTransformerContainer(LayerContainer):
    """
        Transformer layer container for the Phi model.
    """
    qkv_w: FusedQKVParameter
    qkv_b: FusedQKVParameter
    attn_out_w: AttentionOutputParameter
    attn_out_b: AttentionOutputParameter
    mlp_1_w: MLP1Parameter
    mlp_1_b: MLP1Parameter
    mlp_2_w: MLP2Parameter
    mlp_2_b: MLP2Parameter
    ln_gamma: NormParameter
    ln_beta: NormParameter

    PARAM_MAPPING = {
        "mixer.Wqkv.weight": "qkv_w.params",
        "mixer.Wqkv.bias": "qkv_b.params",
        "mixer.out_proj.weight": "attn_out_w.params",
        "mixer.out_proj.bias": "attn_out_b.params",
        "mlp.fc1.weight": "mlp_1_w.params",
        "mlp.fc1.bias": "mlp_1_b.params",
        "mlp.fc2.weight": "mlp_2_w.params",
        "mlp.fc2.bias": "mlp_2_b.params",
        "ln.weight": "ln_gamma.params",
        "ln.bias": "ln_beta.params",
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
        "transformer.embd.wte.weight": "word_emb.params",
        "lm_head.ln.weight": "final_norm_gamma.params",
        "lm_head.ln.bias": "final_norm_beta.params",
        "lm_head.linear.weight": "word_unembed_w.params",
        "lm_head.linear.bias": "word_unembed_b.params",
    }

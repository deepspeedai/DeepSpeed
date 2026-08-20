# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.inference.v2.model_implementations.common_parameters import *
from deepspeed.inference.v2.model_implementations.layer_container_base import LayerContainer


class Exaone4_5NonTransformerContainer(LayerContainer):
    """
        Non-Transformer layer container for the EXAONE 4.5 model.

        Identical tensors to EXAONE 4.0, but the EXAONE 4.5 checkpoint is a
        multimodal (``Exaone4_5ForConditionalGeneration``) checkpoint in which the
        language-model weights are nested under ``model.language_model.``. The
        ``lm_head`` remains at the top level.
    """
    word_emb: EmbeddingParameter
    word_unembed: UnembedParameter
    final_norm: NormParameter

    PARAM_MAPPING = {
        "model.language_model.embed_tokens.weight": "word_emb.params",
        "model.language_model.norm.weight": "final_norm.params",
        "lm_head.weight": "word_unembed.params",
    }

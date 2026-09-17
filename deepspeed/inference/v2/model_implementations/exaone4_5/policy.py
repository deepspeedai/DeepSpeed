# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any

from ...config_v2 import RaggedInferenceEngineConfig
from ..inference_policy_base import ContainerMap, InferenceV2Policy
from ..exaone4.container import Exaone4TransformerContainer
from .container import Exaone4_5NonTransformerContainer
from .model import Exaone4_5InferenceModel


class Exaone4_5Policy(InferenceV2Policy):

    def instantiate_model(self, engine_config: RaggedInferenceEngineConfig, mp_group: Any) -> Exaone4_5InferenceModel:
        # EXAONE 4.5 ships a multimodal (Exaone4_5ForConditionalGeneration) config;
        # the language model to serve lives in ``text_config``.
        text_config = getattr(self._model_config, "text_config", self._model_config)
        return Exaone4_5InferenceModel(config=text_config, engine_config=engine_config, base_mp_group=mp_group)

    def build_container_map(self) -> ContainerMap:
        map = ContainerMap()

        # The transformer parameter layout matches EXAONE 4.0; only the checkpoint
        # prefix differs (nested under ``model.language_model.``).
        transformer_containers = [Exaone4TransformerContainer(self.model) for _ in range(self.model.num_layers)]
        map.set_transformer_params(['model.language_model.layers'], transformer_containers)

        map.set_non_transformer_params(Exaone4_5NonTransformerContainer(self.model))

        # Text-only serving: the vision tower (``model.visual.``) and the MTP
        # self-speculative head (``mtp.``) are present in the checkpoint but are
        # intentionally left unmapped so the loader skips them.
        map.set_unmapped_params(['model.visual.', 'mtp.'])

        return map

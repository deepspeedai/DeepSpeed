# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any

from ...config_v2 import RaggedInferenceEngineConfig
from ..inference_policy_base import ContainerMap, InferenceV2Policy
from .container import QwenNonTransformerContainer, QwenTransformerContainer
from .model import QwenInferenceModel


class QwenPolicy(InferenceV2Policy):

    def instantiate_model(self, engine_config: RaggedInferenceEngineConfig, mp_group: Any) -> QwenInferenceModel:
        return QwenInferenceModel(config=self._model_config, engine_config=engine_config, base_mp_group=mp_group)

    def build_container_map(self) -> ContainerMap:
        map = ContainerMap()

        transformer_containers = [QwenTransformerContainer(self.model) for _ in range(self.model.num_layers)]

        map.set_transformer_params(['transformer.h'], transformer_containers)

        map.set_non_transformer_params(QwenNonTransformerContainer(self.model))

        map.set_unmapped_params(['transformer.rotary_emb.inv_freq'])

        return map

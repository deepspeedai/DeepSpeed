# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse

from typing import Any

from ...checkpoint import CheckpointEngineBase
from ...config_v2 import RaggedInferenceEngineConfig
from ...model_implementations.inference_policy_base import ContainerMap, InferenceV2Policy
from ...model_implementations.opt.container import OPTNonTransformerContainer, OPTTransformerContainer
from ...model_implementations.opt.model import OPTInferenceModel


class OPTPolicy(InferenceV2Policy):

    def __init__(self, checkpoint_engine: CheckpointEngineBase, model_config: argparse.Namespace) -> None:
        super().__init__(checkpoint_engine, model_config)

    def instantiate_model(self, engine_config: RaggedInferenceEngineConfig, mp_group: Any) -> OPTInferenceModel:
        return OPTInferenceModel(config=self._model_config, engine_config=engine_config, base_mp_group=mp_group)

    def build_container_map(self) -> ContainerMap:
        map = ContainerMap()

        transformer_containers = [OPTTransformerContainer(self.model) for _ in range(self.model.num_layers)]

        map.set_transformer_params(['model.decoder.layers'], transformer_containers)

        map.set_non_transformer_params(OPTNonTransformerContainer(self.model))

        map.set_unmapped_params([])

        return map

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any

from ...config_v2 import RaggedInferenceEngineConfig
from ..inference_policy_base import ContainerMap, InferenceV2Policy
from .container import ExaoneNonTransformerContainer, ExaoneTransformerContainer
from .model import ExaoneInferenceModel


class ExaonePolicy(InferenceV2Policy):
    """
    Policy for EXAONE 4.0 model inference.

    Handles the mapping between HuggingFace checkpoint parameters and DeepSpeed containers,
    and instantiates the EXAONE inference model with hybrid attention support.
    """

    def instantiate_model(self, engine_config: RaggedInferenceEngineConfig, mp_group: Any) -> ExaoneInferenceModel:
        """
        Instantiate the EXAONE 4.0 inference model.

        Arguments:
            engine_config: DeepSpeed inference engine configuration
            mp_group: Multi-processing group for tensor parallelism

        Returns:
            ExaoneInferenceModel: Configured EXAONE inference model
        """
        return ExaoneInferenceModel(config=self._model_config, engine_config=engine_config, base_mp_group=mp_group)

    def build_container_map(self) -> ContainerMap:
        """
        Build the container map for EXAONE 4.0 parameter mapping.

        Maps HuggingFace parameter names to DeepSpeed container dependencies.

        Returns:
            ContainerMap: Configured container map for EXAONE parameters
        """
        map = ContainerMap()

        # Create transformer containers for each layer (64 layers for EXAONE-4.0-32B)
        transformer_containers = [
            ExaoneTransformerContainer(self.model) for _ in range(self._model_config.num_hidden_layers)
        ]
        map.set_transformer_params(['model.layers'], transformer_containers)

        # Create non-transformer container for embedding/output/norm parameters
        map.set_non_transformer_params(ExaoneNonTransformerContainer(self.model))

        # Set unmapped parameters that we want to ignore
        # EXAONE 4.0 doesn't use rotary_emb parameters since RoPE is conditional
        unmapped_params = []

        # Add rotary embedding inverse frequency parameters if they exist
        for i in range(self._model_config.num_hidden_layers):
            unmapped_params.append(f'model.layers.{i}.self_attn.rotary_emb.inv_freq')

        # Add any other parameters that don't need mapping
        map.set_unmapped_params(unmapped_params)

        return map

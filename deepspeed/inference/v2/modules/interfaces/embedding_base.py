# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import abstractmethod
from typing import Any, Dict, Optional, Type

import torch

from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from ...ragged import RaggedBatchWrapper
from ..ds_module import DSModuleBase
from ..module_registry import DSModuleRegistryBase
from ..configs import DSEmbeddingsConfig
from ...inference_parameter import InferenceParameter


class DSEmbeddingBase(DSModuleBase):
    """
    Base mixin for embedding modules. The interface represented by this module is:

    hidden_out = embedding(input_ids) +
                 position_embedding(position_ids) +
                 token_type_embedding(token_type_ids)
    with optional normalization.
    """

    @staticmethod
    def config_class() -> Type[DeepSpeedConfigModel]:
        return DSEmbeddingsConfig

    def __init__(self, config: DSEmbeddingsConfig, implementation_config: Dict[str, Any]) -> None:
        super().__init__(config, implementation_config)

    def transform_param(self, embed_param: torch.Tensor) -> InferenceParameter:
        """
        Perform any necessary transformations on an embedding parameter. This module assumes
        that all embedding parameters would require the same set of transformations.

        Parameters:
            embed_param (torch.Tensor): Embedding parameter. Shape is of [vocab_size, hidden_size]
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def output(self) -> torch.Tensor:
        """
        Pre-allocated output Tensor. This currently needs to be exposed for gather operations
        on the output.

        TODO(cmikeh2): This is not ideal. We need a better abstraction for this, such as giving
        access to the inference comm object to the DSModule.
        """
        raise NotImplementedError()

    def forward(self,
                ragged_batch: RaggedBatchWrapper,
                word_embeddings: torch.Tensor,
                position_embeddings: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                token_type_embeddings: Optional[torch.Tensor] = None) -> InferenceParameter:
        """
        Parameters:
            ragged_batch (torch.Tensor): Ragged batch of token ids + associated metadata.
            word_embeddings (torch.Tensor): Word embeddings.
            position_embeddings (torch.Tensor): Position embeddings. If passed, IDs will be
                inferred from the ragged batch itself.
            token_type_ids (torch.Tensor): Token type ids.
            token_type_embeddings (torch.Tensor): Token type embeddings.

        Returns:
            torch.Tensor: Hidden states. This should be the sum of the relevant
                encodings for the model.
        """
        raise NotImplementedError()


class DSEmbeddingRegistry(DSModuleRegistryBase):
    registry: Dict = {}

    @staticmethod
    def associated_class() -> Type[DSModuleBase]:
        return DSEmbeddingBase

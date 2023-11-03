# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Dict, Optional, Type

import torch

from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from ...ragged import RaggedBatchWrapper
from ..ds_module import DSModuleBase
from ..module_registry import DSModuleRegistryBase
from ..configs import DSUnembedConfig


class DSUnembedBase(DSModuleBase):
    """
    Base mixin for unmebedding modules. The interface represented by this module is:

    if config.do_normalization
        hidden = layer_norm(hidden)
    logits = hidden @ projection
    """

    @staticmethod
    def config_class() -> Type[DeepSpeedConfigModel]:
        return DSUnembedConfig

    def __init__(self, config: DSUnembedConfig, implementation_config: Dict[str, Any]) -> None:
        super().__init__(config, implementation_config)

    def forward(self,
                hidden_states: torch.Tensor,
                vocab_embedding: torch.Tensor,
                ragged_metadata: RaggedBatchWrapper,
                gamma: Optional[torch.Tensor] = None,
                beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward interface. Gamma and beta are optional parameters passed depending on
        `self.config.do_normalization`.

        Args:
            hidden_states (torch.Tensor): Hidden states of shape [tokens, model_dim]
            vocab_embedding (torch.Tensor): Embedding matrix of shape [vocab_size, model_dim]
            ragged_metadata (RaggedBatchWrapper): Metadata for the ragged batch.
            gamma (Optional[torch.Tensor]): Gamma parameter for layer norm.
            beta (Optional[torch.Tensor]): Beta parameter for layer norm.

        Returns:
            torch.Tensor: Unembedded hidden states of shape [n_seqs, model_dim]
        """
        raise NotImplementedError()


class DSUnembedRegistry(DSModuleRegistryBase):
    registry: Dict = {}

    @staticmethod
    def associated_class() -> Type[DSModuleBase]:
        return DSUnembedBase

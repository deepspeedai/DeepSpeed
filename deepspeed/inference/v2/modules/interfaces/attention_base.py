# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Dict, Optional, Tuple, Type

import torch

from ...ragged import RaggedBatchWrapper
from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from ..ds_module import DSModuleBase
from ..module_registry import DSModuleRegistryBase
from ..configs import DSSelfAttentionConfig


class DSSelfAttentionBase(DSModuleBase):
    """
    Base mixin for all attention modules. The interface represented by this module
    is broadly:

    output = attention(query_key_value,
                       Optional[kv_cache],
                       Optional[attention_mask],
                       Optional[attention_bias])
    """

    @staticmethod
    def config_class() -> Type[DeepSpeedConfigModel]:
        return DSSelfAttentionConfig

    def __init__(self, config: DSSelfAttentionConfig, implementation_config: Dict[str, Any]) -> None:
        super().__init__(config, implementation_config)

    @property
    def kv_block_size(self) -> int:
        """
        Return preferred granulatity for blocked KV-cache implementation.
        """
        raise NotImplementedError()

    @property
    def q_block_size(self) -> int:
        """
        Property to calculate blocking granularity for the query dimension.
        This has no impact on the KV-cache structure, but will  affect the
        number of attention atoms associated with a batch.
        """
        raise NotImplementedError()

    def build_atoms(self, ragged_batch: RaggedBatchWrapper) -> None:
        """
        Build the atoms for this module. This is not a strict requirement for the class,
        so this method is a no-op by default rather than abstract.
        """
        pass

    def forward(self,
                q_k_v: torch.Tensor,
                kv_cache: torch.Tensor,
                batch: RaggedBatchWrapper,
                attention_mask: Optional[torch.Tensor] = None,
                attention_bias: Optional[torch.Tensor] = None,
                inv_freqs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
            q_k_v (torch.Tensor): Query, key, and value tensors. Expected shape is:
                [
                    batch,
                    seq_len,
                    2 * self._config.n_heads_kv + self._config.n_heads_q,
                    self._config.head_size
                ].
            kv_cache (Optional[torch.Tensor]): Key and value cache tensor. Expected shape is
                [
                    2,
                    batch,
                    kv_cache_len,
                    self._config.n_heads_kv,
                    self._config.head_size
                ]. If None, cache is disabled. The `kv_cache_len` dimension does not need to
                be contiguous (it should expand stride by `max_out_tokens`).
            batch (RaggedBatchWrapper): Ragged batch metadata.
            attention_mask (Optional[torch.Tensor]): Attention mask tensor. If None, masking is
                disabled. This will defer to the config in the case of conflicting information.
                This means if the config class is implying causal attention, the mask will be ignored.
            attention_bias (Optional[torch.Tensor]): Attention bias tensor. If None, bias is disabled.
        """
        raise NotImplementedError()


class DSSelfAttentionRegistry(DSModuleRegistryBase):
    registry: Dict = {}

    @staticmethod
    def associated_class() -> Type[DSModuleBase]:
        return DSSelfAttentionBase

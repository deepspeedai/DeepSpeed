# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple, Type

import torch

from ..ragged import DSStateManager, RaggedBatchWrapper
from ..ragged.manager_configs import KVCacheConfig
from ..ragged import DSSequenceDescriptor
from ..model_implementations.layer_container_base import LayerContainer
from ..config_v2 import RaggedInferenceEngineConfig
"""
This abstract class defines the interfaces that a model implementation should implement
in order to include anything that may be called by the engine. Most models should be able
to inherit from `DSInferenceTransformerModelBase` to reduce implementation work so it is recommended
to begin there.
"""
"""
Placeholder for typing the model config, which can vary based on model implementation/
"""
DSModelImplementationConfig = Type['DSModelImplementationConfig']
"""
Placeholder for typing the distributed comm object.

TODO(cmikeh2): Replace when we have a more defined API for the inference communication system.
"""
MPType = Type["MPType"]


class DSInferenceModelBase(torch.nn.Module, ABC):
    """
    Implementation of a model for inference composable with ragged batching.
    """

    _config: DSModelImplementationConfig
    """
    Model-specific configuration. No abstraction surrounds this yet.
    """

    _engine_config: RaggedInferenceEngineConfig
    """
    Engine configuration.
    """

    _base_mp_group: MPType
    """
    Base communication group for Tensor-parallel inference.
    """

    _non_transformer: Optional[LayerContainer]
    """
    Abstract container for storing both embedding (pre-transformer) and unembedding (post-transformer)
    parameters. This attribute should be None at model instantiation until the Policy sets
    the model parameters. These parameters are grouped together since many model implementations
    will tie the embedding and unembedding parameters together.
    """

    _transformer: Optional[Iterable[LayerContainer]]
    """
    List of abstract containers (1 per layer) for storing transformer (transformer)
    parameters. This attribute should be None at model instantiation until the Policy
    sets the model parameters.
    """

    state_manager: Optional[DSStateManager]
    """
    Since the state manager is lazy initialized, by the engine, it is not guaranteed to be present
    until full initialization.
    """

    def __init__(self, config: DSModelImplementationConfig, engine_config: RaggedInferenceEngineConfig,
                 base_mp_group: MPType) -> None:
        """
        Minimal initialization of the model.

        Arguments:
            config (DSModelImplementationConfig): Model-specific configuration. No assumptions
                should be made about this config that are not closely tied to the specific
                model implementation.
            engine_config (RaggedInferenceEngineConfig): Engine configuration.
            base_mp_group (MPType): Base communication group for Tensor-parallel inference.
        """
        super().__init__()
        self._config = config
        self._engine_config = engine_config
        self._base_mp_group = base_mp_group

        # Set to None until the Policy sets the model parameters
        self._non_transformer = None
        self._transformer = None

    def set_parameters(self, transformer: Iterable[LayerContainer], non_transformer: LayerContainer):
        """
        Set the model parameters for the embedding, transformer, and unembedding containers.
        """
        self._transformer = transformer
        self._non_transformer = non_transformer

    def set_state_manager(self, state_manager: DSStateManager):
        """
        Sets the state manager attribute. This is called by the inference engine after
        the model is fully initialized.
        """
        self.state_manager = state_manager

    @abstractmethod
    def get_kv_requirements(self, sequence: DSSequenceDescriptor, max_new_tokens: int,
                            max_new_blocks: int) -> Tuple[int, int]:
        """
        Given a sequence and the number of new tokens in the sequence, determine the
        number of new KV blocks needed to support the sequence. This method is
        used to help the engine provide schedulability APIs and can be used as a helper
        for ``maybe_allocate_kv``.

        Args:
            sequence (DSSequenceDescriptor): The sequence for which to allocate KV-storage.
            max_new_tokens (int): Maximum number of tokens to hypothetically schedule.
            max_new_blocks (int): Maximum number of blocks to hypothetically allocate.

        Returns:
            Tuple[int, int]: The tuple of number of tokens scheduled and number
                of blocks allocated. In general, only one of these numbers will match the
                corresponding input argument, but this is not guaranteed.
        """
        raise NotImplementedError()

    @abstractmethod
    def maybe_allocate_kv(self, sequence: DSSequenceDescriptor, n_new_tokens: int) -> None:
        """
        Given a sequence and the number of new tokens in the sequence, determine
        whether or not additional KV-storage is needed and allocate it if so.

        Args:
            sequence (DSSequenceDescriptor): The sequence for which to allocate KV-storage.
            n_new_tokens (int): The number of new tokens in the sequence.
        """
        raise NotImplementedError()

    @abstractmethod
    def kv_cache_config(self) -> KVCacheConfig:
        """
        Return the KV-cache configuration for this model.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def max_sequence_length(self) -> int:
        """
        The maximum sequence length supported by the model.
        """
        ...

    def maybe_free_kv(self, sequence: DSSequenceDescriptor):
        """
        After completing a forward pass, determine whether or not the there are any KV blocks
        that maybe freed since they are no longer in use.

        Consider the following example:

        We have a block size of 4 and a local window size of 8. At the beginning of the forward
        pass there 10 tokens had been seen and the new forward has a size of 4. This would lend
        itself to the following cache structure prior to the forward:
            [[0, 1, 2*, 3*] [4*, 5*, 6*, 7*] [8*, 9*, x, x] [x x x x]]
        Where x's denote empty cache locations and * denote values that are needed for attention
        of the next open slot. After the forward, the cache would look like the following:
            [[0, 1, 2, 3] [4, 5, 6*, 7*] [8*, 9*, 10*, 11*] [12* 13* x x]]
        In this case, the first block is no longer needed since it is not needed for any future
        local attention windows. This function would be responsible for freeing that block.

        Default behavior assumes no local patterns that require freeing and in general should
        be sufficient.
        """
        pass

    @abstractmethod
    def prepare_batch(self, wrapped_batch: RaggedBatchWrapper) -> None:
        """
        This will be called before each forward with the intent of building forward-specific metadata
        about a batch. The intent here is to build data structures like attention atoms without necessarily
        needing to implement graphable kernels to do so.

        Abstract so as to force model implementations to opt out of doing anything here explicitly.
        """
        raise NotImplementedError()

    def forward(wrapped_batch: RaggedBatchWrapper) -> torch.Tensor:
        """
        Complete a forward pass of the model. This interface should be graphable, so it
        should not rely on the ability to use python control flow.
        """
        raise NotImplementedError()

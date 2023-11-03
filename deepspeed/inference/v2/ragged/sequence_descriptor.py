# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Union

import torch


class BaseSequenceDescriptor:

    @property
    def seen_tokens(self) -> int:
        """
        The number of tokens for this sequence that have completed a forward pass.
        """
        raise NotImplementedError()

    @property
    def cur_allocated_blocks(self) -> int:
        """
        The number of KV blocks currently allocated for this sequence.
        """
        raise NotImplementedError()

    @property
    def kv_blocks_ptr(self) -> int:
        """
        The pointer to the KV blocks for this sequence.
        """
        raise NotImplementedError()


class PlaceholderSequenceDescriptor(BaseSequenceDescriptor):
    """
    The DummySequenceDescriptor is an empty object that allows us to perform schedulability
    checks before formally tracking a sequence.
    """

    def __init__(self, seen_tokens=0, cur_allocated_blocks=0, kv_blocks_ptr=0) -> None:
        self._seen_tokens = seen_tokens
        self._cur_allocated_blocks = cur_allocated_blocks
        self._kv_blocks_ptr = kv_blocks_ptr

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    @property
    def cur_allocated_blocks(self) -> int:
        return self._cur_allocated_blocks

    @property
    def kv_blocks_ptr(self) -> int:
        return self._kv_blocks_ptr


class DSSequenceDescriptor(BaseSequenceDescriptor):

    _seen_tokens: int
    """
    Number of tokens in the sequence that have completed a forward pass.
    """

    _in_flight_tokens: int
    """
    Number of tokens that have begun a forward pass but not yet completed it.
    """

    _max_context: int
    """
    Maximum number of tokens this sequence may eventually include. Currently unused but
    may be used in future implementations for speculative caching.
    """

    _num_allocation_groups: int
    """
    Number of unique allocation groups associated with the sequence.
    """

    _blocks_per_allocation_group: torch.IntTensor
    """
    Number of blocks allocated for each allocation group.
    """

    # Padded list of KV-cache IDs for the sequence.
    _kv_cache_ids: torch.Tensor
    _kv_cache_ids_shadow: torch.Tensor
    """
    Padded list of KV-cache IDs for the sequence. The padded shape is [num_allocation_groups, max_blocks_per_allocation_group].
    """

    # The location in the broader ID tensor where the KV-cache IDs for the sequence
    # are stored. Used on flush.
    _tracking_id: int

    def __init__(self,
                 tracking_id: int,
                 kv_cache_ids: torch.Tensor,
                 kv_cache_ids_shadow: torch.Tensor,
                 max_context: int = -1) -> None:
        self._tracking_id = tracking_id
        self._kv_cache_ids = kv_cache_ids
        self._kv_cache_ids_shadow = kv_cache_ids_shadow
        self._max_context = max_context

        self._seen_tokens = 0
        self._in_flight_tokens = 0

        self._num_allocation_groups = kv_cache_ids_shadow.shape[0]
        self._blocks_per_allocation_group = torch.zeros(self._num_allocation_groups, dtype=torch.int32, device="cpu")

        assert kv_cache_ids.shape[0] == self._num_allocation_groups
        assert len(kv_cache_ids.shape) == 2

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    @property
    def in_flight_tokens(self) -> int:
        return self._in_flight_tokens

    @property
    def max_context(self) -> int:
        return self._max_context

    @property
    def cur_allocated_blocks(self) -> int:
        return self._blocks_per_allocation_group.sum()

    @property
    def tracking_id(self) -> int:
        return self._tracking_id

    def kv_cache_ids(self, on_device: bool = False) -> torch.Tensor:
        """
        Returns the Tensor containing the block IDs for this sequence on the appropriate device.
        """
        if on_device:
            return self._kv_cache_ids
        else:
            return self._kv_cache_ids_shadow

    @property
    def kv_blocks_ptr(self) -> int:
        return self._kv_cache_ids.data_ptr()

    @property
    def all_block_ids(self) -> torch.Tensor:
        block_ids = []
        for allocation_group, num_blocks in zip(self._kv_cache_ids, self._blocks_per_allocation_group):
            block_ids.append(allocation_group[:num_blocks])
        return torch.cat(block_ids)

    def pre_forward(self, num_tokens: int) -> None:
        """
        Update the state of the sequence before a forward pass.
        """
        self._in_flight_tokens = num_tokens

    def post_forward(self) -> None:
        """
        Update the state of the sequence after a forward pass.
        """
        self._seen_tokens += self._in_flight_tokens
        self._in_flight_tokens = 0

    def extend_kv_cache(self, new_ids: Union[List[torch.IntTensor], torch.IntTensor]) -> None:
        """
        Extend the KV-cache for the sequence.

        Args:
            new_ids (Union[List[torch.IntTensor], torch.IntTensor]): For each allocation group, the IDs
                to add to the KV-cache. If there is only one allocation group, a single tensor can be
                provided. Otherwise, a list of tensors should be provided. The tensors do not need
                to have the same shape.
        """
        if isinstance(new_ids, torch.Tensor):
            new_ids = [new_ids]

        if len(new_ids) != self._num_allocation_groups:
            raise ValueError(f"Only {len(new_ids)} allocation groups provided, expected {self._num_allocation_groups}")

        for group_id, new_group_ids in enumerate(new_ids):
            new_blocks = new_group_ids.numel()

            if new_blocks == 0:
                # If we have multiple groups, it's possible to have an empty group.
                continue

            shadow_alloc_group = self._kv_cache_ids_shadow[group_id]
            alloc_group = self._kv_cache_ids[group_id]
            cur_blocks = self._blocks_per_allocation_group[group_id]

            shadow_alloc_group[cur_blocks:cur_blocks + new_blocks].copy_(new_group_ids)
            alloc_group[cur_blocks:cur_blocks + new_blocks].copy_(shadow_alloc_group[cur_blocks:cur_blocks +
                                                                                     new_blocks],
                                                                  non_blocking=True)

            self._blocks_per_allocation_group[group_id] += new_blocks

    def free_kv_cache(self, free_ids: Union[List[torch.IntTensor], torch.IntTensor]) -> None:
        """
        Free blocks from the KV-cache for the sequence.

        Args:
            free_ids (Union[List[torch.IntTensor], torch.IntTensor]): The ids of blocks to free
                from the KV-cache. If there is only one allocation group, a single tensor can be
                provided. Otherwise, a list of tensors should be provided. The tensors do not need
                to have the same shape.
        """
        raise NotImplementedError("Partial KV-cache freeing is not yet supported.")

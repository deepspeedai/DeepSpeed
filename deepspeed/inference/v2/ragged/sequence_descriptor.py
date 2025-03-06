# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Tuple, Union

import torch


class BaseSequenceDescriptor:

    @property
    def seen_tokens(self) -> int:
        """
        The number of tokens for this sequence that have completed a forward pass.
        """
        raise NotImplementedError()

    @property
    def cur_allocated_blocks(self, cache_group: int = 0) -> int:
        """
        The number of KV blocks currently allocated for this sequence.
        """
        raise NotImplementedError()

    @property
    def kv_blocks_ptr(self, cache_group: int = 0) -> int:
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
    def cur_allocated_blocks(self, cache_group: int = 0) -> int:
        return self._cur_allocated_blocks

    @property
    def kv_blocks_ptr(self, cache_group: int = 0) -> int:
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

    _num_allocation_groups: Tuple[int, ...]
    """
    Number of unique allocation groups associated with the sequence for each cache group.
    """

    _blocks_per_allocation_group: Tuple[torch.IntTensor, ...]
    """
    Number of blocks allocated for each allocation group in each cache group.
    """

    # Padded list of KV-cache IDs for the sequence.
    _kv_cache_ids: Tuple[torch.Tensor, ...]
    _kv_cache_ids_shadow: Tuple[torch.Tensor, ...]
    """
    Padded list of KV-cache IDs for the sequence. The padded shape is [num_allocation_groups, max_blocks_per_allocation_group].
    """

    # The location in the broader ID tensor where the KV-cache IDs for the sequence
    # are stored. Used on flush.
    _tracking_id: int

    def __init__(self,
                 tracking_id: int,
                 kv_cache_ids: Tuple[torch.Tensor, ...],
                 kv_cache_ids_shadow: Tuple[torch.Tensor, ...],
                 max_context: int = -1) -> None:
        """
        Create the metadata to track a single sequence in the system.

        Arguments:
            tracking_id (int): The slot in the tracking buffers used to track this sequence.
            kv_cache_ids (Tuple[torch.Tensor, ...]): The KV-cache IDs for the sequence. The shape
                of the tensor should be [num_allocation_groups, max_blocks_per_allocation_group].
                There should be one tensor per cache group.
            kv_cache_ids_shadow (Tuple[torch.Tensor, ...]): The shadow tensor for the KV-cache IDs.
                This tensor should be allocated on the host and should have the same shape as the
                tensor provided in ``kv_cache_ids``. There should be one tensor per cache group.
            max_context (int): The maximum number of tokens this sequence may eventually include.
                Currently unused but may be used in future implementations for speculative caching.
        """
        self._tracking_id = tracking_id
        self._kv_cache_ids = kv_cache_ids
        self._kv_cache_ids_shadow = kv_cache_ids_shadow
        self._max_context = max_context
        self._n_cache_groups = len(kv_cache_ids)

        self._seen_tokens = 0
        self._in_flight_tokens = 0

        self._num_allocation_groups = tuple(kv_cache_ids_shadow.shape[0]
                                            for kv_cache_ids_shadow in kv_cache_ids_shadow)
        self._blocks_per_allocation_group = tuple(
            torch.zeros(num_groups, dtype=torch.int32, device="cpu") for num_groups in self._num_allocation_groups)

        for cache_group, kv_cache_ids in enumerate(kv_cache_ids):
            assert self._num_allocation_groups[cache_group] == kv_cache_ids.shape[0]
            assert len(kv_cache_ids.shape) == 2

    @property
    def seen_tokens(self) -> int:
        """
        Number of tokens in the sequence that have completed a forward pass.
        """
        return self._seen_tokens

    @property
    def in_flight_tokens(self) -> int:
        """
        Number of tokens that have begun a forward pass but not yet completed it.
        """
        return self._in_flight_tokens

    @property
    def max_context(self) -> int:
        """
        Maximum number of tokens for this sequence. Currently unused.
        """
        return self._max_context

    @property
    def tracking_id(self) -> int:
        """
        Return the slot in the tracking buffers used to track this sequence.
        """
        return self._tracking_id

    @property
    def cur_allocated_blocks(self, cache_group: int = 0) -> int:
        """
        Returns the number of blocks currently allocated for this sequence in the specified cache group.

        Arguments:
            cache_group (int): The cache group to query.
        """
        # Currently, there is only one allocation group.
        # A shortcut is used here to bypass the overhead of sum().
        if len(self._blocks_per_allocation_group) == 1:
            return self._blocks_per_allocation_group[0].item()
        return self._blocks_per_allocation_group[cache_group].sum().item()

    def kv_cache_ids(self, cache_group: int = 0, on_device: bool = False) -> torch.Tensor:
        """
        Returns the Tensor containing the block IDs for this sequence on the appropriate device
        for the specified cache group.

        Arguments:
            cache_group (int): The cache group to query.
            on_device (bool): Whether or not to return the Tensor on the device or on the host.
        """
        if on_device:
            return self._kv_cache_ids[cache_group]
        else:
            return self._kv_cache_ids_shadow[cache_group]

    @property
    def kv_blocks_ptr(self, cache_group: int = 0) -> int:
        """
        Get the device pointer to the base of the KV-cache ids for the specified cache group and
        sequence.

        Arguments:
            cache_group (int): The cache group to query.
        """
        return self._kv_cache_ids[cache_group].data_ptr()

    #TODO: this was previously a property but causing issues with PR-4668 need to consult w. Connor
    def all_block_ids(self, cache_group: int = 0) -> torch.Tensor:
        """
        Return the Tensor containing all block IDs for this sequence in the specified cache group.

        Arguments:
            cache_group (int): The cache group to query.
        """
        block_ids = []
        for allocation_group, num_blocks in zip(self._kv_cache_ids[cache_group],
                                                self._blocks_per_allocation_group[cache_group]):
            block_ids.append(allocation_group[:num_blocks])
        return torch.cat(block_ids)

    def pre_forward(self, num_tokens: int) -> None:
        """
        Update the state of the sequence before a forward pass.

        Arguments:
            num_tokens (int): The number of tokens in the sequence that will be executed during the
                next forward pass of the model.
        """
        self._in_flight_tokens = num_tokens

    def post_forward(self) -> None:
        """
        Update the state of the sequence after a forward pass. This should be called after the forward
        pass completes. NOTE: due to the asynchronous nature of the accelerator, this may be called
        before the forward pass completes on the device itself.
        """
        self._seen_tokens += self._in_flight_tokens
        self._in_flight_tokens = 0

    def extend_kv_cache(self, new_ids: Union[List[torch.IntTensor], torch.IntTensor], cache_group: int = 0) -> None:
        """
        Extend the KV-cache for the sequence.

        Arguments:
            new_ids (Union[List[torch.IntTensor], torch.IntTensor]): For each allocation group, the IDs
                to add to the KV-cache. If there is only one allocation group, a single tensor can be
                provided. Otherwise, a list of tensors should be provided. The tensors do not need
                to have the same shape.
        """
        if isinstance(new_ids, torch.Tensor):
            new_ids = [new_ids]

        if len(new_ids) != self._num_allocation_groups[cache_group]:
            raise ValueError(
                f"Only {len(new_ids)} allocation groups provided, expected {self._num_allocation_groups[cache_group]}")

        for group_id, new_group_ids in enumerate(new_ids):
            new_blocks = new_group_ids.numel()

            if new_blocks == 0:
                # If we have multiple groups, it's possible to have an empty group.
                continue

            shadow_alloc_group = self._kv_cache_ids_shadow[cache_group][group_id]
            alloc_group = self._kv_cache_ids[cache_group][group_id]
            cur_blocks = self._blocks_per_allocation_group[cache_group][group_id]

            shadow_alloc_group[cur_blocks:cur_blocks + new_blocks].copy_(new_group_ids)
            alloc_group[cur_blocks:cur_blocks + new_blocks].copy_(shadow_alloc_group[cur_blocks:cur_blocks +
                                                                                     new_blocks],
                                                                  non_blocking=True)

            self._blocks_per_allocation_group[cache_group][group_id] += new_blocks

    def free_kv_cache(self, free_ids: Union[List[torch.IntTensor], torch.IntTensor], cache_group: int = 0) -> None:
        """
        Free blocks from the KV-cache for the sequence.

        Arguments:
            free_ids (Union[List[torch.IntTensor], torch.IntTensor]): The ids of blocks to free
                from the KV-cache. If there is only one allocation group, a single tensor can be
                provided. Otherwise, a list of tensors should be provided. The tensors do not need
                to have the same shape.
        """
        raise NotImplementedError("Partial KV-cache freeing is not yet supported.")

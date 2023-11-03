# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from typing import Any, Dict, Optional, Tuple

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import RaggedUtilsBuilder
from deepspeed.utils.logging import logger

from .blocked_allocator import BlockedAllocator
from .kv_cache import BlockedKVCache
from .manager_configs import DSStateManagerConfig, KVCacheConfig
from .sequence_descriptor import DSSequenceDescriptor


class DSStateManager:
    """
    Base abstract class for managing blocked KV caches. Will probably have a single
    implementation for now.
    """

    _config: DSStateManagerConfig
    """
    Config for state management. See DSStateManagerConfig for more details. The arguments here
    should come from the engine config.
    """

    _kv_config: KVCacheConfig
    """
    Config for the KV cache. See KVCacheConfig for more details. These arguments should derive
    from the model implementation.
    """

    _kv_cache: BlockedKVCache
    """
    Persistent KV cache store.
    """

    # Container for tracking all sequences in the system.
    _seqs: Dict[int, DSSequenceDescriptor]
    """
    Container for tracking all sequences in the system.

    TODO(cmikeh2): Evaluate if this has any performance implications.
    """

    # Allocator for tracking sequences.
    _tracking_allocator: BlockedAllocator
    _all_block_ids: torch.Tensor
    _all_block_ids_shadow: torch.Tensor

    # TODO(cmikeh2): This interface both needs to be more flexible and more concrete.
    def __init__(self,
                 config: DSStateManagerConfig,
                 kv_config: KVCacheConfig,
                 base_mp_group: Optional[Any] = None) -> None:
        """
        The key

        Parameters:
            block_size (int): The number of tokens to allocate in each block.
        """
        self._config = config
        self._kv_config = kv_config

        # Load our helpers for host allocation.
        self._ragged_utils = RaggedUtilsBuilder().load()

        # Initialize the allocator for tracking sequences (so this doesn't need to be ad-hoc).
        self._tracking_allocator = BlockedAllocator(self._config.max_tracked_sequences)

        # Storage to back tracking the KV cache allocation.
        ids_shape = (
            self._config.max_tracked_sequences,
            self._kv_config.num_allocation_groups,
            self._kv_config.max_blocks_per_allocation_group,
        )
        self._all_block_ids = torch.zeros(ids_shape, dtype=torch.int32, device=get_accelerator().current_device())
        self._all_block_ids_shadow = self._ragged_utils.allocate_fast_host_buffer(self._all_block_ids)

        # Initialize the sequence container.
        self._seqs = {}

        # Finally initialize the KV cache.
        self._kv_cache = BlockedKVCache(self._kv_config,
                                        self._config.memory_config,
                                        mp_group=base_mp_group,
                                        offload=self._config.offload)

    def get_cache(self, cache_id: int) -> torch.Tensor:
        """
        Return the Tensor associated with the given cache id.
        """
        return self._kv_cache.get_cache(cache_id)

    def query(self, uid: Optional[int] = None) -> Tuple[int, int, Optional[int]]:
        """
        Query the state of the KV cache for occupancy.

        Parameters:
            seq_id (Optional[int]): The sequence id to query. If None, the last
                return value will be None.

        Returns:
            Tuple[int, int, Optional[Tuple[int, int]]: A tuple of the block size, the number of
                free blocks, and the number of cached tokens for the given sequence.
        """
        if uid is not None:
            cached_toks = self._seqs[uid].cached_tokens
            free_toks = cached_toks % self._block_size
            return (self._block_size, self._kv_cache.free_blocks, free_toks)
        else:
            return (self._block_size, self._kv_cache.free_blocks, None)

    def flush_sequence(self, uid: int) -> None:
        """
        Free all resources associated with the given sequence id.
        """
        if uid not in self._seqs:
            logger.warning(f"Attempting to flush sequence {uid} which does not exist.")
            return

        seq = self._seqs[uid]
        self._kv_cache.free(seq.all_block_ids)
        self._tracking_allocator.free(seq.tracking_id)
        del self._seqs[uid]

    def get_sequence(self, uid: int) -> Optional[DSSequenceDescriptor]:
        """
        Get the sequence descriptor for the given sequence id. If the sequence does not exist,
        then None is returned.
        """
        if uid not in self._seqs:
            return None

        return self._seqs[uid]

    def get_or_create_sequence(self, uid: int) -> DSSequenceDescriptor:
        """
        Get the existing sequence descriptor for a given uid or initialize one if
        it does not exist. NOTE: This will always return a valid sequence descriptor
        if one may be allocated and should not be used from APIs that are attempting
        to test the schedulability of a hypothetical batch.
        """
        if uid in self._seqs:
            return self._seqs[uid]
        else:
            return self._create_sequence(uid)

    def _create_sequence(self, uid: int) -> DSSequenceDescriptor:
        """
        Create a new sequence descriptor for the given sequence id.
        """
        if uid in self._seqs:
            raise ValueError(f"Sequence {uid} already exists.")

        try:
            tracking_slot = self._tracking_allocator.allocate(1).item()
        except ValueError:
            raise RuntimeError(
                f"Unable to create tracking slot for sequence {uid} since the metadata buffers are full.")

        seq_block_ids = self._all_block_ids[tracking_slot]
        seq_block_ids_shadow = self._all_block_ids_shadow[tracking_slot]
        self._seqs[uid] = DSSequenceDescriptor(tracking_slot,
                                               seq_block_ids,
                                               seq_block_ids_shadow,
                                               max_context=self._config.max_context)
        logger.debug(f"Created sequence {uid} with tracking slot {tracking_slot}.")
        return self._seqs[uid]

    @property
    def tracked_sequences(self) -> Dict[int, DSSequenceDescriptor]:
        """
        Return the tracked sequences.
        """
        return self._seqs

    @property
    def n_tracked_sequences(self) -> int:
        """
        Return the number of sequences currently tracked.
        """
        return len(self._seqs)

    @property
    def kv_block_size(self) -> int:
        """
        Return the block size of the KV cache.
        """
        return self._kv_config.block_size

    @property
    def free_blocks(self) -> int:
        """
        Return the number of free blocks in the KV cache.
        """
        return self._kv_cache.free_blocks

    def allocate_blocks(self, n_blocks: int) -> torch.Tensor:
        return self._kv_cache.reserve(n_blocks)

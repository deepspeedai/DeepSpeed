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

    _kv_configs: Tuple[KVCacheConfig]
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
    _all_block_ids: Tuple[torch.Tensor, ...]
    _all_block_ids_shadow: Tuple[torch.Tensor, ...]

    def __init__(self,
                 config: DSStateManagerConfig,
                 kv_configs: Tuple[KVCacheConfig, ...],
                 base_mp_group: Optional[Any] = None) -> None:
        """
        The key

        Parameters:
            block_size (int): The number of tokens to allocate in each block.
        """
        self._config = config
        self._kv_configs = kv_configs

        # Load our helpers for host allocation.
        self._ragged_utils = RaggedUtilsBuilder().load()

        # Initialize the allocator for tracking sequences (so this doesn't need to be ad-hoc).
        self._tracking_allocator = BlockedAllocator(self._config.max_tracked_sequences)

        all_block_ids = []
        all_block_ids_shadow = []

        for cache_config in self._kv_configs:
            # Storage to back tracking the KV cache allocation.
            ids_shape = (
                self._config.max_tracked_sequences,
                cache_config.num_allocation_groups,
                cache_config.max_blocks_per_allocation_group,
            )

            all_block_ids.append(torch.zeros(ids_shape, dtype=torch.int32, device=get_accelerator().current_device()))
            all_block_ids_shadow.append(self._ragged_utils.allocate_fast_host_buffer(all_block_ids[-1]))

        self._all_block_ids = tuple(all_block_ids)
        self._all_block_ids_shadow = tuple(all_block_ids_shadow)

        # Initialize the sequence container.
        self._seqs = {}

        # Finally initialize the KV cache.
        self._kv_cache = BlockedKVCache(self._kv_configs,
                                        self._config.memory_config,
                                        mp_group=base_mp_group,
                                        offload=self._config.offload)

    def get_cache(self, cache_id: int, cache_group: int = 0) -> torch.Tensor:
        """
        Return the Tensor associated with the given cache id in the specified cache group.

        Arguments:
            cache_group (str): The KV cache group.
            cache_id (int): The cache id within that group.
        """
        return self._kv_cache.get_cache(cache_id, cache_group=cache_group)

    def flush_sequence(self, uid: int) -> None:
        """
        Free all resources associated with the given sequence id.
        """
        if uid not in self._seqs:
            logger.warning(f"Attempting to flush sequence {uid} which does not exist.")
            return

        seq = self._seqs[uid]
        for i in range(self.n_kv_cache_groups):
            self._kv_cache.free(seq.all_block_ids(cache_group=i), cache_group=i)

        self._tracking_allocator.free(seq.tracking_id)
        del self._seqs[uid]

    def get_sequence(self, uid: int) -> Optional[DSSequenceDescriptor]:
        """
        Get the sequence descriptor for the given sequence id. If the sequence does not exist,
        then None is returned.
        """
        return self._seqs.get(uid, None)

    def get_or_create_sequence(self, uid: int) -> DSSequenceDescriptor:
        """
        Get the existing sequence descriptor for a given uid or initialize one if
        it does not exist. NOTE: This will always return a valid sequence descriptor
        if one may be allocated and should not be used from APIs that are attempting
        to test the schedulability of a hypothetical batch.
        """
        seq = self.get_sequence(uid)
        if seq is not None:
            return seq
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

        seq_block_ids = tuple(all_block_ids[tracking_slot] for all_block_ids in self._all_block_ids)
        seq_block_ids_shadow = tuple(all_block_ids_shadow[tracking_slot]
                                     for all_block_ids_shadow in self._all_block_ids_shadow)

        self._seqs[uid] = DSSequenceDescriptor(tracking_slot,
                                               seq_block_ids,
                                               seq_block_ids_shadow,
                                               max_context=self._config.max_context)
        # TODO(cmikeh2): Debug call here might be unnecessary and is potentially on critical path.
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
    def n_kv_cache_groups(self) -> int:
        """
        Return the number of KV caches.
        """
        return self._kv_cache.num_caches

    @property
    def free_blocks(self) -> torch.Tensor:
        """
        Return the number of free blocks in the KV cache.
        """
        return self._kv_cache.free_blocks

    def allocate_blocks(self, n_blocks: int, cache_group: int = 0) -> torch.Tensor:
        return self._kv_cache.reserve(n_blocks, cache_group=cache_group)

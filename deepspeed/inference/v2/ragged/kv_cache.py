# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import operator
from functools import reduce
from typing import Any, Iterable, Optional, Tuple

import torch

import deepspeed.comm as dist
from deepspeed.comm.reduce_op import ReduceOp

from deepspeed.accelerator import get_accelerator
from ..inference_utils import elem_size
from ..logging import inference_logger
from .blocked_allocator import BlockedAllocator
from .manager_configs import AllocationMode, KVCacheConfig, MemoryConfig


def split_kv(kv_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split a KV cache instance into its key and value components.

    Parameters:
        kv_cache (torch.Tensor): The KV-cache to split. This should be a 5D tensor with the
            following shape: [num_blocks, block_size, 2, num_heads, head_size]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The key and value components of the KV-cache. Both
            tensors will have the shape [num_blocks, block_size, num_heads, head_size].
    """
    if kv_cache.ndim != 5:
        raise ValueError(f"KV-cache must have 5 dimensions, got {kv_cache.ndim}.")

    return kv_cache[:, :, 0, :, :], kv_cache[:, :, 1, :, :]


class BlockedKVCache:

    _caches: Tuple[torch.Tensor, ...]
    """
    Backing storage for all KV caches. This is a 6D tensor with the following shape:
        (num_caches, num_blocks, block_size, 2, num_heads, head_size)
    """

    _allocators: Tuple[BlockedAllocator, ...]
    """
    Block allocator for tracking cache usage. This manages the GPU cache.
    """

    _configs: Tuple[KVCacheConfig, ...]
    """
    Configuration of the KV cache(s). See ``KVCacheConfig`` for more details. This enables the support
    for different types/shapes of KV-caches (i.e. the alternating local and global attention in
    GPT-Neo).
    """

    def __init__(self,
                 configs: Tuple[KVCacheConfig, ...],
                 memory_config: MemoryConfig,
                 mp_group: Optional[Any] = None,
                 offload: bool = False) -> None:
        """
        Create a container that will maintain the storage and allocations for a set of
        blocked KV-caches.

        Parameters:
            config (KVCacheConfig): The configuration of the KV-cache.
            slack (int): The amount of slack space to reserve in GPU memory for the cache.
            enable_offload (bool): Whether to enable offloading of the cache to the host.
            blocks (int): The number of blocks to pre-allocate for the cache. If this is set,
                slack will be ignored.
        """
        self._configs = configs
        self._memory_config = memory_config
        self._enable_offload = offload

        if self._enable_offload:
            raise NotImplementedError("Offloading of KV-caches is not yet supported.")

        if AllocationMode(self._memory_config.mode) is AllocationMode.RESERVE:
            # TODO(cmikeh2): Change the weighting based on the type of the KV-cache

            total_per_block_footprint = 0
            for config in self._configs:
                per_block_footprint = reduce(operator.mul, config.cache_shape, config.block_size)
                per_block_footprint *= 2  # for key and value
                total_per_block_footprint += per_block_footprint * elem_size(config.cache_dtype)

            # Perform a dummy nccl call before calculating available memory, on some systems (H100) we've observed higher memory allocations from NCCL
            if dist.get_world_size(group=mp_group) > 1:
                dummy_tensor = torch.tensor(0, dtype=torch.int32, device=get_accelerator().current_device())
                dist.all_reduce(dummy_tensor, op=ReduceOp.MIN, group=mp_group)

            get_accelerator().empty_cache()
            available_kv_memory = get_accelerator().available_memory() - self._memory_config.size
            total_memory = get_accelerator().total_memory()

            inference_logger().debug(
                f"Memory usage before KV-cache allocation: total_memory={total_memory}, available_kv_memory={available_kv_memory}, total_per_block_footprint={total_per_block_footprint}"
            )

            if available_kv_memory < total_per_block_footprint:
                raise ValueError(
                    f"Insufficient memory to allocate KV-caches. Required: {total_per_block_footprint}, Available: {available_kv_memory}"
                )

            num_blocks = available_kv_memory // total_per_block_footprint

            # In a multi-process setting, we need to ensure that all processes have the same
            # KV cache capacity to ensure scheduling guarantees are equivalent on all ranks.
            if dist.get_world_size(group=mp_group) > 1:
                reduce_tensor = torch.tensor(num_blocks, dtype=torch.int32, device=get_accelerator().current_device())
                dist.all_reduce(reduce_tensor, op=ReduceOp.MIN, group=mp_group)
                num_blocks = reduce_tensor.item()

                # This is ugly but don't want the fragmentation of the 8 byte Tensor maybe
                # hanging around.
                del reduce_tensor
                get_accelerator().empty_cache()
        else:  # AllocationMode.ALLOCATE
            num_blocks = self._memory_config.size

        caches = []
        allocators = []

        for cache_group_id, config in enumerate(self._configs):
            num_caches = config.cache_shape[0]
            num_heads = config.cache_shape[1]
            head_size = config.cache_shape[2]

            alloc_shape = (num_caches, num_blocks, config.block_size, 2, num_heads, head_size)
            inference_logger().info(
                f"Allocating KV-cache {cache_group_id} with shape: {alloc_shape} consisting of {num_blocks} blocks.")
            caches.append(torch.empty(alloc_shape, dtype=config.cache_dtype,
                                      device=get_accelerator().current_device()))
            allocators.append(BlockedAllocator(num_blocks))

        self._caches = tuple(caches)
        self._allocators = tuple(allocators)

    def reserve(self, num_blocks: int, cache_group: int = 0) -> torch.Tensor:
        """
        Reserve a number of blocks from the cache. This will return a 1D tensor of
        block_ids that have been marked as reserved.

        Parameters:
            num_blocks (int): The number of blocks to reserve.
            cache_group (int): The cache group to reserve from. Default is 0.
        """
        return self._allocators[cache_group].allocate(num_blocks)

    def free(self, blocks: Iterable[int], cache_group: int = 0) -> None:
        """
        Free a set of blocks from the cache. This will mark the blocks as free in the
        allocator.

        Parameters:
            blocks (Iterable[int]): The blocks to free.
            cache_group (int): The cache group to free from. Default is 0.
        """
        self._allocators[cache_group].free(blocks)

    def offload(self, blocks: Iterable[int], cache_group: int = 0) -> torch.Tensor:
        """
        Offload KV-cache blocks from accelerator memory to the host.

        Parameters:
            blocks (Iterable[int]): The blocks to offload.
            cache_group (int): The cache group to offload from. Default is 0.
        """
        raise NotImplementedError("Offloading is not yet supported.")

    def restore(self, blocks: Iterable[int], cache_group: int = 0) -> torch.Tensor:
        """
        Restore KV-cache blocks from the host to accelerator memory.

        Parameters:
            blocks (Iterable[int]): The blocks to restore.
            cache_group (int): The cache group to restore to. Default is 0.
        """
        raise NotImplementedError("Offloading is not yet supported.")

    def get_cache(self, cache_id: int, cache_group: int = 0) -> torch.Tensor:
        """
        Get the tensor associated with the given cache ID.

        Parameters:
            cache_id (int): The ID of the cache tensor to get.
            cache_group (int): The cache group to get from. Default is 0.
        """
        return self._caches[cache_group][cache_id]

    @property
    def free_blocks(self) -> torch.Tensor:
        """
        Return the number of free blocks in each cache
        """
        return [allocator.free_blocks for allocator in self._allocators]

    @property
    def num_caches(self) -> int:
        """
        Return the number of caches
        """
        return len(self._caches)

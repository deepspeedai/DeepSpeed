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

    _caches: torch.Tensor
    """
    Backing storage for all KV caches. This is a 6D tensor with the following shape:
        (num_caches, num_blocks, block_size, 2, num_heads, head_size)
    """

    _allocator: BlockedAllocator
    """
    Block allocator for tracking cache usage. This manages the GPU cache.
    """

    _config: KVCacheConfig
    """
    Configuration of the KV cache. See ``KVCacheConfig`` for more details.
    """

    def __init__(self,
                 config: KVCacheConfig,
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
        self._config = config
        self._memory_config = memory_config
        self._enable_offload = offload

        if self._enable_offload:
            raise NotImplementedError("Offloading of KV-caches is not yet supported.")

        if AllocationMode(self._memory_config.mode) is AllocationMode.RESERVE:
            per_block_footprint = reduce(operator.mul, self._config.cache_shape, self._config.block_size)
            per_block_footprint *= 2  # for key and value
            per_block_footprint *= elem_size(self._config.cache_dtype)

            # Perform a dummy nccl call before calculating available memory, on some systems (H100) we've observed higher memory allocations from NCCL
            if dist.get_world_size(group=mp_group) > 1:
                dummy_tensor = torch.tensor(0, dtype=torch.int32, device=get_accelerator().current_device())
                dist.all_reduce(dummy_tensor, op=ReduceOp.MIN, group=mp_group)

            get_accelerator().empty_cache()
            available_kv_memory = get_accelerator().available_memory() - self._memory_config.size
            total_memory = get_accelerator().total_memory()

            inference_logger().debug(
                f"Memory usage before KV-cache allocation: total_memory={total_memory}, available_kv_memory={available_kv_memory}, per_block_footprint={per_block_footprint}"
            )

            if available_kv_memory < per_block_footprint:
                raise ValueError(
                    f"Insufficient memory to allocate KV-caches. Required: {per_block_footprint}, Available: {available_kv_memory}"
                )

            num_blocks = available_kv_memory // per_block_footprint

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

        num_caches = self._config.cache_shape[0]
        num_heads = self._config.cache_shape[1]
        head_size = self._config.cache_shape[2]

        alloc_shape = (num_caches, num_blocks, self._config.block_size, 2, num_heads, head_size)
        inference_logger().info(f"Allocating KV-cache with shape: {alloc_shape} consisting of {num_blocks} blocks.")
        self._caches = torch.empty(alloc_shape,
                                   dtype=self._config.cache_dtype,
                                   device=get_accelerator().current_device())
        self._allocator = BlockedAllocator(num_blocks)

    def reserve(self, num_blocks: int) -> torch.Tensor:
        """
        Reserve a number of blocks from the cache. This will return a 1D tensor of
        block_ids that have been marked as reserved.
        """
        return self._allocator.allocate(num_blocks)

    def free(self, blocks: Iterable[int]) -> None:
        """
        Free a set of blocks from the cache. This will mark the blocks as free in the
        allocator.
        """
        self._allocator.free(blocks)

    def offload(self, blocks: Iterable[int]) -> torch.Tensor:
        """
        Offload KV-cache blocks from accelerator memory to the host.
        """
        raise NotImplementedError("Offloading is not yet supported.")

    def restore(self, blocks: Iterable[int]) -> torch.Tensor:
        """
        Restore KV-cache blocks from the host to accelerator memory.
        """
        raise NotImplementedError("Offloading is not yet supported.")

    def get_cache(self, cache_id: int) -> torch.Tensor:
        """
        Get the tensor associated with the given cache ID.
        """
        return self._caches[cache_id]

    @property
    def free_blocks(self):
        return self._allocator.free_blocks

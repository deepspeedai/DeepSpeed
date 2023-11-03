# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Iterable, Union

import torch


class BlockedAllocator:
    """
    Allocator class for managing which blocks are free/used in the
    blocked KV-cache. This is a simple allocator that uses a linked list
    to keep track of which blocks are free/used. The cost of allocation/deallocation
    is O(blocks), where blocks is the number of blocks to allocate/deallocate.

    TODO(cmikeh2): Evaluate performance of this allocator and migrate
    to C++ if necessary.
    """
    # Number of blocks in the KV-cache(s).
    _num_blocks: int

    # Array of blocks, where each element is the next block in the linked list.
    _blocks: torch.Tensor

    # Index of the head of the linked list.
    _head: int

    # Number of free blocks in the KV-cache.
    _free_blocks: int

    def __init__(self, num_blocks: int) -> None:
        """
        Initialize an allocator with `num_blocks` blocks. This requires at least
        `num_blocks` * 4 bytes of host memory.

        Parameters:
            num_blocks (int): The number of blocks to allocate.
        """

        if num_blocks < 1:
            raise ValueError(f'Blocked KV-cache must have at least 1 block, provided {num_blocks}')

        self._num_blocks = num_blocks
        self._blocks = torch.arange(1, num_blocks + 1, dtype=torch.int32, device='cpu', pin_memory=True)
        self._head = 0
        self._free_blocks = num_blocks

    def allocate(self, num_blocks: int) -> torch.Tensor:
        """
        Allocate a list of blocks from the associated KV-caches. This will
        return `num_blocks` blocks from the KV-cache if they are available,
        or raise an exception if there are not enough free blocks.

        Parameters:
            num_blocks (int): The number of blocks to allocate.

        Returns:
            List[int]: The list of blocks allocated.
        """
        if num_blocks > self._free_blocks:
            raise ValueError(f'Not enough free blocks in the KV-cache to allocate {num_blocks} blocks')

        allocated_blocks = torch.zeros(num_blocks, dtype=torch.int32)
        for i in range(num_blocks):
            allocated_blocks[i] = self._head
            self._head = self._blocks[self._head].item()
            self._blocks[allocated_blocks[i]] = -1  # Mark as used
            self._free_blocks -= 1

        return allocated_blocks

    def free(self, blocks: Union[Iterable[int], int]) -> None:
        """
        Return a list of blocks to the free pool. If a single invalid block is provided (i.e.,
        one that is out of range of the allocator or is already free), then an exception is raised
        and no blocks are freed.

        Parameters:
            blocks (Union[Iterable[int], int]): The list of blocks to free. If only one block
                is to be freed, this can be alone as an integer.
        """
        if isinstance(blocks, int):
            blocks = [blocks]

        for block in blocks:
            # Parse all blocks for validity before mutating the list.
            if block < 0 or block >= self._num_blocks:
                raise ValueError(f'Invalid block {block} provided to free')

            if self._blocks[block] != -1:
                raise ValueError(f'Block {block} is already free')

        for block in blocks:
            self._blocks[block] = self._head
            self._head = block
            self._free_blocks += 1

    @property
    def free_blocks(self) -> int:
        """
        Return the number of free blocks in the KV-cache.
        """
        return self._free_blocks

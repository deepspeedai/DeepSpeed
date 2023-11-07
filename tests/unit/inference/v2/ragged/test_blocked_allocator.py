# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import random
from typing import List

import pytest
import torch

from deepspeed.inference.v2.ragged.blocked_allocator import BlockedAllocator


@pytest.mark.inference_v2
@pytest.mark.parametrize('bad_size', [0, -1])
def test_bad_initialization(bad_size: int) -> None:
    with pytest.raises(ValueError):
        BlockedAllocator(bad_size)


@pytest.mark.inference_v2
def test_allocation() -> None:

    allocator = BlockedAllocator(16)

    a1 = allocator.allocate(4)
    assert a1.numel() == 4
    assert allocator.free_blocks == 12

    a2_allocs = []
    for i in range(3):
        a2_allocs.append(allocator.allocate(2))
        assert allocator.free_blocks == 12 - (i + 1) * 2

    a3 = allocator.allocate(6)
    assert a3.numel() == 6

    assert allocator.free_blocks == 0

    # Test that we can't allocate more blocks than we have.
    with pytest.raises(ValueError):
        allocator.allocate(1)

    all_vals = torch.cat([a1, *a2_allocs, a3], dim=0)
    unique_vals = torch.unique(all_vals, sorted=False)
    assert unique_vals.numel() == all_vals.numel()


@pytest.mark.inference_v2
def test_too_large_allocation():
    allocator = BlockedAllocator(16)

    with pytest.raises(ValueError):
        allocator.allocate(17)


@pytest.mark.inference_v2
def test_deallocation() -> None:
    allocator = BlockedAllocator(16)

    # Allocate
    all_blocks = allocator.allocate(16)
    assert allocator.free_blocks == 0

    # Deallocate all blocks
    allocator.free(all_blocks)
    assert allocator.free_blocks == 16

    # Get all the blocks again
    all_blocks = allocator.allocate(16)

    # Deallocate in chunks
    c1 = all_blocks[:4]
    c2 = all_blocks[4:8]

    allocator.free(c1)
    assert allocator.free_blocks == 4

    allocator.free(c2)
    assert allocator.free_blocks == 8

    with pytest.raises(ValueError):
        allocator.free(c1)

    with pytest.raises(ValueError):
        allocator.free(c2)


@pytest.mark.inference_v2
@pytest.mark.parametrize('index', [-1, 2])
def test_invalid_dealloc_indices(index: int):
    allocator = BlockedAllocator(1)

    with pytest.raises(ValueError):
        allocator.free(torch.tensor([index]))


@pytest.mark.inference_v2
@pytest.mark.parametrize('index', [-1, 2])
def test_invalid_alloc_indices(index: int):
    allocator = BlockedAllocator(1)
    allocator.allocate(1)

    to_free = [0, index]

    with pytest.raises(ValueError):
        allocator.free(torch.tensor(to_free))

    # Block 0 should not be freed if passed with an invalid index.
    assert allocator.free_blocks == 0

    allocator.free(torch.tensor([0]))
    assert allocator.free_blocks == 1


@pytest.mark.inference_v2
@pytest.mark.parametrize('test_iters', [8192])
def test_long_running_allocation(test_iters: int) -> None:
    """
    Evaluate the stability of the allocator over a longer sequence of allocations/deallocations.
    """
    TOTAL_BLOCKS = 128

    allocator = BlockedAllocator(TOTAL_BLOCKS)

    def validate_uniqueness(all_blocks: List[torch.Tensor]) -> None:
        all_vals = torch.cat(all_blocks, dim=0)
        assert all_vals.numel() <= TOTAL_BLOCKS

        unique_vals = torch.unique(all_vals, sorted=False)
        assert unique_vals.numel() == all_vals.numel()

    all_allocs: List[torch.Tensor] = []
    num_allocs = 0
    num_frees = 0
    num_blocks_allocated = 0
    num_blocks_freed = 0

    for _ in range(test_iters):
        decision = random.randint(0, 1)

        if decision == 0:
            blocks_to_allocate = random.randint(1, 24)
            if blocks_to_allocate > allocator.free_blocks:
                with pytest.raises(ValueError):
                    allocator.allocate(blocks_to_allocate)
            else:
                all_allocs.append(allocator.allocate(blocks_to_allocate))
                num_allocs += 1
                num_blocks_allocated += blocks_to_allocate
        else:
            if len(all_allocs) > 0:
                idx = random.randint(0, len(all_allocs) - 1)
                allocator.free(all_allocs[idx])

                num_frees += 1
                num_blocks_freed += all_allocs[idx].numel()

                del all_allocs[idx]

        if len(all_allocs) > 0:
            validate_uniqueness(all_allocs)

    assert num_allocs == num_frees + len(all_allocs)
    assert num_blocks_allocated == num_blocks_freed + (TOTAL_BLOCKS - allocator.free_blocks)

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

import torch
import deepspeed.comm as dist

from deepspeed.runtime.utils import partition_uniform
from deepspeed.runtime.utils import partition_balanced
from deepspeed.runtime.utils import prefix_sum_inc
from deepspeed.runtime.utils import PartitionedTensor
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest


class TestPartitionedTensor(DistributedTest):
    world_size = 4

    def test(self):
        world = dist.get_world_size()

        group = dist.new_group(ranks=list(range(world)))

        rows = world * 4
        cols = 3

        full = torch.rand(rows, cols).to(get_accelerator().device_name())
        dist.broadcast(full, src=0, group=group)
        part = PartitionedTensor(full, group=group)

        assert len(part.local_size()) == 1
        assert part.local_size()[0] * world == full.numel()

        reconstructed = part.full()
        assert torch.equal(full, reconstructed)


class TestPartitionedTensorUnEven(DistributedTest):
    world_size = 4

    def test(self):
        world = dist.get_world_size()

        group = dist.new_group(ranks=list(range(world)))

        rows = world * 4 - 1
        cols = world + 1

        full = torch.rand(rows, cols).to(get_accelerator().device_name())
        dist.broadcast(full, src=0, group=group)
        part = PartitionedTensor(full, group=group)

        assert len(part.local_size()) == 1

        reconstructed = part.full()
        assert torch.equal(full, reconstructed)


class TestPartitionedTensorMeta(DistributedTest):
    world_size = 4

    def test(self):
        world = dist.get_world_size()

        group = dist.new_group(ranks=list(range(world)))

        rows = world * 7
        cols = 3

        full = torch.rand(rows, cols).to(get_accelerator().device_name())
        dist.broadcast(full, src=0, group=group)
        part = PartitionedTensor(full, group=group)

        my_meta = PartitionedTensor.from_meta(part.to_meta(), part.local_data, group)
        assert torch.equal(full, my_meta.full())


def assert_valid_partition(weights, parts, P):
    N = len(weights)
    assert len(parts) == P + 1
    assert parts[0] == 0
    assert parts[P] == N
    for idx in range(P):
        assert parts[idx] <= parts[idx + 1]


def get_partition_weights(weights, parts):
    """ Return the amount of weight in each partition. """
    costs = [0] * (len(parts) - 1)
    P = len(parts) - 1
    for p in range(P):
        start = parts[p]
        stop = parts[p + 1]
        costs[p] = sum(weights[start:stop])
    return costs


def test_prefix_sum():
    x = [3, 4, 5]
    psum = prefix_sum_inc(x)
    assert psum == [3, 7, 12]


def test_valid_partition():
    N = 10
    P = 1
    weights = [1] * N
    parts = partition_balanced(weights, P)
    assert_valid_partition(weights, parts, P)


def test_short_partition_uniform():
    N = 2
    P = 4
    weights = [1] * N
    parts = partition_uniform(len(weights), P)
    assert_valid_partition(weights, parts, P)


def test_short_partition():
    N = 2
    P = 4
    weights = [1] * N
    parts = partition_balanced(weights, P)
    assert_valid_partition(weights, parts, P)


def test_easy_balance_uniform():
    weights = [1] * 8
    P = 4
    parts = partition_uniform(len(weights), P)
    assert_valid_partition(weights, parts, P)
    costs = get_partition_weights(weights, parts)
    assert all(c == 2 for c in costs)


def test_easy_balance_balanced():
    weights = [1] * 8
    P = 4
    parts = partition_balanced(weights, P)
    assert_valid_partition(weights, parts, P)
    costs = get_partition_weights(weights, parts)
    assert all(c == 2 for c in costs), costs


def test_int_balanced():
    weights = [0, 1, 2, 3, 3, 3]
    P = 4
    parts = partition_balanced(weights, P)
    assert parts == [0, 3, 4, 5, 6]

    assert_valid_partition(weights, parts, P)
    costs = get_partition_weights(weights, parts)
    assert all(c == 3 for c in costs)


def test_float_balanced():
    weights = [0., 1.1, 1.9, 3., 3., 3.]
    P = 4
    parts = partition_balanced(weights, P)
    assert_valid_partition(weights, parts, P)
    assert parts == [0, 3, 4, 5, 6]


@pytest.mark.skip(reason="Variance-minimizing partitioning returns different result.")
def test_float_lastheavy():
    weights = [0., 1.1, 1.9, 3., 30.]
    P = 2
    parts = partition_balanced(weights, P)
    assert_valid_partition(weights, parts, P)
    assert parts == [0, 4, 5]


def test_float_midheavy():
    weights = [0., 1.1, 30, 3.]
    P = 3
    parts = partition_balanced(weights, P)
    assert_valid_partition(weights, parts, P)
    assert parts == [0, 2, 3, 4]


def test_balance_bert():
    # Parameters per layer for a transformer model with 24 transformers and hidden dim 1024
    weights = [
        52559872, 12596224, 12596224, 12596224, 12596224, 12596224, 12596224, 12596224, 12596224, 12596224, 12596224,
        12596224, 12596224, 12596224, 12596224, 12596224, 12596224, 12596224, 12596224, 12596224, 12596224, 12596224,
        12596224, 12596224, 12596224, 0, 52559872
    ]
    P = 8
    parts = partition_balanced(weights, P)
    assert_valid_partition(weights, parts, P)

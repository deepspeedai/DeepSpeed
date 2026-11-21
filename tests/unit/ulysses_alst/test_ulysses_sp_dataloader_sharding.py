# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPDataLoaderAdapter

SP_WORLD_SIZE = 4


class MapDataset(Dataset):

    def __len__(self):
        return 16

    def __getitem__(self, idx):
        return torch.tensor([idx])


class StreamDataset(IterableDataset):

    def __iter__(self):
        return iter([torch.tensor([i]) for i in range(16)])


def build_adapter(dl, sp_world_size=SP_WORLD_SIZE):
    # __init__ only wraps the DataLoader in an iterator, so no process group is
    # needed to exercise the sharding check.
    return UlyssesSPDataLoaderAdapter(dl, sp_rank=0, sp_group=None, sp_world_size=sp_world_size, device='cpu')


def test_unsharded_dataloader_warns(caplog):
    """A plain DataLoader hands every SP rank the same samples.

    Without this warning the duplication is silent: each batch is processed
    sp_world_size times and the rest of the epoch is never read.
    """
    dl = DataLoader(MapDataset(), batch_size=1)

    build_adapter(dl)

    assert 'does not appear to be sharded' in caplog.text
    assert f'All {SP_WORLD_SIZE} SP ranks' in caplog.text


def test_distributed_sampler_does_not_warn(caplog):
    dataset = MapDataset()
    sampler = DistributedSampler(dataset, num_replicas=SP_WORLD_SIZE, rank=0)
    dl = DataLoader(dataset, batch_size=1, sampler=sampler)

    build_adapter(dl)

    assert 'does not appear to be sharded' not in caplog.text


def test_single_sp_rank_does_not_warn(caplog):
    """With sp_world_size == 1 a rank only ever replays its own batch."""
    dl = DataLoader(MapDataset(), batch_size=1)

    build_adapter(dl, sp_world_size=1)

    assert 'does not appear to be sharded' not in caplog.text


def test_iterable_dataset_does_not_warn(caplog):
    """An IterableDataset shards inside the dataset, which cannot be inspected."""
    dl = DataLoader(StreamDataset(), batch_size=1)

    build_adapter(dl)

    assert 'does not appear to be sharded' not in caplog.text


@pytest.mark.parametrize('num_replicas', [2, SP_WORLD_SIZE, 8])
def test_any_sharded_sampler_does_not_warn(caplog, num_replicas):
    dataset = MapDataset()
    sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=0)
    dl = DataLoader(dataset, batch_size=1, sampler=sampler)

    build_adapter(dl)

    assert 'does not appear to be sharded' not in caplog.text

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.utils import RepeatingLoader
from deepspeed.runtime.dataloader import DeepSpeedDataLoader
from torch.utils.data import DistributedSampler
import torch
import pytest
import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataset


def test_repeating_loader():
    loader = [1, 2, 3]
    loader = RepeatingLoader(loader)

    for idx in range(50):
        assert next(loader) == 1
        assert next(loader) == 2
        assert next(loader) == 3


def _epoch_orders(loader, num_epochs):
    return [[int(batch) for batch in loader] for _ in range(num_epochs)]


def _make_loader(data_sampler=None, world_size=1, rank=0):
    return DeepSpeedDataLoader(torch.arange(8),
                               batch_size=1,
                               pin_memory=False,
                               local_rank=0,
                               tput_timer=None,
                               num_local_io_workers=0,
                               data_sampler=data_sampler,
                               data_parallel_world_size=world_size,
                               data_parallel_rank=rank)


def _reference_order(epoch, world_size=1, rank=0):
    sampler = DistributedSampler(torch.arange(8), num_replicas=world_size, rank=rank)
    sampler.set_epoch(epoch)
    return list(sampler)


def test_dataloader_reshuffles_every_epoch():
    orders = _epoch_orders(_make_loader(), 3)
    assert orders == [_reference_order(epoch) for epoch in range(3)]


def test_dataloader_keeps_epoch_set_by_caller():
    loader = _make_loader()
    assert _epoch_orders(loader, 1) == [_reference_order(0)]
    loader.data_sampler.set_epoch(5)
    assert _epoch_orders(loader, 2) == [_reference_order(5), _reference_order(6)]


def test_dataloader_keeps_epoch_set_to_same_value():
    loader = _make_loader()
    orders = []
    for _ in range(3):
        loader.data_sampler.set_epoch(0)
        orders += _epoch_orders(loader, 1)
    assert orders == [_reference_order(0)] * 3


def test_dataloader_ranks_stay_disjoint_in_every_epoch():
    loaders = [_make_loader(world_size=2, rank=rank) for rank in range(2)]
    for epoch in range(3):
        per_rank = [_epoch_orders(loader, 1)[0] for loader in loaders]
        assert per_rank == [_reference_order(epoch, world_size=2, rank=rank) for rank in range(2)]
        assert sorted(per_rank[0] + per_rank[1]) == list(range(8))


def test_dataloader_leaves_caller_sampler_alone():
    sampler = DistributedSampler(torch.arange(8), num_replicas=1, rank=0)
    orders = _epoch_orders(_make_loader(data_sampler=sampler), 2)
    assert orders == [_reference_order(0), _reference_order(0)]
    assert sampler.epoch == 0


@pytest.mark.parametrize('train_batch_size, drop_last', [(1, True), (4, True), (1, False), (4, False)])
class TestDataLoaderDropLast(DistributedTest):
    world_size = 1

    def test(self, train_batch_size, drop_last):
        config_dict = {"train_batch_size": train_batch_size, "dataloader_drop_last": drop_last, "steps_per_print": 1}
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.AdamW(params=model.parameters())
        # TODO: no way to set DeepSpeedEngine.deepspeed_io params, need to use
        # pin_memory=False for cuda device
        train_dataset = random_dataset(total_samples=50,
                                       hidden_dim=hidden_dim,
                                       device=torch.device('cpu'),
                                       dtype=torch.float32)
        model, _, training_dataloader, _ = deepspeed.initialize(config=config_dict,
                                                                model=model,
                                                                training_data=train_dataset,
                                                                optimizer=optimizer)
        training_dataloader.num_local_io_workers = 0  # We can't do nested mp.pool
        for n, batch in enumerate(training_dataloader):
            x = batch[0].to(get_accelerator().current_device_name())
            y = batch[1].to(get_accelerator().current_device_name())
            loss = model(x, y)
            model.backward(loss)
            model.step()


class TestDataLoaderEpochs(DistributedTest):
    world_size = 1

    def test(self):
        config_dict = {"train_micro_batch_size_per_gpu": 1, "steps_per_print": 1}
        train_dataset = [(torch.tensor([float(i)]), ) for i in range(8)]
        model, _, training_dataloader, _ = deepspeed.initialize(config=config_dict,
                                                                model=SimpleModel(1),
                                                                model_parameters=SimpleModel(1).parameters(),
                                                                training_data=train_dataset)
        training_dataloader.num_local_io_workers = 0  # We can't do nested mp.pool
        orders = [[int(batch[0]) for batch in training_dataloader] for _ in range(3)]
        assert orders == [_reference_order(epoch) for epoch in range(3)]

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.utils import RepeatingLoader
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

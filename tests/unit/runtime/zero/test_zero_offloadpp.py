# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import pytest
import deepspeed.comm as dist
from unit.common import DistributedTest
from unit.simple_model import random_dataloader

import deepspeed

from deepspeed.runtime.zero.offload_config import DeepSpeedZeroOffloadOptimizerConfig

import torch.nn as nn


class NNModel(nn.Module):

    def __init__(self, h_dim=1024, n_layers=2):
        super(NNModel, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(h_dim, h_dim) for i in range(n_layers)])
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        for layer in self.layers:
            x = layer(x)
        return self.cross_entropy_loss(x, y)


def test_zero_partial_offload_config():
    config = DeepSpeedZeroOffloadOptimizerConfig(**{"ratio": 0.3})
    assert config.ratio == 0.3


#Large sweep along hidden dim, num_layers of different sizes
@pytest.mark.parametrize("h_dim", [1024])
@pytest.mark.parametrize("n_layers", [4, 8])
class TestZeroPartialOffloadConfigSweep(DistributedTest):
    world_size = 4

    def test(self, h_dim: int, n_layers: int) -> None:

        config_dict = {
            "train_batch_size": 256,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015,
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 15
            },
            "zero_optimization": {
                "stage": 3,
                "sub_group_size": 8,
                "reduce_bucket_size": 20,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                    "ratio": 0.3
                }
            }
        }

        model = NNModel(h_dim, n_layers)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        data_loader = random_dataloader(model=model, total_samples=20, hidden_dim=h_dim, device=model.device)
        dist.barrier()

        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

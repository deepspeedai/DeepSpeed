# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import pytest
import deepspeed.comm as dist
from unit.common import DistributedTest
from unit.simple_model import random_dataloader

import deepspeed
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


#Large sweep along hidden dim, num_layers of different sizes for qgZeRO.
@pytest.mark.parametrize("h_dim", [1024, 2000])
@pytest.mark.parametrize("n_layers", [8, 20])
class TesthpZeroConfigSweep(DistributedTest):
    world_size = 4

    def test(self, h_dim: int, n_layers: int) -> None:
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "reduce_scatter": True,
                "zero_quantized_gradients": True
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1.
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 1.,
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

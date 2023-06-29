# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import pytest
import deepspeed.comm as dist
from torch.nn import Module

from unit.common import DistributedTest
from unit.simple_model import random_dataloader

import deepspeed

from deepspeed.runtime.zero.config import DeepSpeedZeroConfig

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


def test_zero_hpz_partition_size_config():
    config = DeepSpeedZeroConfig(**{"zero_hpz_partition_size": 4})
    assert config.zero_hpz_partition_size == 4


def _assert_no_secondary_tensor_group(model: Module) -> None:
    for _, param in model.named_parameters():
        assert param.ds_secondary_tensor is None
        assert param.ds_zero_param_process_group is None


def _assert_secondary_tensor_size(model: Module) -> None:
    for _, param in model.named_parameters():
        assert param.ds_secondary_tensor is not None
        assert param.ds_secondary_tensor.size()[0] % param.ds_tensor.size()[0] == 0


#Large sweep along hidden dim, num_layers, and zpg of different sizes
#Assert when zpg=1 that secondary group and tensors are invalid
@pytest.mark.sequential
@pytest.mark.parametrize("h_dim", [1024])
@pytest.mark.parametrize("n_layers", [4, 9])
@pytest.mark.parametrize("zpg", [1, 2, 4])
class TestZeroPPConfigSweep(DistributedTest):
    world_size = 4

    def test(self, h_dim: int, n_layers: int, zpg: int) -> None:
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "zero_hpz_partition_size": zpg,
                "zero_quantized_weights": True,
                "zero_quantized_gradients": True,
                "contiguous_gradients": True,
                "overlap_comm": True,
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
        if zpg == 1:
            _assert_no_secondary_tensor_group(model)

        for n, batch in enumerate(data_loader):
            if n == 0 and zpg != 1:
                _assert_secondary_tensor_size(model)
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

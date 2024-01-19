# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed.comm as dist
import torch

from unit.common import DistributedTest
from unit.simple_model import random_dataloader

import deepspeed
from deepspeed.utils import set_z3_leaf_modules, z3_leaf_module


class MyModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(MyModel, self).__init__()
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim, bias=False),
             torch.nn.Linear(hidden_dim, hidden_dim, bias=False)])
        self.act = torch.nn.ReLU()
        self.cel = torch.nn.CrossEntropyLoss()
        self.counter = 0

    def forward(self, x, y):
        # This fails without setting this module as a leaf module.
        # See the comment in `set_z3_leaf_modules()`.
        x = self.linears[self.counter % len(self.linears)](x)
        x = self.act(x)
        loss = self.cel(x, y)
        self.counter += 1
        return x, loss


def run_model(model, config_dict, hidden_dim, dtype):
    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
    data_loader = random_dataloader(model=model,
                                    total_samples=10,
                                    hidden_dim=hidden_dim,
                                    device=model.device,
                                    dtype=dtype)
    dist.barrier()
    for batch in data_loader:
        loss = model(batch[0], batch[1])
        loss = loss[1]
        model.backward(loss)
        model.step()

    # Needed in ZeRO 3. Not doing so can give memory leak
    model.destroy()


class TestSetZ3LeafModule(DistributedTest):
    # Need multiple gpus to test possible hanging
    world_size = 2
    reuse_dist_env = True

    def test_set_z3_leaf_modules(self):
        hidden_dim = 128

        # `stage3_max_reuse_distance` is set to 0 to cause an error if the module is not set as a leaf module
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 3,
                "stage3_prefetch_bucket_size": hidden_dim**2,
                "stage3_param_persistence_threshold": 0,
                "stage3_max_reuse_distance": 0,
            }
        }

        model = MyModel(hidden_dim)

        assert not z3_leaf_module(model)
        set_z3_leaf_modules(model, [MyModel])
        assert z3_leaf_module(model)

        run_model(model, config_dict, hidden_dim, torch.float16)

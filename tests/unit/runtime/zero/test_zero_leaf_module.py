# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import deepspeed.comm as dist
import torch

from unit.common import DistributedTest, preferred_dtype
from unit.simple_model import random_dataloader

import deepspeed
from deepspeed.utils import set_z3_leaf_modules, unset_z3_leaf_modules, get_z3_leaf_modules, z3_leaf_module
from deepspeed.accelerator import get_accelerator
from torch import nn
import time


class ChooseModuleByCounter(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(ChooseModuleByCounter, self).__init__()
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


class ChooseModuleByRankModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(ChooseModuleByRankModel, self).__init__()
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim, bias=False),
             torch.nn.Linear(hidden_dim, hidden_dim, bias=False)])
        self.act = torch.nn.ReLU()
        self.cel = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        # Each rank runs only one of the linear layers
        x = self.linears[dist.get_rank() % len(self.linears)](x)
        x = self.act(x)
        loss = self.cel(x, y)
        return x, loss


class MLPBlock(nn.Module):

    def __init__(self, hidden_dim):
        super(MLPBlock, self).__init__()
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FineGrainedBlock(nn.Module):

    def __init__(self, hidden_dim, num_block):
        super(FineGrainedBlock, self).__init__()
        self.num_block = num_block
        self.mlp_layers = torch.nn.ModuleList([MLPBlock(hidden_dim=hidden_dim) for _ in range(self.num_block)])

    def forward(self, x):
        for i in range(self.num_block):
            x = self.mlp_layers[i](x)
        return x


class modelWithFineGrainedBlock(nn.Module):

    def __init__(self, hidden_dim, num_block):
        super(modelWithFineGrainedBlock, self).__init__()
        self.coarse_grained_layer1 = nn.Linear(hidden_dim, 8 * hidden_dim)
        self.coarse_grained_layer2 = nn.Linear(8 * hidden_dim, hidden_dim)
        self.fine_grained_layer = FineGrainedBlock(hidden_dim, num_block)
        self.cel = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.coarse_grained_layer1(x)
        x = self.coarse_grained_layer2(x)
        x = self.fine_grained_layer(x)
        loss = self.cel(x, y)
        return x, loss


def run_model(model, config_dict, hidden_dim, dtype, requires_grad):
    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
    data_loader = random_dataloader(model=model,
                                    total_samples=10,
                                    hidden_dim=hidden_dim,
                                    device=model.device,
                                    dtype=dtype)
    dist.barrier()
    for batch in data_loader:
        batch[0].requires_grad = requires_grad
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

    def _test_set_z3_leaf_modules(self, cls, requires_grad):
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
            "zero_optimization": {
                "stage": 3,
                "stage3_prefetch_bucket_size": hidden_dim**2,
                "stage3_param_persistence_threshold": 0,
                "stage3_max_reuse_distance": 0,
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        model = cls(hidden_dim)

        assert not z3_leaf_module(model)
        set_z3_leaf_modules(model, [cls])
        assert z3_leaf_module(model)

        run_model(model, config_dict, hidden_dim, preferred_dtype(), requires_grad)

    def test_choose_module_by_counter(self):
        self._test_set_z3_leaf_modules(ChooseModuleByCounter, True)

    def test_choose_module_by_rank(self):
        self._test_set_z3_leaf_modules(ChooseModuleByRankModel, True)

    def test_no_grad_input_error(self):
        try:
            self._test_set_z3_leaf_modules(ChooseModuleByCounter, False)
            raise AssertionError(
                "Expected RuntimeError: inputs with requires_grad=False is not supported for a leaf module")
        except RuntimeError as e:
            pass

    def test_set_unset_leaf_modules(self):
        hidden_dim = 128
        model = ChooseModuleByCounter(hidden_dim)
        assert len(set_z3_leaf_modules(model, [torch.nn.ModuleList])) == 1, \
            "Expected only one module to be set as a leaf module"
        assert len(get_z3_leaf_modules(model)) == 1, "Expected there is only one leaf module"

        assert len(unset_z3_leaf_modules(model, [torch.nn.ModuleList])) == 1, \
            "Expected only one module to be unset as a leaf module"
        assert len(get_z3_leaf_modules(model)) == 0, "Expected there is no leaf module"

    def test_set_no_match_class(self):
        hidden_dim = 128
        model = ChooseModuleByCounter(hidden_dim)
        try:
            set_z3_leaf_modules(model, [torch.nn.Conv2d])
            raise AssertionError("Expected error that no module is set as a leaf module")
        except ValueError as e:
            pass


@pytest.mark.parametrize("module_granularity_threshold", [0, 100, 12100, 10000000])
class TestZ3LeafOptimization(DistributedTest):
    world_size = 2
    reuse_dist_env = True

    def test_finegrained_optimization(self, module_granularity_threshold: int):
        hidden_dim = 128
        num_block = 16
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "zero_optimization": {
                "stage": 3,
                "stage3_prefetch_bucket_size": hidden_dim**2,
                "stage3_param_persistence_threshold": 0,
                "stage3_max_reuse_distance": 0,
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        def bench_loss_and_time(config):
            warm_up_step = 10
            model = modelWithFineGrainedBlock(hidden_dim, num_block)
            model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            data_loader = random_dataloader(model=model,
                                            total_samples=20,
                                            hidden_dim=hidden_dim,
                                            device=model.device,
                                            dtype=preferred_dtype())
            dist.barrier()
            loss_list = []

            for i, batch in enumerate(data_loader):
                if i == warm_up_step:
                    dist.barrier()
                    get_accelerator().synchronize()
                    start_time = time.time()
                batch[0].requires_grad = True
                loss = model(batch[0], batch[1])
                loss = loss[1]
                loss_list.append(loss)
                model.backward(loss)
                model.step()
            get_accelerator().synchronize()
            end_time = time.time()
            duration = end_time - start_time
            model.destroy()
            return loss_list, duration

        baseline_loss_list, baseline_exec_time = bench_loss_and_time(config_dict)

        config_dict["zero_optimization"]["stage3_module_granularity_threshold"] = module_granularity_threshold
        loss, duration = bench_loss_and_time(config_dict)

        if dist.get_rank() == 0:
            print("baseline exec time:", baseline_exec_time)
            print(
                f"finegrained optimziation exec time: {duration},granularity threshold:{module_granularity_threshold} "
            )
            assert baseline_loss_list == loss, f"incorrect loss value with threshold:{module_granularity_threshold}"

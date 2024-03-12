# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed.comm as dist
import torch

from unit.common import DistributedTest, preferred_dtype
from unit.simple_model import random_dataloader

import deepspeed
from deepspeed.utils import set_z3_leaf_modules, unset_z3_leaf_modules, get_z3_leaf_modules, z3_leaf_module
from deepspeed.accelerator import get_accelerator


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
        if get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True}
        elif get_accelerator().is_bf16_supported():
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

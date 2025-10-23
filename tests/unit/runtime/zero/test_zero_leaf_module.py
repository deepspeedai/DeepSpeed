# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import deepspeed.comm as dist
import torch

from unit.common import DistributedTest, preferred_dtype
from unit.simple_model import random_dataloader

import deepspeed
from deepspeed.utils import set_z3_leaf_modules, unset_z3_leaf_modules, get_z3_leaf_modules, z3_leaf_module, \
    set_z3_leaf_modules_by_name, set_z3_leaf_modules_by_suffix
from deepspeed.runtime.zero.config import DeepSpeedZeroConfig
from deepspeed.runtime.zero.leaf_module_config import (DEFAULT_LEAF_MODULE_CLASSES, DEFAULT_LEAF_MODULE_NAMES,
                                                       DEFAULT_LEAF_MODULE_NAME_SUFFIXES)
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


class BaseLeafModule(nn.Module):

    def __init__(self):
        super(BaseLeafModule, self).__init__()


class SubLeafModule(BaseLeafModule):

    def __init__(self, hidden_dim):
        super(SubLeafModule, self).__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.proj(x)


class WrapperLeafModule(nn.Module):

    def __init__(self, hidden_dim):
        super(WrapperLeafModule, self).__init__()
        self.child = SubLeafModule(hidden_dim)

    def forward(self, x):
        return self.child(x)


def test_set_leaf_modules_with_fully_qualified_name():
    hidden_dim = 16
    model = WrapperLeafModule(hidden_dim)
    fq_name = f"{SubLeafModule.__module__}.{SubLeafModule.__qualname__}"

    matched = set_z3_leaf_modules(model, [fq_name])

    assert len(matched) == 1
    assert matched[0] is model.child
    assert z3_leaf_module(model.child)
    assert not z3_leaf_module(model)


def test_set_leaf_modules_no_raise_when_missing():
    hidden_dim = 16
    model = WrapperLeafModule(hidden_dim)

    matched = set_z3_leaf_modules(model, ["NonExistentClass"], raise_if_not_found=False)

    assert matched == []
    assert not z3_leaf_module(model.child)


def test_set_leaf_modules_by_name():
    hidden_dim = 16
    model = WrapperLeafModule(hidden_dim)

    matched, missing = set_z3_leaf_modules_by_name(model, ["child"])

    assert matched == [model.child]
    assert missing == []
    assert z3_leaf_module(model.child)


def test_set_leaf_modules_by_name_missing():
    hidden_dim = 16
    model = WrapperLeafModule(hidden_dim)

    matched, missing = set_z3_leaf_modules_by_name(model, ["missing"], raise_if_not_found=False)

    assert matched == []
    assert missing == ["missing"]


def test_set_leaf_modules_by_suffix():
    hidden_dim = 16
    model = WrapperLeafModule(hidden_dim)

    matched, missing = set_z3_leaf_modules_by_suffix(model, ["child"])

    assert missing == []
    assert matched == [model.child]
    assert z3_leaf_module(model.child)


def test_set_leaf_modules_by_suffix_missing():
    hidden_dim = 16
    model = WrapperLeafModule(hidden_dim)

    matched, missing = set_z3_leaf_modules_by_suffix(model, ["missing"], raise_if_not_found=False)

    assert matched == []
    assert missing == ["missing"]


def test_zero_leaf_module_default_config():
    config = DeepSpeedZeroConfig()
    assert config.leaf_module.classes == DEFAULT_LEAF_MODULE_CLASSES
    assert config.leaf_module.names == DEFAULT_LEAF_MODULE_NAMES
    assert config.leaf_module.name_suffixes == DEFAULT_LEAF_MODULE_NAME_SUFFIXES


def test_zero_leaf_module_custom_config():
    payload = {
        "leaf_module": {
            "classes": ["custom.module.CustomClass"],
            "names": ["transformer.layer"],
            "name_suffixes": ["experts"]
        }
    }

    config = DeepSpeedZeroConfig(**payload)

    assert config.leaf_module.classes == ["custom.module.CustomClass"]
    assert config.leaf_module.names == ["transformer.layer"]
    assert config.leaf_module.name_suffixes == ["experts"]


def test_zero_leaf_module_string_coercion():
    payload = {"leaf_module": {"classes": "my.Class", "names": "submodule", "name_suffixes": "tail"}}

    config = DeepSpeedZeroConfig(**payload)

    assert config.leaf_module.classes == ["my.Class"]
    assert config.leaf_module.names == ["submodule"]
    assert config.leaf_module.name_suffixes == ["tail"]


@pytest.mark.skip(reason="Requires Hugging Face transformers; run manually when validating defaults.")
def test_default_leaf_module_classes_exist():
    import importlib

    from deepspeed.runtime.zero.leaf_module_config import DEFAULT_LEAF_MODULE_CLASSES

    for cls_path in DEFAULT_LEAF_MODULE_CLASSES:
        module_name, _, class_name = cls_path.rpartition('.')
        module = importlib.import_module(module_name)
        assert hasattr(module, class_name), f"Expected {class_name} in {module_name}"


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

    def _create_zero_config(self, hidden_dim, leaf_module=None):
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
        if leaf_module is not None:
            config_dict["zero_optimization"]["leaf_module"] = leaf_module

        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        return config_dict

    def _test_set_z3_leaf_modules(self, cls, requires_grad):
        hidden_dim = 128
        config_dict = self._create_zero_config(hidden_dim)

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

    def test_set_leaf_modules_with_subclass(self):
        hidden_dim = 32
        model = WrapperLeafModule(hidden_dim)

        leaf_modules = set_z3_leaf_modules(model, [BaseLeafModule])

        assert len(leaf_modules) == 1, "Expected the subclass instance to be marked as leaf"
        assert leaf_modules[0] is model.child, "Expected the subclass instance to be returned"
        assert z3_leaf_module(model.child), "Expected subclass instance flagged as leaf"
        assert not z3_leaf_module(model), "Expected wrapper module to remain non-leaf"

    def test_set_no_match_class(self):
        hidden_dim = 128
        model = ChooseModuleByCounter(hidden_dim)
        try:
            set_z3_leaf_modules(model, [torch.nn.Conv2d])
            raise AssertionError("Expected error that no module is set as a leaf module")
        except ValueError as e:
            pass

    def test_leaf_module_enabled_via_config(self):
        hidden_dim = 128
        leaf_class_fqn = f"{ChooseModuleByCounter.__module__}.{ChooseModuleByCounter.__qualname__}"
        config_dict = self._create_zero_config(hidden_dim,
                                               leaf_module={
                                                   "classes": [leaf_class_fqn],
                                                   "name_suffixes": ["linears"]
                                               })

        model = ChooseModuleByCounter(hidden_dim)
        assert not z3_leaf_module(model)

        run_model(model, config_dict, hidden_dim, preferred_dtype(), True)

        assert z3_leaf_module(model)
        modules_by_name = dict(model.named_modules())
        assert "linears" in modules_by_name
        assert z3_leaf_module(modules_by_name["linears"])


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

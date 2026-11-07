# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
import math

import pytest
import torch

import deepspeed
from deepspeed.runtime.zero.utils import is_zero_param
from deepspeed.utils import set_z3_leaf_module
from unit.common import DistributedTest


class _LateModuleModel(torch.nn.Module):

    def __init__(self, hidden_dim=8):
        super().__init__()
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim)
        self.extra = None

    def forward(self, x):
        x = self.fc(x)
        if self.extra is not None:
            x = self.extra(x)
        return x


class _LeafBlock(torch.nn.Module):

    def __init__(self, hidden_dim=8):
        super().__init__()
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x)


class _LeafModel(torch.nn.Module):

    def __init__(self, hidden_dim=8):
        super().__init__()
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim)
        self.block = _LeafBlock(hidden_dim)

    def forward(self, x):
        return self.block(self.fc(x))


class _PassthroughWrapper(torch.nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _ModulesToSaveWrapper(torch.nn.Module):
    """Minimal stand-in for PEFT's ``ModulesToSaveWrapper``.

    It keeps a frozen copy of the wrapped module next to the trainable copy
    that is actually used in the forward pass, like PEFT does for
    ``modules_to_save``.
    """

    def __init__(self, module):
        super().__init__()
        self.original_module = copy.deepcopy(module)
        self.original_module.requires_grad_(False)
        self.modules_to_save = torch.nn.ModuleDict({"default": module})

    def forward(self, *args, **kwargs):
        return self.modules_to_save["default"](*args, **kwargs)


def _config():
    return {
        "train_micro_batch_size_per_gpu": 1,
        "zero_optimization": {
            "stage": 3,
        },
        "zero_allow_untested_optimizer": True,
    }


class TestZero3LateParameterPartition(DistributedTest):
    world_size = 1

    def test_late_module_is_partitioned_before_forward(self):
        dtype = torch.float32
        model = _LateModuleModel()
        engine, *_ = deepspeed.initialize(model=model,
                                          optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                                                    lr=0.1),
                                          model_parameters=model.parameters(),
                                          config=_config())

        late = torch.nn.Linear(8, 8).to(device=engine.device, dtype=dtype)
        late.requires_grad_(False)
        engine.module.extra = late

        assert not is_zero_param(late.weight)

        x = torch.randn(2, 8, dtype=dtype, device=engine.device)
        engine.backward(engine(x).sum())
        engine.step()

        assert is_zero_param(late.weight)
        assert late.weight.ds_numel == 8 * 8
        assert late.weight.ds_tensor.numel() == math.ceil(8 * 8 / deepspeed.comm.get_world_size())

    def test_frozen_modules_to_save_style_wrapper_is_partitioned(self):
        dtype = torch.float32
        model = _LateModuleModel()
        engine, *_ = deepspeed.initialize(model=model,
                                          optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                                                    lr=0.1),
                                          model_parameters=model.parameters(),
                                          config=_config())

        wrapped = torch.nn.Linear(8, 8).to(device=engine.device, dtype=dtype)
        wrapped.requires_grad_(False)
        engine.module.extra = _ModulesToSaveWrapper(wrapped)

        original = engine.module.extra.original_module.weight
        saved = engine.module.extra.modules_to_save["default"].weight
        assert not is_zero_param(original)
        assert not is_zero_param(saved)

        x = torch.randn(2, 8, dtype=dtype, device=engine.device)
        engine.backward(engine(x).sum())
        engine.step()

        world_size = deepspeed.comm.get_world_size()
        for param in (original, saved):
            assert is_zero_param(param)
            assert param.ds_numel == 8 * 8
            assert param.ds_tensor.numel() == math.ceil(8 * 8 / world_size)

    def test_trainable_late_parameter_is_rejected(self):
        dtype = torch.float32
        model = _LateModuleModel()
        engine, *_ = deepspeed.initialize(model=model,
                                          optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                                                    lr=0.1),
                                          model_parameters=model.parameters(),
                                          config=_config())

        late = torch.nn.Linear(8, 8).to(device=engine.device, dtype=dtype)
        engine.module.extra = late

        x = torch.randn(2, 8, dtype=dtype, device=engine.device)
        with pytest.raises(RuntimeError, match="cannot optimize parameters added"):
            engine(x)

    def test_repeated_forwards_do_not_duplicate_hooks(self):
        dtype = torch.float32
        model = _LateModuleModel()
        engine, *_ = deepspeed.initialize(model=model,
                                          optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                                                    lr=0.1),
                                          model_parameters=model.parameters(),
                                          config=_config())

        late = torch.nn.Linear(8, 8).to(device=engine.device, dtype=dtype)
        late.requires_grad_(False)
        engine.module.extra = late
        offload = engine.optimizer.parameter_offload

        x = torch.randn(2, 8, dtype=dtype, device=engine.device)
        engine(x)
        hook_count = len(offload.forward_hooks)
        engine(x)
        assert len(offload.forward_hooks) == hook_count
        assert offload.partition_pending_parameters() == 0

    def test_parameter_added_to_existing_module_is_partitioned_before_forward(self):
        dtype = torch.float32
        model = _LateModuleModel()
        engine, *_ = deepspeed.initialize(model=model,
                                          optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                                                    lr=0.1),
                                          model_parameters=model.parameters(),
                                          config=_config())

        extra = torch.nn.Parameter(torch.zeros(8, 8, dtype=dtype, device=engine.device), requires_grad=False)
        engine.module.fc.extra_weight = extra
        assert not is_zero_param(extra)

        x = torch.randn(2, 8, dtype=dtype, device=engine.device)
        engine.backward(engine(x).sum())
        engine.step()

        assert is_zero_param(extra)
        assert extra.ds_numel == 8 * 8

    def test_parameter_added_to_existing_module_is_partitioned_at_step(self):
        dtype = torch.float32
        model = _LateModuleModel()
        engine, *_ = deepspeed.initialize(model=model,
                                          optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                                                    lr=0.1),
                                          model_parameters=model.parameters(),
                                          config=_config())

        extra = torch.nn.Parameter(torch.zeros(8, 8, dtype=dtype, device=engine.device), requires_grad=False)
        engine.module.fc.extra_weight = extra
        assert not is_zero_param(extra)

        engine.optimizer.parameter_offload.partition_all_parameters()

        assert is_zero_param(extra)
        assert extra.ds_numel == 8 * 8

    def test_existing_leaf_descendants_are_not_registered(self):
        dtype = torch.float32
        model = _LeafModel()
        set_z3_leaf_module(model.block, True)
        engine, *_ = deepspeed.initialize(model=model,
                                          optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                                                    lr=0.1),
                                          model_parameters=model.parameters(),
                                          config=_config())

        offload = engine.optimizer.parameter_offload
        hook_count = len(offload.forward_hooks)
        assert hasattr(model.block, "ds_id")
        assert not hasattr(model.block.fc, "ds_id")

        x = torch.randn(2, 8, dtype=dtype, device=engine.device)
        engine(x)

        assert len(offload.forward_hooks) == hook_count
        assert not hasattr(model.block.fc, "ds_id")
        assert offload.partition_pending_parameters() == 0

    def test_wrapping_existing_module_refreshes_param_names(self):
        dtype = torch.float32
        model = _LateModuleModel()
        engine, *_ = deepspeed.initialize(model=model,
                                          optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                                                    lr=0.1),
                                          model_parameters=model.parameters(),
                                          config=_config())

        # Wrapping an existing module does not add parameters, but it renames
        # them. The checkpoint registry still has to be refreshed.
        engine.module.fc = _PassthroughWrapper(engine.module.fc)

        x = torch.randn(2, 8, dtype=dtype, device=engine.device)
        engine(x)

        names = set(engine.param_names.values())
        assert "fc.module.weight" in names
        assert "fc.module.bias" in names

    def test_frozen_late_parameter_checkpoint_round_trip(self, tmpdir):
        dtype = torch.float32
        model = _LateModuleModel()
        engine, *_ = deepspeed.initialize(model=model,
                                          optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                                                    lr=0.1),
                                          model_parameters=model.parameters(),
                                          config=_config())

        late = torch.nn.Linear(8, 8).to(device=engine.device, dtype=dtype)
        late.requires_grad_(False)
        engine.module.extra = late

        x = torch.randn(2, 8, dtype=dtype, device=engine.device)
        engine.backward(engine(x).sum())
        engine.step()

        with deepspeed.zero.GatheredParameters([late.weight]):
            expected = late.weight.detach().cpu().clone()

        checkpoint_dir = str(tmpdir)
        engine.save_checkpoint(checkpoint_dir, tag="late-frozen")
        engine.destroy()

        model2 = _LateModuleModel()
        extra = torch.nn.Linear(8, 8).to(device=engine.device, dtype=dtype)
        extra.requires_grad_(False)
        model2.extra = extra
        engine2, *_ = deepspeed.initialize(model=model2,
                                           optimizer=torch.optim.SGD(
                                               [p for p in model2.parameters() if p.requires_grad], lr=0.1),
                                           model_parameters=model2.parameters(),
                                           config=_config())
        engine2.load_checkpoint(checkpoint_dir, tag="late-frozen")

        with deepspeed.zero.GatheredParameters([extra.weight]):
            torch.testing.assert_close(extra.weight.detach().cpu(), expected)

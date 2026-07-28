# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn.functional as F

import deepspeed
from deepspeed.runtime.zero.partition_parameters import (InsertPostInitMethodToModuleSubClasses, ZeroParamStatus,
                                                         zero3_linear_wrap)
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad
from unit.common import DistributedTest


def _zero3_config(memory_efficient_linear):
    return {
        "train_micro_batch_size_per_gpu": 2,
        "zero_optimization": {
            "stage": 3,
            "memory_efficient_linear": memory_efficient_linear,
            "stage3_param_persistence_threshold": 0,
            "stage3_max_reuse_distance": 0,
        },
    }


def _storage_id(tensor):
    return tensor.untyped_storage()._cdata


class TestZero3LinearDirectInit(DistributedTest):
    world_size = 2

    def test_wrapper_and_parameter_storage_lifecycle(self):
        torch.manual_seed(20260728)
        assert not hasattr(InsertPostInitMethodToModuleSubClasses, "linear_bk")
        original_linear = F.linear
        model = torch.nn.Linear(4, 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        reference_input = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
        reference_weight = model.weight.detach().clone().requires_grad_()
        reference_bias = model.bias.detach().clone().requires_grad_()
        reference_output = original_linear(reference_input, reference_weight, reference_bias)
        reference_output.sum().backward()

        engine, *_ = deepspeed.initialize(model=model,
                                          optimizer=optimizer,
                                          config_params=_zero3_config(memory_efficient_linear=True))
        assert F.linear is zero3_linear_wrap
        assert InsertPostInitMethodToModuleSubClasses.linear_bk is original_linear

        weight = engine.module.weight
        saved = []
        unpacked = []

        def pack_hook(tensor):
            is_weight = tensor is weight
            if is_weight:
                saved.append((_storage_id(tensor), tensor.numel()))
            return is_weight, tensor

        def unpack_hook(packed):
            is_weight, tensor = packed
            if is_weight:
                unpacked.append((_storage_id(tensor), tensor.numel()))
            return tensor

        inputs = reference_input.to(engine.device)
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            output = engine(inputs)
            assert saved and saved[0][1] == reference_weight.numel()
            p0_storage = saved[0][0]
            assert weight.ds_status == ZeroParamStatus.NOT_AVAILABLE
            assert weight.numel() == 0
            engine.backward(output.sum())

        assert unpacked and unpacked[0][1] == reference_weight.numel()
        assert unpacked[0][0] != p0_storage
        assert all(storage != p0_storage for storage, _ in unpacked)
        assert weight.ds_status == ZeroParamStatus.NOT_AVAILABLE
        torch.testing.assert_close(output.detach().cpu(), reference_output.detach())
        torch.testing.assert_close(safe_get_full_grad(weight).cpu(), reference_weight.grad)
        torch.testing.assert_close(safe_get_full_grad(engine.module.bias).cpu(), reference_bias.grad)

        before_step = safe_get_full_fp32_param(weight).clone()
        engine.step()
        after_step = safe_get_full_fp32_param(weight)
        assert not torch.equal(before_step, after_step)


class TestZero3LinearDirectInitDisabled(DistributedTest):
    world_size = 2

    def test_disabled_preserves_builtin_linear(self):
        torch.manual_seed(20260728)
        assert not hasattr(InsertPostInitMethodToModuleSubClasses, "linear_bk")
        original_linear = F.linear
        model = torch.nn.Linear(4, 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

        engine, *_ = deepspeed.initialize(model=model,
                                          optimizer=optimizer,
                                          config_params=_zero3_config(memory_efficient_linear=False))
        assert F.linear is original_linear
        assert not hasattr(InsertPostInitMethodToModuleSubClasses, "linear_bk")

        output = engine(torch.ones(2, 4, device=engine.device))
        engine.backward(output.sum())
        assert safe_get_full_grad(engine.module.weight) is not None
        engine.step()


class TestZero3LinearInitContextPersistence(DistributedTest):
    world_size = 1

    def test_context_activation_is_idempotent_and_persistent(self):
        assert not hasattr(InsertPostInitMethodToModuleSubClasses, "linear_bk")
        original_linear = F.linear
        config = _zero3_config(memory_efficient_linear=True)

        init = deepspeed.zero.Init(config_dict_or_path=config)
        assert F.linear is original_linear
        with init:
            assert F.linear is zero3_linear_wrap
            with deepspeed.zero.Init(config_dict_or_path=config):
                assert F.linear is zero3_linear_wrap
                assert InsertPostInitMethodToModuleSubClasses.linear_bk is original_linear

        assert F.linear is zero3_linear_wrap
        assert InsertPostInitMethodToModuleSubClasses.linear_bk is original_linear

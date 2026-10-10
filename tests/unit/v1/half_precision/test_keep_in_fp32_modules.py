# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""data_types.keep_in_fp32_modules: buffers a model names for fp32 stay fp32 under bf16/fp16 training.

The toy model follows transformers: it lists its fp32 tensors in `_keep_in_fp32_modules_strict`. The listed
buffer holds values near 8, where bf16 values are 0.0625 apart, so a lost fp32 value shows up as a value at a
bf16 grid point.
"""

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.config import DeepSpeedConfig, DeepSpeedConfigError
from deepspeed.runtime.keep_in_fp32 import keep_in_fp32_pattern
from unit.common import DistributedTest

HIDDEN = 64
# Distinct in fp32, but bf16 rounds all of them to 8.0.
EXACT_VALUES = 8.0 + torch.arange(HIDDEN, dtype=torch.float32) * 1e-4


class ToyModel(torch.nn.Module):
    _keep_in_fp32_modules_strict = ["router_bias"]

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.register_buffer("router_bias", torch.zeros(HIDDEN))
        self.register_buffer("scale", torch.ones(HIDDEN))

    def forward(self, x):
        return ((self.linear(x) * self.scale + self.router_bias)**2).mean()


class WrappedToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.child = torch.nn.Module()
        self.child._keep_in_fp32_modules_strict = ["router_bias"]
        self.child.inner = torch.nn.Module()
        self.child.inner.linear = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.child.inner.register_buffer("router_bias", torch.zeros(HIDDEN))
        self.child.inner.register_buffer("scale", torch.ones(HIDDEN))
        self.other_child = torch.nn.Module()
        self.other_child.inner = torch.nn.Module()
        self.other_child.inner.register_buffer("router_bias", torch.zeros(HIDDEN))
        self.child2 = torch.nn.Module()
        self.child2.inner = torch.nn.Module()
        self.child2.inner.register_buffer("router_bias", torch.zeros(HIDDEN))

    def forward(self, x):
        inner = self.child.inner
        return ((inner.linear(x) * inner.scale + inner.router_bias)**2).mean()


def _config(stage, keep="auto", buffer_dtype=None):
    data_types = {"keep_in_fp32_modules": keep}
    if buffer_dtype is not None:
        data_types["buffer_dtype"] = buffer_dtype
    return {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "zero_optimization": {
            "stage": stage
        },
        "bf16": {
            "enabled": True
        },
        "data_types": data_types,
    }


class TestKeepInFp32Pattern:

    def test_auto_uses_strict_list_under_bf16(self):
        pattern = keep_in_fp32_pattern(ToyModel(), "auto", torch.bfloat16)
        assert pattern.search("layers.0.router_bias")
        assert not pattern.search("scale")

    def test_auto_adds_fp16_only_list_under_fp16(self):
        model = ToyModel()
        model._keep_in_fp32_modules = ["scale"]
        assert not keep_in_fp32_pattern(model, "auto", torch.bfloat16).search("scale")
        assert keep_in_fp32_pattern(model, "auto", torch.float16).search("scale")

    def test_auto_uses_lists_from_nested_modules(self):
        pattern = keep_in_fp32_pattern(WrappedToyModel(), "auto", torch.bfloat16)
        assert pattern.search("child.inner.router_bias")
        assert not pattern.search("child.inner.scale")
        assert not pattern.search("other_child.inner.router_bias")
        assert not pattern.search("child2.inner.router_bias")

    def test_explicit_list_and_glob(self):
        pattern = keep_in_fp32_pattern(ToyModel(), ["layers.*.gate.bias"], torch.bfloat16)
        assert pattern.search("model.layers.3.gate.bias")
        assert not pattern.search("router_bias")

    @pytest.mark.parametrize("keep,dtype", [([], torch.bfloat16), ("auto", torch.float32)])
    def test_nothing_to_keep(self, keep, dtype):
        assert keep_in_fp32_pattern(ToyModel(), keep, dtype) is None


class TestKeepInFp32Config(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("keep", ["yes", ["router_bias", 3], {"router_bias": True}])
    def test_invalid_setting_rejected(self, keep):
        with pytest.raises(DeepSpeedConfigError, match="keep_in_fp32_modules"):
            DeepSpeedConfig(_config(0, keep=keep))


@pytest.mark.skipif(torch.bfloat16 not in get_accelerator().supported_dtypes(), reason="bf16 not supported")
class TestZeroInitKeepsBuffersFp32(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("keep", ["auto", []])
    def test_loaded_values_survive(self, keep):
        with deepspeed.zero.Init(config_dict_or_path=_config(3, keep=keep)):
            model = ToyModel()
        # What a checkpoint loader does under ZeRO-3 (transformers' _load_state_dict_into_zero3_model).
        model.router_bias.copy_(EXACT_VALUES)

        if keep == "auto":
            assert model.router_bias.dtype == torch.float32
            assert torch.equal(model.router_bias.cpu(), EXACT_VALUES)
        else:
            # Without the list the values are rounded to 8.0: the behavior this option fixes.
            assert model.router_bias.dtype == torch.bfloat16
            assert torch.equal(model.router_bias.float().cpu(), torch.full((HIDDEN, ), 8.0))
        assert model.scale.dtype == torch.bfloat16

    def test_kept_through_initialize_and_training(self):
        config = _config(3)
        with deepspeed.zero.Init(config_dict_or_path=config):
            model = ToyModel()
        model.router_bias.copy_(EXACT_VALUES)
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
        x = torch.randn(1, HIDDEN, device=engine.device, dtype=torch.bfloat16)
        for _ in range(2):
            loss = engine(x)
            engine.backward(loss)
            engine.step()
        assert torch.isfinite(loss)
        assert engine.module.router_bias.dtype == torch.float32
        assert torch.equal(engine.module.router_bias.cpu(), EXACT_VALUES)

    def test_existing_wrapped_module_uses_child_list(self):
        model = WrappedToyModel().bfloat16()
        deepspeed.zero.Init(module=model, config_dict_or_path=_config(3))
        assert model.child.inner.router_bias.dtype == torch.float32
        assert model.child.inner.scale.dtype == torch.bfloat16
        assert model.other_child.inner.router_bias.dtype == torch.bfloat16
        assert model.child2.inner.router_bias.dtype == torch.bfloat16


@pytest.mark.skipif(torch.bfloat16 not in get_accelerator().supported_dtypes(), reason="bf16 not supported")
@pytest.mark.parametrize("zero_stage", [0, 3])
class TestEngineKeepsBuffersFp32(DistributedTest):
    world_size = 1

    def test_buffer_dtype_does_not_cast_listed_buffers(self, zero_stage):
        model = ToyModel()
        model.router_bias.copy_(EXACT_VALUES)
        engine, _, _, _ = deepspeed.initialize(config=_config(zero_stage, buffer_dtype="bf16"),
                                               model=model,
                                               model_parameters=model.parameters())
        assert engine.module.router_bias.dtype == torch.float32
        assert torch.equal(engine.module.router_bias.cpu(), EXACT_VALUES)
        assert engine.module.scale.dtype == torch.bfloat16

    def test_listed_buffer_loaded_in_bf16_is_upcast(self, zero_stage):
        model = ToyModel()
        model.router_bias.data = model.router_bias.data.bfloat16()
        engine, _, _, _ = deepspeed.initialize(config=_config(zero_stage),
                                               model=model,
                                               model_parameters=model.parameters())
        assert engine.module.router_bias.dtype == torch.float32

    def test_empty_list_keeps_old_behavior(self, zero_stage):
        model = ToyModel()
        engine, _, _, _ = deepspeed.initialize(config=_config(zero_stage, keep=[], buffer_dtype="bf16"),
                                               model=model,
                                               model_parameters=model.parameters())
        assert engine.module.router_bias.dtype == torch.bfloat16

    def test_wrapped_model_uses_child_list(self, zero_stage):
        model = WrappedToyModel()
        model.child.inner.router_bias.copy_(EXACT_VALUES)
        engine, _, _, _ = deepspeed.initialize(config=_config(zero_stage, buffer_dtype="bf16"),
                                               model=model,
                                               model_parameters=model.parameters())
        assert engine.module.child.inner.router_bias.dtype == torch.float32
        assert torch.equal(engine.module.child.inner.router_bias.cpu(), EXACT_VALUES)
        assert engine.module.child.inner.scale.dtype == torch.bfloat16
        assert engine.module.other_child.inner.router_bias.dtype == torch.bfloat16
        assert engine.module.child2.inner.router_bias.dtype == torch.bfloat16

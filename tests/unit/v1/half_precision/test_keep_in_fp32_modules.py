# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""data_types.keep_in_fp32_modules: tensors a model names for fp32 stay fp32 under bf16/fp16 training.

The toy models follow transformers: they list their fp32 tensors in `_keep_in_fp32_modules_strict`. The listed
tensors hold values near 8, where bf16 values are 0.0625 apart, so a lost fp32 value or update shows up as a
value at a bf16 grid point.
"""

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.config import DeepSpeedConfig, DeepSpeedConfigError
from deepspeed.runtime.keep_in_fp32 import keep_in_fp32_pattern
from deepspeed.utils import safe_get_full_fp32_param
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


class ToyModelWithParams(torch.nn.Module):
    _keep_in_fp32_modules_strict = ["sink", "router_bias"]

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.sink = torch.nn.Parameter(torch.full((HIDDEN, ), 8.0))
        self.register_buffer("router_bias", torch.zeros(HIDDEN))

    def forward(self, x):
        return ((self.linear(x) * self.sink + self.router_bias)**2).mean()


def _params_config(stage,
                   keep="auto",
                   dtype="bf16",
                   offload_optimizer=False,
                   offload_param=False,
                   contiguous=True,
                   allreduce_fetch=False,
                   grad_accum=1):
    zero = {"stage": stage, "contiguous_gradients": contiguous}
    if stage == 3:
        # Partition every parameter, including the small ones.
        zero.update({"stage3_param_persistence_threshold": 0, "use_all_reduce_for_fetch_params": allreduce_fetch})
    if offload_optimizer:
        zero["offload_optimizer"] = {"device": "cpu"}
    if offload_param:
        zero["offload_param"] = {"device": "cpu"}
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": grad_accum,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "zero_optimization": zero,
        "data_types": {
            "keep_in_fp32_modules": keep
        },
    }
    config[dtype] = {"enabled": True}
    if dtype == "fp16":
        config["fp16"]["initial_scale_power"] = 8
    return config


def _gathered(param):
    with deepspeed.zero.GatheredParameters([param]):
        return param.data.detach().clone()


def _stored_dtype(tensor):
    return tensor.ds_tensor.dtype if hasattr(tensor, "ds_tensor") else tensor.dtype


def _train(engine, steps):
    dtype = torch.bfloat16 if engine.bfloat16_enabled() else torch.half
    x = torch.randn(1, HIDDEN, device=engine.device, dtype=dtype)
    for _ in range(steps * engine.gradient_accumulation_steps()):
        loss = engine(x)
        engine.backward(loss)
        engine.step()
    return loss


@pytest.mark.skipif(torch.bfloat16 not in get_accelerator().supported_dtypes(), reason="bf16 not supported")
class TestZeroInitKeepsParamsFp32(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("keep", ["auto", []])
    def test_loaded_values_survive(self, keep):
        with deepspeed.zero.Init(config_dict_or_path=_params_config(3, keep=keep)):
            model = ToyModelWithParams()
        # What a checkpoint loader does under ZeRO-3 (transformers' _load_state_dict_into_zero3_model).
        with deepspeed.zero.GatheredParameters([model.sink], modifier_rank=0):
            model.sink.data.copy_(EXACT_VALUES)

        if keep == "auto":
            assert _stored_dtype(model.sink) == torch.float32
            assert torch.equal(_gathered(model.sink).cpu(), EXACT_VALUES)
        else:
            # Without the list the value is rounded to 8.0: the behavior this option fixes.
            assert _stored_dtype(model.sink) == torch.bfloat16
            assert torch.equal(_gathered(model.sink).float().cpu(), torch.full((HIDDEN, ), 8.0))
        assert _stored_dtype(model.linear.weight) == torch.bfloat16


@pytest.mark.skipif(torch.bfloat16 not in get_accelerator().supported_dtypes(), reason="bf16 not supported")
class TestZero3TrainsKeptParamsInFp32(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("zero_init", [True, False])
    @pytest.mark.parametrize("contiguous", [True, False])
    @pytest.mark.parametrize("offload_optimizer", [False, True])
    @pytest.mark.parametrize("grad_accum", [1, 2])
    def test_update_below_bf16_resolution_is_kept(self, zero_init, contiguous, offload_optimizer, grad_accum):
        config = _params_config(3, contiguous=contiguous, offload_optimizer=offload_optimizer, grad_accum=grad_accum)
        if zero_init:
            with deepspeed.zero.Init(config_dict_or_path=config):
                model = ToyModelWithParams()
        else:
            model = ToyModelWithParams()
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
        loss = _train(engine, steps=1)

        assert torch.isfinite(loss)
        sink = engine.module.sink
        assert _stored_dtype(sink) == torch.float32
        assert _stored_dtype(engine.module.linear.weight) == torch.bfloat16
        assert engine.module.router_bias.dtype == torch.float32
        # Adam moves each element by about the learning rate, 1e-3: invisible in bf16 at 8.0.
        forward_value = _gathered(sink).float()
        master_value = safe_get_full_fp32_param(sink).float().to(forward_value.device)
        assert torch.equal(forward_value, master_value)
        assert (forward_value - 8.0).abs().max() > 5e-4

    def test_all_reduce_fetch(self):
        config = _params_config(3, allreduce_fetch=True)
        with deepspeed.zero.Init(config_dict_or_path=config):
            model = ToyModelWithParams()
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
        _train(engine, steps=2)
        assert (_gathered(engine.module.sink) - 8.0).abs().max() > 5e-4

    def test_checkpoint_round_trip(self, tmpdir):
        config = _params_config(3)
        with deepspeed.zero.Init(config_dict_or_path=config):
            model = ToyModelWithParams()
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
        _train(engine, steps=1)
        saved = _gathered(engine.module.sink)
        engine.save_checkpoint(str(tmpdir))

        with deepspeed.zero.Init(config_dict_or_path=config):
            model2 = ToyModelWithParams()
        engine2, _, _, _ = deepspeed.initialize(config=config, model=model2, model_parameters=model2.parameters())
        engine2.load_checkpoint(str(tmpdir))
        assert torch.equal(_gathered(engine2.module.sink), saved)
        assert torch.equal(safe_get_full_fp32_param(engine2.module.sink).to(saved.device), saved)
        assert torch.isfinite(_train(engine2, steps=1))

    def test_param_offload_keeps_only_buffers(self):
        config = _params_config(3, offload_param=True, offload_optimizer=True)
        with deepspeed.zero.Init(config_dict_or_path=config):
            model = ToyModelWithParams()
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
        assert torch.isfinite(_train(engine, steps=1))
        assert _stored_dtype(engine.module.sink) == torch.bfloat16
        assert engine.module.router_bias.dtype == torch.float32


@pytest.mark.skipif(not get_accelerator().is_fp16_supported(), reason="fp16 not supported")
class TestZero3Fp16KeptParams(DistributedTest):
    world_size = 2

    def test_fp16_trains(self):
        config = _params_config(3, dtype="fp16")
        with deepspeed.zero.Init(config_dict_or_path=config):
            model = ToyModelWithParams()
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
        assert torch.isfinite(_train(engine, steps=2))
        assert _stored_dtype(engine.module.sink) == torch.float32
        assert _stored_dtype(engine.module.linear.weight) == torch.half


@pytest.mark.skipif(torch.bfloat16 not in get_accelerator().supported_dtypes(), reason="bf16 not supported")
class TestStagesBelowThreeKeptParams(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("stage", [0, 1, 2])
    def test_auto_casts_params_keeps_buffers(self, stage):
        model = ToyModelWithParams()
        engine, _, _, _ = deepspeed.initialize(config=_params_config(stage),
                                               model=model,
                                               model_parameters=model.parameters())
        assert torch.isfinite(_train(engine, steps=1))
        assert engine.module.sink.dtype == torch.bfloat16
        assert engine.module.router_bias.dtype == torch.float32

    def test_explicit_parameter_pattern_rejected(self):
        model = ToyModelWithParams()
        # Not pytest.raises: when it fails it raises a BaseException, which the DistributedTest worker pool
        # drops, so the test would hang instead of failing.
        try:
            deepspeed.initialize(config=_params_config(2, keep=["sink"]),
                                 model=model,
                                 model_parameters=model.parameters())
        except ValueError as error:
            assert "keep_in_fp32_modules" in str(error)
        else:
            raise AssertionError("stage 2 accepted an explicit keep_in_fp32_modules list that names a parameter")

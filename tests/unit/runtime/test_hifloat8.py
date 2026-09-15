# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from deepspeed.runtime.config import DeepSpeedConfigError, get_hifloat8_config
from deepspeed.runtime.engine import DeepSpeedEngine


def _has_torch_npu_hifloat8_helper():
    try:
        return find_spec("torch_npu.utils.hifloat8_train.hifloat8_linear") is not None
    except (AttributeError, ImportError):
        return False


requires_torch_npu_hifloat8 = pytest.mark.skipif(not _has_torch_npu_hifloat8_helper(),
                                                 reason="torch_npu HiFloat8 training helper is not installed")


def test_hifloat8_config_defaults_disabled():
    assert get_hifloat8_config({}) == {
        "enabled": False,
        "module_name_patterns": (),
        "min_numel": 0,
    }


@pytest.mark.parametrize(
    "config",
    [
        {
            "hifloat8": {
                "enabled": True
            }
        },
        {
            "hifloat8": {
                "enabled": "yes",
                "module_name_patterns": ["*"]
            }
        },
        {
            "hifloat8": {
                "enabled": True,
                "module_name_patterns": [""],
                "min_numel": 0
            }
        },
        {
            "hifloat8": {
                "enabled": True,
                "module_name_patterns": ["*"],
                "min_numel": -1
            }
        },
        {
            "hifloat8": {
                "enabled": True,
                "module_name_patterns": ["*"],
                "probe_kernel": False
            }
        },
        {
            "hifloat8": {
                "enabled": True,
                "module_name_patterns": ["*"],
                "unknown": 1
            }
        },
    ],
)
def test_hifloat8_config_rejects_invalid_values(config):
    with pytest.raises(DeepSpeedConfigError):
        get_hifloat8_config(config)


class _ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.mlp = nn.ModuleDict({
            "gate_proj": nn.Linear(4, 8, bias=False),
            "up_proj": nn.Linear(4, 8, bias=False),
            "down_proj": nn.Linear(8, 4, bias=False),
        })
        self.model.self_attn = nn.ModuleDict({"q_proj": nn.Linear(4, 4, bias=False)})


def _make_engine(model, patterns, min_numel=0):
    engine = object.__new__(DeepSpeedEngine)
    nn.Module.__init__(engine)
    engine._config = SimpleNamespace(
        hifloat8_config={
            "enabled": True,
            "module_name_patterns": tuple(patterns),
            "min_numel": min_numel,
        },
        bfloat16_config=SimpleNamespace(enabled=True),
        tensor_parallel_config=SimpleNamespace(autotp_size=1),
        zero_optimization_stage=2,
    )
    engine.pipeline_parallelism = False
    engine.has_moe_layers = False
    engine.mpu = None
    engine.device = torch.device("cpu")
    engine._set_client_model(model)
    return engine


@requires_torch_npu_hifloat8
def test_engine_selects_routed_experts_without_replacing_parameters(monkeypatch):
    # Catches silently selecting the router or orphaning packed expert weights.
    from deepspeed.moe.ep_experts import GroupedExperts
    import torch_npu.utils.hifloat8_train as implementation

    model = nn.ModuleDict({
        "experts": GroupedExperts(8, 16, 2, use_grouped_mm=False),
        "router": nn.Linear(8, 2, bias=False),
    })
    model.experts.w1.requires_grad_(False)
    parameters = dict(model.named_parameters())
    keys = tuple(model.state_dict())
    monkeypatch.setattr(implementation, "assert_hifloat8_grouped_training_available", lambda **kwargs: None)
    engine = _make_engine(model, ["experts"])
    engine._configure_hifloat8()
    assert engine.hifloat8_grouped_module_names == ("experts", )
    assert engine.hifloat8_converted_module_names == ()
    assert model.experts.hifloat8_enabled
    assert type(model.router) is nn.Linear
    assert not model.experts.w1.requires_grad
    assert tuple(model.state_dict()) == keys
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters.items())


@requires_torch_npu_hifloat8
def test_engine_converts_only_selected_modules_and_preserves_parameters(monkeypatch):
    model = _ToyModel()
    before = dict(model.named_parameters())
    probe_calls = []
    import deepspeed.runtime.hifloat8 as bridge

    HiFloat8Linear = bridge.get_hifloat8_linear_class()

    monkeypatch.setattr(
        bridge,
        "assert_hifloat8_training_available",
        lambda **kwargs: probe_calls.append(kwargs),
    )
    engine = _make_engine(model, ["*.mlp.gate_proj", "*.mlp.up_proj", "*.mlp.down_proj"])
    engine._configure_hifloat8()

    assert probe_calls == [{"probe_kernel": True, "device": torch.device("cpu")}]
    assert engine.hifloat8_converted_module_names == (
        "model.mlp.gate_proj",
        "model.mlp.up_proj",
        "model.mlp.down_proj",
    )
    assert all(isinstance(model.model.mlp[name], HiFloat8Linear) for name in model.model.mlp)
    assert type(model.model.self_attn["q_proj"]) is nn.Linear
    after = dict(model.named_parameters())
    assert before.keys() == after.keys()
    assert all(after[name] is parameter for name, parameter in before.items())


def test_engine_fails_when_module_selection_is_empty(monkeypatch):
    import deepspeed.runtime.hifloat8 as bridge

    monkeypatch.setattr(bridge, "assert_hifloat8_training_available", lambda **_kwargs: None)
    engine = _make_engine(_ToyModel(), ["*.does_not_exist"])
    with pytest.raises(RuntimeError, match="matched no nn.Linear"):
        engine._configure_hifloat8()


def test_engine_rejects_partially_unmatched_patterns_before_conversion():
    # A typo must not silently leave part of the requested model in BF16.
    model = _ToyModel()
    engine = _make_engine(model, ["*.mlp.gate_proj", "*.missing_projection"])
    with pytest.raises(RuntimeError, match="patterns matched no eligible modules"):
        engine._configure_hifloat8()
    assert type(model.model.mlp["gate_proj"]) is nn.Linear


def test_hifloat8_validation_rejects_autotp():
    engine = _make_engine(_ToyModel(), ["*"])
    engine._config.tensor_parallel_config.autotp_size = 2

    with pytest.raises(RuntimeError, match="AutoTP"):
        engine._validate_hifloat8_configuration()


def test_hifloat8_validation_rejects_custom_model_parallel_unit():
    engine = _make_engine(_ToyModel(), ["*"])
    engine.mpu = object()

    with pytest.raises(RuntimeError, match="model-parallel unit"):
        engine._validate_hifloat8_configuration()


@pytest.mark.parametrize("error", [RuntimeError("unsupported device"), ImportError("missing torch_npu helper")])
def test_engine_kernel_failure_reports_selection_and_does_not_mutate(monkeypatch, error):
    import deepspeed.runtime.hifloat8 as bridge

    model = _ToyModel()
    before = dict(model.named_parameters())

    def fail_probe(**_kwargs):
        raise error

    monkeypatch.setattr(bridge, "assert_hifloat8_training_available", fail_probe)
    engine = _make_engine(model, ["*.mlp.gate_proj"])

    with pytest.raises(RuntimeError, match=r"selected 1 Linear modules \(32 matrix elements\).+before conversion"):
        engine._configure_hifloat8()

    assert type(model.model.mlp["gate_proj"]) is nn.Linear
    after = dict(model.named_parameters())
    assert before.keys() == after.keys()
    assert all(after[name] is parameter for name, parameter in before.items())

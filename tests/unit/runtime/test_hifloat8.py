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

requires_torchao_npu = pytest.mark.skipif(find_spec("torchao_npu") is None, reason="torchao_npu is not installed")


def test_hifloat8_config_defaults_disabled():
    assert get_hifloat8_config({}) == {
        "enabled": False,
        "backend": "torchao_npu",
        "module_name_patterns": (),
        "min_numel": 0,
        "expected_module_count": None,
        "config": None,
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


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "84"])
def test_hifloat8_config_rejects_invalid_expected_count(count):
    with pytest.raises(DeepSpeedConfigError):
        get_hifloat8_config({"hifloat8": {"expected_module_count": count}})


def test_hifloat8_config_accepts_dense_selection_contract():
    # Existing Swift Dense configurations require exactly 28 * 3 projections.
    config = get_hifloat8_config({"hifloat8": {"expected_module_count": 84}})
    assert config["expected_module_count"] == 84


@pytest.mark.parametrize("backend", ["", "unknown", None, 1])
def test_hifloat8_config_rejects_unknown_backend(backend):
    with pytest.raises(DeepSpeedConfigError, match="hifloat8.backend"):
        get_hifloat8_config({"hifloat8": {"backend": backend}})


def test_hifloat8_config_selects_torchao_npu():
    assert get_hifloat8_config({"hifloat8": {"backend": "torchao_npu"}})["backend"] == "torchao_npu"


@requires_torchao_npu
@pytest.mark.parametrize("policy", [
    None, {}, {
        "input_dst_type_max": 15,
        "weight_dst_type_max": 15,
        "grad_dst_type_max": 224,
        "scale_policy": "pertensor",
        "compute_dtype": "bfloat16"
    }, {
        "input_dst_type_max": 31,
        "weight_dst_type_max": 31,
        "grad_dst_type_max": 127,
        "compute_dtype": "bfloat16"
    }
])
def test_torchao_npu_backend_preserves_dense_and_grouped_parameters(policy):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    from deepspeed.moe.ep_experts import GroupedExperts
    from torchao_npu.hifloat8 import HiFloat8Linear

    model = nn.ModuleDict({
        "dense": nn.Linear(16, 16, bias=False),
        "experts": GroupedExperts(16, 32, 2, use_grouped_mm=True),
    }).to("npu").bfloat16()
    parameters_before = dict(model.named_parameters())
    keys_before = tuple(model.state_dict())
    engine = _make_engine(model, ["dense", "experts"])
    engine.device = torch.device("npu:0")
    engine._config.hifloat8_config["backend"] = "torchao_npu"
    engine._config.hifloat8_config["config"] = policy
    engine._configure_hifloat8()

    assert isinstance(model["dense"], HiFloat8Linear)
    assert model["experts"].hifloat8_enabled
    assert model["dense"].config is model["experts"].hifloat8_config
    assert model["dense"].config.grad_dst_type_max == (policy.get("grad_dst_type_max", 224) if policy else 224)
    assert tuple(model.state_dict()) == keys_before
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters_before.items())


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


@requires_torchao_npu
def test_engine_selects_routed_experts_without_replacing_parameters(monkeypatch):
    # Catches silently selecting the router or orphaning packed expert weights.
    from deepspeed.moe.ep_experts import GroupedExperts
    import torchao_npu.hifloat8 as implementation

    model = nn.ModuleDict({
        "experts": GroupedExperts(8, 16, 2, use_grouped_mm=False),
        "router": nn.Linear(8, 2, bias=False),
    })
    model.experts.w1.requires_grad_(False)
    parameters = dict(model.named_parameters())
    keys = tuple(model.state_dict())
    monkeypatch.setattr(implementation, "assert_hifloat8_training_available", lambda **kwargs: None)
    engine = _make_engine(model, ["experts"])
    engine._configure_hifloat8()
    assert engine.hifloat8_grouped_module_names == ("experts", )
    assert engine.hifloat8_converted_module_names == ()
    assert model.experts.hifloat8_enabled
    assert type(model.router) is nn.Linear
    assert not model.experts.w1.requires_grad
    assert tuple(model.state_dict()) == keys
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters.items())


@requires_torchao_npu
def test_engine_converts_only_selected_modules_and_preserves_parameters(monkeypatch):
    model = _ToyModel()
    before = dict(model.named_parameters())
    probe_calls = []
    import torchao_npu.hifloat8 as implementation

    HiFloat8Linear = implementation.HiFloat8Linear

    monkeypatch.setattr(
        implementation,
        "assert_hifloat8_training_available",
        lambda **kwargs: probe_calls.append(kwargs),
    )
    engine = _make_engine(model, ["*.mlp.gate_proj", "*.mlp.up_proj", "*.mlp.down_proj"])
    engine._config.hifloat8_config["expected_module_count"] = 3
    engine._configure_hifloat8()

    assert len(probe_calls) == 1
    assert probe_calls[0]["device"] == torch.device("cpu")
    assert probe_calls[0]["linear"] and not probe_calls[0]["grouped"]
    assert probe_calls[0]["config"] is model.model.mlp["gate_proj"].config
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


def test_engine_fails_when_module_selection_is_empty():
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


def test_engine_rejects_wrong_module_count_before_kernel_probe(monkeypatch):
    # A partially selected model must fail before native initialization or mutation.
    import builtins

    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name.startswith("torchao_npu"):
            pytest.fail("module-count mismatch must fail before loading the optional backend")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    model = _ToyModel()
    engine = _make_engine(model, ["*.mlp.gate_proj"])
    engine._config.hifloat8_config["expected_module_count"] = 3
    with pytest.raises(RuntimeError, match="expected_module_count"):
        engine._configure_hifloat8()
    assert type(model.model.mlp["gate_proj"]) is nn.Linear


def test_hifloat8_validation_rejects_autotp():
    engine = _make_engine(_ToyModel(), ["*"])
    engine._config.tensor_parallel_config.autotp_size = 2

    with pytest.raises(RuntimeError, match="AutoTP"):
        engine._validate_hifloat8_configuration()


def test_missing_optional_backend_reports_selection_before_model_mutation(monkeypatch):
    # Direct lazy imports must preserve the diagnostic and parameter contract
    # when the optional backend is absent, including on a CUDA/CPU installation.
    import builtins

    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name.startswith("torchao_npu"):
            raise ImportError("torchao_npu is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    model = _ToyModel()
    parameters = dict(model.named_parameters())
    engine = _make_engine(model, ["*.mlp.gate_proj"])
    with pytest.raises(RuntimeError, match="native kernel validation failed before conversion"):
        engine._configure_hifloat8()
    assert type(model.model.mlp["gate_proj"]) is nn.Linear
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters.items())


def test_hifloat8_validation_rejects_custom_model_parallel_unit():
    engine = _make_engine(_ToyModel(), ["*"])
    engine.mpu = object()

    with pytest.raises(RuntimeError, match="model-parallel unit"):
        engine._validate_hifloat8_configuration()


@requires_torchao_npu
@pytest.mark.parametrize("error", [RuntimeError("unsupported device"), ImportError("missing torchao_npu backend")])
def test_engine_kernel_failure_reports_selection_and_does_not_mutate(monkeypatch, error):
    import torchao_npu.hifloat8 as implementation

    model = _ToyModel()
    before = dict(model.named_parameters())

    def fail_probe(**_kwargs):
        raise error

    monkeypatch.setattr(implementation, "assert_hifloat8_training_available", fail_probe)
    engine = _make_engine(model, ["*.mlp.gate_proj"])

    with pytest.raises(RuntimeError, match=r"selected 1 Linear modules \(32 matrix elements\).+before conversion"):
        engine._configure_hifloat8()

    assert type(model.model.mlp["gate_proj"]) is nn.Linear
    after = dict(model.named_parameters())
    assert before.keys() == after.keys()
    assert all(after[name] is parameter for name, parameter in before.items())


@pytest.mark.parametrize("policy", [True, 1, "bf16", []])
def test_hifloat8_config_rejects_non_object_policy(policy):
    with pytest.raises(DeepSpeedConfigError, match="hifloat8.config"):
        get_hifloat8_config({"hifloat8": {"backend": "torchao_npu", "config": policy}})


def test_hifloat8_legacy_backend_is_rejected_with_migration_instruction():
    with pytest.raises(DeepSpeedConfigError, match="legacy torch_npu helper was removed"):
        get_hifloat8_config({"hifloat8": {"backend": "torch_npu"}})
    policy = {"grad_dst_type_max": 15, "compute_dtype": "bfloat16"}
    assert get_hifloat8_config({"hifloat8": {"config": policy}})["config"] == policy


def test_minimal_hifloat8_json_requires_no_numerical_fields():
    config = get_hifloat8_config({
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2
        },
        "hifloat8": {
            "enabled": True,
            "backend": "torchao_npu",
            "module_name_patterns": ["*.experts", "*.shared_experts.*_proj"],
            "min_numel": 65536,
        },
    })
    assert config["enabled"] and config["config"] is None
    assert config["backend"] == "torchao_npu"
    assert config["min_numel"] == 65536

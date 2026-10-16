# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from deepspeed.runtime.config import DeepSpeedConfigError, get_hifloat8_config
from deepspeed.runtime.engine import DeepSpeedEngine

# Native tests require the public API; a missing extension must fail there.
# Gate only device availability, without importing TorchAO during collection.
requires_npu = pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="NPU unavailable")


def test_hifloat8_config_defaults_disabled():
    assert get_hifloat8_config({}) == {
        "enabled": False,
        "backend": "torchao",
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
    assert get_hifloat8_config({"hifloat8": {"backend": "torchao"}})["backend"] == "torchao"


@requires_npu
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
def test_torchao_backend_preserves_dense_and_grouped_parameters(policy, monkeypatch):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    from deepspeed.moe.ep_experts import GroupedExperts
    from torchao.hifloat8 import HiFloat8Linear
    import torchao.hifloat8 as api

    policies = []
    original = api.hifloat8_grouped_mm

    def recorded(*args, **kwargs):
        policies.append(kwargs["config"])
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "hifloat8_grouped_mm", recorded)
    model = nn.ModuleDict({
        "dense": nn.Linear(16, 16, bias=False),
        "experts": GroupedExperts(16, 32, 2, use_grouped_mm=True),
    }).to("npu").bfloat16()
    with torch.no_grad():
        for parameter in model["experts"].parameters():
            parameter.normal_(std=0.02)
    parameters_before = dict(model.named_parameters())
    keys_before = tuple(model.state_dict())
    engine = _make_engine(model, ["dense", "experts"])
    engine.device = torch.device("npu:0")
    engine._config.hifloat8_config["backend"] = "torchao"
    engine._config.hifloat8_config["config"] = policy
    engine._configure_hifloat8()

    assert isinstance(model["dense"], HiFloat8Linear)
    assert model["experts"].grouped_mm is not None
    x = torch.randn(16, 16, device="npu", dtype=torch.bfloat16)
    model["experts"](model["dense"](x), torch.tensor([0, 16], device="npu", dtype=torch.int64))
    assert policies and all(value is model["dense"].config for value in policies)
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
    engine._config.hifloat8_config = get_hifloat8_config({"hifloat8": engine._config.hifloat8_config})
    engine._config.float16_config = SimpleNamespace(enabled=False)
    engine.pipeline_parallelism = False
    engine.has_moe_layers = False
    engine.mpu = None
    engine.device = torch.device("cpu")
    engine._set_client_model(model)
    return engine


@requires_npu
def test_engine_selects_routed_experts_without_replacing_parameters(monkeypatch):
    if not hasattr(torch, "npu"):
        pytest.skip("NPU accelerator required")
    # Catches silently selecting the router or orphaning packed expert weights.
    from deepspeed.moe.ep_experts import GroupedExperts
    import torchao.hifloat8 as implementation

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
    assert model.experts.grouped_mm is not None
    assert type(model.router) is nn.Linear
    assert not model.experts.w1.requires_grad
    assert tuple(model.state_dict()) == keys
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters.items())


@requires_npu
def test_engine_converts_only_selected_modules_and_preserves_parameters(monkeypatch):
    if not hasattr(torch, "npu"):
        pytest.skip("NPU accelerator required")
    model = _ToyModel()
    before = dict(model.named_parameters())
    probe_calls = []
    import torchao.hifloat8 as implementation

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
    from deepspeed.accelerator import get_accelerator

    def unexpected_provider(backend, format):
        pytest.fail("module-count mismatch must fail before resolving the optional backend")

    monkeypatch.setattr(get_accelerator(), "get_hifloat8_provider", unexpected_provider)
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
    # Provider policy parsing may report a missing optional dependency on any
    # accelerator; the engine must retain its diagnostic and parameter contract.
    from deepspeed.accelerator import get_accelerator

    error = ImportError("optional low-precision backend is not installed")

    def missing_policy(values):
        raise error

    provider = SimpleNamespace(HiFloat8Config=SimpleNamespace(from_dict=missing_policy))
    monkeypatch.setattr(get_accelerator(), "get_hifloat8_provider", lambda: provider)
    model = _ToyModel()
    parameters = dict(model.named_parameters())
    engine = _make_engine(model, ["*.mlp.gate_proj"])
    with pytest.raises(RuntimeError,
                       match=r"selected 1 Linear modules \(32 matrix elements\).+before conversion") as caught:
        engine._configure_hifloat8()
    assert caught.value.__cause__ is error
    assert type(model.model.mlp["gate_proj"]) is nn.Linear
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters.items())


def test_hifloat8_validation_rejects_custom_model_parallel_unit():
    engine = _make_engine(_ToyModel(), ["*"])
    engine.mpu = object()

    with pytest.raises(RuntimeError, match="model-parallel unit"):
        engine._validate_hifloat8_configuration()


@requires_npu
@pytest.mark.parametrize("error", [RuntimeError("unsupported device"), ImportError("missing torchao_npu backend")])
def test_engine_kernel_failure_reports_selection_and_does_not_mutate(monkeypatch, error):
    if not hasattr(torch, "npu"):
        pytest.skip("NPU accelerator required")
    import torchao.hifloat8 as implementation

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
    assert config["backend"] == "torchao"
    assert config["min_numel"] == 65536


def test_hifloat8_config_keeps_policy_opaque_and_normalizes_legacy_backend():
    policy = {"backend_specific_option": 7}
    assert get_hifloat8_config({"hifloat8": {"backend": "torchao_npu", "config": policy}})["backend"] == "torchao"
    assert get_hifloat8_config({"hifloat8": {"config": policy}})["config"] == policy


def test_accelerator_without_hifloat8_fails_before_conversion(monkeypatch):
    from deepspeed.accelerator import get_accelerator

    monkeypatch.setattr(get_accelerator(), "get_hifloat8_provider", lambda: None)
    model = _ToyModel()
    engine = _make_engine(model, ["*.mlp.gate_proj"])
    with pytest.raises(RuntimeError, match="No HiFloat8 provider"):
        engine._configure_hifloat8()
    assert type(model.model.mlp["gate_proj"]) is nn.Linear


@requires_npu
def test_public_grouped_quantize_rejects_missing_accelerator_capability(monkeypatch):
    # Direct quantize_ must fail before changing experts when no device kernel is available.
    from torchao.quantization import quantize_
    from torchao.hifloat8 import HiFloat8Config
    from deepspeed.accelerator import get_accelerator
    from deepspeed.moe.ep_experts import GroupedExperts
    from deepspeed.runtime.torchao import GroupedExpertsConfig

    experts = GroupedExperts(8, 16, 2, use_grouped_mm=False)
    parameters = dict(experts.named_parameters())
    monkeypatch.setattr(get_accelerator(), "get_hifloat8_provider", lambda: None)
    with pytest.raises(RuntimeError, match="No HiFloat8 provider"):
        quantize_(experts, GroupedExpertsConfig(HiFloat8Config()), filter_fn=lambda module, name: name == "")
    assert experts.grouped_mm is None
    assert all(dict(experts.named_parameters())[name] is value for name, value in parameters.items())


def test_npu_provider_missing_dependency_has_install_hint(monkeypatch):
    from importlib.metadata import PackageNotFoundError
    from deepspeed.accelerator.npu import hifloat8

    def missing_version(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(hifloat8, "version", missing_version)
    with pytest.raises(ImportError, match="Install torchao"):
        hifloat8.get_hifloat8_provider()


def test_removed_configuration_extension_reports_existing_schema():
    # Do not silently disable a run using the short-lived generic section.
    with pytest.raises(DeepSpeedConfigError, match="hifloat8"):
        get_hifloat8_config({"low_precision": {"enabled": False}})


@requires_npu
@pytest.mark.parametrize("kind", ["mixed", "linear", "grouped"])
def test_public_quantize_entry_converts_dense_and_grouped_in_place(monkeypatch, kind):
    # A split expert path would bypass the public API; wrap the real collaborator.
    import torchao.quantization as quantization
    from torchao.hifloat8 import HiFloat8Linear
    from deepspeed.moe.ep_experts import GroupedExperts

    torch.manual_seed(59)
    if kind == "mixed":
        model = nn.ModuleDict({"dense": nn.Linear(16, 16), "experts": GroupedExperts(16, 32, 2, use_grouped_mm=False)})
        patterns = ["dense", "experts"]
    elif kind == "linear":
        model = nn.Linear(16, 16)
        patterns = ["*"]
    else:
        model = GroupedExperts(16, 32, 2, use_grouped_mm=False)
        patterns = ["*"]
    model = model.to("npu").bfloat16()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.02)
    before = dict(model.named_parameters())
    keys = tuple(model.state_dict())
    calls = []
    policies = []
    import torchao.hifloat8 as api
    original_grouped = api.hifloat8_grouped_mm

    def recorded_grouped(*args, **kwargs):
        policies.append(kwargs["config"])
        return original_grouped(*args, **kwargs)

    monkeypatch.setattr(api, "hifloat8_grouped_mm", recorded_grouped)
    original = quantization.quantize_

    def counted(actual_model, config, **kwargs):
        calls.append(actual_model)
        return original(actual_model, config, **kwargs)

    monkeypatch.setattr(quantization, "quantize_", counted)
    engine = _make_engine(model, patterns)
    engine.device = torch.device("npu:0")
    engine._configure_hifloat8()
    assert calls == [model]
    assert engine.module is model
    assert tuple(model.state_dict()) == keys
    assert all(dict(model.named_parameters())[name] is value for name, value in before.items())
    x = torch.randn(16, 16, device="npu", dtype=torch.bfloat16, requires_grad=True)
    counts = torch.tensor([0, 16], device="npu", dtype=torch.int64)
    if kind == "mixed":
        assert isinstance(model["dense"], HiFloat8Linear)
        output = model["experts"](model["dense"](x), counts)
    elif kind == "linear":
        assert isinstance(model, HiFloat8Linear)
        output = model(x)
    else:
        assert model.grouped_mm is not None
        output = model(x, counts)
    if kind == "mixed":
        assert policies and all(value is model["dense"].config for value in policies)
    assert output.dtype == x.dtype
    output.float().square().mean().backward()
    assert bool(torch.isfinite(x.grad).all())
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters())

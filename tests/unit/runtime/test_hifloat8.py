# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.config import DeepSpeedConfigError
from deepspeed.runtime.quantization.config import get_quantization_config
from deepspeed.runtime.engine import DeepSpeedEngine

# Native tests require the public API; a missing extension must fail there.
# Gate only device availability, without importing TorchAO during collection.
requires_npu = pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="NPU unavailable")


def get_hifloat8_config(config):
    return get_quantization_config(config).hifloat8


def test_hifloat8_config_defaults_disabled():
    assert get_hifloat8_config({}) == {
        "enabled": False,
        "backend": "torchao",
        "module_name_patterns": (),
        "min_numel": 0,
        "expected_module_count": None,
        "config": None,
        "module_configs": None,
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
def test_torchao_backend_preserves_dense_and_grouped_parameters(policy):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    from deepspeed.moe.ep_experts import GroupedExperts
    from torchao.hifloat8 import HiFloat8GMM, HiFloat8Linear

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
    assert isinstance(model["experts"].grouped_mm, HiFloat8GMM)
    assert vars(model["experts"].grouped_mm.config) == vars(model["dense"].config)
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


def _make_engine(model, patterns, min_numel=0, module_configs=None):
    engine = object.__new__(DeepSpeedEngine)
    nn.Module.__init__(engine)
    engine._config = SimpleNamespace(
        hifloat8_config={
            "enabled": True,
            "module_name_patterns": tuple(patterns),
            "min_numel": min_numel,
            "module_configs": module_configs,
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
    engine.device = torch.device(get_accelerator().device_name())
    engine._set_client_model(model)
    return engine


@requires_npu
def test_engine_selects_routed_experts_without_replacing_parameters():
    if not hasattr(torch, "npu"):
        pytest.skip("NPU accelerator required")
    # Catches silently selecting the router or orphaning packed expert weights.
    from deepspeed.moe.ep_experts import GroupedExperts
    from torchao.hifloat8 import HiFloat8GMM

    model = nn.ModuleDict({
        "experts": GroupedExperts(8, 16, 2, use_grouped_mm=False),
        "router": nn.Linear(8, 2, bias=False),
    })
    model.experts.w1.requires_grad_(False)
    parameters = dict(model.named_parameters())
    keys = tuple(model.state_dict())
    engine = _make_engine(model, ["experts"])
    engine._configure_hifloat8()
    assert engine.hifloat8_grouped_module_names == ("experts", )
    assert engine.hifloat8_converted_module_names == ()
    assert isinstance(model.experts.grouped_mm, HiFloat8GMM)
    assert type(model.router) is nn.Linear
    assert not model.experts.w1.requires_grad
    assert tuple(model.state_dict()) == keys
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters.items())


@requires_npu
def test_engine_converts_only_selected_modules_and_preserves_parameters():
    if not hasattr(torch, "npu"):
        pytest.skip("NPU accelerator required")
    model = _ToyModel()
    before = dict(model.named_parameters())
    from torchao.hifloat8 import HiFloat8Linear

    engine = _make_engine(model, ["*.mlp.gate_proj", "*.mlp.up_proj", "*.mlp.down_proj"])
    engine._config.hifloat8_config["expected_module_count"] = 3
    engine._configure_hifloat8()

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


def test_engine_rejects_wrong_module_count_before_backend_load(monkeypatch):
    # A partially selected model must fail before native initialization or mutation.
    def unexpected_provider(_format):
        pytest.fail("module-count mismatch must fail before resolving the optional backend")

    monkeypatch.setattr(get_accelerator(), "get_training_quantization_config", unexpected_provider)
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
        engine._validate_hifloat8_configuration(engine.module)


def test_missing_optional_backend_reports_selection_before_model_mutation(monkeypatch):
    # Provider policy parsing may report a missing optional dependency on any
    # accelerator; the engine must retain its diagnostic and parameter contract.
    error = ImportError("optional low-precision backend is not installed")

    def missing_policy(_format):
        raise error

    monkeypatch.setattr(get_accelerator(), "get_training_quantization_config", missing_policy)
    model = _ToyModel()
    parameters = dict(model.named_parameters())
    engine = _make_engine(model, ["*.mlp.gate_proj"])
    with pytest.raises(RuntimeError, match="selected 1 Linear and 0 AutoEP modules.+before conversion") as caught:
        engine._configure_hifloat8()
    assert caught.value.__cause__ is error
    assert type(model.model.mlp["gate_proj"]) is nn.Linear
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters.items())


def test_hifloat8_validation_rejects_custom_model_parallel_unit():
    engine = _make_engine(_ToyModel(), ["*"])
    engine.mpu = object()

    with pytest.raises(RuntimeError, match="model-parallel unit"):
        engine._validate_hifloat8_configuration(engine.module)


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
    monkeypatch.setattr(get_accelerator(), "get_training_quantization_config", lambda _format: None)
    model = _ToyModel()
    engine = _make_engine(model, ["*.mlp.gate_proj"])
    with pytest.raises(RuntimeError, match="No hifloat8 training backend"):
        engine._configure_hifloat8()
    assert type(model.model.mlp["gate_proj"]) is nn.Linear


@requires_npu
def test_invalid_recipe_fails_before_conversion():
    model = _ToyModel()
    engine = _make_engine(model, ["*.mlp.gate_proj"])
    engine._config.hifloat8_config["config"] = {"compute_dtype": "float32"}
    with pytest.raises(RuntimeError, match="recipe failed before conversion"):
        engine._configure_hifloat8()
    assert type(model.model.mlp["gate_proj"]) is nn.Linear


def test_npu_provider_missing_dependency_has_install_hint(monkeypatch):
    from importlib.metadata import PackageNotFoundError
    from deepspeed.accelerator.npu import hifloat8

    def missing_version(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(hifloat8, "version", missing_version)
    with pytest.raises(ImportError, match="Install torchao"):
        hifloat8.get_hifloat8_config_class()


def test_removed_configuration_extension_reports_existing_schema():
    # Do not silently disable a run using the short-lived generic section.
    with pytest.raises(DeepSpeedConfigError, match="hifloat8"):
        get_hifloat8_config({"low_precision": {"enabled": False}})


@requires_npu
@pytest.mark.parametrize("kind", ["mixed", "linear", "grouped"])
@pytest.mark.parametrize("use_mapping", [False, True])
def test_public_quantize_entry_converts_dense_and_grouped_in_place(monkeypatch, kind, use_mapping):
    # A split expert path would bypass the public API; wrap the real collaborator.
    import torchao.quantization as quantization
    from torchao.hifloat8 import HiFloat8GMM, HiFloat8Linear
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
    original = quantization.quantize_

    def counted(actual_model, config, **kwargs):
        calls.append(actual_model)
        return original(actual_model, config, **kwargs)

    monkeypatch.setattr(quantization, "quantize_", counted)
    module_configs = None
    if use_mapping:
        module_configs = {name: {} for name in (["dense", "experts"] if kind == "mixed" else [""])}
        patterns = []
    engine = _make_engine(model, patterns, module_configs=module_configs)
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
        assert isinstance(model.grouped_mm, HiFloat8GMM)
        output = model(x, counts)
    assert output.dtype == x.dtype
    output.float().square().mean().backward()
    assert bool(torch.isfinite(x.grad).all())
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters())


def test_quantization_hifloat8_and_legacy_config_normalize_identically():
    options = {"enabled": True, "module_name_patterns": ["*.experts"], "config": {"grad_dst_type_max": None}}
    assert get_quantization_config({"quantization": {
        "hifloat8": options
    }}) == get_quantization_config({"hifloat8": options})
    with pytest.raises(DeepSpeedConfigError, match="quantization.hifloat8"):
        get_quantization_config({"quantization": {"hifloat8": options}, "hifloat8": options})


def test_quantization_recipes_share_one_configuration():
    config = get_quantization_config({
        "quantization": {
            "subchannel": {
                "enabled": True,
                "forward": {
                    "block_size": 32
                }
            },
            "hifloat8": {
                "enabled": False
            },
        }
    })
    assert config.subchannel.enabled
    assert config.subchannel.forward.block_size == 32
    assert not config.hifloat8["enabled"]


def test_quantization_rejects_both_recipes_enabled():
    with pytest.raises(DeepSpeedConfigError, match="Enable only one"):
        get_quantization_config({
            "hifloat8": {
                "enabled": True,
                "module_name_patterns": ["*.experts"]
            },
            "quantization": {
                "subchannel": {
                    "enabled": True
                }
            },
        })


@pytest.mark.parametrize("option,value", [("overlap_weight_grad", True), ("dw_staging", object())])
def test_hifloat8_rejects_deferred_expert_gradients_before_conversion(option, value):
    from deepspeed.moe.ep_experts import GroupedExperts

    model = nn.Module()
    model.experts = GroupedExperts(16, 32, 2, use_grouped_mm=False)
    setattr(model, option, value)
    original = model.experts.grouped_mm
    engine = _make_engine(model, ["experts"])
    with pytest.raises(RuntimeError, match="deferred or staged weight gradients"):
        engine._configure_hifloat8()
    assert model.experts.grouped_mm is original


@pytest.mark.parametrize("options", [[], None, {"fp8": {"enabled": True}}])
def test_invalid_training_quantization_sections_fail_closed(options):
    with pytest.raises((DeepSpeedConfigError, ValueError)):
        get_quantization_config({"quantization": options})


@pytest.mark.parametrize("module_configs", [[], True, {1: {}}, {"dense": []}, {"re:(": {}}])
def test_invalid_module_recipe_structure_fails_during_config_parsing(module_configs):
    with pytest.raises(DeepSpeedConfigError):
        get_quantization_config({"quantization": {"hifloat8": {"module_configs": module_configs}}})


@pytest.mark.parametrize("module_configs", [{"missing": {}}, {"re:.*missing": {}}, {"_default": None}])
def test_module_recipe_selection_errors_preserve_the_model(module_configs):
    model = _ToyModel()
    before = dict(model.named_parameters())
    engine = _make_engine(model, [], module_configs=module_configs)
    with pytest.raises(RuntimeError, match="matched no"):
        engine._configure_hifloat8()
    assert all(type(module) is nn.Linear for module in model.model.mlp.values())
    assert all(dict(model.named_parameters())[name] is value for name, value in before.items())


@requires_npu
@pytest.mark.parametrize("scoped", [False, True])
def test_module_recipes_match_explicit_ao_reference_with_one_conversion(monkeypatch, scoped):
    # Catches wrong precedence, ignored per-module recipes and lost optimizer parameters.
    import copy
    import torchao.quantization as quantization
    from torchao.hifloat8 import HiFloat8Config
    from deepspeed.moe.ep_experts import GroupedExperts

    torch.manual_seed(113)
    model = nn.ModuleDict({
        "dense_a": nn.Linear(64, 64),
        "dense_b": nn.Linear(64, 64),
        "dense_c": nn.Linear(64, 64),
        "experts_a": GroupedExperts(64, 64, 2, use_grouped_mm=False),
        "experts_b": GroupedExperts(64, 64, 2, use_grouped_mm=False),
        "untouched": nn.Linear(64, 64),
        "embedding": nn.Embedding(8, 64),
    }).to(device="npu", dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.02)
    model["experts_b"].w1.requires_grad_(False)
    reference = copy.deepcopy(model)
    parameters = dict(model.named_parameters())
    keys = tuple(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    common = {"input_dst_type_max": 15, "weight_dst_type_max": 31, "grad_dst_type_max": 224}
    reference_configs = {}
    for name in ("dense_a", "dense_b", "dense_c", "experts_a", "experts_b"):
        policy = dict(common)
        grouped = name.startswith("experts")
        policy["grad_dst_type_max"] = 31 if grouped else 127
        if name == "dense_a":
            policy.update(grad_dst_type_max=15, compute_dtype="float16")
        recipe = HiFloat8Config.from_dict({"grouped" if grouped else "linear": policy})
        reference_configs[f"{name}.grouped_mm" if grouped else name] = recipe
    quantization.quantize_(reference, quantization.FqnToConfig(reference_configs), filter_fn=None)

    calls = []
    original = quantization.quantize_

    def counted(actual_model, config, **kwargs):
        calls.append(actual_model)
        return original(actual_model, config, **kwargs)

    monkeypatch.setattr(quantization, "quantize_", counted)
    module_configs = {
        "re:experts_[ab]": {
            "grad_dst_type_max": 31
        },
        "re:.*": {
            "grad_dst_type_max": 127
        },
        "dense_a": {
            "grad_dst_type_max": 15,
            "compute_dtype": "float16"
        },
        "untouched": None,
        "_default": None,
    }
    patterns = ["dense_*", "experts_*", "untouched"] if scoped else []
    engine = _make_engine(model, patterns, module_configs=module_configs)
    engine._config.hifloat8_config["config"] = {"weight_dst_type_max": 31}
    engine._config.hifloat8_config["expected_module_count"] = 5
    engine._configure_hifloat8()
    assert calls == [model]
    assert tuple(model.state_dict()) == keys
    assert type(model["untouched"]) is nn.Linear
    assert all(dict(model.named_parameters())[name] is value for name, value in parameters.items())

    def forward(module, value):
        counts = torch.tensor([0, value.shape[0]], device=value.device, dtype=torch.int64)
        hidden = module["dense_a"](value) + module["dense_b"](value)
        return module["experts_a"](hidden, counts) + module["experts_b"](module["dense_c"](value), counts)

    value = torch.randn(16, 64, device="npu", dtype=torch.bfloat16, requires_grad=True)
    ref_value = value.detach().clone().requires_grad_()
    actual, expected = forward(model, value), forward(reference, ref_value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    torch.testing.assert_close(value.grad, ref_value.grad, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        ref_parameter = dict(reference.named_parameters())[name]
        torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=0, atol=0)
    optimizer.step()
    reference_optimizer.step()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)


@requires_npu
def test_late_invalid_module_recipe_fails_before_any_conversion():
    # A valid first recipe must not mutate the model before a later recipe fails AO validation.
    model = nn.ModuleDict({"first": nn.Linear(8, 8), "last": nn.Linear(8, 8)})
    engine = _make_engine(model, [], module_configs={"first": {}, "last": {"compute_dtype": "float32"}})
    with pytest.raises(RuntimeError, match="recipe failed before conversion"):
        engine._configure_hifloat8()
    assert all(type(module) is nn.Linear for module in model.values())


@requires_npu
def test_default_module_recipe_converts_only_eligible_modules():
    model = nn.ModuleDict({"dense": nn.Linear(8, 8), "skip": nn.Linear(8, 8), "embedding": nn.Embedding(8, 8)})
    engine = _make_engine(model, [], module_configs={"skip": None, "_default": {"grad_dst_type_max": 15}})
    engine._config.hifloat8_config["expected_module_count"] = 1
    engine._configure_hifloat8()
    assert model["dense"].config.grad_dst_type_max == 15
    assert type(model["skip"]) is nn.Linear
    assert type(model["embedding"]) is nn.Embedding

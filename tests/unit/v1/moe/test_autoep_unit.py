# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Compact critical-path tests for AutoEP."""

import ast
import inspect
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import deepspeed.runtime.engine as ds_engine
import deepspeed.runtime.zero.stage3 as zero_stage3
import deepspeed.moe.ep_repack as ep_repack
from deepspeed.module_inject.auto_ep import AutoEP, _resolve_route_scale
from deepspeed.module_inject.auto_ep_config import (
    AutoEPConfig,
    MoELayerSpec,
    PRESET_MODELS,
    fill_autoep_config_from_hf,
    parse_autoep_config,
    validate_autoep_config,
    validate_autoep_post_detection,
)
from deepspeed.module_inject.auto_ep_layer import (
    AutoEPMoELayer,
    apply_scores_before_experts_if_enabled,
    combine_from_routed,
    resolve_score_apply_mode,
)
from deepspeed.module_inject.auto_ep_preset_adapters import get_preset_adapter
from deepspeed.module_inject.auto_ep_presets.registry import (
    preset_name_for_hf_model_type,
    unsupported_preset_for_hf_model_type,
)
from deepspeed.moe.layer import MoE
from deepspeed.moe.ep_experts import GroupedExperts
from deepspeed.moe.ep_kernels import TokenReorderer
from deepspeed.moe.ep_repack import repack_expert_weights
from deepspeed.moe.ep_router import TokenChoiceTopKRouter
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.utils import groups
from unit.v1.moe.autoep_test_utils import (
    MockMoEBlock,
    MockMoETransformer,
    UNSUPPORTED_LOAD_BALANCE_VALUES,
    assert_causal_lm_outputs_close,
    assert_load_balance_coeff_rejection_message,
    replace_autoep_layers,
    skip_unless_transformers_has,
    state_matched_models,
    tiny_mixtral_config,
)


def _runtime_config(**kwargs):
    kwargs.setdefault("use_grouped_mm", False)
    return AutoEPConfig(**kwargs)


def _make_spec(**kwargs):
    defaults = dict(
        moe_module_name="model.layers.0.mlp",
        model_family="mixtral",
        router_name="gate",
        experts_name="experts",
        expert_storage="fused_3d",
        expert_w1_name="gate_up_proj",
        expert_w2_name="down_proj",
        expert_w3_name=None,
        num_experts=4,
        top_k=2,
        hidden_size=64,
        ffn_hidden_size=128,
        score_func="softmax",
        score_apply="post",
        route_norm=True,
        gate_bias=False,
        return_router_logits=False,
        router_logits_capture_target="none",
        router_logits_capture_index=None,
        router_logits_capture_layer_name=None,
        has_shared_experts=False,
        shared_experts_name="",
        shared_experts_gate_name="",
    )
    defaults.update(kwargs)
    return MoELayerSpec(**defaults)


def _assert_same_dtype_device(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device


def _mark_fake_zero_param(param, full_data, partition_data=None, ds_id=0, name="param"):
    param.ds_id = ds_id
    param.ds_shape = torch.Size(full_data.shape)
    param._autoep_test_full_data = full_data.detach().clone()
    param._autoep_test_name = name
    if partition_data is None:
        partition_data = torch.zeros(1, dtype=full_data.dtype, device=full_data.device)
    param.data = partition_data.detach().clone()
    return param


class FakeGatheredParameters:
    calls = []

    def __init__(self, params, modifier_rank=None, fwd_module=None, enabled=True):
        self.params = list(params)
        self.modifier_rank = modifier_rank
        self.enabled = enabled
        self._saved_data = []
        FakeGatheredParameters.calls.append({
            "names": [getattr(param, "_autoep_test_name", f"param{param.ds_id}") for param in self.params],
            "modifier_rank":
            modifier_rank,
            "enabled":
            enabled,
        })

    def __enter__(self):
        if not self.enabled:
            return
        for param in self.params:
            self._saved_data.append((param, param.data))
            param.data = param._autoep_test_full_data.detach().clone()

    def __exit__(self, *exc):
        if not self.enabled:
            return
        for param, data in self._saved_data:
            param.data = data


class MockSharedExpert(nn.Module):

    def __init__(self, hidden_size=64):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class MockDeepSeekV3Config:
    model_type = "deepseek_v3"
    n_routed_experts = 8
    num_experts_per_tok = 2
    hidden_size = 64
    moe_intermediate_size = 128
    n_group = 4
    topk_group = 2
    routed_scaling_factor = 2.5


class MockDeepSeekV3Expert(nn.Module):

    def __init__(self, hidden_size=64, ffn_hidden=128):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden, bias=False)
        self.down_proj = nn.Linear(ffn_hidden, hidden_size, bias=False)


class MockDeepSeekV3MoEBlock(nn.Module):

    def __init__(self, num_experts=8, ffn_hidden=128, hidden_size=64):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([MockDeepSeekV3Expert(hidden_size, ffn_hidden) for _ in range(num_experts)])
        self.shared_experts = MockSharedExpert(hidden_size)


class MockDeepSeekV3Transformer(nn.Module):

    def __init__(self, num_layers=2, num_experts=8):
        super().__init__()
        self.config = MockDeepSeekV3Config()
        self.config.n_routed_experts = num_experts
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([self._make_layer(num_experts) for _ in range(num_layers)])

    @staticmethod
    def _make_layer(num_experts):
        layer = nn.Module()
        layer.mlp = MockDeepSeekV3MoEBlock(num_experts)
        return layer


class TestAutoEPConfig:

    def test_parse_and_validate_enabled_size_contract(self):
        disabled = parse_autoep_config({})
        assert disabled.enabled is False
        assert disabled.autoep_size == 1
        assert disabled.validate_folding_routing is False
        assert disabled.load_balance_coeff is None
        assert disabled._load_balance_coeff_explicit is False

        config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 4,
            "preset_model": "mixtral",
            "load_balance_coeff": None,
            "score_apply": "pre",
            "route_scale": 2.0,
            "validate_folding_routing": True,
        })

        assert config.enabled is True
        assert config.autoep_size == 4
        assert config.preset_model == "mixtral"
        assert config.validate_folding_routing is True
        assert config.load_balance_coeff is None
        assert config._load_balance_coeff_explicit is True
        assert config.score_apply == "pre"
        assert config.route_scale == 2.0
        validate_autoep_config(config, world_size=4, pp_size=1, tp_size=1, sp_size=1)

    def test_validate_folding_routing_requires_boolean(self):
        with pytest.raises(ValueError, match="validate_folding_routing"):
            validate_autoep_config(AutoEPConfig(enabled=True, validate_folding_routing="true"),
                                   world_size=1,
                                   pp_size=1,
                                   tp_size=1,
                                   sp_size=1)

    @pytest.mark.parametrize("value", UNSUPPORTED_LOAD_BALANCE_VALUES)
    def test_load_balance_coeff_rejected_at_parse(self, value):
        with pytest.raises(ValueError) as exc_info:
            parse_autoep_config({"enabled": True, "load_balance_coeff": value})
        assert_load_balance_coeff_rejection_message(exc_info.value, value)

    @pytest.mark.parametrize("enabled", [True, False])
    @pytest.mark.parametrize("value", [0.01, False, "0.01"])
    def test_load_balance_coeff_rejected_by_validate(self, enabled, value):
        config = AutoEPConfig(enabled=enabled, load_balance_coeff=value)

        with pytest.raises(ValueError) as exc_info:
            validate_autoep_config(config, world_size=1, pp_size=1, tp_size=1, sp_size=1)
        assert_load_balance_coeff_rejection_message(exc_info.value, value)

    def test_ep_size_validation_rejects_invalid_topology(self):
        validate_autoep_config(AutoEPConfig(enabled=True, autoep_size=2),
                               world_size=8,
                               pp_size=1,
                               tp_size=2,
                               sp_size=1)
        with pytest.raises(ValueError, match="must divide the stage size"):
            validate_autoep_config(AutoEPConfig(enabled=True, autoep_size=3),
                                   world_size=8,
                                   pp_size=1,
                                   tp_size=1,
                                   sp_size=1)
        with pytest.raises(ValueError, match="exceeds num_experts"):
            validate_autoep_post_detection(AutoEPConfig(enabled=True, autoep_size=16), [_make_spec(num_experts=8)])

    def test_expert_tensor_parallel_size_is_parsed_but_limited_to_one(self):
        config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 2,
            "expert_tensor_parallel_size": 1,
        })
        assert config.expert_tensor_parallel_size == 1

        config.expert_tensor_parallel_size = 2
        with pytest.raises(ValueError, match="expert_tensor_parallel_size=1"):
            validate_autoep_config(config, world_size=4, pp_size=1, tp_size=1, sp_size=1)

    def test_configure_expert_parallel_uses_engine_mpu_sequence_parallel_size(self, monkeypatch):

        class SequenceParallelMPU:

            def get_model_parallel_world_size(self):
                return 1

            def get_sequence_parallel_world_size(self):
                return 2

        class EmptyAutoEP:

            def __init__(self, model, config):
                pass

            def ep_parser(self):
                return []

        observed = {}

        def record_validate(config, world_size, pp_size, tp_size, sp_size):
            observed["validate"] = {
                "world_size": world_size,
                "pp_size": pp_size,
                "tp_size": tp_size,
                "sp_size": sp_size,
            }

        def record_create(**kwargs):
            observed["create"] = kwargs

        monkeypatch.setattr(groups, "mpu", None)
        monkeypatch.setattr(groups, "_get_sequence_parallel_world_size", lambda: 1)
        monkeypatch.setattr(groups, "_create_expert_and_data_parallel", record_create)
        monkeypatch.setattr(groups, "_get_expert_parallel_group", lambda name: object())
        monkeypatch.setattr(ds_engine.dist, "get_world_size", lambda: 4)
        monkeypatch.setattr(ds_engine.dist, "get_rank", lambda group=None: 0)
        monkeypatch.setattr("deepspeed.module_inject.auto_ep.AutoEP", EmptyAutoEP)
        monkeypatch.setattr("deepspeed.module_inject.auto_ep_config.validate_autoep_config", record_validate)

        engine = object.__new__(DeepSpeedEngine)
        engine.mpu = SequenceParallelMPU()
        engine._config = SimpleNamespace(
            expert_parallel_config=AutoEPConfig(enabled=True, autoep_size=2),
            tensor_parallel_config=SimpleNamespace(autotp_size=1),
            use_data_before_expert_parallel_=False,
            zero_config=SimpleNamespace(offload_optimizer=None, offload_param=None),
            zero_optimization_stage=0,
        )

        engine._configure_expert_parallel(model=nn.Module())

        assert groups.mpu is None
        assert observed["validate"]["sp_size"] == 2
        assert observed["create"]["mp_size"] == 2
        assert observed["create"]["mp_mode"] == "sp"

    def test_configure_expert_parallel_rejects_bwc_tensor_model_parallel_mpu(self, monkeypatch):

        class TensorParallelMPU:

            def get_tensor_model_parallel_world_size(self):
                return 2

        monkeypatch.setattr(groups, "_get_sequence_parallel_world_size", lambda: 1)

        engine = object.__new__(DeepSpeedEngine)
        engine.mpu = TensorParallelMPU()
        engine._config = SimpleNamespace(
            expert_parallel_config=AutoEPConfig(enabled=True, autoep_size=2),
            tensor_parallel_config=SimpleNamespace(autotp_size=1),
            use_data_before_expert_parallel_=False,
        )

        with pytest.raises(ValueError, match="bwc_tensor_model_parallel_world_size=2"):
            engine._configure_expert_parallel(model=nn.Module())

    def test_autoep_sequence_parallel_size_falls_back_to_groups_helper(self, monkeypatch):
        monkeypatch.setattr(groups, "_get_sequence_parallel_world_size", lambda: 3)

        engine = object.__new__(DeepSpeedEngine)
        engine.mpu = object()

        assert engine._autoep_sequence_parallel_world_size() == 3

    def test_zero3_compatibility_gate_rejects_native_moe(self):
        engine = object.__new__(DeepSpeedEngine)
        engine.__dict__["module"] = nn.Sequential(MoE(hidden_size=4, expert=nn.Linear(4, 4), num_experts=1))
        engine.has_moe_layers = True
        engine.sequence_parallel_size = 1
        engine._config = SimpleNamespace(
            tensor_parallel_config=SimpleNamespace(autotp_size=1),
            expert_parallel_config=AutoEPConfig(enabled=True, autoep_size=1),
        )

        with pytest.raises(AssertionError, match="Native DeepSpeed MoE"):
            engine._validate_zero3_moe_compatibility()

    def test_zero3_compatibility_gate_allows_constrained_autoep(self):
        model = MockMoETransformer(num_layers=1)
        replace_autoep_layers(model, "mixtral")
        engine = object.__new__(DeepSpeedEngine)
        engine.__dict__["module"] = model
        engine.has_moe_layers = True
        engine.sequence_parallel_size = 1
        engine.zero_quantized_gradients = lambda: False
        engine._config = SimpleNamespace(
            tensor_parallel_config=SimpleNamespace(autotp_size=1),
            expert_parallel_config=AutoEPConfig(enabled=True, autoep_size=1),
        )

        engine._validate_zero3_moe_compatibility()

    def test_zero3_compatibility_gate_rejects_sequence_parallel(self):
        model = MockMoETransformer(num_layers=1)
        replace_autoep_layers(model, "mixtral")
        engine = object.__new__(DeepSpeedEngine)
        engine.__dict__["module"] = model
        engine.has_moe_layers = True
        engine.sequence_parallel_size = 2
        engine.zero_quantized_gradients = lambda: False
        engine._config = SimpleNamespace(
            tensor_parallel_config=SimpleNamespace(autotp_size=1),
            expert_parallel_config=AutoEPConfig(enabled=True, autoep_size=1),
        )

        with pytest.raises(AssertionError, match="sequence parallelism"):
            engine._validate_zero3_moe_compatibility()

    def test_zero3_compatibility_gate_rejects_active_autotp(self):
        model = MockMoETransformer(num_layers=1)
        replace_autoep_layers(model, "mixtral")
        engine = object.__new__(DeepSpeedEngine)
        engine.__dict__["module"] = model
        engine.has_moe_layers = True
        engine.sequence_parallel_size = 1
        engine.zero_quantized_gradients = lambda: False
        engine._config = SimpleNamespace(
            tensor_parallel_config=SimpleNamespace(autotp_size=2),
            expert_parallel_config=AutoEPConfig(enabled=True, autoep_size=1),
        )

        with pytest.raises(AssertionError, match="AutoTP"):
            engine._validate_zero3_moe_compatibility()

    def test_zero3_compatibility_gate_rejects_quantized_gradients(self):
        model = MockMoETransformer(num_layers=1)
        replace_autoep_layers(model, "mixtral")
        engine = object.__new__(DeepSpeedEngine)
        engine.__dict__["module"] = model
        engine.has_moe_layers = True
        engine.sequence_parallel_size = 1
        engine.zero_quantized_gradients = lambda: True
        engine._config = SimpleNamespace(
            tensor_parallel_config=SimpleNamespace(autotp_size=1),
            expert_parallel_config=AutoEPConfig(enabled=True, autoep_size=1),
        )

        with pytest.raises(AssertionError, match="zero_quantized_gradients"):
            engine._validate_zero3_moe_compatibility()

    def test_zero3_compatibility_gate_rejects_mics(self):
        model = MockMoETransformer(num_layers=1)
        replace_autoep_layers(model, "mixtral")
        engine = object.__new__(DeepSpeedEngine)
        engine.__dict__["module"] = model
        engine.has_moe_layers = True
        engine.sequence_parallel_size = 1
        engine.zero_quantized_gradients = lambda: False
        engine._config = SimpleNamespace(
            mics_shard_size=2,
            zero_config=SimpleNamespace(zero_hpz_partition_size=1),
            tensor_parallel_config=SimpleNamespace(autotp_size=1),
            expert_parallel_config=AutoEPConfig(enabled=True, autoep_size=1),
        )

        with pytest.raises(AssertionError, match="MiCS"):
            engine._validate_zero3_moe_compatibility()

    def test_zero3_compatibility_gate_rejects_hpzero(self):
        model = MockMoETransformer(num_layers=1)
        replace_autoep_layers(model, "mixtral")
        engine = object.__new__(DeepSpeedEngine)
        engine.__dict__["module"] = model
        engine.has_moe_layers = True
        engine.sequence_parallel_size = 1
        engine.zero_quantized_gradients = lambda: False
        engine._config = SimpleNamespace(
            mics_shard_size=0,
            zero_config=SimpleNamespace(zero_hpz_partition_size=2),
            tensor_parallel_config=SimpleNamespace(autotp_size=1),
            expert_parallel_config=AutoEPConfig(enabled=True, autoep_size=1),
        )

        with pytest.raises(AssertionError, match="hpZeRO"):
            engine._validate_zero3_moe_compatibility()

    def test_autoep_layer_marks_zero3_param_placement_families(self):
        model = MockMoETransformer(num_layers=1)
        replace_autoep_layers(model, "mixtral")
        autoep_layer = next(module for module in model.modules() if isinstance(module, AutoEPMoELayer))

        for param in autoep_layer.experts.parameters():
            assert param.ds_zero_placement_family == "autoep_expert"
            assert param.ds_zero_partition_group_name == autoep_layer.ep_group_name

        for param in autoep_layer.router.parameters():
            assert param.ds_zero_placement_family == "replicated"

    def test_zero3_checkpoint_metadata_includes_partition_group_ranks(self):
        optimizer = object.__new__(DeepSpeedZeroOptimizer_Stage3)
        param = nn.Parameter(torch.empty(1))
        param.ds_zero_placement_family = "autoep_expert"
        param.ds_zero_partition_group_name = "ep_size_2"
        optimizer.fp16_groups = [[param]]
        optimizer._get_sub_group_partition_count = lambda _: 2
        optimizer._get_sub_group_partition_rank = lambda _: 1
        optimizer._get_sub_group_partition_ranks = lambda _: [1, 3]

        metadata = optimizer._zero3_partition_group_metadata()

        assert metadata == [{
            "sub_group": 0,
            "partition_count": 2,
            "partition_rank": 1,
            "partition_ranks": [1, 3],
            "families": ["autoep_expert"],
            "group_names": ["ep_size_2"],
        }]

        param.ds_zero_placement_family = "replicated"
        param.ds_zero_partition_group_name = None
        assert optimizer._zero3_partition_group_metadata() is None

    def test_zero3_cpu_offload_grad_norm_reduces_autoep_expert_parallel_group(self, monkeypatch):
        optimizer = object.__new__(DeepSpeedZeroOptimizer_Stage3)
        param = nn.Parameter(torch.empty(1))
        param.ds_zero_placement_family = "autoep_expert"
        param.ds_zero_partition_group_name = "ep_size_2"
        optimizer.model_parallel_rank = 0
        optimizer.norm_for_param_grads = {7: 3.0}
        optimizer.get_param_id = lambda _: 7
        optimizer._assert_same_partition_group = lambda _: None
        optimizer._get_param_partition_group = lambda _: "expert_data_parallel"
        optimizer._model_parallel_all_reduce = lambda tensor, op: None
        optimizer._autoep_expert_parallel_group = lambda _: "expert_parallel"
        calls = []

        def fake_all_reduce(tensor, op=None, group=None):
            calls.append(group)

        class FakeAccelerator:

            def FloatTensor(self, values):
                return torch.FloatTensor(values)

        monkeypatch.setattr(zero_stage3, "get_accelerator", lambda: FakeAccelerator())
        monkeypatch.setattr(zero_stage3.dist, "all_reduce", fake_all_reduce)

        norm = optimizer.complete_grad_norm_calculation_for_cpu_offload([param])

        assert calls == ["expert_data_parallel", "expert_parallel"]
        assert torch.isfinite(norm)

    def test_zero3_autoep_reduce_scatter_grads_average_by_global_dp(self, monkeypatch):
        optimizer = object.__new__(DeepSpeedZeroOptimizer_Stage3)
        optimizer.dp_process_group = "global_data_parallel"
        optimizer.dtype = torch.float32
        optimizer.gradient_accumulation_dtype = torch.float32
        optimizer.postscale_gradients = True
        optimizer.gradient_predivide_factor = 1.0
        optimizer.all2all_process_group = None
        optimizer._assert_same_partition_group = lambda _: None
        optimizer._get_param_partition_group = lambda _: "expert_data_parallel"
        optimizer._autoep_expert_parallel_group = lambda _: "expert_parallel"
        param = nn.Parameter(torch.ones(4))
        param.grad = torch.ones(4)

        class FakeAccelerator:

            def device_count(self):
                return 4

        def fake_get_world_size(group=None):
            return 2 if group == "expert_data_parallel" else 4

        def fake_reduce_scatter(grads, process_group):
            assert process_group == "expert_data_parallel"
            return [torch.full((2, ), 8.0)]

        monkeypatch.setattr(zero_stage3, "get_accelerator", lambda: FakeAccelerator())
        monkeypatch.setattr(zero_stage3.dist, "get_world_size", fake_get_world_size)
        monkeypatch.setattr(zero_stage3, "reduce_scatter_coalesced", fake_reduce_scatter)

        grad_partitions = optimizer._DeepSpeedZeroOptimizer_Stage3__avg_scatter_grads([param], torch.float32)

        torch.testing.assert_close(grad_partitions[0], torch.full((2, ), 4.0))

    def test_zero3_autoep_contiguous_grads_average_by_global_dp(self, monkeypatch):
        optimizer = object.__new__(DeepSpeedZeroOptimizer_Stage3)
        optimizer.dp_process_group = "global_data_parallel"
        optimizer.ipg_buckets = {torch.float32: SimpleNamespace(params=[], process_group="expert_data_parallel")}
        optimizer.postscale_gradients = True
        optimizer.gradient_predivide_factor = 1.0
        optimizer.sequence_parallel_size = 1
        optimizer.gradient_accumulation_dtype = torch.float32
        optimizer._assert_same_partition_group = lambda _: None
        optimizer._autoep_expert_parallel_group = lambda _: "expert_parallel"
        optimizer._apply_distributed_muon_update = lambda communication_data_type, buffer: None
        param = nn.Parameter(torch.empty(2))
        param.grad = torch.zeros(2)
        param.partition_numel = lambda: 1
        optimizer.ipg_buckets[torch.float32].params = [param]

        def fake_get_world_size(group=None):
            return 2 if group == "expert_data_parallel" else 4

        def fake_all_reduce(tensor, group=None):
            assert group == "expert_data_parallel"
            tensor.mul_(2)

        monkeypatch.setattr(zero_stage3.dist, "get_world_size", fake_get_world_size)
        monkeypatch.setattr(zero_stage3.dist, "get_rank", lambda group=None: 0)
        monkeypatch.setattr(zero_stage3.dist, "all_reduce", fake_all_reduce)

        grad_partitions = optimizer._DeepSpeedZeroOptimizer_Stage3__avg_scatter_contiguous_grads(
            torch.tensor([4.0, 8.0]), torch.float32)

        torch.testing.assert_close(grad_partitions[0], torch.tensor([2.0]))

    def test_pipeline_load_module_state_dict_accepts_autoep_zero3_fetch_kwarg(self):
        from deepspeed.runtime.pipe.engine import PipelineEngine

        signature = inspect.signature(PipelineEngine.load_module_state_dict)

        assert "z3_params_to_fetch" in signature.parameters
        assert "allowed_missing_keys" in signature.parameters

    def test_load_module_state_dict_nonstrict_keeps_nonstrict_semantics_with_allowed_missing_keys(self):
        engine = object.__new__(DeepSpeedEngine)
        # bypass nn.Module.__setattr__, which requires Module.__init__
        object.__setattr__(engine, "module", nn.Linear(2, 2))
        checkpoint = {"module": {"unexpected_key": torch.zeros(1)}}

        # strict=False must keep the documented non-strict load semantics even
        # when AutoEP expert keys are allowed to be missing.
        engine.load_module_state_dict(checkpoint, strict=False, allowed_missing_keys=["weight"])

        with pytest.raises(RuntimeError, match="outside AutoEP expert"):
            engine.load_module_state_dict(checkpoint, strict=True, allowed_missing_keys=["weight"])

    def test_resolve_zero3_param_placement_rejects_pre_partitioned_expert_on_wrong_group(self, monkeypatch):
        engine = object.__new__(DeepSpeedEngine)
        model = nn.Linear(2, 2, bias=False)
        # bypass nn.Module.__setattr__, which requires Module.__init__
        object.__setattr__(engine, "module", model)

        expert_group = object()
        other_group = object()
        monkeypatch.setattr(ds_engine.groups, "_get_expert_data_parallel_group", lambda name: expert_group)
        monkeypatch.setattr(ds_engine.dist, "get_rank", lambda group=None: 0)
        monkeypatch.setattr(ds_engine.dist, "get_world_size", lambda group=None: 1)
        monkeypatch.setattr(ds_engine.dist,
                            "get_all_ranks_from_group",
                            lambda group: [0] if group is expert_group else [0, 1],
                            raising=False)

        param = model.weight
        param.ds_zero_placement_family = "autoep_expert"
        param.ds_zero_partition_group_name = "ep_size_2"
        param.ds_id = 0
        param.ds_process_group = other_group

        with pytest.raises(AssertionError, match="already ZeRO-partitioned over a non-expert process group"):
            engine._resolve_zero3_param_placement()

        # A pre-partitioned expert param over the matching group is accepted
        # and keeps metadata derived from its actual partition group.
        param.ds_process_group = expert_group
        engine._resolve_zero3_param_placement()
        assert param.ds_zero_partition_process_group is expert_group

    def test_autoep_zero3_16bit_export_guard_directs_to_universal_conversion(self):
        engine = object.__new__(DeepSpeedEngine)
        engine.zero_optimization_partition_weights = lambda: True
        engine._has_autoep_layers = lambda: True

        with pytest.raises(NotImplementedError, match="ds_to_universal.py"):
            engine._raise_if_autoep_zero3_consolidated_export("save_16bit_model")

    def test_universal_converter_detects_zero3_partitioned_autoep_model_state(self, tmp_path):
        from deepspeed.checkpoint.constants import (
            AUTOEP_LAYERS_KEY,
            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION,
            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION_KEY,
            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_KEY,
            AUTOEP_ZERO3_PARTITIONED_EXPERT_STATE_FORMAT,
        )
        from deepspeed.checkpoint.ds_to_universal import (
            _autoep_expert_param_names_by_rank,
            _get_zero3_model_state_files,
            _uses_zero3_partitioned_autoep_metadata,
        )

        zero3_model_file = tmp_path / "zero_pp_rank_0_mp_rank_00_model_states.pt"
        expert_file = tmp_path / "layer_0_expert_0_mp_rank_00_model_states.pt"
        metadata = [{
            "moe_layer_id": 0,
            "module_path": "model.layers.0.mlp",
            "num_experts": 4,
            "num_local_experts": 2,
            "ep_size": 2,
            "expert_key_prefix": "model.layers.0.mlp.experts",
            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_KEY: AUTOEP_ZERO3_PARTITIONED_EXPERT_STATE_FORMAT,
            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION_KEY: AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION,
            "ep_group_name": "ep_size_2",
            "ep_rank": 0,
            "expert_data_parallel_rank": 0,
            "expert_data_parallel_world_size": 1,
            "global_expert_start": 0,
            "global_expert_end": 2,
        }]
        torch.save({AUTOEP_LAYERS_KEY: metadata}, zero3_model_file)
        torch.save({"expert": torch.empty(1)}, expert_file)

        model_files = _get_zero3_model_state_files(str(tmp_path))
        expert_param_names, metadata_by_rank = _autoep_expert_param_names_by_rank(model_files)

        assert model_files == [str(zero3_model_file)]
        assert expert_param_names == {
            "model.layers.0.mlp.experts.w1",
            "model.layers.0.mlp.experts.w2",
            "model.layers.0.mlp.experts.w3",
        }
        assert _uses_zero3_partitioned_autoep_metadata(metadata_by_rank[0])

    def test_universal_stage3_extract_accepts_tuple_param_shapes(self, tmp_path):
        from deepspeed.checkpoint.constants import OPTIMIZER_STATE_DICT
        from deepspeed.checkpoint.ds_to_universal import extract_zero_shards_stage3

        optim_file = tmp_path / "zero_pp_rank_0_mp_rank_00_optim_states.pt"
        torch.save(
            {
                OPTIMIZER_STATE_DICT: {
                    "optimizer_state_dict": {
                        "state": [{
                            "exp_avg": torch.arange(6, dtype=torch.float32),
                            "exp_avg_sq": torch.arange(6, dtype=torch.float32) + 10,
                        }]
                    },
                    "fp32_flat_groups": [torch.arange(6, dtype=torch.float32) + 20],
                }
            },
            optim_file,
        )

        temp_dir = tmp_path / "tmp"
        extract_zero_shards_stage3([str(optim_file)], [OrderedDict([("dense.weight", (2, 3))])], 1, str(temp_dir), 0)

        fp32_fragment = torch.load(temp_dir / "dense.weight" / "0" / "fp32.00", weights_only=False)
        exp_avg_fragment = torch.load(temp_dir / "dense.weight" / "0" / "exp_avg.00", weights_only=False)
        torch.testing.assert_close(fp32_fragment, torch.arange(6, dtype=torch.float32) + 20)
        torch.testing.assert_close(exp_avg_fragment, torch.arange(6, dtype=torch.float32))

    def test_zero_to_fp32_rejects_zero3_partitioned_autoep_checkpoint(self, tmp_path):
        from deepspeed.checkpoint.constants import (
            AUTOEP_LAYERS_KEY,
            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION,
            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION_KEY,
            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_KEY,
            AUTOEP_ZERO3_PARTITIONED_EXPERT_STATE_FORMAT,
            BUFFER_NAMES,
            PARAM_SHAPES,
        )
        from deepspeed.utils.zero_to_fp32 import _raise_if_autoep_zero3_partitioned_checkpoint

        model_file = tmp_path / "zero_pp_rank_0_mp_rank_00_model_states.pt"
        torch.save(
            {
                BUFFER_NAMES: [],
                PARAM_SHAPES: [],
                "module": {},
                "shared_params": {},
                AUTOEP_LAYERS_KEY: [{
                    "moe_layer_id": 0,
                    "module_path": "model.layers.0.mlp",
                    "num_experts": 4,
                    "num_local_experts": 2,
                    "ep_size": 2,
                    "expert_key_prefix": "model.layers.0.mlp.experts",
                    AUTOEP_ZERO3_EXPERT_STATE_FORMAT_KEY: AUTOEP_ZERO3_PARTITIONED_EXPERT_STATE_FORMAT,
                    AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION_KEY: AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION,
                    "ep_group_name": "ep_size_2",
                    "ep_rank": 0,
                    "expert_data_parallel_rank": 0,
                    "expert_data_parallel_world_size": 1,
                    "global_expert_start": 0,
                    "global_expert_end": 2,
                }],
            },
            model_file,
        )

        with pytest.raises(NotImplementedError, match="ds_to_universal.py"):
            _raise_if_autoep_zero3_partitioned_checkpoint([str(model_file)])

        # parse_model_states is the guard point used by
        # _get_fp32_state_dict_from_zero_checkpoint, which loads each model
        # state file only once.
        from deepspeed.utils.zero_to_fp32 import parse_model_states
        with pytest.raises(NotImplementedError, match="ds_to_universal.py"):
            parse_model_states([str(model_file)])

    def test_preset_registry_core_contracts(self):
        assert set(PRESET_MODELS) == {"mixtral", "qwen3_moe", "qwen3_5_moe", "deepseek_v2", "deepseek_v3"}
        assert preset_name_for_hf_model_type("mixtral") == "mixtral"
        assert preset_name_for_hf_model_type("qwen2_moe") == "qwen3_moe"
        assert preset_name_for_hf_model_type("llama4_text") is None

        qwen35 = unsupported_preset_for_hf_model_type("qwen3_5_moe")
        assert qwen35 is not None
        assert "qwen3_5_moe_text" in qwen35[1].unsupported_hf_model_type_notes["qwen3_5_moe"]
        assert PRESET_MODELS["deepseek_v2"].supports_expert_bias is False
        assert PRESET_MODELS["deepseek_v3"].unsupported_router_bias_names == ()

    def test_fill_autoep_config_from_hf_defaults(self):
        config = AutoEPConfig(enabled=True, autoep_size=2)

        fill_autoep_config_from_hf(config, MockDeepSeekV3Config())

        assert config.num_expert_groups == 4
        assert config.num_limited_groups == 2
        assert config.route_scale == pytest.approx(2.5)

    def test_fill_autoep_config_from_hf_preserves_explicit_values(self):
        config = AutoEPConfig(enabled=True,
                              autoep_size=2,
                              num_expert_groups=8,
                              num_limited_groups=1,
                              routed_scaling_factor=3.0,
                              route_scale=3.0)

        fill_autoep_config_from_hf(config, MockDeepSeekV3Config())

        assert config.num_expert_groups == 8
        assert config.num_limited_groups == 1
        assert config.route_scale == pytest.approx(3.0)

    @pytest.mark.parametrize("value", ["2.5", True, float("nan"), float("inf")])
    def test_invalid_routed_scaling_factor_rejected(self, value):
        with pytest.raises(ValueError, match="routed_scaling_factor"):
            _resolve_route_scale(AutoEPConfig(enabled=True, routed_scaling_factor=value), None)


class TestRoutingAndLayerSemantics:

    def test_router_route_scale_and_group_limited_routing(self):
        base = TokenChoiceTopKRouter(64, 8, 4, 2, 2, "softmax", False, 1.0, False)
        scaled = TokenChoiceTopKRouter(64, 8, 4, 2, 2, "softmax", False, 2.5, False)
        scaled.load_state_dict(base.state_dict())
        x = torch.randn(50, 64)

        base_scores, base_experts, base_counts = base(x)
        scaled_scores, scaled_experts, scaled_counts = scaled(x)

        assert torch.equal(scaled_experts, base_experts)
        assert torch.allclose(scaled_scores, base_scores * 2.5, atol=1e-5)
        assert torch.equal(scaled_counts, base_counts)
        assert base_counts.shape == (8, )

    def test_grouped_experts_and_token_reorderer(self):
        experts = GroupedExperts(dim=64, hidden_dim=128, num_experts=4, use_grouped_mm=False)
        nn.init.normal_(experts.w1, std=0.02)
        nn.init.normal_(experts.w2, std=0.02)
        nn.init.normal_(experts.w3, std=0.02)
        out = experts(torch.randn(8, 64), torch.tensor([2, 2, 2, 2]))
        assert out.shape == (8, 64)
        assert not torch.isnan(out).any()

        top_scores = torch.randn(20, 2)
        selected_experts = torch.randint(0, 4, (20, 2))
        scores_sorted, indices_sorted, counts = TokenReorderer(num_experts=4, top_k=2)(top_scores, selected_experts)
        assert scores_sorted.shape == (40, )
        assert set(indices_sorted.tolist()) == set(range(40))
        assert torch.equal(counts, torch.bincount(selected_experts.reshape(-1), minlength=4).to(counts.dtype))

    def test_score_application_and_combine(self):
        x = torch.randn(4, 8)
        scores = torch.tensor([0.25, 0.5, 0.75, 1.0])
        expected = x.float() * scores.reshape(-1, 1)
        torch.testing.assert_close(apply_scores_before_experts_if_enabled(x, scores, "pre"), expected.to(x.dtype))

        spec = _make_spec(score_apply="post")
        assert resolve_score_apply_mode(spec, "auto") == "post"
        expert_output = torch.ones(4, 8)
        top_scores = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
        out = combine_from_routed(expert_output, top_scores, torch.arange(4), 2, "post", "weighted_sum", (1, 2, 8))
        torch.testing.assert_close(out[0, 0], torch.ones(8))

    def test_autoep_layer_forward_and_expert_bias_rejection(self):
        source = MockMoEBlock(num_experts=4, ffn_hidden=128, hidden_size=64)
        layer = AutoEPMoELayer(_make_spec(route_scale=2.5),
                               source,
                               ep_size=1,
                               ep_rank=0,
                               config=_runtime_config(enabled=True, autoep_size=1))
        out = layer(torch.randn(2, 8, 64))
        assert layer._is_autoep_layer is True
        assert layer.num_experts == 4
        assert layer.router.route_scale == pytest.approx(2.5)
        assert out.shape == (2, 8, 64)
        assert not torch.isnan(out).any()

        with pytest.raises(ValueError, match="load_balance_coeff/expert_bias"):
            AutoEPMoELayer(_make_spec(model_family="no_bias_family", supports_expert_bias=False),
                           source,
                           ep_size=1,
                           ep_rank=0,
                           config=AutoEPConfig(enabled=True, autoep_size=1, load_balance_coeff=0.02))


class TestModelDetectionAndReplacement:

    def test_mixtral_detect_replace_and_mock_forward(self):
        model = MockMoETransformer(num_layers=2, moe_every_n=1)
        auto_ep = AutoEP(model, _runtime_config(enabled=True, autoep_size=1, preset_model="mixtral"))
        specs = auto_ep.ep_parser()

        assert len(specs) == 2
        assert specs[0].model_family == "mixtral"
        auto_ep.replace_moe_layer(specs[0], ep_size=1, ep_rank=0)
        assert isinstance(model.model.layers[0].mlp, AutoEPMoELayer)
        assert model(torch.randn(1, 4, 64)).shape == (1, 4, 100)

    def test_fused_replacement_preserves_frozen_experts_and_trainable_router(self):
        model = MockMoETransformer(num_layers=1, num_experts=4, moe_every_n=1).to(dtype=torch.bfloat16)
        source = model.model.layers[0].mlp
        source.experts.gate_up_proj.requires_grad_(False)
        source.experts.down_proj.requires_grad_(False)
        source.gate.weight.requires_grad_(True)

        auto_ep = AutoEP(model, _runtime_config(enabled=True, autoep_size=1, preset_model="mixtral"))
        spec = auto_ep.ep_parser()[0]
        auto_ep.replace_moe_layer(spec, ep_size=1, ep_rank=0)

        replaced = model.model.layers[0].mlp
        assert isinstance(replaced, AutoEPMoELayer)
        assert replaced.experts.w1.requires_grad is False
        assert replaced.experts.w2.requires_grad is False
        assert replaced.experts.w3.requires_grad is False
        assert replaced.router.gate.weight.requires_grad is True
        _assert_same_dtype_device(replaced.router.gate.weight, source.gate.weight)
        _assert_same_dtype_device(replaced.experts.w1, source.experts.gate_up_proj)
        _assert_same_dtype_device(replaced.experts.w2, source.experts.down_proj)
        _assert_same_dtype_device(replaced.experts.w3, source.experts.gate_up_proj)

    def test_zero_init_source_gathered_for_parser_router_and_fused_repack(self, monkeypatch):
        FakeGatheredParameters.calls = []
        monkeypatch.setattr(ep_repack, "GatheredParameters", FakeGatheredParameters)

        model = MockMoETransformer(num_layers=1, num_experts=4, moe_every_n=1)
        source = model.model.layers[0].mlp
        expected_gate = source.gate.weight.detach().clone()
        expected_gate_up = source.experts.gate_up_proj.detach().clone()
        expected_down = source.experts.down_proj.detach().clone()

        _mark_fake_zero_param(source.gate.weight, expected_gate, ds_id=1, name="router.weight")
        _mark_fake_zero_param(source.experts.gate_up_proj, expected_gate_up, ds_id=2, name="experts.gate_up_proj")
        _mark_fake_zero_param(source.experts.down_proj, expected_down, ds_id=3, name="experts.down_proj")

        auto_ep = AutoEP(model, _runtime_config(enabled=True, autoep_size=1, preset_model="mixtral"))
        specs = auto_ep.ep_parser()
        assert len(specs) == 1
        assert specs[0].expert_storage == "fused_3d"
        assert specs[0].num_experts == 4
        assert specs[0].hidden_size == 64

        auto_ep.replace_moe_layer(specs[0], ep_size=1, ep_rank=0)

        replaced = model.model.layers[0].mlp
        torch.testing.assert_close(replaced.router.gate.weight, expected_gate)
        torch.testing.assert_close(replaced.experts.w1, expected_gate_up[:, :128, :])
        torch.testing.assert_close(replaced.experts.w3, expected_gate_up[:, 128:, :])
        torch.testing.assert_close(replaced.experts.w2, expected_down)
        assert [call["names"] for call in FakeGatheredParameters.calls] == [
            ["router.weight"],
            ["experts.gate_up_proj", "experts.down_proj"],
        ]
        assert all(call["modifier_rank"] is None for call in FakeGatheredParameters.calls)

    def test_module_list_replacement_preserves_frozen_experts_and_trainable_router(self, monkeypatch):
        monkeypatch.setattr(get_preset_adapter("deepseek_v3"), "_installed_transformers_version", lambda: "5.0.0")
        model = MockDeepSeekV3Transformer(num_layers=1, num_experts=4).to(dtype=torch.bfloat16)
        source = model.model.layers[0].mlp
        for expert in source.experts:
            for param in expert.parameters():
                param.requires_grad_(False)
        source.gate.weight.requires_grad_(True)
        source.gate.e_score_correction_bias = nn.Parameter(torch.zeros(4,
                                                                       dtype=source.gate.weight.dtype,
                                                                       device=source.gate.weight.device),
                                                           requires_grad=True)

        auto_ep = AutoEP(model, _runtime_config(enabled=True, autoep_size=2))
        spec = auto_ep.ep_parser()[0]
        auto_ep.replace_moe_layer(spec, ep_size=2, ep_rank=0)

        replaced = model.model.layers[0].mlp
        assert isinstance(replaced, AutoEPMoELayer)
        assert replaced.experts.w1.requires_grad is False
        assert replaced.experts.w2.requires_grad is False
        assert replaced.experts.w3.requires_grad is False
        assert replaced.router.gate.weight.requires_grad is True
        assert replaced.router.e_score_correction_bias.requires_grad is True
        _assert_same_dtype_device(replaced.router.gate.weight, source.gate.weight)
        _assert_same_dtype_device(replaced.router.e_score_correction_bias, source.gate.e_score_correction_bias)
        _assert_same_dtype_device(replaced.experts.w1, source.experts[0].gate_proj.weight)
        _assert_same_dtype_device(replaced.experts.w2, source.experts[0].down_proj.weight)
        _assert_same_dtype_device(replaced.experts.w3, source.experts[0].up_proj.weight)

    def test_module_list_zero_source_gathers_all_experts_in_global_order(self, monkeypatch):
        FakeGatheredParameters.calls = []
        monkeypatch.setattr(ep_repack, "GatheredParameters", FakeGatheredParameters)
        monkeypatch.setattr(get_preset_adapter("deepseek_v3"), "_installed_transformers_version", lambda: "5.0.0")

        model = MockDeepSeekV3Transformer(num_layers=1, num_experts=4)
        source = model.model.layers[0].mlp
        for expert_idx, expert in enumerate(source.experts):
            for offset, (suffix, param) in enumerate((
                ("w1", expert.gate_proj.weight),
                ("w2", expert.down_proj.weight),
                ("w3", expert.up_proj.weight),
            )):
                full_data = param.detach().clone()
                _mark_fake_zero_param(param,
                                      full_data,
                                      ds_id=10 + 3 * expert_idx + offset,
                                      name=f"e{expert_idx}.{suffix}")

        auto_ep = AutoEP(model, _runtime_config(enabled=True, autoep_size=2))
        spec = auto_ep.ep_parser()[0]
        w1, w2, w3 = repack_expert_weights(source.experts, spec, ep_rank=1, ep_size=2)

        expected_w1 = torch.stack([
            source.experts[2].gate_proj.weight._autoep_test_full_data,
            source.experts[3].gate_proj.weight._autoep_test_full_data
        ])
        expected_w2 = torch.stack([
            source.experts[2].down_proj.weight._autoep_test_full_data,
            source.experts[3].down_proj.weight._autoep_test_full_data
        ])
        expected_w3 = torch.stack([
            source.experts[2].up_proj.weight._autoep_test_full_data,
            source.experts[3].up_proj.weight._autoep_test_full_data
        ])

        torch.testing.assert_close(w1, expected_w1)
        torch.testing.assert_close(w2, expected_w2)
        torch.testing.assert_close(w3, expected_w3)
        assert [call["names"] for call in FakeGatheredParameters.calls] == [
            ["e0.w1", "e0.w2", "e0.w3"],
            ["e1.w1", "e1.w2", "e1.w3"],
            ["e2.w1", "e2.w2", "e2.w3"],
            ["e3.w1", "e3.w2", "e3.w3"],
        ]

    def test_module_list_mixed_expert_requires_grad_flags_are_rejected(self, monkeypatch):
        monkeypatch.setattr(get_preset_adapter("deepseek_v3"), "_installed_transformers_version", lambda: "5.0.0")
        model = MockDeepSeekV3Transformer(num_layers=1, num_experts=4)
        source = model.model.layers[0].mlp
        source.experts[0].gate_proj.weight.requires_grad_(False)
        source.experts[1].gate_proj.weight.requires_grad_(True)

        auto_ep = AutoEP(model, _runtime_config(enabled=True, autoep_size=2))
        spec = auto_ep.ep_parser()[0]
        with pytest.raises(ValueError, match="mixed requires_grad flags"):
            auto_ep.replace_moe_layer(spec, ep_size=2, ep_rank=0)

        model = MockDeepSeekV3Transformer(num_layers=1, num_experts=4)
        source = model.model.layers[0].mlp
        source.experts[1].gate_proj.to(dtype=torch.float64)

        auto_ep = AutoEP(model, _runtime_config(enabled=True, autoep_size=2))
        spec = auto_ep.ep_parser()[0]
        with pytest.raises(ValueError, match="mixed dtype/device"):
            auto_ep.replace_moe_layer(spec, ep_size=2, ep_rank=0)

    def test_hf_mixtral_causal_lm_matches_autoep_with_router_logits(self):
        transformers = pytest.importorskip("transformers")
        skip_unless_transformers_has(transformers,
                                     "MixtralConfig",
                                     "MixtralForCausalLM",
                                     min_version="5.0.0",
                                     reason="Mixtral AutoEP router-logit capture")

        torch.manual_seed(1234)
        config = tiny_mixtral_config(transformers)
        native_model, autoep_model = state_matched_models(transformers.MixtralForCausalLM, config)
        replace_autoep_layers(autoep_model, "mixtral")
        assert_causal_lm_outputs_close(native_model,
                                       autoep_model,
                                       output_router_logits=True,
                                       compare_router_logits=True,
                                       compare_aux_loss=True,
                                       compare_logits=False)

    def test_qwen_adapter_guards(self, monkeypatch):
        monkeypatch.setattr(get_preset_adapter("qwen3_moe"), "_installed_transformers_version", lambda: "5.0.0")
        model = MockMoETransformer(num_layers=1, num_experts=4, moe_every_n=1)
        model.config.model_type = "qwen2_moe"
        model.config.num_experts = model.config.num_local_experts

        specs = AutoEP(model, _runtime_config(enabled=True, autoep_size=1)).ep_parser()

        assert len(specs) == 1
        assert specs[0].model_family == "qwen3_moe"

        model.config.model_type = "qwen3_5_moe"
        with pytest.raises(ValueError, match="qwen3_5_moe_text"):
            AutoEP(model, _runtime_config(enabled=True, autoep_size=1))._resolve_presets()

    def test_deepseek_v3_detection_and_score_correction_bias_copy(self, monkeypatch):
        FakeGatheredParameters.calls = []
        monkeypatch.setattr(ep_repack, "GatheredParameters", FakeGatheredParameters)
        monkeypatch.setattr(get_preset_adapter("deepseek_v3"), "_installed_transformers_version", lambda: "5.0.0")
        model = MockDeepSeekV3Transformer(num_layers=1, num_experts=8)
        auto_ep = AutoEP(model, _runtime_config(enabled=True, autoep_size=2))
        specs = auto_ep.ep_parser()

        assert len(specs) == 1
        assert specs[0].model_family == "deepseek_v3"
        assert specs[0].expert_storage == "module_list"
        assert specs[0].expert_w1_name == "gate_proj"
        assert specs[0].has_shared_experts is True

        source_bias = torch.arange(8, dtype=torch.float32)
        model.model.layers[0].mlp.gate.e_score_correction_bias = nn.Parameter(source_bias.clone())
        _mark_fake_zero_param(model.model.layers[0].mlp.gate.e_score_correction_bias,
                              source_bias,
                              ds_id=100,
                              name="router.e_score_correction_bias")

        auto_ep.replace_moe_layer(specs[0], ep_size=2, ep_rank=0)

        replaced = model.model.layers[0].mlp
        assert isinstance(replaced, AutoEPMoELayer)
        assert replaced.router.e_score_correction_bias is not None
        torch.testing.assert_close(replaced.router.e_score_correction_bias, source_bias)
        assert ["router.e_score_correction_bias"] in [call["names"] for call in FakeGatheredParameters.calls]


def _eager_pep604_lines(module):
    """Line numbers where a module evaluates PEP 604 unions at import time."""
    tree = ast.parse(inspect.getsource(module))
    defers_annotations = any(
        isinstance(node, ast.ImportFrom) and node.module == "__future__" and any(alias.name == "annotations"
                                                                                 for alias in node.names)
        for node in tree.body)
    if defers_annotations:
        return []
    offending_lines = []
    for node in ast.walk(tree):
        annotations = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            for arg in arguments.args + arguments.posonlyargs + arguments.kwonlyargs + [
                    arguments.vararg, arguments.kwarg
            ]:
                if arg is not None and arg.annotation is not None:
                    annotations.append(arg.annotation)
            if node.returns is not None:
                annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        for annotation in annotations:
            for sub_node in ast.walk(annotation):
                if isinstance(sub_node, ast.BinOp) and isinstance(sub_node.op, ast.BitOr):
                    offending_lines.append(sub_node.lineno)
    return sorted(set(offending_lines))


class TestPy39AnnotationSafety:

    def test_autoep_import_chain_defers_pep604_annotations(self):
        """PEP 604 unions (``int | None``) in def signatures or class-level
        annotations are evaluated at import time, so on Python 3.9 they raise
        TypeError while the module is imported; that escapes the engine's
        ``except ImportError`` guards around AutoEP and breaks every
        ``deepspeed.initialize()`` (issue #8102). Every module in the AutoEP
        import chain must defer annotation evaluation with
        ``from __future__ import annotations``."""
        import deepspeed.moe.ep_count as ep_count
        import deepspeed.moe.ep_experts as ep_experts
        import deepspeed.moe.ep_kernels as ep_kernels
        import deepspeed.moe.ep_router as ep_router
        import deepspeed.module_inject.auto_ep as auto_ep
        import deepspeed.module_inject.auto_ep_config as auto_ep_config
        import deepspeed.module_inject.auto_ep_layer as auto_ep_layer
        import deepspeed.module_inject.auto_ep_preset_adapters as preset_adapters
        import deepspeed.module_inject.auto_ep_presets.base as presets_base
        import deepspeed.module_inject.auto_ep_presets.registry as presets_registry

        autoep_import_chain = [
            ep_count, ep_experts, ep_kernels, ep_router, ep_repack, auto_ep, auto_ep_config, auto_ep_layer,
            preset_adapters, presets_base, presets_registry
        ]
        for module in autoep_import_chain:
            offending_lines = _eager_pep604_lines(module)
            assert not offending_lines, (
                f"{module.__name__} evaluates PEP 604 unions at import time (lines {offending_lines}); "
                f"on Python 3.9 this raises TypeError during import and escapes the engine's "
                f"except-ImportError guards (issue #8102). Add 'from __future__ import annotations'.")

# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

import torch
import torch.nn as nn

import deepspeed.runtime.engine as ds_engine
from deepspeed import set_optimizer_flags
from deepspeed.module_inject.auto_ep import AutoEP
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
from deepspeed.runtime.engine import DeepSpeedEngine


class _Experts(nn.Module):

    def __init__(self, layer_index):
        super().__init__()
        self.w1 = nn.Parameter(torch.full((2, 2), float(layer_index + 1)))
        self.w2 = nn.Parameter(torch.full((2, 2), float(layer_index + 2)), requires_grad=False)
        for param in self.parameters():
            param.ds_zero_placement_family = "autoep_expert"
            param.ds_zero_partition_group_name = f"expert_group_{layer_index}"


class _Replacement(nn.Module):

    def __init__(self, layer_index):
        super().__init__()
        self.layer_index = layer_index
        self.experts = _Experts(layer_index)
        self.router = nn.Linear(2, 2, bias=False)
        self.shared_experts = nn.Linear(2, 2, bias=False)
        self.dense = nn.Linear(2, 2, bias=False)
        self.num_local_experts = 1


def _spec(layer_index):
    return SimpleNamespace(
        moe_module_name=f"layers.{layer_index}.mlp",
        preset_adapter="test",
        model_family="test",
    )


def test_autoep_zero3_partitions_each_replacement_before_next_allocation(monkeypatch):
    auto_ep = object.__new__(AutoEP)
    events = []
    replacements = []

    def construct(spec, ep_size, ep_rank):
        layer_index = len(replacements)
        if layer_index:
            assert events[-1] == ("converted", layer_index - 1)
        events.append(("construct", layer_index))
        replacement = _Replacement(layer_index)
        replacements.append(replacement)
        return replacement

    auto_ep._replace_moe_layer_without_retarget = construct
    auto_ep._retarget_transformers_output_recorders = lambda spec, replacement: events.append(
        ("retarget", replacement.layer_index))

    resolved_groups = {}

    def resolve_group(group_name):
        return resolved_groups.setdefault(group_name, object())

    converted_batches = []

    def convert_to_zero_parameters(param_list):
        layer_index = len(converted_batches)
        replacement = replacements[layer_index]
        expected_params = list(replacement.experts.parameters())
        assert [id(param) for param in param_list] == [id(param) for param in expected_params]
        assert all(param.ds_zero_placement_family == "autoep_expert" for param in param_list)
        assert [param.requires_grad for param in param_list] == [True, False]
        assert [param.tolist() for param in param_list] == [
            [[float(layer_index + 1)] * 2] * 2,
            [[float(layer_index + 2)] * 2] * 2,
        ]
        assert all(param.ds_zero_partition_process_group is resolved_groups[param.ds_zero_partition_group_name]
                   for param in param_list)
        converted_batches.append(param_list)
        events.append(("converted", layer_index))

    monkeypatch.setattr(ds_engine.groups, "_get_expert_data_parallel_group", resolve_group)

    def on_moe_layer_replaced(replacement):
        events.append(("callback", replacement.layer_index))
        DeepSpeedEngine._partition_autoep_zero3_experts(replacement, convert_to_zero_parameters)

    auto_ep.replace_moe_layers([_spec(0), _spec(1)], ep_size=2, ep_rank=0, on_moe_layer_replaced=on_moe_layer_replaced)

    assert events == [
        ("construct", 0),
        ("callback", 0),
        ("converted", 0),
        ("construct", 1),
        ("callback", 1),
        ("converted", 1),
        ("retarget", 0),
    ]
    converted_ids = {id(param) for batch in converted_batches for param in batch}
    expected_ids = {id(param) for replacement in replacements for param in replacement.experts.parameters()}
    excluded_ids = {
        id(param)
        for replacement in replacements
        for module in (replacement.router, replacement.shared_experts, replacement.dense)
        for param in module.parameters()
    }
    assert converted_ids == expected_ids
    assert converted_ids.isdisjoint(excluded_ids)
    assert set(resolved_groups) == {"expert_group_0", "expert_group_1"}


def test_autoep_zero3_eager_conversion_gates():
    engine = object.__new__(DeepSpeedEngine)
    source = nn.Linear(2, 2, bias=False)
    converter = lambda param_list: None
    source.weight.convert_to_zero_parameters = converter

    engine.zero_optimization_partition_weights = lambda: False
    assert engine._autoep_zero3_param_converter(source) is None

    engine.zero_optimization_partition_weights = lambda: True
    assert engine._autoep_zero3_param_converter(source) is converter

    ordinary_source = nn.Linear(2, 2, bias=False)
    assert engine._autoep_zero3_param_converter(ordinary_source) is None


def test_autoep_zero3_partitioned_experts_keep_muon_assignment(monkeypatch):
    replacement = _Replacement(0)
    replacement.ordinary_partition = nn.Parameter(torch.empty(0))
    replacement.ordinary_partition.ds_shape = torch.Size((2, 2))
    monkeypatch.setattr(ds_engine.groups, "_get_expert_data_parallel_group", lambda group_name: object())

    def convert_to_zero_parameters(param_list):
        for param in param_list:
            param.ds_shape = param.shape
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)

    DeepSpeedEngine._partition_autoep_zero3_experts(replacement, convert_to_zero_parameters)
    assert all(param.ndim == 1 for param in replacement.experts.parameters())

    set_optimizer_flags(SimpleNamespace(optimizer_name="muon"), replacement)

    assert all(param.use_muon for param in replacement.experts.parameters())
    assert not replacement.ordinary_partition.use_muon


def test_autoep_zero3_partitioned_experts_keep_optimizer_grouping():

    def make_experts(partitioned):
        params = []
        for _ in range(4):
            param = nn.Parameter(torch.empty(0) if partitioned else torch.ones(4))
            param.allreduce = False
            param.group_name = "ep_size_2"
            if partitioned:
                param.ds_numel = 4
            params.append(param)
        return params

    def group_lengths(params):
        groups = split_params_into_different_moe_groups_for_optimizer({
            "name": "dense-params",
            "params": params,
        },
                                                                      max_group_size=10)
        return [len(group["params"]) for group in groups if group.get("moe")]

    assert group_lengths(make_experts(partitioned=False)) == [2, 2]
    assert group_lengths(make_experts(partitioned=True)) == [2, 2]

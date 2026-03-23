# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.checkpoint.constants import (PARAMETER_WITH_ROW_PARALLELISM_PATTERNS, PARAMETER_WITH_SUB_PARAMS,
                                            TP_REPLICATED_PARAMETER_PATTERNS, DS_AUTOTP_UC_META)
from deepspeed.module_inject.layers import (_build_param_uc_restore_meta, _get_param_uc_conversion_meta,
                                            LinearAllreduce, LinearLayer, SubParamLinearLayer,
                                            collect_autotp_universal_checkpoint_info)


def test_collect_autotp_universal_checkpoint_info_row_parallel():
    layer = LinearAllreduce(torch.nn.Linear(16, 8, bias=True), mp_group=None, name="proj")
    model = torch.nn.Module()
    model.proj = layer

    uc_info = collect_autotp_universal_checkpoint_info(model)

    # collect_autotp_universal_checkpoint_info() stores regex patterns like r"^proj\.weight$"
    assert r"^proj\.weight$" in uc_info[PARAMETER_WITH_ROW_PARALLELISM_PATTERNS]
    # bias in LinearAllreduce is marked replicated, so it should appear in replicated patterns
    assert r"^proj\.bias$" in uc_info[TP_REPLICATED_PARAMETER_PATTERNS]


def test_collect_autotp_universal_checkpoint_info_subparams():
    layer = SubParamLinearLayer(torch.nn.Linear(12, 12, bias=True),
                                mp_group=None,
                                shape=(3, -1),
                                partition_dim=0,
                                name="qkv")
    model = torch.nn.Module()
    model.qkv = layer

    uc_info = collect_autotp_universal_checkpoint_info(model)

    assert len(uc_info[PARAMETER_WITH_SUB_PARAMS]) == 1
    assert uc_info[PARAMETER_WITH_SUB_PARAMS][0]["partition_dim"] == 0


def test_collect_autotp_universal_checkpoint_info_column_parallel_bias_not_replicated():
    layer = LinearLayer(torch.nn.Linear(16, 8, bias=True), mp_group=None, name="dense")
    model = torch.nn.Module()
    model.dense = layer

    uc_info = collect_autotp_universal_checkpoint_info(model)

    assert not any("dense.weight" in p for p in uc_info[PARAMETER_WITH_ROW_PARALLELISM_PATTERNS])
    assert not any("dense.bias" in p for p in uc_info[TP_REPLICATED_PARAMETER_PATTERNS])


def test_collect_autotp_universal_checkpoint_info_subparams_preserves_shape_metadata():
    layer = SubParamLinearLayer(torch.nn.Linear(12, 12, bias=True),
                                mp_group=None,
                                shape=((2, 10), 12),
                                partition_dim=0,
                                name="fused")
    model = torch.nn.Module()
    model.fused = layer

    uc_info = collect_autotp_universal_checkpoint_info(model)

    assert uc_info[PARAMETER_WITH_SUB_PARAMS][0]["shape"] == [(2, 10), 12]


def test_subparam_layer_marks_standardized_param_metadata():
    layer = SubParamLinearLayer(torch.nn.Linear(12, 12, bias=True),
                                mp_group=None,
                                shape=(3, -1),
                                partition_dim=0,
                                name="packed")

    weight_meta = getattr(layer.weight, DS_AUTOTP_UC_META)
    bias_meta = getattr(layer.bias, DS_AUTOTP_UC_META)

    assert weight_meta["sub_param_sizes"] == (4, 4, 4)
    assert tuple(weight_meta["target_partition_shape"]) == tuple(layer.weight.shape)
    assert tuple(bias_meta["target_partition_shape"]) == tuple(layer.bias.shape)


def test_universal_checkpoint_info_excludes_param_level_recovery_fields():
    layer = SubParamLinearLayer(torch.nn.Linear(12, 12, bias=True),
                                mp_group=None,
                                shape=(3, -1),
                                partition_dim=0,
                                name="packed")
    model = torch.nn.Module()
    model.packed = layer

    uc_info = collect_autotp_universal_checkpoint_info(model)
    subparam_entry = uc_info[PARAMETER_WITH_SUB_PARAMS][0]

    assert "shape" in subparam_entry
    assert "partition_dim" in subparam_entry
    assert "patterns" in subparam_entry
    assert "sub_param_sizes" not in subparam_entry
    assert "target_partition_shape" not in subparam_entry


def test_collect_uses_conversion_view_not_recovery_fields():
    layer = SubParamLinearLayer(torch.nn.Linear(12, 12, bias=True),
                                mp_group=None,
                                shape=(3, -1),
                                partition_dim=0,
                                name="packed")
    model = torch.nn.Module()
    model.packed = layer

    meta = getattr(layer.weight, "ds_autotp_universal_checkpoint_meta")
    meta["partition_dim"] = 99
    meta["sub_param_shape"] = (999, -1)

    uc_info = collect_autotp_universal_checkpoint_info(model)
    subparam_entry = uc_info[PARAMETER_WITH_SUB_PARAMS][0]

    assert subparam_entry["partition_dim"] == 0
    assert subparam_entry["shape"] == [3, -1]


def test_param_uc_restore_builder_normalizes_shapes_and_nests_conversion_view():
    restore_meta = _build_param_uc_restore_meta(partition_type="column",
                                                partition_dim=0,
                                                logical_shape=[12, 8],
                                                output_shape=[12],
                                                sub_param_shape=[3, -1],
                                                sub_param_sizes=[4, 4, 4],
                                                target_partition_shape=torch.Size([4, 8]),
                                                original_shape=torch.Size([12, 8]),
                                                is_bias=False,
                                                replicated=False)

    assert restore_meta["logical_shape"] == (12, 8)
    assert restore_meta["output_shape"] == (12, )
    assert restore_meta["sub_param_shape"] == (3, -1)
    assert restore_meta["sub_param_sizes"] == (4, 4, 4)
    assert restore_meta["target_partition_shape"] == (4, 8)
    assert restore_meta["original_shape"] == (12, 8)
    assert restore_meta["conversion"] == {
        "partition_type": "column",
        "partition_dim": 0,
        "sub_param_shape": (3, -1),
        "original_shape": (12, 8),
        "is_bias": False,
        "replicated": False,
    }


def test_conversion_helper_reads_builder_nested_view():
    param = torch.nn.Parameter(torch.zeros(4, 8))
    param.ds_autotp_universal_checkpoint_meta = _build_param_uc_restore_meta(partition_type="row",
                                                                             partition_dim=1,
                                                                             logical_shape=[4, 16],
                                                                             output_shape=[4],
                                                                             original_shape=[4, 16],
                                                                             is_bias=False,
                                                                             replicated=False)

    assert _get_param_uc_conversion_meta(param) == param.ds_autotp_universal_checkpoint_meta["conversion"]

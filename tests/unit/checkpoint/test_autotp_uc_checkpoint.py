# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import glob
import os
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import deepspeed
import deepspeed.comm as dist
import deepspeed.checkpoint.ds_to_universal as ds_to_universal
from deepspeed.checkpoint.constants import (
    AUTOTP_UNSUPPORTED_PARAMETER_PATTERNS, CAT_DIM, FP32_FLAT_GROUPS, FP32_WEIGHT_KEY, OPTIMIZER_STATE_DICT, PARAM,
    PARAM_GROUPS, PARAM_SHAPES, PARAMETER_WITH_ROW_PARALLELISM_PATTERNS, PARAMETER_WITH_SUB_PARAMS, SUB_PARAM_SHAPE,
    SUB_PARAM_SHARD_WIDTHS, TP_REPLICATED_PARAMETER_PATTERNS, UNIVERSAL_CHECKPOINT_INFO,
    UNIVERSAL_CHECKPOINT_VERSION_KEY, UNIVERSAL_CHECKPOINT_VERSION_VALUE, VOCABULARY_PARAMETER_PATTERNS, ZERO_STAGE)
from deepspeed.checkpoint.universal_checkpoint import SubparamShape as CheckpointSubparamShape
from deepspeed.checkpoint.ds_to_universal import (_group_per_tp_shapes, _validate_autotp_conversion_support, main as
                                                  convert_to_universal, merge_tp_slices)
from deepspeed.checkpoint.universal_checkpoint import (_get_param_uc_restore_meta, _resolve_autotp_partition,
                                                       load_hp_checkpoint_state)
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.utils import RepeatingLoader, groups
from deepspeed.module_inject.tp_shard import get_shard_size_list

from unit.common import DistributedTest


class _DummyAddress:

    def __init__(self, start, numel):
        self.start = start
        self.numel = numel


class _DummyHPMapping:

    def __init__(self, param):
        self.lp_fragment_address = _DummyAddress(0, param.numel())
        self._param = param
        self.optim_fragment = {}

    def get_hp_fragment(self):
        return self._param.view(-1)

    def get_optim_state_keys(self):
        return []


def _make_param(shape, meta=None):
    param = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float32))
    param._hp_mapping = _DummyHPMapping(param)
    if meta is not None:
        setattr(param, 'ds_autotp_universal_checkpoint_meta', meta)
    return param


def test_validate_autotp_conversion_support_rejects_unsupported_layout():
    uc_info = {AUTOTP_UNSUPPORTED_PARAMETER_PATTERNS: {r"^qkv\.weight$": "CodeGenBlock layout is interleaved"}}

    with pytest.raises(ValueError, match="CodeGenBlock layout is interleaved"):
        _validate_autotp_conversion_support(uc_info)


def test_validate_autotp_conversion_support_accepts_legacy_schema():
    _validate_autotp_conversion_support({})


def test_inject_missing_state_still_validates_unsupported_layout(monkeypatch):
    uc_info = {AUTOTP_UNSUPPORTED_PARAMETER_PATTERNS: {r"^qkv\.weight$": "unsupported layout"}}

    class FakeCheckpoint:

        def get_checkpoint_info(self, key):
            assert key == UNIVERSAL_CHECKPOINT_INFO
            return uc_info

    checkpoint = FakeCheckpoint()
    injected = []
    monkeypatch.setattr(ds_to_universal, "_get_optim_files", lambda _: ["optim.pt"])
    monkeypatch.setattr(ds_to_universal, "_filter_zero3_optim_files", lambda _: [])
    monkeypatch.setattr(ds_to_universal, "_get_zero_stage", lambda _: 1)
    monkeypatch.setattr(ds_to_universal, "DeepSpeedCheckpoint", lambda _: checkpoint)
    monkeypatch.setattr(ds_to_universal, "_inject_missing_state", lambda value: injected.append(value))
    args = SimpleNamespace(input_folder="input", output_folder="output", inject_missing_state=True)

    with pytest.raises(ValueError, match="unsupported layout"):
        ds_to_universal.main(args)
    assert injected == [checkpoint]


def test_resolve_autotp_partition_row_parallel_weight():
    param = _make_param(
        (4, 4), {
            'partition_type': 'row',
            'partition_dim': 1,
            'logical_shape': (4, 8),
            'output_shape': (4, ),
            'sub_param_shape': None,
            'original_shape': (4, 8),
            'is_bias': False,
            'replicated': False,
        })
    full_hp_param = torch.arange(32, dtype=torch.float32).view(4, 8)

    slice_flat = _resolve_autotp_partition(param, {PARAM: full_hp_param}, full_hp_param, tp_rank=1, tp_world_size=2)

    expected = full_hp_param.chunk(2, dim=1)[1].flatten()
    assert torch.equal(slice_flat, expected)


def test_resolve_autotp_partition_subparam_column_weight():
    param = _make_param(
        (3, 4), {
            'partition_type': 'column',
            'partition_dim': 0,
            'logical_shape': (6, 4),
            'output_shape': (6, ),
            'sub_param_shape': ((2, 2, 2), 4),
            'original_shape': (6, 4),
            'is_bias': False,
            'replicated': False,
        })
    full_hp_param = torch.arange(24, dtype=torch.float32).view(6, 4)

    slice_flat = _resolve_autotp_partition(param, {PARAM: full_hp_param}, full_hp_param, tp_rank=0, tp_world_size=2)

    chunks = [sub.chunk(2, dim=0)[0] for sub in full_hp_param.view(3, 2, 4)]
    expected = torch.cat(chunks, dim=0).flatten()
    assert torch.equal(slice_flat, expected)


def test_resolve_autotp_partition_subparam_sizes_uneven_gqa_like():
    # Simulate a fused QKV weight where Q/K/V have uneven sizes along partition_dim=0.
    # Example (GQA-like):
    #   Q: 8
    #   K: 4
    #   V: 4
    # Total: 16
    #
    # With tp_world_size=2, correct slicing is:
    #   Q chunk -> 4 per rank
    #   K chunk -> 2 per rank
    #   V chunk -> 2 per rank
    # Each rank gets 8 rows total, but importantly boundaries must align with Q/K/V.
    sub_param_sizes = [8, 4, 4]
    tp_world_size = 2
    tp_rank = 1

    param = _make_param(
        (8, 2),
        {
            "partition_type": "column",
            "partition_dim": 0,
            "logical_shape": (sum(sub_param_sizes), 2),  # (16, 2)
            "output_shape": (sum(sub_param_sizes), ),  # (16,)
            "sub_param_shape": (tuple(sub_param_sizes), 2),
            "sub_param_sizes": sub_param_sizes,
            "original_shape": (sum(sub_param_sizes), 2),
            "is_bias": False,
            "replicated": False,
        })

    # Full (unsharded) HP parameter: shape (16, 2)
    full_hp_param = torch.arange(sum(sub_param_sizes) * 2, dtype=torch.float32).view(sum(sub_param_sizes), 2)

    slice_flat = _resolve_autotp_partition(param, {PARAM: full_hp_param},
                                           full_hp_param,
                                           tp_rank=tp_rank,
                                           tp_world_size=tp_world_size)

    # Expected: split into Q/K/V blocks, chunk each block by TP, take tp_rank slice, concat back.
    q, k, v = torch.split(full_hp_param, sub_param_sizes, dim=0)
    expected = torch.cat([
        q.chunk(tp_world_size, dim=0)[tp_rank],
        k.chunk(tp_world_size, dim=0)[tp_rank],
        v.chunk(tp_world_size, dim=0)[tp_rank]
    ],
                         dim=0).flatten()

    assert torch.equal(slice_flat, expected)


def test_resolve_autotp_partition_uses_uneven_partition_sizes():
    full_hp_param = torch.arange(101 * 4, dtype=torch.float32).view(101, 4)
    param = _make_param(
        (50, 4), {
            'partition_type': 'column',
            'partition_dim': 0,
            'logical_shape': (101, 4),
            'output_shape': (101, ),
            'partition_sizes': (51, 50),
            'original_shape': (101, 4),
            'is_bias': False,
            'replicated': False,
        })

    slice_flat = _resolve_autotp_partition(param, {PARAM: full_hp_param}, full_hp_param, tp_rank=1, tp_world_size=2)

    expected = full_hp_param.narrow(0, 51, 50).flatten()
    assert torch.equal(slice_flat, expected)


def test_resolve_autotp_partition_uses_uneven_partition_sizes_for_bias():
    full_hp_param = torch.arange(101, dtype=torch.float32)
    param = _make_param(
        (50, ), {
            'partition_type': 'column',
            'partition_dim': 0,
            'logical_shape': (101, ),
            'output_shape': (101, ),
            'partition_sizes': (51, 50),
            'original_shape': (101, ),
            'is_bias': True,
            'replicated': False,
        })

    slice_flat = _resolve_autotp_partition(param, {PARAM: full_hp_param}, full_hp_param, tp_rank=1, tp_world_size=2)

    expected = full_hp_param.narrow(0, 51, 50).flatten()
    assert torch.equal(slice_flat, expected)


def test_resolve_autotp_partition_prefers_resolved_subparam_sizes():
    # Standard AutoTP records sub_param_shape as a logical view spec: the partition_dim entry
    # is the sub-parameter *count* (3), not a width. Only sub_param_sizes carries the real
    # widths, so reading the count as a width would restore a single row instead of six.
    param = _make_param(
        (6, 8), {
            'partition_type': 'column',
            'partition_dim': 0,
            'logical_shape': (12, 8),
            'output_shape': (12, ),
            'sub_param_shape': (3, -1),
            'sub_param_sizes': (4, 4, 4),
            'original_shape': (12, 8),
            'is_bias': False,
            'replicated': False,
        })
    full_hp_param = torch.arange(96, dtype=torch.float32).view(12, 8)

    slice_flat = _resolve_autotp_partition(param, {PARAM: full_hp_param}, full_hp_param, tp_rank=0, tp_world_size=2)

    a, b, c = torch.split(full_hp_param, [4, 4, 4], dim=0)
    expected = torch.cat([a.narrow(0, 0, 2), b.narrow(0, 0, 2), c.narrow(0, 0, 2)], dim=0).flatten()
    assert torch.equal(slice_flat, expected)


def test_resolve_autotp_partition_rejects_uneven_subparams_without_widths():
    # Without recorded widths an even split is assumed; for uneven sub-parameters that would
    # shift every offset and silently drop the trailing rows, so it must be refused instead.
    param = _make_param(
        (5, 2), {
            'partition_type': 'column',
            'partition_dim': 0,
            'logical_shape': (16, 2),
            'output_shape': (16, ),
            'sub_param_shape': None,
            'sub_param_sizes': (8, 4, 4),
            'sub_param_shard_widths': None,
            'original_shape': (16, 2),
            'is_bias': False,
            'replicated': False,
        })
    full_hp_param = torch.arange(32, dtype=torch.float32).view(16, 2)

    with pytest.raises(AssertionError, match="not divisible by tp_world_size"):
        _resolve_autotp_partition(param, {PARAM: full_hp_param}, full_hp_param, tp_rank=0, tp_world_size=3)


def test_resolve_autotp_partition_rejects_uneven_without_partition_sizes():
    # The chunk() fallback sizes blocks as ceil(size / tp) and shrinks the last one, while
    # AutoTP gives the remainder to the low ranks. For 10/4 that is [3, 3, 3, 1] instead of
    # [3, 3, 2, 2], so an uneven checkpoint lacking partition_sizes must not be guessed at.
    param = _make_param(
        (3, 2), {
            'partition_type': 'column',
            'partition_dim': 0,
            'logical_shape': (10, 2),
            'output_shape': (10, ),
            'sub_param_shape': None,
            'sub_param_sizes': None,
            'partition_sizes': None,
            'original_shape': (10, 2),
            'is_bias': False,
            'replicated': False,
        })
    full_hp_param = torch.arange(20, dtype=torch.float32).view(10, 2)

    with pytest.raises(AssertionError, match="records no partition_sizes"):
        _resolve_autotp_partition(param, {PARAM: full_hp_param}, full_hp_param, tp_rank=3, tp_world_size=4)


def test_resolve_autotp_partition_replicated_bias():
    full_hp_param = torch.arange(8, dtype=torch.float32)
    param = _make_param(
        (8, ), {
            'partition_type': 'row',
            'partition_dim': None,
            'logical_shape': (8, ),
            'output_shape': (8, ),
            'sub_param_shape': None,
            'original_shape': (8, ),
            'is_bias': True,
            'replicated': True,
        })

    slice_flat = _resolve_autotp_partition(param, {PARAM: full_hp_param}, full_hp_param, tp_rank=1, tp_world_size=2)

    assert torch.equal(slice_flat, full_hp_param)


def test_load_hp_checkpoint_state_prefers_autotp_metadata(tmp_path, monkeypatch):
    param = _make_param(
        (4, 4), {
            'partition_type': 'row',
            'partition_dim': 1,
            'logical_shape': (4, 8),
            'output_shape': (4, ),
            'sub_param_shape': None,
            'original_shape': (4, 8),
            'is_bias': False,
            'replicated': False,
        })
    param.load_hp_checkpoint_state = types.MethodType(load_hp_checkpoint_state, param)

    import deepspeed.checkpoint.universal_checkpoint as uc
    monkeypatch.setattr(uc, "current_param", param, raising=False)

    ckpt_dir = tmp_path / "weight"
    ckpt_dir.mkdir(parents=True)
    full_hp_param = torch.arange(32, dtype=torch.float32).view(4, 8)
    torch.save({PARAM: full_hp_param}, ckpt_dir / f"{FP32_WEIGHT_KEY}.pt")

    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {PARAM: full_hp_param} if str(args[0]).endswith("fp32.pt") else 0,
    )

    step = param.load_hp_checkpoint_state(str(ckpt_dir), tp_rank=1, tp_world_size=2)

    assert step is None
    expected = full_hp_param.chunk(2, dim=1)[1].flatten()
    assert torch.equal(param.data.flatten(), expected)


def test_load_hp_checkpoint_state_keeps_tp_topology_for_single_kv_head(tmp_path, monkeypatch):
    # MQA: a single key/value head lives entirely on tp rank 0, so that rank's local shape
    # matches the universal tensor. The recorded per-rank widths must still be honored.
    param = _make_param(
        (3, 4), {
            UNIVERSAL_CHECKPOINT_VERSION_KEY: UNIVERSAL_CHECKPOINT_VERSION_VALUE,
            'partition_type': 'column',
            'partition_dim': 0,
            'logical_shape': (3, 4),
            'output_shape': (3, ),
            'sub_param_sizes': (1, 1, 1),
            'sub_param_shard_widths': [[1, 0], [1, 0], [1, 0]],
            'original_shape': (3, 4),
            'is_bias': False,
            'replicated': False,
        })
    param.load_hp_checkpoint_state = types.MethodType(load_hp_checkpoint_state, param)

    ckpt_dir = tmp_path / "weight"
    ckpt_dir.mkdir(parents=True)
    full_hp_param = torch.arange(12, dtype=torch.float32).view(3, 4)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {PARAM: full_hp_param} if str(args[0]).endswith("fp32.pt") else 0,
    )
    torch.save({PARAM: full_hp_param}, ckpt_dir / f"{FP32_WEIGHT_KEY}.pt")

    param.load_hp_checkpoint_state(str(ckpt_dir), tp_rank=0, tp_world_size=2)

    assert torch.equal(param.data.flatten(), full_hp_param.flatten())


def _write_tp_slice(base_dir, param_name, tp_idx, state_name, tensor):
    shard_dir = base_dir / param_name / str(tp_idx)
    shard_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.reshape(-1), shard_dir / f"{state_name}.00")


def _write_tp_states(base_dir, param_name, tp_idx, fp32_tensor):
    # merge_tp_slices attempts to merge these three states, so the test must write all of them.
    _write_tp_slice(base_dir, param_name, tp_idx, "fp32", fp32_tensor)
    _write_tp_slice(base_dir, param_name, tp_idx, "exp_avg", torch.zeros_like(fp32_tensor))
    _write_tp_slice(base_dir, param_name, tp_idx, "exp_avg_sq", torch.zeros_like(fp32_tensor))


def test_merge_tp_slices_emits_subparam_shape_metadata(tmp_path):
    slice_dir = tmp_path / "slices"
    output_dir = tmp_path / "out"
    param_name = "module.qkv.weight"

    tp0 = torch.arange(12, dtype=torch.float32).view(3, 4)
    tp1 = torch.arange(12, 24, dtype=torch.float32).view(3, 4)
    _write_tp_states(slice_dir, param_name, 0, tp0)
    _write_tp_states(slice_dir, param_name, 1, tp1)

    uc_info = {
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [],
        TP_REPLICATED_PARAMETER_PATTERNS: [],
        PARAMETER_WITH_SUB_PARAMS: [{
            "patterns": [rf"^{param_name}$"],
            "shape": [(2, 2, 2), 4],
            "partition_dim": 0,
        }],
    }

    unmatched = merge_tp_slices(uc_info, str(output_dir), str(slice_dir), 2,
                                (param_name, [torch.Size([3, 4]), torch.Size([3, 4])]))

    ckpt = torch.load(output_dir / param_name / "fp32.pt", weights_only=False)
    assert not unmatched
    assert isinstance(ckpt[SUB_PARAM_SHAPE], CheckpointSubparamShape)
    assert ckpt[SUB_PARAM_SHAPE].partition_dim == 0


def test_merge_tp_slices_rejects_current_version_without_shard_widths(tmp_path):
    # A checkpoint declaring the current version always records the widths, so a missing entry
    # means inconsistent metadata rather than an old checkpoint. Falling back to an even split
    # there would silently reshuffle an uneven fused weight.
    slice_dir = tmp_path / "slices"
    output_dir = tmp_path / "out"
    param_name = "module.qkv.weight"

    _write_tp_states(slice_dir, param_name, 0, torch.zeros(3, 4))
    _write_tp_states(slice_dir, param_name, 1, torch.zeros(3, 4))

    uc_info = {
        UNIVERSAL_CHECKPOINT_VERSION_KEY: UNIVERSAL_CHECKPOINT_VERSION_VALUE,
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [],
        TP_REPLICATED_PARAMETER_PATTERNS: [],
        PARAMETER_WITH_SUB_PARAMS: [{
            "patterns": [rf"^{param_name}$"],
            "shape": [(2, 2, 2), 4],
            "partition_dim": 0,
        }],
    }

    with pytest.raises(AssertionError, match="records no sub_param_shard_widths"):
        merge_tp_slices(uc_info, str(output_dir), str(slice_dir), 2,
                        (param_name, [torch.Size([3, 4]), torch.Size([3, 4])]))


def test_merge_tp_slices_decodes_legacy_subparam_count(tmp_path):
    slice_dir = tmp_path / "slices"
    output_dir = tmp_path / "out"
    param_name = "module.qkv.weight"
    full = torch.arange(24, dtype=torch.float32).view(12, 2)

    # Legacy shape=(3, -1) means three equal Q/K/V sub-parameters, not a
    # three-row physical width. Each TP rank stores its piece of every sub-parameter.
    rows = {0: [0, 1, 4, 5, 8, 9], 1: [2, 3, 6, 7, 10, 11]}
    for tp_index, row_ids in rows.items():
        _write_tp_states(slice_dir, param_name, tp_index, full[row_ids].contiguous())

    uc_info = {
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [],
        TP_REPLICATED_PARAMETER_PATTERNS: [],
        PARAMETER_WITH_SUB_PARAMS: [{
            "patterns": [rf"^{param_name}$"],
            "shape": [3, -1],
            "partition_dim": 0,
        }],
    }

    merge_tp_slices(uc_info, str(output_dir), str(slice_dir), 2,
                    (param_name, [torch.Size([6, 2]), torch.Size([6, 2])]))

    ckpt = torch.load(output_dir / param_name / "fp32.pt", weights_only=False)
    torch.testing.assert_close(ckpt[PARAM], full)
    assert ckpt[SUB_PARAM_SHAPE].shape == ((4, 4, 4), -1)


def test_merge_tp_slices_uses_uneven_sub_param_widths(tmp_path):
    # A fused QKV with 6 query and 3 key/value heads over 2 ranks: the heads go 2/1, so
    # rank 0 owns 4 query and 2+2 key/value rows while rank 1 owns 2 and 1+1.
    slice_dir = tmp_path / "slices"
    output_dir = tmp_path / "out"
    param_name = "module.qkv_proj.weight"
    shard_widths = [[4, 2], [2, 1], [2, 1]]

    full = torch.arange(24, dtype=torch.float32).view(12, 2)
    rows = {0: [0, 1, 2, 3, 6, 7, 9, 10], 1: [4, 5, 8, 11]}
    for tp_index, row_ids in rows.items():
        _write_tp_states(slice_dir, param_name, tp_index, full[row_ids].contiguous())

    uc_info = {
        UNIVERSAL_CHECKPOINT_VERSION_KEY: UNIVERSAL_CHECKPOINT_VERSION_VALUE,
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [],
        TP_REPLICATED_PARAMETER_PATTERNS: [],
        PARAMETER_WITH_SUB_PARAMS: [{
            "patterns": [rf"^{param_name}$"],
            "shape": [(6, 3, 3), 2],
            "partition_dim": 0,
        }],
        SUB_PARAM_SHARD_WIDTHS: {
            rf"^{param_name}$": shard_widths
        },
    }

    merge_tp_slices(uc_info, str(output_dir), str(slice_dir), 2,
                    (param_name, [torch.Size([8, 2]), torch.Size([4, 2])]))

    ckpt = torch.load(output_dir / param_name / "fp32.pt", weights_only=False)
    torch.testing.assert_close(ckpt[PARAM], full)


def test_merge_tp_slices_restores_uneven_fused_bias_order(tmp_path):
    # The bias of a fused QKV is cut per sub-parameter exactly like its weight. Concatenating
    # the rank slices end to end instead would interleave Q/K/V and restore a wrong bias.
    slice_dir = tmp_path / "slices"
    output_dir = tmp_path / "out"
    param_name = "module.qkv_proj.bias"
    shard_widths = [[4, 2], [2, 1], [2, 1]]

    full = torch.arange(12, dtype=torch.float32)
    rows = {0: [0, 1, 2, 3, 6, 7, 9, 10], 1: [4, 5, 8, 11]}
    for tp_index, row_ids in rows.items():
        _write_tp_states(slice_dir, param_name, tp_index, full[row_ids].contiguous())

    uc_info = {
        UNIVERSAL_CHECKPOINT_VERSION_KEY: UNIVERSAL_CHECKPOINT_VERSION_VALUE,
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [],
        TP_REPLICATED_PARAMETER_PATTERNS: [],
        PARAMETER_WITH_SUB_PARAMS: [{
            "patterns": [rf"^{param_name}$"],
            "shape": [(6, 3, 3)],
            "partition_dim": 0,
        }],
        SUB_PARAM_SHARD_WIDTHS: {
            rf"^{param_name}$": shard_widths
        },
    }

    merge_tp_slices(uc_info, str(output_dir), str(slice_dir), 2, (param_name, [torch.Size([8]), torch.Size([4])]))

    ckpt = torch.load(output_dir / param_name / "fp32.pt", weights_only=False)
    torch.testing.assert_close(ckpt[PARAM], full)


def test_merge_tp_slices_handles_rank_without_any_sub_param_rows(tmp_path):
    # With more ranks than key/value heads a rank can own none of the parameter. Its slice is
    # empty, and an empty slice cannot infer the placeholder dimension of a (n, -1) view spec.
    slice_dir = tmp_path / "slices"
    output_dir = tmp_path / "out"
    param_name = "module.qkv_proj.weight"
    shard_widths = [[2, 0], [1, 0], [1, 0]]

    full = torch.arange(8, dtype=torch.float32).view(4, 2)
    rows = {0: [0, 1, 2, 3], 1: []}
    for tp_index, row_ids in rows.items():
        _write_tp_states(slice_dir, param_name, tp_index, full[row_ids].contiguous())

    uc_info = {
        UNIVERSAL_CHECKPOINT_VERSION_KEY: UNIVERSAL_CHECKPOINT_VERSION_VALUE,
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [],
        TP_REPLICATED_PARAMETER_PATTERNS: [],
        PARAMETER_WITH_SUB_PARAMS: [{
            "patterns": [rf"^{param_name}$"],
            "shape": [(2, 1, 1), -1],
            "partition_dim": 0,
        }],
        SUB_PARAM_SHARD_WIDTHS: {
            rf"^{param_name}$": shard_widths
        },
    }

    merge_tp_slices(uc_info, str(output_dir), str(slice_dir), 2,
                    (param_name, [torch.Size([4, 2]), torch.Size([0, 2])]))

    ckpt = torch.load(output_dir / param_name / "fp32.pt", weights_only=False)
    torch.testing.assert_close(ckpt[PARAM], full)


def test_merge_tp_slices_uses_row_parallel_cat_dim(tmp_path):
    slice_dir = tmp_path / "slices"
    output_dir = tmp_path / "out"
    param_name = "module.proj.weight"

    # Uneven row-parallel shards: rank 0 owns 3 input columns, rank 1 owns 2.
    tp0 = torch.arange(12, dtype=torch.float32).view(4, 3)
    tp1 = torch.arange(12, 20, dtype=torch.float32).view(4, 2)
    _write_tp_states(slice_dir, param_name, 0, tp0)
    _write_tp_states(slice_dir, param_name, 1, tp1)

    uc_info = {
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [rf"^{param_name}$"],
        TP_REPLICATED_PARAMETER_PATTERNS: [],
        PARAMETER_WITH_SUB_PARAMS: [],
    }

    merge_tp_slices(uc_info, str(output_dir), str(slice_dir), 2,
                    (param_name, [torch.Size([4, 3]), torch.Size([4, 2])]))

    ckpt = torch.load(output_dir / param_name / "fp32.pt", weights_only=False)
    assert ckpt[CAT_DIM] == 1
    assert torch.equal(ckpt[PARAM], torch.cat([tp0, tp1], dim=1))


def test_zero_optimizer_uc_info_comes_from_cached_state():
    param = _make_param((2, 2))
    expected_uc_info = {"key": "value"}
    setattr(param, UNIVERSAL_CHECKPOINT_INFO, expected_uc_info)

    optimizer = object.__new__(DeepSpeedZeroOptimizer)
    optimizer.bit16_groups = [[param]]
    optimizer._enable_universal_checkpoint()
    delattr(param, UNIVERSAL_CHECKPOINT_INFO)

    assert optimizer._get_universal_checkpoint_info() == expected_uc_info


def test_bf16_optimizer_uc_info_comes_from_cached_state():
    param = _make_param((2, 2))
    expected_uc_info = {"key": "value"}
    setattr(param, UNIVERSAL_CHECKPOINT_INFO, expected_uc_info)

    optimizer = object.__new__(BF16_Optimizer)
    optimizer.bf16_groups = [[param]]
    optimizer._enable_universal_checkpoint()
    delattr(param, UNIVERSAL_CHECKPOINT_INFO)

    assert optimizer._get_universal_checkpoint_info() == expected_uc_info


def test_get_param_uc_restore_meta_returns_top_level_restore_schema():
    meta = {
        "partition_dim": 1,
        "logical_shape": (4, 8),
        "output_shape": (4, ),
        "sub_param_shape": None,
        "sub_param_sizes": None,
        "target_partition_shape": (4, 4),
        "is_bias": False,
        "replicated": False,
        "conversion": {
            "partition_dim": 999
        },
    }
    param = _make_param((4, 4), meta)

    restore_meta = _get_param_uc_restore_meta(param)

    assert restore_meta["partition_dim"] == 1
    assert restore_meta["conversion"]["partition_dim"] == 999


def _write_stage3_checkpoint(ckpt_dir, shards_by_param, dp_degree, uc_info):
    # Build a synthetic AutoTP + ZeRO-3 checkpoint: one optim/model_states file per
    # (tp_rank, dp_rank), where each file holds the ZeRO-DP partition of one TP shard.
    # shards_by_param maps each parameter name to its per-TP-rank shard tensors (one
    # tensor per tp_rank). Each tp rank's model_states stores its OWN shard shapes, so
    # uneven TP splits are represented faithfully.
    os.makedirs(str(ckpt_dir), exist_ok=True)
    param_names = list(shards_by_param.keys())
    tp_degree = len(shards_by_param[param_names[0]])
    for tp_rank in range(tp_degree):
        shards = [shards_by_param[name][tp_rank] for name in param_names]
        flat = torch.cat([s.reshape(-1) for s in shards])
        param_shapes = [{name: list(shards_by_param[name][tp_rank].shape) for name in param_names}]
        per_dp = len(flat) // dp_degree
        for dp_rank in range(dp_degree):
            partition = flat[dp_rank * per_dp:(dp_rank + 1) * per_dp].clone()
            optim_outer = {
                OPTIMIZER_STATE_DICT: {
                    'state': [{
                        'exp_avg': partition * 10,
                        'exp_avg_sq': partition * 100,
                    }],
                    PARAM_GROUPS: [{}],
                },
                FP32_FLAT_GROUPS: [partition],
                ZERO_STAGE: 3,
            }
            stem = str(ckpt_dir / f"zero_pp_rank_{dp_rank}_mp_rank_{tp_rank:02d}")
            torch.save({PARAM_SHAPES: param_shapes, UNIVERSAL_CHECKPOINT_INFO: uc_info}, stem + "_model_states.pt")
            torch.save({OPTIMIZER_STATE_DICT: optim_outer}, stem + "_optim_states.pt")


def test_stage3_autotp_universal_conversion_reassembles_full_weight(tmp_path):
    # Reproduces the documented bug: AutoTP + ZeRO-3 column-parallel weight must be
    # reassembled across BOTH the TP and DP dimensions. The buggy stage3 path dropped
    # TP shards (numel == one shard); the fix reuses the stage<=2 TP-aware merge.
    full_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    tp_shards = [full_weight[0:2].contiguous(), full_weight[2:4].contiguous()]  # column-parallel, tp=2
    dp_degree = 2

    uc_info = {
        UNIVERSAL_CHECKPOINT_VERSION_KEY: UNIVERSAL_CHECKPOINT_VERSION_VALUE,
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [],  # column-parallel -> default cat dim 0
        TP_REPLICATED_PARAMETER_PATTERNS: [],
        VOCABULARY_PARAMETER_PATTERNS: [],
        PARAMETER_WITH_SUB_PARAMS: [],
    }

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    _write_stage3_checkpoint(ckpt_dir, {'fc.weight': tp_shards}, dp_degree, uc_info)

    out_dir = tmp_path / "universal"
    args = SimpleNamespace(input_folder=str(ckpt_dir),
                           output_folder=str(out_dir),
                           num_extract_workers=1,
                           num_merge_workers=1,
                           keep_temp_folder=False,
                           strict=True,
                           inject_missing_state=False)
    convert_to_universal(args)

    for state_name, scale in [('fp32', 1), ('exp_avg', 10), ('exp_avg_sq', 100)]:
        ckpt = torch.load(out_dir / "zero" / "fc.weight" / f"{state_name}.pt", weights_only=False)
        assert ckpt[CAT_DIM] == 0
        torch.testing.assert_close(ckpt[PARAM], (full_weight * scale).reshape(4, 4))


def test_stage3_autotp_universal_conversion_uneven_tp(tmp_path):
    # Uneven TP split: a column-parallel dim (5) not divisible by tp_degree (2) yields
    # shards of different shapes (3x4 and 2x4). Each rank stores its own shard shape, and
    # the merge must reshape each TP slice to its own shape before concatenating.
    full_weight = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    tp_shards = [full_weight[0:3].contiguous(), full_weight[3:5].contiguous()]  # (3,4) + (2,4)
    dp_degree = 2

    uc_info = {
        UNIVERSAL_CHECKPOINT_VERSION_KEY: UNIVERSAL_CHECKPOINT_VERSION_VALUE,
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [],
        TP_REPLICATED_PARAMETER_PATTERNS: [],
        VOCABULARY_PARAMETER_PATTERNS: [],
        PARAMETER_WITH_SUB_PARAMS: [],
    }

    ckpt_dir = tmp_path / "ckpt"
    _write_stage3_checkpoint(ckpt_dir, {'fc.weight': tp_shards}, dp_degree, uc_info)

    out_dir = tmp_path / "universal"
    args = SimpleNamespace(input_folder=str(ckpt_dir),
                           output_folder=str(out_dir),
                           num_extract_workers=1,
                           num_merge_workers=1,
                           keep_temp_folder=False,
                           strict=True,
                           inject_missing_state=False)
    convert_to_universal(args)

    for state_name, scale in [('fp32', 1), ('exp_avg', 10), ('exp_avg_sq', 100)]:
        ckpt = torch.load(out_dir / "zero" / "fc.weight" / f"{state_name}.pt", weights_only=False)
        assert ckpt[CAT_DIM] == 0
        assert tuple(ckpt[PARAM].shape) == (5, 4)
        torch.testing.assert_close(ckpt[PARAM], (full_weight * scale).reshape(5, 4))


def test_stage3_no_tp_universal_conversion_unchanged(tmp_path):
    # Regression guard: plain ZeRO-3 (tp_degree == 1) must keep using the DP-only path,
    # producing a plain {PARAM: tensor} entry with no TP merge metadata.
    full_weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    dp_degree = 2
    # No UNIVERSAL_CHECKPOINT_INFO and a single (tp=0) rank -> tp_degree detected as 1.
    _write_stage3_checkpoint(ckpt_dir=tmp_path / "ckpt",
                             shards_by_param={'fc.weight': [full_weight]},
                             dp_degree=dp_degree,
                             uc_info={})

    ckpt_dir = tmp_path / "ckpt"
    out_dir = tmp_path / "universal"
    args = SimpleNamespace(input_folder=str(ckpt_dir),
                           output_folder=str(out_dir),
                           num_extract_workers=1,
                           num_merge_workers=1,
                           keep_temp_folder=False,
                           strict=True,
                           inject_missing_state=False)
    convert_to_universal(args)

    fp32_ckpt = torch.load(out_dir / "zero" / "fc.weight" / "fp32.pt", weights_only=False)
    # DP-only merge writes the flat raw tensor (no reshape / no {PARAM: ...} dict).
    torch.testing.assert_close(fp32_ckpt, full_weight.reshape(-1))


def test_stage3_autotp_universal_conversion_replicated_param(tmp_path):
    # Regression test for params AutoTP leaves unchanged: LayerNorm/RMSNorm weights are
    # TP-replicated. collect_autotp_universal_checkpoint_info now lists them in
    # TP_REPLICATED_PARAMETER_PATTERNS, so the converter must reduce them to a single
    # copy instead of concatenating TP duplicates ([H] -> [H * tp_degree]).
    fc_full = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    fc_shards = [fc_full[0:2].contiguous(), fc_full[2:4].contiguous()]  # column-parallel, tp=2
    ln_vec = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    ln_shards = [ln_vec.clone(), ln_vec.clone()]  # replicated: identical on every tp rank

    uc_info = {
        UNIVERSAL_CHECKPOINT_VERSION_KEY: UNIVERSAL_CHECKPOINT_VERSION_VALUE,
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [],
        TP_REPLICATED_PARAMETER_PATTERNS: [r"^ln\.weight$"],
        VOCABULARY_PARAMETER_PATTERNS: [],
        PARAMETER_WITH_SUB_PARAMS: [],
    }

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    _write_stage3_checkpoint(ckpt_dir, {'fc.weight': fc_shards, 'ln.weight': ln_shards}, dp_degree=1, uc_info=uc_info)

    out_dir = tmp_path / "universal"
    args = SimpleNamespace(input_folder=str(ckpt_dir),
                           output_folder=str(out_dir),
                           num_extract_workers=1,
                           num_merge_workers=1,
                           keep_temp_folder=False,
                           strict=False,
                           inject_missing_state=False)
    convert_to_universal(args)

    # Column-parallel weight: default cat(dim=0) reassembles the full [4, 4].
    fc_ckpt = torch.load(out_dir / "zero" / "fc.weight" / "fp32.pt", weights_only=False)
    assert fc_ckpt[CAT_DIM] == 0
    torch.testing.assert_close(fc_ckpt[PARAM], fc_full)

    # Replicated weight: reduced to a single copy, NOT concatenated to [8].
    ln_ckpt = torch.load(out_dir / "zero" / "ln.weight" / "fp32.pt", weights_only=False)
    assert tuple(ln_ckpt[PARAM].shape) == (4, )
    torch.testing.assert_close(ln_ckpt[PARAM], ln_vec)


def test_group_per_tp_shapes_handles_pp_local_params():
    # mp_rank_files are pp-major in the stage<=2 path: pp0_tp0, pp0_tp1, ..., pp1_tp0, ...
    # This matches meg_2d_parallel_map.simple_init(): index i -> (pp=i // tp_degree,
    # tp=i % tp_degree). PP-local params (only in some PP stages) must survive the
    # per-TP PP-union.
    slice_shapes_by_tp = [
        {
            'fc.weight': (3, 4),
            'layer0.weight': (2, 4)
        },  # pp0_tp0
        {
            'fc.weight': (2, 4),
            'layer0.weight': (2, 4)
        },  # pp0_tp1 (uneven fc.shape)
        {
            'fc.weight': (3, 4),
            'layer1.weight': (2, 4)
        },  # pp1_tp0
        {
            'fc.weight': (2, 4),
            'layer1.weight': (2, 4)
        },  # pp1_tp1
    ]
    result = _group_per_tp_shapes(slice_shapes_by_tp, pp_degree=2, tp_degree=2)

    assert 'layer0.weight' in result, 'PP-local param lost'
    assert 'layer1.weight' in result, 'PP-local param lost'
    assert result['fc.weight'] == [(3, 4), (2, 4)]
    assert result['layer0.weight'] == [(2, 4), (2, 4)]
    assert result['layer1.weight'] == [(2, 4), (2, 4)]


def test_group_per_tp_shapes_rejects_disagreeing_pipeline_replicas():
    # A tied parameter is stored in every PP stage's file. Replicas that disagree mean the
    # tie was partitioned inconsistently, which must not be papered over by keeping the
    # stage that happened to be read last.
    slice_shapes_by_tp = [
        {
            'tied_modules.embed.weight': (3, 4)
        },  # pp0_tp0
        {
            'tied_modules.embed.weight': (2, 4)
        },  # pp0_tp1
        {
            'tied_modules.embed.weight': (1, 4)
        },  # pp1_tp0 disagrees with pp0_tp0
        {
            'tied_modules.embed.weight': (2, 4)
        },  # pp1_tp1
    ]

    with pytest.raises(AssertionError, match='disagree on shape'):
        _group_per_tp_shapes(slice_shapes_by_tp, pp_degree=2, tp_degree=2)


class TestRealCheckpointUniversalConversionTPxPP(DistributedTest):
    # Generate a real ZeRO-1 checkpoint with TP=2, PP=2 on CPU (gloo) using the
    # production DeepSpeed writer, then convert it to a universal checkpoint.
    # The writer numbers mp_rank files PP-major (mp_rank = pp * tp_degree + tp,
    # matching reshape_meg_2d.simple_init); _group_per_tp_shapes must use the same
    # ordering. The TP-major index bug dropped a shape (None) for one TP rank of
    # every parameter, so the assertion below fails fast on regression and the
    # converter would crash on its reshape.
    world_size = 4
    backend = "gloo"
    requires_cuda_env = False

    def _launch_procs(self, num_procs, init_method):
        # CPU/gloo test: the number of processes is not bound to the accelerator's
        # device_count() (CPU sockets), so bypass the base class's per-device gate
        # that would otherwise skip a 4-process test on a single-socket CPU box.
        torch.multiprocessing.set_start_method('forkserver', force=True)
        self._launch_daemonic_procs(num_procs, init_method)

    def test(self, tmpdir):
        hidden = 8
        # plain nn.Linear is not truly TP-sharded (that needs Megatron layers), but
        # the topology/grid, checkpoint format, and PP-major mp_rank numbering are
        # all real production code, which is exactly the contract under test.
        topology = PipeModelDataParallelTopology(num_pp=2, num_mp=2, num_dp=1)
        layers = [nn.Linear(hidden, hidden) for _ in range(4)] + [nn.Linear(hidden, 1)]
        model = PipelineModule(layers=layers, topology=topology)

        config = {
            "train_batch_size": self.world_size,
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "zero_optimization": {
                "stage": 1
            },
            "pipeline": {
                "activation_checkpoint_interval": 0
            },
        }
        model_engine, _, _, _ = deepspeed.initialize(model=model,
                                                     config=config,
                                                     model_parameters=list(model.parameters()))

        if model_engine.is_first_stage or model_engine.is_last_stage:
            batch = (torch.randn(1, hidden), torch.zeros(1))
            data_iter = iter(RepeatingLoader([batch]))
        else:
            data_iter = None
        model_engine.train_batch(data_iter=data_iter)

        ckpt_dir = str(tmpdir)
        model_engine.save_checkpoint(ckpt_dir, tag="e1")
        dist.barrier()

        # Every rank loads the shared mp_rank files and checks the grouping on real
        # writer data. The grouping assertion is symmetric, so all ranks fail
        # together (no hang) if the index bug regresses.
        stage_dir = os.path.join(ckpt_dir, "e1")
        mp_files = sorted(glob.glob(os.path.join(stage_dir, "mp_rank_*_model_states.pt")),
                          key=lambda p: int(os.path.basename(p).split("_")[2]))
        assert len(mp_files) == 4, f"expected 4 mp_rank files, got {len(mp_files)}"
        shapes_by_tp = []
        for f in mp_files:
            sd = torch.load(f, map_location="cpu", weights_only=False)
            shapes_by_tp.append(dict((k, v) for d in sd[PARAM_SHAPES] for k, v in d.items()))
        grouped = _group_per_tp_shapes(shapes_by_tp, pp_degree=2, tp_degree=2)
        assert grouped, "no parameters grouped from checkpoint"
        for name, per_tp in grouped.items():
            assert len(per_tp) == 2, f"{name}: expected 2 TP shapes, got {per_tp}"
            assert all(s is not None for s in per_tp), f"{name}: missing TP shape in {per_tp}"

        # Rank 0 also exercises the full converter end-to-end on the real checkpoint.
        if dist.get_rank() == 0:
            out_dir = os.path.join(str(tmpdir), "universal")
            args = SimpleNamespace(input_folder=stage_dir,
                                   output_folder=out_dir,
                                   num_extract_workers=1,
                                   num_merge_workers=1,
                                   keep_temp_folder=False,
                                   strict=True,
                                   inject_missing_state=True)
            convert_to_universal(args)
            assert os.path.isdir(os.path.join(out_dir, "zero")), "universal 'zero' dir not written"
        dist.barrier()


CP_TAG = "uneven_tp"
UNIVERSAL_TAG = f"{CP_TAG}_universal"


class UnevenVocabLmHeadModel(torch.nn.Module):

    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        return self.lm_head(x).sum()


class GQAAttentionModel(torch.nn.Module):
    """Column-parallel q/k/v feeding a row-parallel o_proj, sharded on kv-head boundaries."""

    class Config:

        def __init__(self, hidden_dim, num_heads):
            self.hidden_size = hidden_dim
            self.num_attention_heads = num_heads
            self.num_key_value_heads = num_heads

    class Attention(torch.nn.Module):

        def __init__(self, hidden_dim):
            super().__init__()
            self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

        def forward(self, x):
            return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))

    class Layer(torch.nn.Module):

        def __init__(self, hidden_dim):
            super().__init__()
            self.self_attn = GQAAttentionModel.Attention(hidden_dim)

        def forward(self, x):
            return self.self_attn(x)

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.layers = torch.nn.ModuleList([GQAAttentionModel.Layer(hidden_dim)])
        self.config = GQAAttentionModel.Config(hidden_dim, num_heads)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.sum()


def _convert_to_universal(checkpoint_dir, universal_dir):
    convert_to_universal(
        SimpleNamespace(input_folder=checkpoint_dir,
                        output_folder=universal_dir,
                        num_extract_workers=1,
                        num_merge_workers=1,
                        keep_temp_folder=False,
                        strict=True,
                        inject_missing_state=False))


def _train_steps(engine, hidden_dim, steps=3):
    for _ in range(steps):
        batch = torch.randn(2, hidden_dim, device=engine.device)
        dist.broadcast(batch, src=0)
        engine.backward(engine(batch))
        engine.step()


def _save_and_convert(engine, tmpdir):
    engine.save_checkpoint(tmpdir, tag=CP_TAG, client_state={"iteration": 3})
    dist.barrier()
    if dist.get_rank() == 0:
        _convert_to_universal(os.path.join(tmpdir, CP_TAG), os.path.join(tmpdir, UNIVERSAL_TAG))
    dist.barrier()


class TestUnevenColumnUniversalCheckpoint(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    def test_save_convert_load_uneven_lm_head(self, tmpdir):
        hidden_dim = 12
        vocab_size = 101  # Not divisible by the two TP ranks, giving shards of 51 and 50.
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "tensor_parallel": {
                "autotp_size": self.world_size,
                "partition_config": {
                    "use_default_specs":
                    False,
                    "layer_specs": [{
                        "patterns": [r".*lm_head\.weight$"],
                        "partition_type": "column",
                        "gather_output": True,
                    }],
                },
            },
            "zero_optimization": {
                "stage": 1
            },
        }

        from copy import deepcopy

        torch.manual_seed(42)
        model = UnevenVocabLmHeadModel(hidden_dim, vocab_size)
        reference = deepcopy(model)  # unsharded, plain nn.Linear; independent source of truth
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

        tp_group = groups.get_tensor_model_parallel_group()
        tp_rank = groups.get_tensor_model_parallel_rank()
        shard_sizes = get_shard_size_list(vocab_size, self.world_size, "lm_head")
        assert shard_sizes == [51, 50], shard_sizes
        assert engine.module.lm_head.weight.shape[0] == shard_sizes[tp_rank]

        dev = engine.device
        dtype = engine.module.lm_head.weight.dtype

        # ====== Phase 1: sharded forward == unsharded reference, element-wise ======
        x = torch.randn(2, hidden_dim, device=dev, dtype=dtype)
        dist.broadcast(x, src=0, group=tp_group)
        x_ref = x.detach().cpu()
        with torch.no_grad():
            out = engine.module.lm_head(x).cpu()
        torch.testing.assert_close(out, reference.lm_head(x_ref), atol=1e-2, rtol=1e-2)

        # ====== Phase 2: universal ckpt save -> convert -> load is lossless ======
        _train_steps(engine, hidden_dim)
        expected_weight = engine.module.lm_head.weight.detach().cpu().clone()
        expected_bias = engine.module.lm_head.bias.detach().cpu().clone()
        # Independently all-gather the trained shards so the merged-file and restored-forward
        # references are built without AutoTP's own gather.
        exp_w_full = _all_gather_cat_dim0(engine.module.lm_head.weight.detach(), tp_group).cpu()
        exp_b_full = _all_gather_cat_dim0(engine.module.lm_head.bias.detach().view(-1, 1), tp_group).view(-1).cpu()
        _save_and_convert(engine, tmpdir)

        # Checked on every rank: a rank-0-only failure would leave the others hanging in
        # the next collective instead of failing the test.
        _assert_merged_matches(tmpdir, "lm_head.weight", exp_w_full)
        _assert_merged_matches(tmpdir, "lm_head.bias", exp_b_full)

        config_dict["checkpoint"] = {"load_universal": True}
        torch.manual_seed(123)
        restored = UnevenVocabLmHeadModel(hidden_dim, vocab_size)
        restored_engine, _, _, _ = deepspeed.initialize(model=restored,
                                                        model_parameters=restored.parameters(),
                                                        config=config_dict)
        restored_engine.load_checkpoint(tmpdir, tag=UNIVERSAL_TAG, load_optimizer_states=True)

        torch.testing.assert_close(restored_engine.module.lm_head.weight.detach().cpu(), expected_weight)
        torch.testing.assert_close(restored_engine.module.lm_head.bias.detach().cpu(), expected_bias)

        # Restored forward == F.linear with the independently gathered expected weights.
        x2 = torch.randn(2, hidden_dim, device=dev, dtype=dtype)
        dist.broadcast(x2, src=0, group=tp_group)
        with torch.no_grad():
            restored_out = restored_engine.module.lm_head(x2).cpu()
        ref2 = torch.nn.functional.linear(x2.cpu(), exp_w_full, exp_b_full)
        torch.testing.assert_close(restored_out, ref2, atol=1e-2, rtol=1e-2)

        # The optimizer must be usable after the restore.
        _train_steps(restored_engine, hidden_dim, steps=1)


class TestUnevenRowUniversalCheckpoint(DistributedTest):
    world_size = 4
    reuse_dist_env = False

    def test_save_convert_load_uneven_row_parallel(self, tmpdir):
        hidden_dim = 384
        num_heads = 6  # Not divisible by the four TP ranks, giving shards of 128/128/64/64.
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "tensor_parallel": {
                "autotp_size": self.world_size
            },
            "zero_optimization": {
                "stage": 1
            },
        }

        from copy import deepcopy

        torch.manual_seed(42)
        model = GQAAttentionModel(hidden_dim, num_heads)
        reference = deepcopy(model)  # unsharded, plain nn.Linear; independent source of truth
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

        tp_group = groups.get_tensor_model_parallel_group()
        tp_rank = groups.get_tensor_model_parallel_rank()
        attn = engine.module.layers[0].self_attn
        # Column and row parallelism must shard the same dimension identically.
        assert attn.q_proj.weight.shape[0] == attn.o_proj.weight.shape[1]
        head_dim = hidden_dim // num_heads
        head_shards = get_shard_size_list(num_heads, self.world_size, attn.q_proj.name)
        assert head_shards == [2, 2, 1, 1], head_shards
        dim_shards = [h * head_dim for h in head_shards]
        assert attn.q_proj.weight.shape[0] == dim_shards[tp_rank], (attn.q_proj.weight.shape, dim_shards)

        dev = engine.device
        dtype = attn.q_proj.weight.dtype

        # ====== Phase 1: sharded forward == unsharded reference, element-wise ======
        x = torch.randn(2, hidden_dim, device=dev, dtype=dtype)
        dist.broadcast(x, src=0, group=tp_group)
        x_ref = x.detach().cpu()
        with torch.no_grad():
            out = attn(x).cpu()
        ref_out = reference.layers[0].self_attn(x_ref)
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

        # ====== Phase 2: universal ckpt save -> convert -> load is lossless ======
        _train_steps(engine, hidden_dim)
        expected_q = attn.q_proj.weight.detach().cpu().clone()
        expected_o = attn.o_proj.weight.detach().cpu().clone()
        # Independently all-gather the trained q/k/v (column, dim 0) and o (row, dim 1) weights
        # so the merged-file and restored-forward references are built without AutoTP's gather.
        exp_q = _all_gather_cat_dim0(attn.q_proj.weight.detach(), tp_group).cpu()
        exp_k = _all_gather_cat_dim0(attn.k_proj.weight.detach(), tp_group).cpu()
        exp_v = _all_gather_cat_dim0(attn.v_proj.weight.detach(), tp_group).cpu()
        exp_o = _all_gather_cat_dim1(attn.o_proj.weight.detach(), tp_group).cpu()
        _save_and_convert(engine, tmpdir)

        # Checked on every rank: a rank-0-only failure would leave the others hanging in
        # the next collective instead of failing the test.
        _assert_merged_matches(tmpdir, "layers.0.self_attn.q_proj.weight", exp_q)
        _assert_merged_matches(tmpdir, "layers.0.self_attn.o_proj.weight", exp_o)

        config_dict["checkpoint"] = {"load_universal": True}
        torch.manual_seed(123)
        restored = GQAAttentionModel(hidden_dim, num_heads)
        restored_engine, _, _, _ = deepspeed.initialize(model=restored,
                                                        model_parameters=restored.parameters(),
                                                        config=config_dict)
        restored_engine.load_checkpoint(tmpdir, tag=UNIVERSAL_TAG, load_optimizer_states=True)

        restored_attn = restored_engine.module.layers[0].self_attn
        torch.testing.assert_close(restored_attn.q_proj.weight.detach().cpu(), expected_q)
        torch.testing.assert_close(restored_attn.o_proj.weight.detach().cpu(), expected_o)

        # Restored forward == F.linear attention with the independently gathered weights.
        x2 = torch.randn(2, hidden_dim, device=dev, dtype=dtype)
        dist.broadcast(x2, src=0, group=tp_group)
        with torch.no_grad():
            restored_out = restored_attn(x2).cpu()
        x2c = x2.cpu()
        ref2 = torch.nn.functional.linear(
            torch.nn.functional.linear(x2c, exp_q) + torch.nn.functional.linear(x2c, exp_k) +
            torch.nn.functional.linear(x2c, exp_v), exp_o)
        torch.testing.assert_close(restored_out, ref2, atol=1e-2, rtol=1e-2)

        _train_steps(restored_engine, hidden_dim, steps=1)


def test_merge_tp_slices_realigns_rank_that_wrote_no_fragment(tmp_path):
    # A rank owning none of a parameter is dropped from the hp mapping and writes no fragment
    # at all. Pairing the remaining slices with the shard widths by position would then shift
    # rank 2 onto rank 1's widths, so the gap has to be filled back in.
    slice_dir = tmp_path / "slices"
    output_dir = tmp_path / "out"
    param_name = "module.qkv_proj.weight"
    shard_widths = [[2, 0, 2], [1, 0, 1], [1, 0, 1]]

    full = torch.arange(16, dtype=torch.float32).view(8, 2)
    rows = {0: [0, 1, 4, 6], 2: [2, 3, 5, 7]}
    for tp_index, row_ids in rows.items():
        _write_tp_states(slice_dir, param_name, tp_index, full[row_ids].contiguous())

    uc_info = {
        UNIVERSAL_CHECKPOINT_VERSION_KEY: UNIVERSAL_CHECKPOINT_VERSION_VALUE,
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [],
        TP_REPLICATED_PARAMETER_PATTERNS: [],
        PARAMETER_WITH_SUB_PARAMS: [{
            "patterns": [rf"^{param_name}$"],
            "shape": [(4, 2, 2), 2],
            "partition_dim": 0,
        }],
        SUB_PARAM_SHARD_WIDTHS: {
            rf"^{param_name}$": shard_widths
        },
    }

    merge_tp_slices(uc_info, str(output_dir), str(slice_dir), 3,
                    (param_name, [torch.Size([4, 2]), torch.Size([0, 2]),
                                  torch.Size([4, 2])]))

    ckpt = torch.load(output_dir / param_name / "fp32.pt", weights_only=False)
    torch.testing.assert_close(ckpt[PARAM], full)


def test_merge_zero_shards_rejects_a_missing_rank_that_owns_elements(tmp_path):
    # A rank whose recorded shape is not empty must have written a fragment. Standing in a
    # zero-sized placeholder there would quietly drop real optimizer state.
    param_name = "module.proj.weight"
    _write_tp_states(tmp_path, param_name, 0, torch.zeros(4, 2))

    with pytest.raises(AssertionError, match="not empty"):
        ds_to_universal._merge_zero_shards(str(tmp_path / param_name), "fp32", 2,
                                           [torch.Size([4, 2]), torch.Size([4, 2])])


def _all_gather_cat_dim0(local, group):
    """Plain all_gather of uneven shards along dim 0, auto-discovering per-rank widths.

    Independent of AutoTP's gather_params: a bug in the gather under test cannot hide
    behind a buggy expected value.
    """
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    local_rows = torch.tensor([local.shape[0]], device=local.device, dtype=torch.long)
    row_counts = [torch.empty(1, dtype=torch.long, device=local.device) for _ in range(world)]
    dist.all_gather(row_counts, local_rows, group=group)
    sizes = [int(c.item()) for c in row_counts]
    max_rows = max(sizes)
    pad = max_rows - local.shape[0]
    if pad > 0:
        local_padded = torch.cat(
            [local, torch.zeros(pad, *local.shape[1:], dtype=local.dtype, device=local.device)], dim=0)
    else:
        local_padded = local
    gathered = [torch.empty_like(local_padded) for _ in range(world)]
    dist.all_gather(gathered, local_padded.contiguous(), group=group)
    return torch.cat([gathered[r][:sizes[r]] for r in range(world)], dim=0)


def _all_gather_cat_dim1(local, group):
    """Same as _all_gather_cat_dim0, but for row-parallel weights sharded along dim 1."""
    return _all_gather_cat_dim0(local.t().contiguous(), group).t().contiguous()


def _assert_merged_matches(tmpdir, param_name, expected):
    """The merged universal slice for `param_name` must equal `expected` element-wise."""
    merged = torch.load(os.path.join(tmpdir, UNIVERSAL_TAG, "zero", param_name, "fp32.pt"), weights_only=False)
    assert merged[PARAM].shape == expected.shape, (param_name, merged[PARAM].shape, expected.shape)
    torch.testing.assert_close(merged[PARAM].cpu().float(), expected.float())


class TestUnevenTp3TrainingAndUniversalCheckpoint(DistributedTest):
    """tp=3 training must not change numerics vs an unsharded reference, and a universal
    checkpoint round-trip must preserve both the weights and the forward output."""

    world_size = 3
    reuse_dist_env = False

    def test_strict_correctness(self, tmpdir):
        hidden_dim = 12
        vocab_size = 10  # 10 / 3 -> [4, 3, 3]

        torch.manual_seed(42)
        reference = nn.Linear(hidden_dim, vocab_size)
        model = UnevenVocabLmHeadModel(hidden_dim, vocab_size)
        with torch.no_grad():
            model.lm_head.weight.copy_(reference.weight)
            model.lm_head.bias.copy_(reference.bias)

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "tensor_parallel": {
                "autotp_size": self.world_size,
                "partition_config": {
                    "use_default_specs":
                    False,
                    "layer_specs": [{
                        "patterns": [r".*lm_head\.weight$"],
                        "partition_type": "column",
                        "gather_output": True,
                    }],
                },
            },
            "zero_optimization": {
                "stage": 1
            },
        }
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

        tp_group = groups.get_tensor_model_parallel_group()
        tp_rank = groups.get_tensor_model_parallel_rank()
        shard_sizes = get_shard_size_list(vocab_size, self.world_size, "lm_head")
        assert sum(shard_sizes) == vocab_size, shard_sizes
        assert shard_sizes == [4, 3, 3], shard_sizes
        assert engine.module.lm_head.weight.shape[0] == shard_sizes[tp_rank]

        dev = engine.device
        dtype = engine.module.lm_head.weight.dtype
        off = sum(shard_sizes[:tp_rank])
        w = shard_sizes[tp_rank]

        # ====== Phase 1: runtime correctness via a STANDALONE LinearLayer ======
        # The engine's optimizer owns .grad after engine.backward(), so gradient
        # correctness is checked on an independent LinearLayer (same pattern as
        # run_tp_layer_fwd_bwd), which leaves .grad on the parameter. The unsharded
        # nn.Linear `reference` is the independent source of truth for both output
        # and gradients.
        from copy import deepcopy
        from deepspeed.module_inject.layers import LinearLayer
        x = torch.randn(2, hidden_dim, device=dev, dtype=dtype)
        dist.broadcast(x, src=0, group=tp_group)
        x_ref = x.detach().cpu()

        linear = LinearLayer(deepcopy(reference).to(device=dev, dtype=dtype), tp_group, gather_output=True)
        out = linear(x)
        torch.testing.assert_close(out.detach().cpu(), reference(x_ref).detach(), atol=1e-2, rtol=1e-2)

        out.sum().backward()
        reference(x_ref).sum().backward()
        torch.testing.assert_close(linear.weight.grad.detach().cpu(),
                                   reference.weight.grad[off:off + w],
                                   atol=1e-3,
                                   rtol=1e-3)
        torch.testing.assert_close(linear.bias.grad.detach().cpu(),
                                   reference.bias.grad[off:off + w],
                                   atol=1e-3,
                                   rtol=1e-3)

        # ====== Phase 2: universal ckpt save -> convert -> load is lossless ======
        _train_steps(engine, hidden_dim, steps=3)
        exp_w_shard = engine.module.lm_head.weight.detach().cpu().clone()
        exp_b_shard = engine.module.lm_head.bias.detach().cpu().clone()
        exp_w_full = _all_gather_cat_dim0(engine.module.lm_head.weight.detach(), tp_group).cpu()
        exp_b_full = _all_gather_cat_dim0(engine.module.lm_head.bias.detach().view(-1, 1), tp_group).view(-1).cpu()

        _save_and_convert(engine, tmpdir)

        _assert_merged_matches(tmpdir, "lm_head.weight", exp_w_full)
        _assert_merged_matches(tmpdir, "lm_head.bias", exp_b_full)

        config_dict["checkpoint"] = {"load_universal": True}
        torch.manual_seed(123)
        restored_model = UnevenVocabLmHeadModel(hidden_dim, vocab_size)
        restored_engine, _, _, _ = deepspeed.initialize(model=restored_model,
                                                        model_parameters=restored_model.parameters(),
                                                        config=config_dict)
        restored_engine.load_checkpoint(tmpdir, tag=UNIVERSAL_TAG, load_optimizer_states=True)

        # 2a: restored per-rank shards == pre-save shards
        torch.testing.assert_close(restored_engine.module.lm_head.weight.detach().cpu(), exp_w_shard)
        torch.testing.assert_close(restored_engine.module.lm_head.bias.detach().cpu(), exp_b_shard)

        # 2b: restored forward (gathered) == F.linear with independently all-gathered expected weights
        x2 = torch.randn(2, hidden_dim, device=dev, dtype=dtype)
        dist.broadcast(x2, src=0, group=tp_group)
        with torch.no_grad():
            restored_out = restored_engine.module.lm_head(x2).cpu()
        ref2 = torch.nn.functional.linear(x2.cpu(), exp_w_full, exp_b_full)
        torch.testing.assert_close(restored_out, ref2, atol=1e-2, rtol=1e-2)

        # 2c: optimizer state usable
        _train_steps(restored_engine, hidden_dim, steps=1)


class TestUnevenTp3RowParallelGQA(DistributedTest):
    """tp=3 row-parallel (GQA-style attention) must not change numerics vs an unsharded
    reference, and a universal checkpoint round-trip must preserve both the column-side
    (q/k/v) and row-side (o_proj) uneven shards.

    num_heads=4 over tp=3 shards heads as [2, 1, 1], so with head_dim=3 the q/k/v
    column shards and the o_proj row shard all become [6, 3, 3] on dim 0/1. This is the
    commit 847c4e4 scenario (column output dim == following row input dim must agree per
    rank) at a non-power-of-two tp_size.
    """

    world_size = 3
    reuse_dist_env = False

    def test_strict_correctness(self, tmpdir):
        from copy import deepcopy

        hidden_dim = 12
        num_heads = 4  # 4 / 3 -> [2, 1, 1] heads -> dim shards [6, 3, 3]

        torch.manual_seed(42)
        model = GQAAttentionModel(hidden_dim, num_heads)
        reference = deepcopy(model)  # unsharded, plain nn.Linear; independent source of truth

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "tensor_parallel": {
                "autotp_size": self.world_size,
            },
            "zero_optimization": {
                "stage": 1
            },
        }
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

        tp_group = groups.get_tensor_model_parallel_group()
        tp_rank = groups.get_tensor_model_parallel_rank()
        attn = engine.module.layers[0].self_attn

        # Column q/k/v output dim must equal row o_proj input dim on every rank, otherwise
        # the row matmul shape-mismatches at runtime (the original 847c4e4 bug).
        assert attn.q_proj.weight.shape[0] == attn.o_proj.weight.shape[1], (attn.q_proj.weight.shape,
                                                                            attn.o_proj.weight.shape)
        head_dim = hidden_dim // num_heads
        head_shards = get_shard_size_list(num_heads, self.world_size, attn.q_proj.name)
        assert head_shards == [2, 1, 1], head_shards
        dim_shards = [h * head_dim for h in head_shards]
        assert attn.q_proj.weight.shape[0] == dim_shards[tp_rank], (attn.q_proj.weight.shape, dim_shards)

        dev = engine.device
        dtype = attn.q_proj.weight.dtype

        # ====== Phase 1: forward == unsharded reference, element-wise ======
        x = torch.randn(2, hidden_dim, device=dev, dtype=dtype)
        dist.broadcast(x, src=0, group=tp_group)
        x_ref = x.detach().cpu()
        with torch.no_grad():
            out = engine.module.layers[0].self_attn(x).cpu()
        ref_out = reference.layers[0].self_attn(x_ref)
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

        # ====== Phase 2: universal ckpt save -> convert -> load is lossless ======
        _train_steps(engine, hidden_dim, steps=3)

        # Independently all-gather the trained q/k/v (column, dim 0) and o (row, dim 1)
        # weights so Phase 2b's reference is built from plain F.linear, not AutoTP.
        exp_q = _all_gather_cat_dim0(attn.q_proj.weight.detach(), tp_group).cpu()
        exp_k = _all_gather_cat_dim0(attn.k_proj.weight.detach(), tp_group).cpu()
        exp_v = _all_gather_cat_dim0(attn.v_proj.weight.detach(), tp_group).cpu()
        exp_o = _all_gather_cat_dim1(attn.o_proj.weight.detach(), tp_group).cpu()
        # Per-rank expected shards for the restore-fidelity check.
        exp_q_shard = attn.q_proj.weight.detach().cpu().clone()
        exp_o_shard = attn.o_proj.weight.detach().cpu().clone()

        _save_and_convert(engine, tmpdir)

        _assert_merged_matches(tmpdir, "layers.0.self_attn.q_proj.weight", exp_q)
        _assert_merged_matches(tmpdir, "layers.0.self_attn.o_proj.weight", exp_o)

        config_dict["checkpoint"] = {"load_universal": True}
        torch.manual_seed(123)
        restored_model = GQAAttentionModel(hidden_dim, num_heads)
        restored_engine, _, _, _ = deepspeed.initialize(model=restored_model,
                                                        model_parameters=restored_model.parameters(),
                                                        config=config_dict)
        restored_engine.load_checkpoint(tmpdir, tag=UNIVERSAL_TAG, load_optimizer_states=True)
        restored_attn = restored_engine.module.layers[0].self_attn

        # 2a: restored per-rank q_proj (column) and o_proj (row) shards == pre-save shards.
        torch.testing.assert_close(restored_attn.q_proj.weight.detach().cpu(), exp_q_shard)
        torch.testing.assert_close(restored_attn.o_proj.weight.detach().cpu(), exp_o_shard)

        # 2b: restored forward == F.linear attention with independently-gathered weights.
        x2 = torch.randn(2, hidden_dim, device=dev, dtype=dtype)
        dist.broadcast(x2, src=0, group=tp_group)
        with torch.no_grad():
            restored_out = restored_engine.module.layers[0].self_attn(x2).cpu()
        x2c = x2.cpu()
        ref2 = torch.nn.functional.linear(
            torch.nn.functional.linear(x2c, exp_q) + torch.nn.functional.linear(x2c, exp_k) +
            torch.nn.functional.linear(x2c, exp_v), exp_o)
        torch.testing.assert_close(restored_out, ref2, atol=1e-2, rtol=1e-2)

        # 2c: optimizer state usable.
        _train_steps(restored_engine, hidden_dim, steps=1)

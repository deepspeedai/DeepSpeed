# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import types
from types import SimpleNamespace

import torch

from deepspeed.checkpoint.constants import (CAT_DIM, FP32_FLAT_GROUPS, FP32_WEIGHT_KEY, OPTIMIZER_STATE_DICT, PARAM,
                                            PARAM_GROUPS, PARAM_SHAPES, PARAMETER_WITH_ROW_PARALLELISM_PATTERNS,
                                            PARAMETER_WITH_SUB_PARAMS, SUB_PARAM_SHAPE,
                                            TP_REPLICATED_PARAMETER_PATTERNS, UNIVERSAL_CHECKPOINT_INFO,
                                            UNIVERSAL_CHECKPOINT_VERSION_KEY, UNIVERSAL_CHECKPOINT_VERSION_VALUE,
                                            VOCABULARY_PARAMETER_PATTERNS, ZERO_STAGE)
from deepspeed.checkpoint.universal_checkpoint import SubparamShape as CheckpointSubparamShape
from deepspeed.checkpoint.ds_to_universal import main as convert_to_universal, merge_tp_slices
from deepspeed.checkpoint.universal_checkpoint import (_get_param_uc_restore_meta, _resolve_autotp_partition,
                                                       load_hp_checkpoint_state)
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer


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


def test_merge_tp_slices_uses_row_parallel_cat_dim(tmp_path):
    slice_dir = tmp_path / "slices"
    output_dir = tmp_path / "out"
    param_name = "module.proj.weight"

    tp0 = torch.arange(16, dtype=torch.float32).view(4, 4)
    tp1 = torch.arange(16, 32, dtype=torch.float32).view(4, 4)
    _write_tp_states(slice_dir, param_name, 0, tp0)
    _write_tp_states(slice_dir, param_name, 1, tp1)

    uc_info = {
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [rf"^{param_name}$"],
        TP_REPLICATED_PARAMETER_PATTERNS: [],
        PARAMETER_WITH_SUB_PARAMS: [],
    }

    merge_tp_slices(uc_info, str(output_dir), str(slice_dir), 2,
                    (param_name, [torch.Size([4, 4]), torch.Size([4, 4])]))

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


def _write_stage3_checkpoint(ckpt_dir, tp_shards, dp_degree, uc_info):
    # Build a synthetic AutoTP + ZeRO-3 checkpoint: one optim/model_states file per
    # (tp_rank, dp_rank), where each file holds the ZeRO-DP partition of one TP shard.
    # Each tp rank's model_states stores its OWN shard shape, so uneven TP splits are
    # represented faithfully.
    os.makedirs(str(ckpt_dir), exist_ok=True)
    for tp_rank, shard in enumerate(tp_shards):
        param_shapes = [{'fc.weight': list(shard.shape)}]
        flat = shard.reshape(-1)
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
    _write_stage3_checkpoint(ckpt_dir, tp_shards, dp_degree, uc_info)

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
    _write_stage3_checkpoint(ckpt_dir, tp_shards, dp_degree, uc_info)

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
    _write_stage3_checkpoint(ckpt_dir=tmp_path / "ckpt", tp_shards=[full_weight], dp_degree=dp_degree, uc_info={})

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

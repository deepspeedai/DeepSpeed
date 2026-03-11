import types
from types import SimpleNamespace

import torch

from deepspeed.checkpoint.constants import (CAT_DIM, FP32_WEIGHT_KEY, PARAM, PARAMETER_WITH_ROW_PARALLELISM_PATTERNS,
                                           PARAMETER_WITH_SUB_PARAMS, SUB_PARAM_SHAPE,
                                           TP_REPLICATED_PARAMETER_PATTERNS, UNIVERSAL_CHECKPOINT_INFO)
from deepspeed.checkpoint.constants import SubparamShape as CheckpointSubparamShape
from deepspeed.checkpoint.ds_to_universal import merge_tp_slices
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
        return self._param

    def get_optim_state_keys(self):
        return []


def _make_param(shape, meta=None):
    param = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float32))
    param._hp_mapping = _DummyHPMapping(param)
    if meta is not None:
        setattr(param, 'ds_autotp_universal_checkpoint_meta', meta)
    return param


def test_resolve_autotp_partition_row_parallel_weight():
    param = _make_param((4, 4), {
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
    param = _make_param((3, 4), {
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


def test_resolve_autotp_partition_replicated_bias():
    full_hp_param = torch.arange(8, dtype=torch.float32)
    param = _make_param((8, ), {
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
    param = _make_param((4, 4), {
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

    ckpt_dir = tmp_path / "weight"
    ckpt_dir.mkdir(parents=True)
    full_hp_param = torch.arange(32, dtype=torch.float32).view(4, 8)
    torch.save({PARAM: full_hp_param}, ckpt_dir / f"{FP32_WEIGHT_KEY}.pt")

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {PARAM: full_hp_param} if str(args[0]).endswith("fp32.pt") else 0)

    step = param.load_hp_checkpoint_state(str(ckpt_dir), tp_rank=1, tp_world_size=2)

    assert step is None
    expected = full_hp_param.chunk(2, dim=1)[1].flatten()
    assert torch.equal(param.data.flatten(), expected)


def _write_tp_slice(base_dir, param_name, tp_idx, state_name, tensor):
    shard_dir = base_dir / param_name / str(tp_idx)
    shard_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, shard_dir / f"{state_name}.00")


def test_merge_tp_slices_emits_subparam_shape_metadata(tmp_path):
    slice_dir = tmp_path / "slices"
    output_dir = tmp_path / "out"
    param_name = "module.qkv.weight"

    tp0 = torch.arange(12, dtype=torch.float32).view(3, 4)
    tp1 = torch.arange(12, 24, dtype=torch.float32).view(3, 4)
    _write_tp_slice(slice_dir, param_name, 0, "fp32", tp0)
    _write_tp_slice(slice_dir, param_name, 1, "fp32", tp1)

    uc_info = {
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [],
        TP_REPLICATED_PARAMETER_PATTERNS: [],
        PARAMETER_WITH_SUB_PARAMS: [{
            "patterns": [rf"^{param_name}$"],
            "shape": [(2, 2, 2), 4],
            "partition_dim": 0,
        }],
    }

    ds_checkpoint = SimpleNamespace(get_checkpoint_info=lambda key: uc_info if key == UNIVERSAL_CHECKPOINT_INFO else {})

    unmatched = merge_tp_slices(ds_checkpoint, str(output_dir), str(slice_dir), 2, (param_name, torch.Size([3, 4])))

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
    _write_tp_slice(slice_dir, param_name, 0, "fp32", tp0)
    _write_tp_slice(slice_dir, param_name, 1, "fp32", tp1)

    uc_info = {
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [rf"^{param_name}$"],
        TP_REPLICATED_PARAMETER_PATTERNS: [],
        PARAMETER_WITH_SUB_PARAMS: [],
    }

    ds_checkpoint = SimpleNamespace(get_checkpoint_info=lambda key: uc_info if key == UNIVERSAL_CHECKPOINT_INFO else {})

    merge_tp_slices(ds_checkpoint, str(output_dir), str(slice_dir), 2, (param_name, torch.Size([4, 4])))

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
        "conversion": {"partition_dim": 999},
    }
    param = _make_param((4, 4), meta)

    restore_meta = _get_param_uc_restore_meta(param)

    assert restore_meta["partition_dim"] == 1
    assert restore_meta["conversion"]["partition_dim"] == 999

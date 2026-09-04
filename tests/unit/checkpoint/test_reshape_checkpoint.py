# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from types import SimpleNamespace

import pytest
import torch

from deepspeed.checkpoint import (DeepSpeedCheckpoint, ZeROCheckpoint, get_model_3d_descriptor, merge_state,
                                  model_3d_desc)
from deepspeed.checkpoint.constants import (AUTOEP_LAYERS_KEY, BASE_OPTIMIZER_STATE, CHECKPOINT_PARALLEL_DIMS,
                                            CHECKPOINT_PP_DEGREE, CHECKPOINT_TP_DEGREE, GROUP_PADDINGS,
                                            OPTIMIZER_STATE_DICT, PARAM_SHAPES, PARTITION_COUNT,
                                            UNIVERSAL_CHECKPOINT_INFO)
from deepspeed.checkpoint.ds_to_universal import _aggregate_autoep_zero12_metadata
from deepspeed.runtime.engine import _checkpoint_parallel_metadata

PP2_TP1 = {CHECKPOINT_PP_DEGREE: 2, CHECKPOINT_TP_DEGREE: 1}
PP1_TP2 = {CHECKPOINT_PP_DEGREE: 1, CHECKPOINT_TP_DEGREE: 2}


def _write_checkpoint_layout(tmpdir, parallel_dimensions, autoep_metadata=None):
    for mp_rank, dimensions in enumerate(parallel_dimensions):
        state = {PARAM_SHAPES: [{}], UNIVERSAL_CHECKPOINT_INFO: {"source": "writer"}, "ds_config": {}}
        if autoep_metadata is not None:
            state[AUTOEP_LAYERS_KEY] = autoep_metadata
        if dimensions is not None:
            state[CHECKPOINT_PARALLEL_DIMS] = dimensions
        torch.save(state, os.path.join(str(tmpdir), f"mp_rank_{mp_rank:02d}_model_states.pt"))
        for dp_rank in range(2):
            torch.save({},
                       os.path.join(str(tmpdir), f"bf16_zero_pp_rank_{dp_rank}_mp_rank_{mp_rank:02d}_optim_states.pt"))


def _do_reshape(src_3d, tgt_3d):
    assert src_3d.can_reshape(tgt_3d)
    new_3d_map = src_3d.reshape(tgt_3d)

    assert len(new_3d_map) == tgt_3d.dp_degree
    for new_2d_map in new_3d_map:
        assert new_2d_map.pp_degree == tgt_3d.pp_degree
        assert new_2d_map.tp_degree == tgt_3d.tp_degree

    return new_3d_map


# Specify 3d shape as pp/tp/dp
def test_reshape_222_to_111():
    src_3d = model_3d_desc(pp_degree=2, tp_degree=2, dp_degree=2)
    tgt_3d = model_3d_desc(pp_degree=1, tp_degree=1, dp_degree=1)

    new_3d_map = _do_reshape(src_3d, tgt_3d)

    assert new_3d_map[0].get_data(pp_index=0, tp_index=0) == [0, 4, 1, 5, 2, 6, 3, 7]


def test_reshape_222_to_121():
    src_3d = model_3d_desc(pp_degree=2, tp_degree=2, dp_degree=2)
    tgt_3d = model_3d_desc(pp_degree=1, tp_degree=2, dp_degree=1)

    new_3d_map = _do_reshape(src_3d, tgt_3d)

    assert new_3d_map[0].get_data(pp_index=0, tp_index=0) == [0, 4, 2, 6]
    assert new_3d_map[0].get_data(pp_index=0, tp_index=1) == [1, 5, 3, 7]


def test_reshape_222_to_122():
    src_3d = model_3d_desc(pp_degree=2, tp_degree=2, dp_degree=2)
    tgt_3d = model_3d_desc(pp_degree=1, tp_degree=2, dp_degree=2)

    new_3d_map = _do_reshape(src_3d, tgt_3d)

    assert new_3d_map[0].get_data(pp_index=0, tp_index=0) == [0, 4]
    assert new_3d_map[0].get_data(pp_index=0, tp_index=1) == [1, 5]
    assert new_3d_map[1].get_data(pp_index=0, tp_index=0) == [2, 6]
    assert new_3d_map[1].get_data(pp_index=0, tp_index=1) == [3, 7]


def test_reshape_222_to_211():
    src_3d = model_3d_desc(pp_degree=2, tp_degree=2, dp_degree=2)
    tgt_3d = model_3d_desc(pp_degree=2, tp_degree=1, dp_degree=1)

    new_3d_map = _do_reshape(src_3d, tgt_3d)

    assert new_3d_map[0].get_data(pp_index=0, tp_index=0) == [0, 4, 1, 5]
    assert new_3d_map[0].get_data(pp_index=1, tp_index=0) == [2, 6, 3, 7]


@pytest.mark.parametrize("dimensions,error", [
    ([None, None], None),
    ([PP2_TP1, None], "missing from model-state files"),
    ([PP2_TP1, PP1_TP2], "disagrees across model-state files"),
])
def test_checkpoint_descriptor_legacy_and_invalid_metadata(tmpdir, dimensions, error):
    _write_checkpoint_layout(tmpdir, dimensions)
    if error:
        with pytest.raises(RuntimeError, match=error):
            get_model_3d_descriptor(str(tmpdir))
    else:
        descriptor = get_model_3d_descriptor(str(tmpdir))
        assert (descriptor.pp_degree, descriptor.tp_degree, descriptor.dp_degree) == (1, 2, 2)
        assert ZeROCheckpoint(str(tmpdir)).get_src_tp_degree() == 2


def test_checkpoint_writer_discovery_and_consumers_load_model_files_once(tmpdir, monkeypatch):
    mpu = SimpleNamespace(get_pipe_parallel_world_size=lambda: 2, get_slice_parallel_world_size=lambda: 1)
    dimensions = _checkpoint_parallel_metadata(mpu)[CHECKPOINT_PARALLEL_DIMS]
    _write_checkpoint_layout(tmpdir, [dimensions, dimensions], autoep_metadata="malformed")

    original_load = torch.load
    model_loads = []

    def tracked_load(path, *args, **kwargs):
        if str(path).endswith("_model_states.pt"):
            model_loads.append(str(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    checkpoint = DeepSpeedCheckpoint(str(tmpdir))
    with pytest.raises(RuntimeError, match="AutoEP metadata must be a list"):
        _aggregate_autoep_zero12_metadata(checkpoint.model_state_metadata)
    assert checkpoint.get_checkpoint_info(UNIVERSAL_CHECKPOINT_INFO) == {"source": "writer"}
    assert (checkpoint.pp_degree, checkpoint.tp_degree, checkpoint.dp_degree) == (2, 1, 2)
    assert model_loads == checkpoint.mp_rank_files


def _write_zero12_optim_checkpoint(tmpdir, dp_degree):
    """Write a dp_degree-way ZeRO 1/2 checkpoint holding a real torch optimizer state_dict."""
    torch.save({PARAM_SHAPES: [{}], "ds_config": {}}, os.path.join(str(tmpdir), "mp_rank_00_model_states.pt"))
    for dp_rank in range(dp_degree):
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.AdamW([param], lr=1e-3)
        param.grad = torch.ones(8)
        optimizer.step()
        state = {
            OPTIMIZER_STATE_DICT: {
                BASE_OPTIMIZER_STATE: optimizer.state_dict(),
                GROUP_PADDINGS: [0],
                PARTITION_COUNT: [dp_degree],
            }
        }
        torch.save(state, os.path.join(str(tmpdir), f"bf16_zero_pp_rank_{dp_rank}_mp_rank_00_optim_states.pt"))


def test_reshape_merges_zero12_optimizer_state_with_scalar_step(tmpdir):
    _write_zero12_optim_checkpoint(tmpdir, dp_degree=2)
    checkpoint = ZeROCheckpoint(str(tmpdir))
    checkpoint.reshape(model_3d_desc(pp_degree=1, tp_degree=1, dp_degree=1))

    merged = checkpoint.get_state_for_rank(0, 0, 0)[OPTIMIZER_STATE_DICT][BASE_OPTIMIZER_STATE]["state"][0]

    assert merged["exp_avg"].numel() == 16
    assert merged["step"].dim() == 0


def test_merge_state_keeps_keys_missing_from_the_second_dict():
    dict_a = {"exp_avg": torch.ones(4), "rank_0_only": torch.ones(2)}
    dict_b = {"exp_avg": torch.ones(4)}

    merged = merge_state(dict_a, dict_b)

    assert list(merged) == ["exp_avg", "rank_0_only"]
    assert merged["exp_avg"].numel() == 8


def test_merge_state_reports_the_full_key_path_on_mismatch(capsys):
    with pytest.raises(ValueError, match="Cannot merge lists of different lengths"):
        merge_state({"opt": {"groups": [1, 2]}}, {"opt": {"groups": [1]}})

    assert "opt.groups" in capsys.readouterr().out

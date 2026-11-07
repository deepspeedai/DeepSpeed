# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from types import SimpleNamespace

import pytest
import torch

from deepspeed.checkpoint import DeepSpeedCheckpoint, ZeROCheckpoint, get_model_3d_descriptor, model_3d_desc
from deepspeed.checkpoint.constants import (AUTOEP_LAYERS_KEY, CHECKPOINT_PARALLEL_DIMS, CHECKPOINT_PP_DEGREE,
                                            CHECKPOINT_TP_DEGREE, PARAM_SHAPES, UNIVERSAL_CHECKPOINT_INFO)
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


@pytest.mark.parametrize("num_layers,source_pp,target_pp,layer_counts,final_norm_idx", [
    (70, 12, 12, [6] * 10 + [5] * 2, -1),
    (70, 12, 6, [12] * 4 + [11] * 2, -1),
    (8, 4, 4, [2] * 4, -1),
    (3, 4, 4, [1, 1, 1, 0], -1),
    (0, 2, 2, [0, 0], -1),
    (5, 1, 1, [5], -1),
    (5, 2, 2, [3, 2], -2),
])
@pytest.mark.parametrize("source_tp,target_tp", [(1, 1), (2, 2), (2, 1)])
def test_pipeline_checkpoint_preserves_transformer_layers(tmpdir, num_layers, source_pp, target_pp, layer_counts,
                                                          final_norm_idx, source_tp, target_tp):
    # Small real checkpoint files exercise discovery and loading without training a model.
    dimensions = {CHECKPOINT_PP_DEGREE: source_pp, CHECKPOINT_TP_DEGREE: source_tp}
    _write_checkpoint_layout(tmpdir, [dimensions] * (source_pp * source_tp))
    num_saved_layers = num_layers + 2 + (final_norm_idx == -2)
    for layer_id in range(num_saved_layers):
        for tp_rank in range(source_tp):
            state = {
                "weight": torch.tensor([[layer_id, tp_rank]], dtype=torch.float32),
                "input_layernorm.weight": torch.tensor([layer_id], dtype=torch.float32),
            }
            torch.save(state, os.path.join(str(tmpdir), f"layer_{layer_id:02d}-model_{tp_rank:02d}-model_states.pt"))

    checkpoint = DeepSpeedCheckpoint(str(tmpdir),
                                     tp_degree=target_tp,
                                     pp_degree=target_pp,
                                     final_layer_norm_idx=final_norm_idx)
    next_layer_id = 1
    all_layer_keys = []
    for pp_rank, count in enumerate(layer_counts):
        layer_ids = list(range(next_layer_id, next_layer_id + count))
        layer_keys = checkpoint.get_pp_transformer_map(pp_rank)
        assert layer_keys == [f"layer_{layer_id:02d}" for layer_id in layer_ids]
        all_layer_keys.extend(layer_keys)
        for tp_rank in range(target_tp):
            states = checkpoint.get_transformer_state(tp_index=tp_rank, pp_index=pp_rank)
            assert len(states) == count
            shards_per_rank = source_tp // target_tp
            source_ranks = range(tp_rank * shards_per_rank, (tp_rank + 1) * shards_per_rank)
            for layer_id, state in zip(layer_ids, states):
                expected = torch.tensor([[layer_id, rank] for rank in source_ranks], dtype=torch.float32)
                torch.testing.assert_close(state["weight"], expected, rtol=0, atol=0)
                torch.testing.assert_close(state["input_layernorm.weight"],
                                           torch.tensor([layer_id], dtype=torch.float32),
                                           rtol=0,
                                           atol=0)
        next_layer_id += count

    assert all_layer_keys == [f"layer_{layer_id:02d}" for layer_id in range(1, num_layers + 1)]
    assert checkpoint.get_embedding_layer_id() == "layer_00"
    assert checkpoint.get_final_norm_layer_id() == f"layer_{num_layers + 1:02d}"

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from types import SimpleNamespace

import pytest
import torch

from deepspeed.checkpoint import DeepSpeedCheckpoint, ZeROCheckpoint, get_model_3d_descriptor, model_3d_desc
from deepspeed.checkpoint.constants import (AUTOEP_LAYERS_KEY, CHECKPOINT_PARALLEL_DIMS, CHECKPOINT_PP_DEGREE,
                                            CHECKPOINT_TP_DEGREE, PARAM_SHAPES, UNIVERSAL_CHECKPOINT_INFO,
                                            UNIVERSAL_CHECKPOINT_VERSION_KEY, UNIVERSAL_CHECKPOINT_VERSION_VALUE)
from deepspeed.checkpoint import ds_to_universal
from deepspeed.checkpoint.ds_to_universal import _aggregate_autoep_zero12_metadata
from deepspeed.runtime.base_optimizer import _universal_checkpoint_tp_info
from deepspeed.runtime.engine import _checkpoint_parallel_metadata, _default_universal_checkpoint_info

PP2_TP1 = {CHECKPOINT_PP_DEGREE: 2, CHECKPOINT_TP_DEGREE: 1}
PP1_TP2 = {CHECKPOINT_PP_DEGREE: 1, CHECKPOINT_TP_DEGREE: 2}
PP1_TP1 = {CHECKPOINT_PP_DEGREE: 1, CHECKPOINT_TP_DEGREE: 1}

# Mirrors deepspeed.runtime.sequence_parallel.parallel_state_sp, whose model-parallel accessors report
# the Ulysses sequence-parallel group.
ULYSSES_SP2_MPU = SimpleNamespace(initialize_sequence_parallel=lambda size: None,
                                  get_model_parallel_world_size=lambda: 2,
                                  get_model_parallel_rank=lambda: 1)
TP2_MPU = SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 2, get_tensor_model_parallel_rank=lambda: 1)


def _write_checkpoint_layout(tmpdir, parallel_dimensions, autoep_metadata=None, mp_world_size=None):
    for mp_rank, dimensions in enumerate(parallel_dimensions):
        state = {PARAM_SHAPES: [{}], UNIVERSAL_CHECKPOINT_INFO: {"source": "writer"}, "ds_config": {}}
        if mp_world_size is not None:
            state["mp_world_size"] = mp_world_size
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


def test_sequence_parallel_mpu_records_no_model_parallelism():
    dimensions = _checkpoint_parallel_metadata(ULYSSES_SP2_MPU)[CHECKPOINT_PARALLEL_DIMS]
    assert dimensions == PP1_TP1
    assert _default_universal_checkpoint_info(dimensions) == {
        UNIVERSAL_CHECKPOINT_VERSION_KEY: UNIVERSAL_CHECKPOINT_VERSION_VALUE
    }
    assert _universal_checkpoint_tp_info(ULYSSES_SP2_MPU) == (0, 1)


def test_model_parallel_checkpoints_get_no_default_universal_info():
    assert _checkpoint_parallel_metadata(TP2_MPU)[CHECKPOINT_PARALLEL_DIMS] == PP1_TP2
    assert _default_universal_checkpoint_info(PP1_TP2) is None
    assert _default_universal_checkpoint_info(PP2_TP1) is None
    assert _universal_checkpoint_tp_info(TP2_MPU) == (1, 2)


def test_legacy_sequence_parallel_checkpoint_converts_as_tp1(tmpdir):
    _write_checkpoint_layout(tmpdir, [PP1_TP2], mp_world_size=1)
    descriptor = get_model_3d_descriptor(str(tmpdir))
    assert (descriptor.pp_degree, descriptor.tp_degree, descriptor.dp_degree) == (1, 1, 2)
    checkpoint = DeepSpeedCheckpoint(str(tmpdir))
    assert (checkpoint.pp_degree, checkpoint.tp_degree, checkpoint.dp_degree) == (1, 1, 2)


@pytest.mark.parametrize("mp_world_size", [None, 2])
def test_missing_tensor_parallel_file_is_not_relabeled(tmpdir, mp_world_size):
    _write_checkpoint_layout(tmpdir, [PP1_TP2], mp_world_size=mp_world_size)
    with pytest.raises(RuntimeError, match="does not match"):
        get_model_3d_descriptor(str(tmpdir))


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


def _random_file(path, size):
    data = os.urandom(size)
    with open(path, "wb") as f:
        f.write(data)
    return data


@pytest.mark.parametrize("size", [0, 1, 4096, 10 * 4096 + 7])
def test_direct_file_copy_is_byte_identical(tmpdir, monkeypatch, size):
    # Interleaved chunks and the unaligned tail must neither drop nor repeat bytes, whatever the file size.
    monkeypatch.setattr(ds_to_universal, "COPY_CHUNK_BYTES", 4096)
    src, dst = os.path.join(tmpdir, "src.pt"), os.path.join(tmpdir, "dst.pt")
    data = _random_file(src, size)
    try:
        ds_to_universal._direct_copy(src, dst, size, num_workers=3)
    except OSError as e:
        pytest.skip(f"no direct I/O in {tmpdir}: {e}")
    with open(dst, "rb") as f:
        assert f.read() == data


@pytest.mark.parametrize("clone", [True, False])
def test_copy_file_is_byte_identical(tmpdir, monkeypatch, clone):
    monkeypatch.setattr(ds_to_universal, "PARALLEL_COPY_MIN_BYTES", 1)
    monkeypatch.setattr(ds_to_universal, "COPY_CHUNK_BYTES", 4096)
    if not clone:
        monkeypatch.setattr(ds_to_universal, "_clone_file", lambda src, dst: False)
    src = os.path.join(tmpdir, "mp_rank_00_model_states.pt")
    data = _random_file(src, 10 * 4096 + 7)
    out = os.path.join(tmpdir, "out")
    os.makedirs(out)
    ds_to_universal._copy_file(src, out, num_workers=3)
    dst = os.path.join(out, os.path.basename(src))
    with open(dst, "rb") as f:
        assert f.read() == data
    assert os.stat(dst).st_mtime == os.stat(src).st_mtime

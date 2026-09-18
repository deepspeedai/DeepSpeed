# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import os
from types import SimpleNamespace

import pytest
import torch

from deepspeed.checkpoint.autoep_affine import make_autoep_placement_descriptor
from deepspeed.checkpoint.autoep_universal import (consolidate_autoep_zero12_expert_states,
                                                   get_autoep_zero12_expert_param_info)
from deepspeed.checkpoint.constants import (
    AUTOEP_EXPERT_PLACEMENT, AUTOEP_LAYERS_KEY, AUTOEP_ZERO3_EXPERT_STATE_FORMAT_KEY,
    AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION, AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION_KEY,
    AUTOEP_ZERO3_PARTITIONED_EXPERT_STATE_FORMAT, AUTOEP_PARAM_LOCAL_EXPERTS, AUTOEP_PLACEMENT_EXPERTS,
    AUTOEP_PLACEMENT_RANK, AUTOEP_PLACEMENT_RANKS, DS_AUTOEP_UC_META, DS_AUTOTP_UC_META, EP_IS_EXPERT_PARAM,
    EP_NUM_EXPERTS, OPTIMIZER_STATE_DICT, PARAM, PARAM_SHAPES)
from deepspeed.checkpoint.ds_to_universal import (_consolidate_zero3_autoep_expert_states,
                                                  _rebuild_zero3_autoep_rank_tensors)
from deepspeed.checkpoint.universal_checkpoint import _resolve_autoep_partition, load_hp_checkpoint_state
from deepspeed.runtime.zero import stage3
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3


def _write_zero12_fragments(temp_dir, param_name, rank_tensors):
    for state_name in ('fp32', 'exp_avg', 'exp_avg_sq'):
        for dp_rank, tensor in rank_tensors.items():
            path = os.path.join(temp_dir, param_name, "0", f"{state_name}.{dp_rank:02d}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(tensor.flatten().float(), path)
    for dp_rank in rank_tensors:
        torch.save(7, os.path.join(temp_dir, param_name, "0", f"step.{dp_rank:02d}"))


def _load_consolidated(output_dir, param_name, state_name='fp32'):
    state = torch.load(os.path.join(output_dir, "zero", param_name, f"{state_name}.pt"),
                       map_location='cpu',
                       weights_only=False)
    assert state[EP_IS_EXPERT_PARAM]
    return state


def test_zero12_affine_consolidation_supports_nonuniform_noncontiguous_placement(tmp_path):
    param_name = "layer.experts.w1"
    placement = make_autoep_placement_descriptor(4, [[2, 0, 3], [1]])
    full = torch.arange(8, dtype=torch.float32).view(4, 2)
    rank_tensors = {
        0: torch.stack([full[2], full[0], full[3]]),
        1: full[1:2],
    }
    _write_zero12_fragments(str(tmp_path / "fragments"), param_name, rank_tensors)
    info = {
        param_name: {
            'num_experts': 4,
            'num_local_experts': 2,
            'ep_size': 2,
            'expert_placement': placement,
        }
    }

    consolidate_autoep_zero12_expert_states(str(tmp_path / "fragments"), str(tmp_path / "output"), info,
                                            {param_name: (2, 2)}, 2, 1, False)

    for state_name in ('fp32', 'exp_avg', 'exp_avg_sq'):
        state = _load_consolidated(str(tmp_path / "output"), param_name, state_name)
        assert state[EP_NUM_EXPERTS] == 4
        assert torch.equal(state[PARAM], full)


def test_zero12_metadata_accepts_explicit_empty_rank():
    placement = make_autoep_placement_descriptor(2, [[], [0, 1]])
    metadata = [{
        'moe_layer_id': 0,
        'module_path': 'layer',
        'num_experts': 2,
        'num_local_experts': 0,
        'ep_size': 2,
        'ep_rank': 0,
        'expert_key_prefix': 'layer.experts',
        AUTOEP_EXPERT_PLACEMENT: placement,
    }]

    param_info = get_autoep_zero12_expert_param_info(metadata)

    assert param_info['layer.experts.w1']['num_local_experts'] == 0


def test_zero12_uniform_affine_consolidation_matches_legacy_order(tmp_path):
    param_name = "layer.experts.w1"
    placement = make_autoep_placement_descriptor(4, [[0, 1], [2, 3]])
    full = torch.arange(8, dtype=torch.float32).view(4, 2)
    _write_zero12_fragments(str(tmp_path / "fragments"), param_name, {0: full[:2], 1: full[2:]})
    info = {
        param_name: {
            'num_experts': 4,
            'num_local_experts': 2,
            'ep_size': 2,
            'expert_placement': placement,
        }
    }

    consolidate_autoep_zero12_expert_states(str(tmp_path / "fragments"), str(tmp_path / "output"), info,
                                            {param_name: (2, 2)}, 2, 1, False)

    assert torch.equal(_load_consolidated(str(tmp_path / "output"), param_name)[PARAM], full)


def test_zero12_affine_consolidation_validates_replicated_experts(tmp_path):
    param_name = "layer.experts.w2"
    placement = make_autoep_placement_descriptor(3, [[0, 1], [1, 2]])
    full = torch.arange(6, dtype=torch.float32).view(3, 2)
    rank_tensors = {
        0: torch.stack([full[0], full[1]]),
        1: torch.stack([full[1] + 1, full[2]]),
    }
    _write_zero12_fragments(str(tmp_path / "fragments"), param_name, rank_tensors)
    info = {
        param_name: {
            'num_experts': 3,
            'num_local_experts': 2,
            'ep_size': 2,
            'expert_placement': placement,
        }
    }

    with pytest.raises(ValueError, match="different data"):
        consolidate_autoep_zero12_expert_states(str(tmp_path / "fragments"), str(tmp_path / "output"), info,
                                                {param_name: (2, 2)}, 2, 1, False)


def test_zero12_legacy_consolidation_still_concatenates_ep_ranks(tmp_path):
    param_name = "layer.experts.w3"
    full = torch.arange(8, dtype=torch.float32).view(4, 2)
    _write_zero12_fragments(str(tmp_path / "fragments"), param_name, {0: full[:2], 1: full[2:]})
    info = {
        param_name: {
            'num_experts': 4,
            'num_local_experts': 2,
            'ep_size': 2,
            'expert_placement': None,
        }
    }

    consolidate_autoep_zero12_expert_states(str(tmp_path / "fragments"), str(tmp_path / "output"), info,
                                            {param_name: (2, 2)}, 2, 1, False)

    assert torch.equal(_load_consolidated(str(tmp_path / "output"), param_name)[PARAM], full)


def test_zero3_affine_rebuild_checks_ranks_shapes_and_replicas():
    placement = make_autoep_placement_descriptor(3, [[2, 0], [1, 2]])
    full = torch.arange(6, dtype=torch.float32).view(3, 2)
    shards = {0: torch.stack([full[2], full[0]]), 1: torch.stack([full[1], full[2]])}

    assert torch.equal(_rebuild_zero3_autoep_rank_tensors(shards, placement, full.shape, "test"), full)
    with pytest.raises(RuntimeError, match="expected nonempty ranks \\[0, 1\\]"):
        _rebuild_zero3_autoep_rank_tensors({0: shards[0]}, placement, full.shape, "test")
    with pytest.raises(RuntimeError, match="different data"):
        _rebuild_zero3_autoep_rank_tensors({
            0: shards[0],
            1: shards[1] + torch.tensor([[0, 0], [1, 0]])
        }, placement, full.shape, "test")


def test_zero3_affine_rebuild_allows_omitted_empty_rank():
    placement = make_autoep_placement_descriptor(3, [[2, 0, 1], []])
    full = torch.arange(6, dtype=torch.float32).view(3, 2)
    shards = {0: torch.stack([full[2], full[0], full[1]])}

    assert torch.equal(_rebuild_zero3_autoep_rank_tensors(shards, placement, full.shape, "test"), full)


def test_zero3_partition_native_consolidation_uses_affine_placement(tmp_path):
    param_names = [f"layer.experts.w{index}" for index in range(1, 4)]
    placement = make_autoep_placement_descriptor(4, [[2, 0, 3], [1]])
    full = torch.arange(8, dtype=torch.float32).view(4, 2)
    rank_tensors = {
        0: torch.stack([full[2], full[0], full[3]]),
        1: full[1:2],
    }
    model_files = []
    optim_files = []
    for rank, local_tensor in rank_tensors.items():
        layer_info = {
            'moe_layer_id': 0,
            'module_path': 'layer',
            'num_experts': 4,
            'num_local_experts': local_tensor.shape[0],
            'ep_size': 2,
            'expert_key_prefix': 'layer.experts',
            AUTOEP_EXPERT_PLACEMENT: placement,
            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_KEY: AUTOEP_ZERO3_PARTITIONED_EXPERT_STATE_FORMAT,
            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION_KEY: AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION,
            'ep_group_name': 'ep_size_2',
            'ep_rank': rank,
            'expert_data_parallel_rank': 0,
            'expert_data_parallel_world_size': 1,
            'global_expert_start': 0,
            'global_expert_end': 0,
        }
        model_file = tmp_path / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt"
        torch.save(
            {
                AUTOEP_LAYERS_KEY: [layer_info],
                PARAM_SHAPES: [{
                    param_name: local_tensor.shape
                } for param_name in param_names],
            }, model_file)
        model_files.append(str(model_file))

        flat = local_tensor.flatten()
        optim_file = tmp_path / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        torch.save(
            {
                OPTIMIZER_STATE_DICT: {
                    'ds_zero_partition_groups': [{
                        'partition_count': 1,
                        'partition_rank': 0,
                    } for _ in param_names],
                    'optimizer_state_dict': {
                        'state': {
                            index: {
                                'exp_avg': flat.clone(),
                                'exp_avg_sq': flat.clone(),
                            }
                            for index in range(len(param_names))
                        }
                    },
                    'fp32_flat_groups': [flat.clone() for _ in param_names],
                }
            }, optim_file)
        optim_files.append(str(optim_file))

    output_dir = tmp_path / "universal"
    _consolidate_zero3_autoep_expert_states(str(output_dir), model_files, optim_files)

    for param_name in param_names:
        for state_name in ('fp32', 'exp_avg', 'exp_avg_sq'):
            state = _load_consolidated(str(output_dir), param_name, state_name)
            assert torch.equal(state[PARAM], full)


def test_zero3_partition_native_consolidation_allows_rank_without_fragments(tmp_path):
    param_names = [f"layer.experts.w{index}" for index in range(1, 4)]
    placement = make_autoep_placement_descriptor(3, [[2, 0, 1], []])
    full = torch.arange(6, dtype=torch.float32).view(3, 2)
    model_files = []
    optim_files = []
    for rank in range(2):
        local_tensor = torch.stack([full[2], full[0], full[1]]) if rank == 0 else None
        layer_info = {
            'moe_layer_id': 0,
            'module_path': 'layer',
            'num_experts': 3,
            'num_local_experts': 3 if rank == 0 else 0,
            'ep_size': 2,
            'expert_key_prefix': 'layer.experts',
            AUTOEP_EXPERT_PLACEMENT: placement,
            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_KEY: AUTOEP_ZERO3_PARTITIONED_EXPERT_STATE_FORMAT,
            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION_KEY: AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION,
            'ep_group_name': 'ep_size_2',
            'ep_rank': rank,
            'expert_data_parallel_rank': 0,
            'expert_data_parallel_world_size': 1,
            'global_expert_start': 0,
            'global_expert_end': 0,
        }
        model_file = tmp_path / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt"
        param_shapes = [] if local_tensor is None else [{param_name: local_tensor.shape} for param_name in param_names]
        torch.save({
            AUTOEP_LAYERS_KEY: [layer_info],
            PARAM_SHAPES: param_shapes,
        }, model_file)
        model_files.append(str(model_file))

        optim_file = tmp_path / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        if local_tensor is None:
            optimizer_state = {
                'ds_zero_partition_groups': [],
                'optimizer_state_dict': {
                    'state': {}
                },
                'fp32_flat_groups': [],
            }
        else:
            flat = local_tensor.flatten()
            optimizer_state = {
                'ds_zero_partition_groups': [{
                    'partition_count': 1,
                    'partition_rank': 0,
                } for _ in param_names],
                'optimizer_state_dict': {
                    'state': {
                        index: {
                            'exp_avg': flat.clone(),
                            'exp_avg_sq': flat.clone(),
                        }
                        for index in range(len(param_names))
                    }
                },
                'fp32_flat_groups': [flat.clone() for _ in param_names],
            }
        torch.save({OPTIMIZER_STATE_DICT: optimizer_state}, optim_file)
        optim_files.append(str(optim_file))

    output_dir = tmp_path / "universal"
    _consolidate_zero3_autoep_expert_states(str(output_dir), model_files, optim_files)

    for param_name in param_names:
        for state_name in ('fp32', 'exp_avg', 'exp_avg_sq'):
            state = _load_consolidated(str(output_dir), param_name, state_name)
            assert torch.equal(state[PARAM], full)


def _target_param(placement, logical_shape, ep_rank):
    entries_by_rank = {entry[AUTOEP_PLACEMENT_RANK]: entry for entry in placement[AUTOEP_PLACEMENT_RANKS]}
    param = torch.nn.Parameter(torch.empty(1))
    setattr(
        param,
        DS_AUTOEP_UC_META,
        {
            AUTOEP_EXPERT_PLACEMENT: placement,
            'logical_shape': list(logical_shape),
            'ep_rank': ep_rank,
            AUTOEP_PARAM_LOCAL_EXPERTS: list(entries_by_rank[ep_rank][AUTOEP_PLACEMENT_EXPERTS]),
        },
    )
    return param


def test_autoep_target_extraction_precedes_tp_and_stage3_partitioning(monkeypatch):
    placement = make_autoep_placement_descriptor(5, [[0, 3], [4, 1, 2]])
    full = torch.arange(20, dtype=torch.float32).view(5, 2, 2)
    checkpoint_state = {
        PARAM: full,
        EP_IS_EXPERT_PARAM: True,
        EP_NUM_EXPERTS: 5,
    }
    param = _target_param(placement, full.shape, 1)
    expected = torch.stack([full[4], full[1], full[2]])

    param.ds_zero_partition_group_name = "ep"
    monkeypatch.setattr(stage3.groups, "_get_expert_parallel_rank", lambda _: 1)

    assert torch.equal(_resolve_autoep_partition(param, checkpoint_state, full, 1), expected)
    assert torch.equal(
        DeepSpeedZeroOptimizer_Stage3._slice_autoep_universal_expert_param(None, checkpoint_state, param), expected)


def test_zero12_restore_loads_target_packed_order(tmp_path):
    placement = make_autoep_placement_descriptor(5, [[0, 3], [4, 1, 2]])
    full = torch.arange(10, dtype=torch.float32).view(5, 2)
    expected = torch.stack([full[4], full[1], full[2]])
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    torch.save({
        PARAM: full,
        EP_IS_EXPERT_PARAM: True,
        EP_NUM_EXPERTS: 5,
    }, checkpoint_dir / "fp32.pt")

    param = torch.nn.Parameter(torch.empty_like(expected))
    setattr(
        param,
        DS_AUTOEP_UC_META,
        {
            AUTOEP_EXPERT_PLACEMENT: placement,
            'logical_shape': list(full.shape),
            'ep_rank': 1,
            AUTOEP_PARAM_LOCAL_EXPERTS: [4, 1, 2],
        },
    )
    destination = torch.empty(expected.numel(), dtype=torch.float32)
    param._hp_mapping = SimpleNamespace(
        optim_fragment={},
        lp_fragment_address=SimpleNamespace(start=0, numel=expected.numel()),
        get_hp_fragment=lambda: destination,
    )

    load_hp_checkpoint_state(param, str(checkpoint_dir), tp_rank=0, tp_world_size=1, ep_rank=1, ep_size=2)

    assert torch.equal(destination.view_as(expected), expected)


def test_zero12_restore_applies_autoep_before_independent_autotp(tmp_path):
    placement = make_autoep_placement_descriptor(4, [[2, 0], [3, 1]])
    full = torch.arange(16, dtype=torch.float32).view(4, 4)
    ep_local = torch.stack([full[2], full[0]])
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    torch.save({
        PARAM: full,
        EP_IS_EXPERT_PARAM: True,
        EP_NUM_EXPERTS: 4,
    }, checkpoint_dir / "fp32.pt")

    param = _target_param(placement, full.shape, 0)
    setattr(
        param,
        DS_AUTOTP_UC_META,
        {
            'logical_shape': list(ep_local.shape),
            'partition_dim': 1,
            'partition_sizes': [2, 2],
        },
    )
    destination = torch.empty(4, dtype=torch.float32)
    param._hp_mapping = SimpleNamespace(
        optim_fragment={},
        lp_fragment_address=SimpleNamespace(start=0, numel=4),
        get_hp_fragment=lambda: destination,
    )

    load_hp_checkpoint_state(param, str(checkpoint_dir), tp_rank=1, tp_world_size=2, ep_rank=0, ep_size=2)

    assert torch.equal(destination.view(2, 2), ep_local[:, 2:])


def test_zero3_restore_extracts_ep_rank_before_zero_partition(tmp_path, monkeypatch):
    placement = make_autoep_placement_descriptor(5, [[0, 3], [4, 1, 2]])
    full = torch.arange(10, dtype=torch.float32).view(5, 2)
    expected_local = torch.stack([full[4], full[1], full[2]])
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    torch.save({
        PARAM: full,
        EP_IS_EXPERT_PARAM: True,
        EP_NUM_EXPERTS: 5,
    }, checkpoint_dir / "fp32.pt")
    param = _target_param(placement, full.shape, 1)
    param.ds_zero_partition_group_name = "ep"
    partition_group = object()
    optimizer = SimpleNamespace(
        _get_param_partition_group=lambda _: partition_group,
        _slice_autoep_universal_expert_param=lambda checkpoint_state, target_param: DeepSpeedZeroOptimizer_Stage3.
        _slice_autoep_universal_expert_param(None, checkpoint_state, target_param),
    )
    monkeypatch.setattr(stage3.dist, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(stage3.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(stage3.groups, "_get_expert_parallel_rank", lambda _: 1)

    actual = DeepSpeedZeroOptimizer_Stage3.load_hp_checkpoint_state(optimizer, str(checkpoint_dir), "fp32", param)

    assert torch.equal(actual, expected_local.flatten()[3:])


@pytest.mark.parametrize(
    'metadata, match',
    [
        ({}, "missing fields"),
        ({
            AUTOEP_EXPERT_PLACEMENT: make_autoep_placement_descriptor(2, [[0], [1]]),
            'logical_shape': [3, 2],
            'ep_rank': 0,
            AUTOEP_PARAM_LOCAL_EXPERTS: [0],
        }, "tensor shape disagrees"),
        ({
            AUTOEP_EXPERT_PLACEMENT: make_autoep_placement_descriptor(2, [[0], [1]]),
            'logical_shape': [2, 2],
            'ep_rank': 2,
            AUTOEP_PARAM_LOCAL_EXPERTS: [],
        }, "Invalid AutoEP"),
    ],
)
def test_autoep_target_metadata_fails_loudly(metadata, match):
    full = torch.arange(4, dtype=torch.float32).view(2, 2)
    checkpoint_state = {PARAM: full, EP_IS_EXPERT_PARAM: True, EP_NUM_EXPERTS: 2}
    param = torch.nn.Parameter(torch.empty(1))
    setattr(param, DS_AUTOEP_UC_META, metadata)

    with pytest.raises(RuntimeError, match=match):
        _resolve_autoep_partition(param, checkpoint_state, full, metadata.get('ep_rank', 0))


def test_autoep_target_metadata_rejects_caller_ep_rank_mismatch():
    placement = make_autoep_placement_descriptor(2, [[0], [1]])
    full = torch.arange(4, dtype=torch.float32).view(2, 2)
    checkpoint_state = {PARAM: full, EP_IS_EXPERT_PARAM: True, EP_NUM_EXPERTS: 2}
    param = _target_param(placement, full.shape, 0)

    with pytest.raises(RuntimeError, match="target ep_rank 1 does not match parameter metadata ep_rank 0"):
        _resolve_autoep_partition(param, checkpoint_state, full, 1)


def test_autoep_target_metadata_rejects_local_expert_list_mismatch():
    placement = make_autoep_placement_descriptor(3, [[2, 0], [1]])
    full = torch.arange(6, dtype=torch.float32).view(3, 2)
    checkpoint_state = {PARAM: full, EP_IS_EXPERT_PARAM: True, EP_NUM_EXPERTS: 3}
    param = _target_param(placement, full.shape, 0)
    getattr(param, DS_AUTOEP_UC_META)[AUTOEP_PARAM_LOCAL_EXPERTS] = [0, 2]

    with pytest.raises(RuntimeError, match="local_experts does not match"):
        _resolve_autoep_partition(param, checkpoint_state, full, 0)

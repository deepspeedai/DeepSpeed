# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Contract tests for AutoEP checkpoint placement metadata."""

import copy

import pytest

from deepspeed.checkpoint.autoep_affine import make_autoep_placement_descriptor
from deepspeed.checkpoint.autoep_zero3_metadata import validate_autoep_zero3_partitioned_metadata
from deepspeed.checkpoint.constants import (AUTOEP_EXPERT_PLACEMENT, AUTOEP_ZERO3_EXPERT_STATE_FORMAT_KEY,
                                            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION,
                                            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION_KEY,
                                            AUTOEP_ZERO3_PARTITIONED_EXPERT_STATE_FORMAT)


def _metadata_entry():
    return {
        "moe_layer_id": 0,
        "module_path": "model.layers.0.mlp",
        "num_experts": 4,
        "num_local_experts": 2,
        "ep_size": 2,
        "expert_key_prefix": "model.layers.0.mlp.experts",
        "ep_rank": 1,
    }


def _partitioned_metadata_entry():
    entry = _metadata_entry()
    entry.update({
        AUTOEP_ZERO3_EXPERT_STATE_FORMAT_KEY: AUTOEP_ZERO3_PARTITIONED_EXPERT_STATE_FORMAT,
        AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION_KEY: AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION,
        "ep_group_name": "ep_size_2",
        "expert_data_parallel_rank": 0,
        "expert_data_parallel_world_size": 1,
        "global_expert_start": 2,
        "global_expert_end": 4,
    })
    return entry


def test_legacy_metadata_synthesizes_uniform_placement():
    placements = validate_autoep_zero3_partitioned_metadata([_metadata_entry()], require_partitioned=False)

    assert placements == [{
        "version": 1,
        "num_experts": 4,
        "ep_size": 2,
        "ranks": [{
            "rank": 0,
            "experts": [0, 1]
        }, {
            "rank": 1,
            "experts": [2, 3]
        }],
    }]


@pytest.mark.parametrize(
    "placement",
    [
        {
            "version": 999,
            "num_experts": 4,
            "ep_size": 2,
            "ranks": [],
        },
        make_autoep_placement_descriptor(6, [[0, 1, 2], [3, 4, 5]]),
        make_autoep_placement_descriptor(4, [[0, 1], [2, 3], []]),
    ],
)
def test_new_invalid_or_mismatched_placement_is_rejected(placement):
    entry = _metadata_entry()
    entry[AUTOEP_EXPERT_PLACEMENT] = placement

    with pytest.raises(RuntimeError, match="placement metadata"):
        validate_autoep_zero3_partitioned_metadata([entry], require_partitioned=False)


def test_rank_local_placement_must_match_runtime_packing():
    entry = _metadata_entry()
    entry[AUTOEP_EXPERT_PLACEMENT] = make_autoep_placement_descriptor(4, [[0, 2], [1, 3]])
    expected_runtime_layers = {
        "model.layers.0.mlp": {
            "ep_rank": 1,
            "local_experts": [2, 3],
        }
    }

    with pytest.raises(RuntimeError, match="expert order"):
        validate_autoep_zero3_partitioned_metadata([entry],
                                                   require_partitioned=False,
                                                   expected_runtime_layers=expected_runtime_layers)


def test_rank_local_expert_count_must_match_layer_entry():
    entry = copy.deepcopy(_metadata_entry())
    entry["num_local_experts"] = 1
    entry[AUTOEP_EXPERT_PLACEMENT] = make_autoep_placement_descriptor(4, [[0, 1], [2, 3]])

    with pytest.raises(RuntimeError, match="expert count"):
        validate_autoep_zero3_partitioned_metadata([entry], require_partitioned=False)


def test_partitioned_metadata_with_explicit_uneven_noncontiguous_placement_is_accepted():
    entry = _partitioned_metadata_entry()
    entry["num_experts"] = 5
    entry["num_local_experts"] = 3
    entry["global_expert_start"] = 99
    entry["global_expert_end"] = 102
    entry[AUTOEP_EXPERT_PLACEMENT] = make_autoep_placement_descriptor(5, [[1, 3], [4, 0, 2]])

    placements = validate_autoep_zero3_partitioned_metadata([entry])

    assert placements == [entry[AUTOEP_EXPERT_PLACEMENT]]


def test_partitioned_legacy_metadata_still_requires_uniform_contiguous_ranges():
    entry = _partitioned_metadata_entry()
    entry["global_expert_start"] = 1

    with pytest.raises(RuntimeError, match="global expert range"):
        validate_autoep_zero3_partitioned_metadata([entry])

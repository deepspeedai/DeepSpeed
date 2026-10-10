# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import json

import pytest
import torch

from deepspeed.checkpoint.affine import AFFINE_MAP_FORMAT_VERSION, ParamAffineMap
from deepspeed.checkpoint.autoep_affine import (AUTOEP_PLACEMENT_VERSION, autoep_experts_for_rank,
                                                autoep_metadata_to_affine_map, autoep_placement_to_affine_map,
                                                extract_autoep_rank_tensor, legacy_uniform_autoep_placement_descriptor,
                                                make_autoep_placement_descriptor, validate_autoep_placement_descriptor)
from deepspeed.checkpoint.constants import (AFFINE_MAP, AFFINE_MAP_PARAMS, AFFINE_MAP_VERSION, AUTOEP_AFFINE_MAPS,
                                            AUTOEP_EXPERT_PLACEMENT)


def _oracle_shards(full_tensor, experts_by_rank):
    return {
        rank:
        torch.stack([full_tensor[expert_id]
                     for expert_id in experts]) if experts else full_tensor.new_empty((0, ) +
                                                                                      tuple(full_tensor.shape[1:]))
        for rank, experts in enumerate(experts_by_rank)
    }


@pytest.mark.parametrize('num_experts, ep_size', [(8, 2), (8, 4)])
def test_uniform_placement_lowers_and_round_trips(num_experts, ep_size):
    num_local_experts = num_experts // ep_size
    descriptor = legacy_uniform_autoep_placement_descriptor(num_experts, num_local_experts, ep_size)
    affine_map = autoep_placement_to_affine_map(descriptor, (num_experts, 3, 2))
    full_tensor = torch.arange(num_experts * 6, dtype=torch.float32).reshape(num_experts, 3, 2)
    experts_by_rank = [
        list(range(rank * num_local_experts, (rank + 1) * num_local_experts)) for rank in range(ep_size)
    ]
    expected_shards = _oracle_shards(full_tensor, experts_by_rank)

    assert affine_map.shard_shapes == {rank: (num_local_experts, 3, 2) for rank in range(ep_size)}
    assert all(len(pieces) == 1 for pieces in affine_map.pieces_by_rank.values())
    for rank in range(ep_size):
        assert torch.equal(affine_map.extract(full_tensor, rank), expected_shards[rank])
    assert torch.equal(affine_map.rebuild(expected_shards), full_tensor)


def test_non_contiguous_non_uniform_placement_preserves_local_order():
    experts_by_rank = [[4, 1, 5], [0], [3, 2]]
    descriptor = make_autoep_placement_descriptor(6, experts_by_rank)
    affine_map = autoep_placement_to_affine_map(descriptor, (6, 2))
    full_tensor = torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41], [50, 51], [60, 61]])
    expected_shards = _oracle_shards(full_tensor, experts_by_rank)

    assert affine_map.shard_shapes == {0: (3, 2), 1: (1, 2), 2: (2, 2)}
    assert torch.equal(affine_map.extract(full_tensor, 0), torch.tensor([[50, 51], [20, 21], [60, 61]]))
    assert torch.equal(affine_map.rebuild(expected_shards), full_tensor)


@pytest.mark.parametrize('use_collection', [False, True])
def test_persisted_autoep_affine_map_must_match_placement_descriptor(use_collection):
    logical_shape = (4, 2)
    persisted_placement = make_autoep_placement_descriptor(4, [[3, 1], [2, 0]])
    legacy_placement = make_autoep_placement_descriptor(4, [[0, 1], [2, 3]])
    persisted_map = autoep_placement_to_affine_map(persisted_placement, logical_shape)

    metadata = {AUTOEP_EXPERT_PLACEMENT: legacy_placement}
    if use_collection:
        metadata[AUTOEP_AFFINE_MAPS] = {
            AFFINE_MAP_VERSION: AFFINE_MAP_FORMAT_VERSION,
            AFFINE_MAP_PARAMS: {
                'experts.w1': persisted_map.to_dict()
            },
        }
    else:
        metadata[AFFINE_MAP] = persisted_map.to_dict()
    with pytest.raises(ValueError, match='disagrees with the expert placement'):
        autoep_metadata_to_affine_map(metadata, logical_shape, 'experts.w1')


def test_explicit_rank_ids_define_entries_independently_of_list_order():
    descriptor = make_autoep_placement_descriptor(4, [[0, 2], [3, 1]])
    descriptor['ranks'].reverse()

    assert autoep_experts_for_rank(descriptor, 0) == [0, 2]
    assert autoep_experts_for_rank(descriptor, 1) == [3, 1]
    affine_map = autoep_placement_to_affine_map(descriptor, (4, 2))
    full_tensor = torch.arange(8).reshape(4, 2)
    assert torch.equal(affine_map.extract(full_tensor, 0), torch.stack([full_tensor[0], full_tensor[2]]))


def test_versioned_autoep_map_collection_loads_by_parameter_name():
    logical_shape = (2, 2)
    descriptor = make_autoep_placement_descriptor(2, [[1], [0]])
    persisted_map = autoep_placement_to_affine_map(descriptor, logical_shape)
    metadata = {
        AUTOEP_AFFINE_MAPS: {
            AFFINE_MAP_VERSION: AFFINE_MAP_FORMAT_VERSION,
            AFFINE_MAP_PARAMS: {
                'experts.w1': persisted_map.to_dict()
            },
        },
    }

    loaded_map = autoep_metadata_to_affine_map(metadata, logical_shape, 'experts.w1')

    assert loaded_map.to_dict() == persisted_map.to_dict()


def test_replication_records_exact_holders_and_stops_piece_merging():
    experts_by_rank = [[0, 1, 2], [1, 2, 3]]
    descriptor = make_autoep_placement_descriptor(4, experts_by_rank)
    affine_map = autoep_placement_to_affine_map(descriptor, (4, 2))
    full_tensor = torch.arange(8).reshape(4, 2)

    assert [piece.locations
            for piece in affine_map.pieces_by_rank[0]] == [frozenset({0}),
                                                           frozenset({0, 1}),
                                                           frozenset({0, 1})]
    assert [piece.shape for piece in affine_map.pieces_by_rank[0]] == [(1, 2), (1, 2), (1, 2)]
    assert [piece.locations
            for piece in affine_map.pieces_by_rank[1]] == [frozenset({0, 1}),
                                                           frozenset({0, 1}),
                                                           frozenset({1})]
    assert torch.equal(affine_map.rebuild(_oracle_shards(full_tensor, experts_by_rank)), full_tensor)


def test_replication_disagreement_with_different_local_order_fails():
    experts_by_rank = [[0, 1, 3, 4], [2, 4, 3]]
    descriptor = make_autoep_placement_descriptor(5, experts_by_rank)
    affine_map = autoep_placement_to_affine_map(descriptor, (5, 2))
    full_tensor = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    shards = _oracle_shards(full_tensor, experts_by_rank)
    shards[1][1].add_(1)

    with pytest.raises(ValueError, match='different data'):
        affine_map.rebuild(shards)


def test_empty_rank_shard_is_representable():
    descriptor = make_autoep_placement_descriptor(3, [[2, 0], [], [1]])
    affine_map = autoep_placement_to_affine_map(descriptor, (3, 4))
    full_tensor = torch.arange(12).reshape(3, 4)

    assert affine_map.shard_shapes[1] == (0, 4)
    assert affine_map.pieces_by_rank[1] == []
    assert extract_autoep_rank_tensor(full_tensor, affine_map, 1).shape == (0, 4)
    assert torch.equal(affine_map.rebuild(_oracle_shards(full_tensor, [[2, 0], [], [1]])), full_tensor)
    shards_without_empty_rank = _oracle_shards(full_tensor, [[2, 0], [], [1]])
    del shards_without_empty_rank[1]
    assert torch.equal(affine_map.rebuild(shards_without_empty_rank), full_tensor)


@pytest.mark.parametrize('descriptor, match', [
    ({
        'version': AUTOEP_PLACEMENT_VERSION,
        'num_experts': 2,
        'ep_size': 2,
        'ranks': [{
            'rank': 0,
            'experts': [0]
        }],
    }, 'exactly 2'),
    ({
        'version': AUTOEP_PLACEMENT_VERSION,
        'num_experts': 2,
        'ep_size': 2,
        'ranks': [{
            'rank': 0,
            'experts': [0]
        }, {
            'rank': 0,
            'experts': [1]
        }],
    }, 'rank 0 appears more than once'),
    ({
        'version': AUTOEP_PLACEMENT_VERSION,
        'num_experts': 2,
        'ep_size': 1,
        'ranks': [{
            'rank': 0,
            'experts': [0, 0, 1]
        }],
    }, 'duplicate expert ID 0'),
    ({
        'version': AUTOEP_PLACEMENT_VERSION,
        'num_experts': 3,
        'ep_size': 2,
        'ranks': [{
            'rank': 0,
            'experts': [0]
        }, {
            'rank': 1,
            'experts': [2]
        }],
    }, 'does not cover global expert IDs \\[1\\]'),
    ({
        'version': AUTOEP_PLACEMENT_VERSION,
        'num_experts': 2,
        'ep_size': 1,
        'ranks': [{
            'rank': 0,
            'experts': [0, 2]
        }],
    }, 'outside \\[0, 2\\)'),
])
def test_malformed_placement_is_rejected(descriptor, match):
    with pytest.raises(ValueError, match=match):
        validate_autoep_placement_descriptor(descriptor)


@pytest.mark.parametrize('num_experts, num_local_experts, ep_size', [(8, 3, 2), (0, 1, 1), (4, 0, 2)])
def test_inconsistent_legacy_metadata_is_rejected(num_experts, num_local_experts, ep_size):
    with pytest.raises(ValueError):
        legacy_uniform_autoep_placement_descriptor(num_experts, num_local_experts, ep_size)


def test_descriptor_and_affine_map_serialization_round_trip():
    descriptor = make_autoep_placement_descriptor(4, [[2, 0], [1, 2, 3]])
    serialized_descriptor = json.loads(json.dumps(descriptor))
    validate_autoep_placement_descriptor(serialized_descriptor)
    affine_map = autoep_placement_to_affine_map(serialized_descriptor, (4, 3))
    restored_map = ParamAffineMap.from_dict(json.loads(json.dumps(affine_map.to_dict())))
    full_tensor = torch.randn(4, 3)
    shards = _oracle_shards(full_tensor, [[2, 0], [1, 2, 3]])

    assert serialized_descriptor == descriptor
    assert restored_map.to_dict() == affine_map.to_dict()
    assert torch.equal(restored_map.rebuild(shards), full_tensor)


def test_extract_target_rank_uses_target_packed_order():
    descriptor = make_autoep_placement_descriptor(5, [[0, 2], [4, 1, 3]])
    target_map = autoep_placement_to_affine_map(descriptor, (5, 2, 2))
    universal_tensor = torch.arange(20).reshape(5, 2, 2)
    expected = torch.stack([universal_tensor[4], universal_tensor[1], universal_tensor[3]])

    actual = extract_autoep_rank_tensor(universal_tensor, target_map, 1)

    assert torch.equal(actual, expected)

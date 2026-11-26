# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import json

import pytest
import torch

from deepspeed.checkpoint.affine import AFFINE_MAP_FORMAT_VERSION, ParamAffineMap
from deepspeed.checkpoint.affine_transfer import plan_transfer
from deepspeed.checkpoint.autoep_affine import (AUTOEP_PLACEMENT_VERSION, autoep_experts_for_rank,
                                                autoep_metadata_to_affine_map, autoep_placement_to_affine_map,
                                                autoep_placements_equal, autoep_storage_affine_map,
                                                extract_autoep_rank_tensor, legacy_uniform_autoep_placement_descriptor,
                                                make_autoep_placement_descriptor, validate_autoep_placement_descriptor)
from deepspeed.checkpoint.constants import (AFFINE_MAP, AFFINE_MAP_PARAMS, AFFINE_MAP_VERSION, AUTOEP_AFFINE_MAPS,
                                            AUTOEP_EXPERT_PLACEMENT, AUTOEP_PLACEMENT_RANKS)


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


def _oracle_storage(full_tensor, experts_by_rank, holders_by_ep_rank, partition_numel_by_ep_rank=None):
    """What each global rank stores, built by indexing the full tensor rather than through a map."""
    storage = {}
    for ep_rank, packed in _oracle_shards(full_tensor, experts_by_rank).items():
        flat = packed.reshape(-1)
        for edp_rank, global_rank in enumerate(holders_by_ep_rank[ep_rank]):
            if partition_numel_by_ep_rank is None:
                storage[global_rank] = flat.clone()
                continue
            partition_numel = partition_numel_by_ep_rank[ep_rank]
            # ZeRO-3 pads the flattened tensor at its end; the padding holds no parameter elements.
            storage[global_rank] = flat[edp_rank * partition_numel:(edp_rank + 1) * partition_numel].clone()
    return storage


def _execute(plan, source_storage, target_storage):
    writes = {rank: torch.zeros_like(storage, dtype=torch.int64) for rank, storage in target_storage.items()}
    for segment in plan.segments:
        source = torch.as_strided(source_storage[segment.source_rank], segment.shape, segment.source_strides,
                                  segment.source_offset)
        torch.as_strided(target_storage[segment.target_rank], segment.shape, segment.target_strides,
                         segment.target_offset).copy_(source)
        torch.as_strided(writes[segment.target_rank], segment.shape, segment.target_strides,
                         segment.target_offset).add_(1)
    return writes


# EP=2 x EDP=2 over global ranks 0..3, then EP=3 x EDP=2 with reordered, replicated and empty
# placements on interleaved global ranks. Partition sizes split experts mid-row and leave padding.
_SOURCE = ([[0, 1], [2, 3]], {0: [0, 2], 1: [1, 3]}, {0: 23, 1: 23})
_TARGET = ([[3, 0], [], [2, 1, 0]], {0: [5, 1], 1: [0, 4], 2: [3, 2]}, {0: 30, 1: 0, 2: 38})


@pytest.mark.parametrize('zero3', [False, True])
@pytest.mark.parametrize('layout', [_SOURCE, _TARGET], ids=['uniform', 'replicated_and_empty'])
def test_storage_map_describes_what_each_global_rank_stores(layout, zero3):
    experts_by_rank, holders, partitions = layout
    partitions = partitions if zero3 else None
    full_tensor = torch.arange(4 * 3 * 5, dtype=torch.float32).reshape(4, 3, 5)
    descriptor = make_autoep_placement_descriptor(4, experts_by_rank)
    expected = _oracle_storage(full_tensor, experts_by_rank, holders, partitions)

    affine_map = autoep_storage_affine_map(descriptor, full_tensor.shape, holders, partitions)

    affine_map.validate_coverage()
    assert affine_map.shard_shapes == {rank: tuple(storage.shape) for rank, storage in expected.items()}
    for rank, storage in expected.items():
        torch.testing.assert_close(affine_map.extract(full_tensor, rank), storage, rtol=0, atol=0)
    torch.testing.assert_close(affine_map.rebuild(expected), full_tensor, rtol=0, atol=0)


def test_storage_map_offers_every_edp_replica_to_the_planner():
    experts_by_rank, holders, _ = _SOURCE
    full_tensor = torch.arange(4 * 6, dtype=torch.float32).reshape(4, 6)
    source_map = autoep_storage_affine_map(make_autoep_placement_descriptor(4, experts_by_rank), full_tensor.shape,
                                           holders)
    target_experts, target_holders, target_partitions = _TARGET
    target_map = autoep_storage_affine_map(make_autoep_placement_descriptor(4, target_experts), full_tensor.shape,
                                           target_holders, target_partitions)
    source_storage = _oracle_storage(full_tensor, experts_by_rank, holders)
    replica_sets = {frozenset(ranks) for ranks in holders.values()}
    offered = []

    def highest_replica(request):
        offered.append(frozenset(candidate.source_rank for candidate in request.candidates))
        return max(candidate.source_rank for candidate in request.candidates)

    for selector, used_ranks in ((None, {0, 1}), (highest_replica, {2, 3})):
        plan = plan_transfer(target_map, source_map, replica_selector=selector)
        assert {segment.source_rank for segment in plan.segments} == used_ranks
        target_storage = {rank: torch.full(shape, float('nan')) for rank, shape in target_map.shard_shapes.items()}
        writes = _execute(plan, source_storage, target_storage)
        expected = _oracle_storage(full_tensor, target_experts, target_holders, target_partitions)
        for rank, storage in expected.items():
            torch.testing.assert_close(target_storage[rank], storage, rtol=0, atol=0)
            assert torch.all(writes[rank] == 1)
    assert offered and set(offered) <= replica_sets


def test_zero3_storage_plans_directly_into_a_new_placement():
    full_tensor = torch.arange(4 * 3 * 5, dtype=torch.float32).reshape(4, 3, 5)
    layouts = []
    for experts_by_rank, holders, partitions in (_SOURCE, _TARGET):
        descriptor = make_autoep_placement_descriptor(4, experts_by_rank)
        layouts.append((autoep_storage_affine_map(descriptor, full_tensor.shape, holders, partitions),
                        _oracle_storage(full_tensor, experts_by_rank, holders, partitions)))
    (source_map, source_storage), (target_map, expected) = layouts

    plan = plan_transfer(target_map, source_map)

    target_storage = {rank: torch.full(shape, float('nan')) for rank, shape in target_map.shard_shapes.items()}
    writes = _execute(plan, source_storage, target_storage)
    for rank, storage in expected.items():
        torch.testing.assert_close(target_storage[rank], storage, rtol=0, atol=0)
        assert torch.all(writes[rank] == 1)
    # Global ranks 0 and 4 hold the empty EP rank; rank 1's ZeRO-3 partition is all padding.
    assert target_map.shard_shapes[0] == target_map.shard_shapes[4] == target_map.shard_shapes[1] == (0, )
    assert not {0, 4, 1} & {segment.target_rank for segment in plan.segments}


def test_storage_map_follows_explicit_rank_ids():
    experts_by_rank, holders, partitions = _TARGET
    descriptor = make_autoep_placement_descriptor(4, experts_by_rank)
    reordered = dict(descriptor)
    reordered[AUTOEP_PLACEMENT_RANKS] = list(reversed(descriptor[AUTOEP_PLACEMENT_RANKS]))

    assert autoep_placements_equal(descriptor, reordered)
    assert not autoep_placements_equal(descriptor, make_autoep_placement_descriptor(4, [[0, 3], [], [2, 1, 0]]))
    for zero3_partitions in (None, partitions):
        expected = autoep_storage_affine_map(descriptor, (4, 6), holders, zero3_partitions)
        actual = autoep_storage_affine_map(reordered, (4, 6), dict(reversed(list(holders.items()))), zero3_partitions)
        assert actual.to_dict() == expected.to_dict()


@pytest.mark.parametrize('holders, partitions, match', [
    ({
        0: [0, 2]
    }, None, 'names EP ranks'),
    ({
        0: [0, 2],
        1: [1, 2]
    }, None, 'more than one EP rank'),
    ({
        0: [0, 2],
        1: []
    }, None, 'no global holder'),
    ({
        0: [0, 2],
        1: [1, 3]
    }, {
        0: 23,
        1: 14
    }, 'cannot store'),
    ({
        0: [0, 2],
        1: [1, 3]
    }, {
        0: 23
    }, 'Partition size'),
])
def test_storage_map_rejects_holders_that_do_not_store_the_placement(holders, partitions, match):
    descriptor = make_autoep_placement_descriptor(4, [[0, 1], [2, 3]])
    with pytest.raises(ValueError, match=match):
        autoep_storage_affine_map(descriptor, (4, 3, 5), holders, partitions)

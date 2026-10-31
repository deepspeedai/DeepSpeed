# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Check ``plan_transfers``: a replica policy may choose where data is read from, never what arrives.

The planner validates every policy answer before it emits a segment: one answer per request, only
validated holders, and parts that tile each request exactly along axis 0. If that check is right,
then whenever the replicas agree, every accepted answer moves the same data -- which is what the
correctness argument for the affine IR (#8252, Theorem 3 of the proof draft) needs from a policy.
The randomized test below is that statement as a check: many random policies, including random
splits, all producing the same target shards. The rejection tests make sure an answer that would
break it is refused rather than planned.
"""

import dataclasses
import random

import pytest
import torch

from deepspeed.checkpoint.affine import contiguous_split_map, replicated_map, segmented_map
from deepspeed.checkpoint.affine_replica_policy import first_holder, local_first_least_loaded, per_request
from deepspeed.checkpoint.affine_transfer import (AffineTransferError, InvalidReplicaPolicyError, PlanBudget,
                                                  ReplicaCandidate, ReplicaPart, ReplicaRequest, plan_transfer,
                                                  plan_transfers)

SENTINEL = -12345.0


def _product(shape):
    count = 1
    for size in shape:
        count *= size
    return count


def _apply_plan(plan, target_map, source_shards):
    """Execute a plan over plain buffers, counting writes. Test scaffolding, never a production path."""
    targets = {}
    for rank, shape in target_map.shard_shapes.items():
        values = torch.full((_product(shape), ), SENTINEL, dtype=torch.float64)
        writes = torch.zeros(_product(shape), dtype=torch.int64)
        targets[rank] = (values, writes)
    for segment in plan.segments:
        source = torch.as_strided(source_shards[segment.source_rank].reshape(-1), segment.shape,
                                  segment.source_strides, segment.source_offset)
        values, writes = targets[segment.target_rank]
        torch.as_strided(values, segment.shape, segment.target_strides, segment.target_offset).copy_(source)
        torch.as_strided(writes, segment.shape, segment.target_strides, segment.target_offset).add_(1)
    return targets


def _assert_moves_exactly(plan, source_map, target_map, full):
    sources = {rank: source_map.extract(full, rank).contiguous() for rank in source_map.shard_shapes}
    for rank, (values, writes) in _apply_plan(plan, target_map, sources).items():
        assert torch.equal(values.view(target_map.shard_shapes[rank]), target_map.extract(full, rank)), rank
        assert bool(torch.all(writes == 1)), rank


def _pairs():
    """Parameters with replicated sources: whole-parameter replicas, a partly replicated fused block,
    a vector and a scalar."""
    bigcode_source = segmented_map((48, 8), [(32, False), (16, True)], 0, 2, split_widths=[[16, 16]])
    bigcode_target = segmented_map((48, 8), [(32, False), (16, True)], 0, 4, split_widths=[[8, 8, 8, 8]])
    return {
        'dense': (contiguous_split_map((12, 6), [2, 4], 1), replicated_map((12, 6), 3)),
        'rows': (contiguous_split_map((10, 4), [3, 7], 0), replicated_map((10, 4), 2)),
        'bigcode': (bigcode_target, bigcode_source),
        'vector': (contiguous_split_map((9, ), [4, 5], 0), replicated_map((9, ), 3)),
        'scalar': (replicated_map((), 2), replicated_map((), 3)),
    }


def _random_policy(rng):

    def policy(requests):
        answers = []
        for request in requests:
            locations = [candidate.source_rank for candidate in request.candidates]
            rows = request.shape[0] if request.shape else 1
            if not request.shape or rows == 1 or rng.random() < 0.4:
                answers.append(rng.choice(locations))
                continue
            cuts = sorted(rng.sample(range(1, rows), rng.randint(1, min(3, rows - 1))))
            starts = [0] + cuts
            ends = cuts + [rows]
            parts = [
                ReplicaPart((request.logical_origin[0] + start, ) + request.logical_origin[1:],
                            (end - start, ) + request.shape[1:], rng.choice(locations))
                for start, end in zip(starts, ends)
            ]
            rng.shuffle(parts)
            answers.append(parts)
        return answers

    return policy


@pytest.mark.parametrize('seed', range(24))
def test_every_accepted_answer_moves_the_same_data(seed):
    pairs = _pairs()
    plans = plan_transfers(pairs, replica_policy=_random_policy(random.Random(seed)))
    assert sorted(plans) == sorted(pairs)
    for name, (target_map, source_map) in pairs.items():
        torch.manual_seed(seed)
        full = torch.randn(tuple(target_map.logical_shape), dtype=torch.float64)
        _assert_moves_exactly(plans[name], source_map, target_map, full)


def test_first_holder_and_per_request_reproduce_plan_transfer():
    pairs = _pairs()

    def select(request):
        return max(candidate.source_rank for candidate in request.candidates)

    for policy, selector in ((first_holder, None), (per_request(select), select)):
        plans = plan_transfers(pairs, replica_policy=policy)
        for name, (target_map, source_map) in pairs.items():
            assert plans[name] == plan_transfer(target_map, source_map, replica_selector=selector), name


def _recorded(policy):
    seen = []

    def wrapper(requests):
        seen.extend(requests)
        return policy(requests)

    return wrapper, seen


def test_requests_arrive_in_a_fixed_order_with_processes_and_bytes():
    pairs = {
        'b': (contiguous_split_map((4, 2), [2, 2], 0), replicated_map((4, 2), 2)),
        'a': (contiguous_split_map((6, ), [6], 0), contiguous_split_map((6, ), [6], 0)),
    }
    policy, seen = _recorded(first_holder)
    plan_transfers(pairs,
                   source_processes={
                       0: 7,
                       1: 8
                   },
                   target_processes={
                       0: 8,
                       1: 9
                   },
                   element_size={
                       'a': 4,
                       'b': 2
                   },
                   replica_policy=policy)

    assert [(request.parameter, request.target_location, request.logical_origin) for request in seen] == [
        ('a', 0, (0, )),
        ('b', 0, (0, 0)),
        ('b', 1, (2, 0)),
    ]
    single = seen[0]
    assert [candidate.source_rank for candidate in single.candidates] == [0]
    assert single.candidate_processes == (7, )
    assert single.target_process == 8
    assert single.bytes == 6 * 4 and single.numel == 6
    replicated = seen[2]
    assert [candidate.source_rank for candidate in replicated.candidates] == [0, 1]
    assert replicated.candidate_processes == (7, 8)
    assert replicated.target_process == 9
    assert replicated.bytes == 4 * 2


def test_without_mappings_requests_carry_no_processes_or_bytes():
    policy, seen = _recorded(first_holder)
    plan_transfers({'w': (contiguous_split_map((4, ), [4], 0), replicated_map((4, ), 2))}, replica_policy=policy)
    assert seen[0].target_process is None
    assert seen[0].candidate_processes == (None, None)
    assert seen[0].bytes is None


def _sources(plans):
    return {name: sorted({segment.source_rank for segment in plan.segments}) for name, plan in plans.items()}


def test_default_policy_copies_locally_when_the_target_process_holds_the_region():
    pairs = {'w': (contiguous_split_map((8, 4), [4, 4], 0), replicated_map((8, 4), 3))}
    plans = plan_transfers(pairs, source_processes={0: 10, 1: 11, 2: 12}, target_processes={0: 12, 1: 11})
    by_target = {segment.target_rank: segment.source_rank for segment in plans['w'].segments}
    assert by_target == {0: 2, 1: 1}


def test_default_policy_balances_and_counts_load_it_cannot_move():
    # 'pinned' has one holder (location 0); its 64 bytes make location 0 the busier sender, so the
    # replicated 'w' goes to location 1 even though 0 is the lower location.
    pairs = {
        'pinned': (contiguous_split_map((16, ), [16], 0), contiguous_split_map((16, ), [16], 0)),
        'w': (contiguous_split_map((4, ), [4], 0), replicated_map((4, ), 2)),
    }
    plans = plan_transfers(pairs,
                           source_processes={
                               0: 0,
                               1: 1
                           },
                           target_processes={0: 5},
                           element_size={
                               'pinned': 4,
                               'w': 4
                           })
    assert _sources(plans) == {'pinned': [0], 'w': [1]}


def test_default_policy_spreads_equal_regions_over_holders():
    pairs = {'w': (contiguous_split_map((8, 2), [2, 2, 2, 2], 0), replicated_map((8, 2), 2))}
    plans = plan_transfers(pairs)
    by_target = {segment.target_rank: segment.source_rank for segment in plans['w'].segments}
    assert by_target == {0: 0, 1: 1, 2: 0, 3: 1}


def test_default_policy_without_processes_never_assumes_a_local_copy():
    # The same number names different processes on the two sides. Without mappings, target
    # location 1 must not prefer source location 1 as if it were local: the tie goes to location 0.
    request = ReplicaRequest(parameter='w',
                             target_location=1,
                             target_process=None,
                             logical_origin=(0, ),
                             shape=(4, ),
                             target_offset=0,
                             target_strides=(1, ),
                             candidates=(ReplicaCandidate(0, 0, (1, )), ReplicaCandidate(1, 0, (1, ))),
                             candidate_processes=(None, None),
                             bytes=None)
    assert local_first_least_loaded([request]) == [0]
    with_processes = dataclasses.replace(request, target_process=11, candidate_processes=(10, 11))
    assert local_first_least_loaded([with_processes]) == [1]


def test_a_split_reads_each_cut_from_its_holder():
    target_map, source_map = contiguous_split_map((6, 3), [6], 0), replicated_map((6, 3), 2)

    def split(requests):
        (request, ) = requests
        assert request.shape == (6, 3)
        return [[ReplicaPart((4, 0), (2, 3), 0), ReplicaPart((0, 0), (4, 3), 1)]]

    plan = plan_transfers({'w': (target_map, source_map)}, replica_policy=split)['w']
    assert sorted((segment.source_rank, segment.target_offset, segment.shape) for segment in plan.segments) == [
        (0, 12, (2, 3)),
        (1, 0, (4, 3)),
    ]
    _assert_moves_exactly(plan, source_map, target_map, torch.randn(6, 3, dtype=torch.float64))


def _one_request_pair():
    return {'w': (contiguous_split_map((6, 4), [6], 0), replicated_map((6, 4), 2))}


def _answering(answer):
    return lambda requests: [answer for _ in requests]


@pytest.mark.parametrize(
    'answer',
    [
        2,  # not a holder
        True,  # a bool is not a location
        1.0,
        None,
        '0',  # a string is not a sequence of parts
        [],
        [3],  # not a ReplicaPart
        [ReplicaPart((0, 0), (6, 4), 5)],  # not a holder
        [ReplicaPart((0, 0), (6, 4), True)],
        [ReplicaPart((0, 0), (2, 4), 0), ReplicaPart((3, 0), (3, 4), 1)],  # gap at row 2
        [ReplicaPart((0, 0), (4, 4), 0), ReplicaPart((3, 0), (3, 4), 1)],  # overlap at row 3
        [ReplicaPart((0, 0), (4, 4), 0)],  # stops early
        [ReplicaPart((0, 0), (7, 4), 0)],  # runs past the region
        [ReplicaPart((0, 0), (6, 2), 0), ReplicaPart((0, 2), (6, 2), 1)],  # a cut along axis 1
        [ReplicaPart((0, 0), (0, 4), 0), ReplicaPart((0, 0), (6, 4), 1)],  # an empty part
        [ReplicaPart((0, ), (6, ), 0)],  # wrong number of axes
        [ReplicaPart((0, 0), (6.0, 4), 0)],
    ])
def test_answers_that_could_change_what_arrives_are_refused(answer):
    with pytest.raises(InvalidReplicaPolicyError):
        plan_transfers(_one_request_pair(), replica_policy=_answering(answer))


def test_one_answer_per_request_is_required():
    pairs = {'w': (contiguous_split_map((4, ), [2, 2], 0), replicated_map((4, ), 2))}
    with pytest.raises(InvalidReplicaPolicyError):
        plan_transfers(pairs, replica_policy=lambda requests: [0])
    with pytest.raises(InvalidReplicaPolicyError):
        plan_transfers(pairs, replica_policy=lambda requests: (0 for _ in requests))
    # Any real sequence will do, for the answers and for the parts of one answer.
    plan_transfers(pairs, replica_policy=lambda requests: tuple(0 for _ in requests))
    plan_transfers(pairs,
                   replica_policy=lambda requests: [(ReplicaPart(request.logical_origin, request.shape, 1), )
                                                    for request in requests])


def test_a_scalar_cannot_be_split():
    pairs = {'s': (replicated_map((), 1), replicated_map((), 2))}
    plan_transfers(pairs, replica_policy=_answering([ReplicaPart((), (), 1)]))
    with pytest.raises(InvalidReplicaPolicyError):
        plan_transfers(pairs, replica_policy=_answering([ReplicaPart((), (), 0), ReplicaPart((), (), 1)]))


def test_min_split_bytes_bounds_every_part_of_a_split():
    split = _answering([ReplicaPart((0, 0), (2, 4), 0), ReplicaPart((2, 0), (4, 4), 1)])
    # The smaller part is 2 x 4 elements x 4 bytes = 32 bytes.
    plan_transfers(_one_request_pair(), element_size={'w': 4}, min_split_bytes=32, replica_policy=split)
    with pytest.raises(InvalidReplicaPolicyError):
        plan_transfers(_one_request_pair(), element_size={'w': 4}, min_split_bytes=33, replica_policy=split)
    # An unsplit answer is never too small.
    plan_transfers(_one_request_pair(), element_size={'w': 4}, min_split_bytes=10**9, replica_policy=_answering(0))


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(min_split_bytes=8),  # needs element_size
        dict(element_size={'w': 4}, min_split_bytes=0),
        dict(element_size={}),
        dict(element_size={'w': True}),
        dict(source_processes={0: 0}),  # location 1 has no process
        dict(target_processes={}),
        dict(source_processes={
            0: 0,
            1: 1.0
        }),
    ])
def test_inconsistent_arguments_are_refused(kwargs):
    with pytest.raises(AffineTransferError):
        plan_transfers(_one_request_pair(), **kwargs)


def test_splits_count_against_the_budget():
    pairs = {'w': (contiguous_split_map((64, 1), [64], 0), replicated_map((64, 1), 2))}
    rows = _answering([ReplicaPart((row, 0), (1, 1), row % 2) for row in range(64)])
    plan_transfers(pairs, PlanBudget(max_pair_checks=64), replica_policy=rows)
    with pytest.raises(AffineTransferError):
        plan_transfers(pairs, PlanBudget(max_pair_checks=63), replica_policy=rows)


def test_maps_are_validated_before_the_policy_runs():
    called = []

    def policy(requests):
        called.append(True)
        return first_holder(requests)

    pairs = {'w': (contiguous_split_map((4, 2), [4], 0), replicated_map((4, 3), 2))}
    with pytest.raises(AffineTransferError):
        plan_transfers(pairs, replica_policy=policy)
    assert not called

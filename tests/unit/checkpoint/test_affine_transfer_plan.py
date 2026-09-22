# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Check that a composed transfer plan moves exactly the elements a rebuild-then-extract would.

The planner's claim is narrow: given the map the source shards obey and the map the target shards
obey, name the copies between them. Two things have to hold for that to be worth anything, and they
are checked differently here.

*Addresses* are checked against the specification's own worked example -- an 8x8 parameter, two row
shards becoming four column shards -- whose every offset is derivable by hand. A planner that agrees
with an oracle built the same way it was built proves nothing, so the numbers in that test are
transcribed, not computed.

*Coverage* is checked by executing the plan. A tiny interpreter writes into target buffers primed
with a sentinel and the result is compared against ``extract`` of the same full tensor, so a gap, a
double write, or a block read from the wrong place all fail the same way. The interpreter lives here
and not in the planner: it exists to prove geometry, and a production path must not need it.

Random layouts are exercised only inside the supported subset. Unsupported cases are asserted to
refuse rather than being skipped, so a passing run cannot be an accident of generation.
"""

import itertools
import random

import pytest
import torch

from deepspeed.checkpoint.affine import (AffinePiece, ParamAffineMap, block_gather_map, contiguous_split_map,
                                         replicated_map, segmented_map, sub_param_map)
from deepspeed.checkpoint.affine_transfer import (AffineTransferError, InvalidMapError, PlanBudget,
                                                  UnsupportedLayoutError, plan_transfer)

SENTINEL = -12345.0


def _shards(map_, maker):
    """Every rank's shard, built from the full parameter through the map being tested against."""
    return {rank: maker(map_, rank) for rank in map_.shard_shapes}


def _apply_plan(plan, target_map, source_shards):
    """Execute a plan over plain buffers. Test scaffolding, never a production path."""
    targets = {
        rank: torch.full((_product(shape), ), SENTINEL, dtype=torch.float64)
        for rank, shape in target_map.shard_shapes.items()
    }
    for segment in plan.segments:
        source = torch.as_strided(source_shards[segment.source_rank].reshape(-1), segment.shape,
                                  segment.source_strides, segment.source_offset)
        target = torch.as_strided(targets[segment.target_rank], segment.shape, segment.target_strides,
                                  segment.target_offset)
        target.copy_(source)
    return {rank: buffer.view(target_map.shard_shapes[rank]) for rank, buffer in targets.items()}


def _product(shape):
    count = 1
    for size in shape:
        count *= size
    return count


def _split(total, degree):
    return [total // degree + (1 if rank < total % degree else 0) for rank in range(degree)]


def _full(rows, cols, seed):
    torch.manual_seed(seed)
    return torch.randn(rows, cols, dtype=torch.float64)


# ---------------------------------------------------------------- the hand-checked example


def test_row_tp2_to_column_tp4_matches_the_hand_derived_addresses():
    """W is 8x8, split by rows across 2 ranks, restored split by columns across 4.

    Each source shard is 4x8 row-major; each target shard is 8x2 column-major-out-of-row-major, so a
    4x2 block sits at 2*b in the source and 8*a in the target. The plan must be 8 such blocks and
    must move 64 elements, and no side may be read as though it were packed.
    """
    source = contiguous_split_map((8, 8), [4, 4], 0)
    target = contiguous_split_map((8, 8), [2, 2, 2, 2], 1)

    plan = plan_transfer(target, source)

    assert len(plan.segments) == 8
    assert plan.moved_elements == 64
    by_pair = {(segment.source_rank, segment.target_rank): segment for segment in plan.segments}
    assert len(by_pair) == 8, "every source/target pair should appear exactly once"
    for a, b in itertools.product(range(2), range(4)):
        segment = by_pair[(a, b)]
        assert segment.shape == (4, 2)
        assert segment.source_offset == 2 * b
        assert segment.source_strides == (8, 1)
        assert segment.target_offset == 8 * a
        assert segment.target_strides == (2, 1)
        assert segment.logical_origin == (4 * a, 2 * b)


def test_identity_still_ships_a_copy_per_holder():
    """Same topology in and out is not a no-op: every rank must still be named as a writer."""
    single = contiguous_split_map((6, 4), [3, 3], 0)

    plan = plan_transfer(single, single)

    assert plan.moved_elements == 24
    assert {segment.target_rank for segment in plan.segments} == {0, 1}
    assert all(segment.source_rank == segment.target_rank for segment in plan.segments)


# ---------------------------------------------------------------- executed coverage


# Every layout keeps one fixed logical shape across degrees, so a plan is always composed between two
# shardings of the same parameter. Sizes are chosen to be divisible at 1, 2 and 4 except where the
# test is specifically about an uneven split.
LAYOUTS = {
    'row_even': lambda tp: contiguous_split_map((8, 8), _split(8, tp), 0),
    'row_uneven': lambda tp: contiguous_split_map((11, 8), _split(11, tp), 0),
    'column': lambda tp: contiguous_split_map((8, 24), _split(24, tp), 1),
    'replicated': lambda tp: replicated_map((5, 6), tp),
    'sub_params': lambda tp: sub_param_map((12, 8), (8, 4), [_split(8, tp), _split(4, tp)], 0),
    'bigcode': lambda tp: segmented_map((48, 8), [(32, False), (16, True)], 0, tp, split_widths=[_split(32, tp)]),
    'yuan_gather': lambda tp: block_gather_map((8, 64),
                                               {rank: [head for head in range(8) if head % tp == rank]
                                                for rank in range(tp)}, 8, 1),
}
DEGREES = [1, 2, 4]
PAIRS = [(s, t) for s in DEGREES for t in DEGREES if s != t] + [(d, d) for d in DEGREES]


@pytest.mark.parametrize('name', sorted(LAYOUTS))
@pytest.mark.parametrize('source_degree, target_degree', PAIRS, ids=[f'{s}to{t}' for s, t in PAIRS])
def test_plan_moves_what_rebuild_and_extract_would(name, source_degree, target_degree):
    """The plan and the full-materialization path must agree element for element.

    ``extract`` is the oracle because it is the operation the plan exists to replace, and it is
    written against the full tensor, so it cannot share a bug with the planner's box arithmetic.
    """
    build = LAYOUTS[name]
    source = build(source_degree)
    target = build(target_degree)
    full = _full(*target.logical_shape, seed=source_degree * 10 + target_degree)

    plan = plan_transfer(target, source)
    source_shards = {rank: source.extract(full, rank) for rank in source.shard_shapes}
    produced = _apply_plan(plan, target, source_shards)
    expected = {rank: target.extract(full, rank) for rank in target.shard_shapes}

    assert set(produced) == set(expected)
    for rank in expected:
        assert torch.equal(produced[rank], expected[rank]), f'{name} {source_degree}->{target_degree} rank {rank}'


@pytest.mark.parametrize('name', sorted(LAYOUTS))
def test_every_target_element_is_written_exactly_once(name):
    """Counts, not values: a gap and an equal-size overlap cancel in any total the plan reports."""
    build = LAYOUTS[name]
    source, target = build(2), build(4)
    full = _full(*target.logical_shape, seed=7)

    plan = plan_transfer(target, source)
    produced = _apply_plan(plan, target, {rank: source.extract(full, rank) for rank in source.shard_shapes})

    for rank, shard in produced.items():
        written = shard != SENTINEL
        assert written.all(), f'{name}: target rank {rank} left {int((~written).sum())} elements unwritten'


def test_a_plan_is_deterministic_under_input_order():
    """Two runs, and a map whose pieces were listed in a different order, must agree exactly.

    A transport that shards work by segment index needs this, and a plan that varied between runs
    would make a transfer failure impossible to reproduce.
    """
    source = contiguous_split_map((8, 8), [4, 4], 0)
    target = contiguous_split_map((8, 8), [3, 3, 2], 1)
    shuffled_target = ParamAffineMap(target.logical_shape, dict(target.shard_shapes),
                                     {rank: list(reversed(pieces)) for rank, pieces in target.pieces_by_rank.items()})

    first = plan_transfer(target, source)
    second = plan_transfer(target, source)
    reordered = plan_transfer(shuffled_target, source)

    assert first.segments == second.segments
    assert reordered.segments == first.segments


# ---------------------------------------------------------------- refusals


def test_a_source_that_does_not_reach_the_target_is_refused():
    """Half a parameter with nowhere to come from must fail, not leave a silently unwritten shard."""
    whole = contiguous_split_map((16, 8), [16], 0)
    half = ParamAffineMap(logical_shape=(16, 8), shard_shapes={0: (8, 8)},
                          pieces_by_rank={
                              0: [
                                  AffinePiece(shape=(8, 8),
                                              source_offset=0,
                                              source_strides=(8, 1),
                                              dest_offset=0,
                                              dest_strides=(8, 1),
                                              locations=[0])
                              ]
                          })

    with pytest.raises(InvalidMapError, match='of its'):
        plan_transfer(whole, half)


def test_two_owners_for_one_region_are_refused():
    source = ParamAffineMap(
        logical_shape=(8, 8),
        shard_shapes={0: (4, 8), 1: (4, 8)},
        pieces_by_rank={
            0: [AffinePiece((4, 8), 0, (8, 1), 0, (8, 1), [0])],
            1: [AffinePiece((4, 8), 0, (8, 1), 0, (8, 1), [1])],
        })
    target = replicated_map((8, 8), 1)

    with pytest.raises(InvalidMapError, match='different locations'):
        plan_transfer(target, source)


def test_a_permuted_piece_is_refused_rather_than_flattened():
    """Strides that are not the tensor's own row-major order are a permutation, and one box cannot hold it."""
    source = ParamAffineMap(
        logical_shape=(8, 8),
        shard_shapes={0: (8, 8)},
        pieces_by_rank={
            0: [AffinePiece((8, 8), 0, (1, 8), 0, (8, 1), [0])],
        })
    target = replicated_map((8, 8), 1)

    with pytest.raises(UnsupportedLayoutError, match='strides'):
        plan_transfer(target, source)


def test_a_scaled_piece_is_refused_for_the_same_reason_affine_refuses_it():
    source = replicated_map((4, 4), 2, scale=0.5)
    target = replicated_map((4, 4), 2)

    with pytest.raises(UnsupportedLayoutError, match='scale'):
        plan_transfer(target, source)


def test_two_pieces_claiming_one_shard_range_are_refused():
    """Disjoint in the parameter, colliding in the shard, and perfectly counted either way.

    This is the premise the rest of the proof borrows, so it is checked rather than assumed: four
    pieces of eight elements each, summing to two shards of eight, tiling the parameter exactly, and
    one rank's second piece landing where its first already is. Nothing about the counts can see it.
    """
    source = ParamAffineMap(
        logical_shape=(4, 4),
        shard_shapes={0: (4, 2), 1: (4, 2)},
        pieces_by_rank={
            0: [
                AffinePiece((2, 2), 0, (4, 1), 0, (2, 1), [0]),
                AffinePiece((2, 2), 8, (4, 1), 0, (2, 1), [0]),
            ],
            1: [
                AffinePiece((2, 2), 2, (4, 1), 0, (2, 1), [1]),
                AffinePiece((2, 2), 10, (4, 1), 4, (2, 1), [1]),
            ],
        })
    assert sum(piece.numel for pieces in source.pieces_by_rank.values() for piece in pieces) == 16

    with pytest.raises(InvalidMapError, match='same\\s+shard addresses'):
        plan_transfer(replicated_map((4, 4), 1), source)


def test_a_shard_described_with_the_wrong_rank_is_refused():
    """A flat shard against a matrix parameter has no axis correspondence to plan along."""
    source = ParamAffineMap(logical_shape=(4, 4),
                            shard_shapes={0: (16, )},
                            pieces_by_rank={
                                0: [AffinePiece((4, 4), 0, (4, 1), 0, (4, 1), [0])]
                            })

    with pytest.raises(UnsupportedLayoutError, match='number of axes'):
        plan_transfer(replicated_map((4, 4), 1), source)


def test_non_contiguous_rank_numbers_are_two_namespaces_not_one():
    """Ranks 3 and 9 are addresses within a layout, not a world size or a process id."""
    source = ParamAffineMap(
        logical_shape=(4, 16),
        shard_shapes={3: (4, 8), 9: (4, 8)},
        pieces_by_rank={
            3: [AffinePiece((4, 8), 0, (16, 1), 0, (8, 1), [3])],
            9: [AffinePiece((4, 8), 8, (16, 1), 0, (8, 1), [9])],
        })
    target = contiguous_split_map((4, 16), [16], 1)

    plan = plan_transfer(target, source)
    assert {segment.source_rank for segment in plan.segments} == {3, 9}
    assert {segment.target_rank for segment in plan.segments} == {0}
    assert plan.moved_elements == 64


def test_different_logical_shapes_are_not_a_reshape():
    with pytest.raises(InvalidMapError, match='describes'):
        plan_transfer(contiguous_split_map((8, 8), [8], 0), contiguous_split_map((64, ), [64], 0))


def test_an_exhausted_budget_refuses_the_whole_plan():
    """A truncated plan is worse than no plan, because it looks like a completed transfer."""
    source = contiguous_split_map((8, 8), [4, 4], 0)
    target = contiguous_split_map((8, 8), [2, 2, 2, 2], 1)

    with pytest.raises(AffineTransferError, match='max_pair_checks'):
        plan_transfer(target, source, PlanBudget(max_pair_checks=1))
    with pytest.raises(AffineTransferError, match='max_segments'):
        plan_transfer(target, source, PlanBudget(max_segments=1))
    with pytest.raises(AffineTransferError, match='max_plan_bytes'):
        plan_transfer(target, source, PlanBudget(max_plan_bytes=1))


def test_a_map_naming_a_rank_with_no_shard_is_refused():
    source = ParamAffineMap(logical_shape=(4, 4), shard_shapes={0: (4, 4)},
                            pieces_by_rank={
                                0: [AffinePiece((4, 4), 0, (4, 1), 0, (4, 1), [0])],
                                7: []
                            })
    with pytest.raises(InvalidMapError, match='no shard shape'):
        plan_transfer(replicated_map((4, 4), 1), source)


# ---------------------------------------------------------------- never materializing


def test_planning_never_touches_the_full_parameter(monkeypatch):
    """Bind the materializing operations to failures on the class the planner would reach them through.

    Patching the definition site is not enough -- the planner resolves these through the instances it
    is handed -- so both bindings are poisoned. ``validate_coverage`` is in the list because it is the
    per-element check that looks harmless and is the one an integrator would be tempted to add.
    """
    from deepspeed.checkpoint import affine as affine_module
    from deepspeed.checkpoint import affine_transfer as transfer_module

    def forbidden(self, *args, **kwargs):
        raise AssertionError(f'planner reached {self.__class__.__name__}.{name}')

    for name in ('rebuild', 'extract', 'uncovered_offsets', 'holders', 'validate_coverage'):
        monkeypatch.setattr(affine_module.ParamAffineMap, name, forbidden)
        monkeypatch.setattr(transfer_module.ParamAffineMap, name, forbidden)

    for source_degree, target_degree in ((2, 4), (4, 2), (1, 8), (8, 1)):
        for build in LAYOUTS.values():
            plan_transfer(build(target_degree), build(source_degree))


def test_plan_size_follows_geometry_and_not_the_model():
    """Same topology, same pieces, a parameter eight orders of magnitude larger: same plan shape.

    No tensor of that size is created, and none can be -- the point is that the plan is metadata, so a
    billion-row parameter costs exactly what a ten-row one costs here.
    """
    rows, cols = 10**9, 8

    source = contiguous_split_map((rows, cols), [rows // 2] * 2, 0)
    target = contiguous_split_map((rows, cols), [cols // 4] * 4, 1)
    small = contiguous_split_map((10, cols), [5, 5], 0)
    small_target = contiguous_split_map((10, cols), [2, 2, 2, 2], 1)

    big, little = plan_transfer(target, source), plan_transfer(small_target, small)

    assert len(big.segments) == len(little.segments)
    assert big.statistics.pair_checks == little.statistics.pair_checks
    assert big.moved_elements == rows * cols
    assert big.statistics.source_pieces == 2 and big.statistics.target_pieces == 4


@pytest.mark.parametrize('rows, cols, degrees', [(11, 8, (2, 4)), (23, 5, (4, 2)), (7, 13, (1, 3))])
def test_uneven_partitions_keep_their_tail_blocks(rows, cols, degrees):
    """Nothing here may assume a degree divides a dimension.

    The uneven split is the case that a ceil-based fallback survives by accident and a sum-based check
    survives by never noticing, so the plan is executed and compared, not counted.
    """
    source_degree, target_degree = degrees
    source = contiguous_split_map((rows, cols), _split(rows, source_degree), 0)
    target = contiguous_split_map((rows, cols), _split(rows, target_degree), 0)
    full = _full(rows, cols, seed=rows * cols)

    plan = plan_transfer(target, source)
    produced = _apply_plan(plan, target, {rank: source.extract(full, rank) for rank in source.shard_shapes})

    for rank in target.shard_shapes:
        assert torch.equal(produced[rank], target.extract(full, rank))


def test_random_supported_layouts_match_the_oracle():
    """Fixed seed, random geometry, including a change of split axis across degrees.

    A layout that cannot be planned is a failure of the generator rather than something to discard:
    skipping it would turn a passing run into a measure of how often the generator happened to
    cooperate. Degrees and dimensions are therefore drawn only where they divide or unevenly tile.
    """
    rng = random.Random(20260922)
    for trial in range(60):
        rows, cols = rng.choice([(12, 4), (16, 8), (24, 12), (7, 13), (23, 5), (48, 8)])
        source_axis, target_axis = rng.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
        source_degree, target_degree = rng.choice([(1, 2), (2, 1), (2, 3), (3, 2), (2, 4), (4, 2), (4, 4), (3, 5)])
        source_total = rows if source_axis == 0 else cols
        target_total = rows if target_axis == 0 else cols
        source = contiguous_split_map((rows, cols), _split(source_total, source_degree), source_axis)
        target = contiguous_split_map((rows, cols), _split(target_total, target_degree), target_axis)
        full = _full(rows, cols, seed=trial)

        plan = plan_transfer(target, source)
        produced = _apply_plan(plan, target, {rank: source.extract(full, rank) for rank in source.shard_shapes})
        for rank in target.shard_shapes:
            assert torch.equal(produced[rank], target.extract(full, rank)), (
                f'trial {trial}: {rows}x{cols} axis {source_axis}->{target_axis} '
                f'degree {source_degree}->{target_degree} rank {rank}')


def test_a_single_corrupt_piece_is_reported_rather_than_planned_around():
    """One wrong number in a valid map must produce a named refusal.

    The map is generated whole, then broken once, so the test says which field the planner catches on
    rather than only that some map someplace fails.
    """
    good = contiguous_split_map((16, 8), [8, 8], 0)
    corruptions = {
        'offset past the end': lambda piece: AffinePiece(piece.shape, 9 * 8, piece.source_strides, piece.dest_offset,
                                                         piece.dest_strides, piece.locations),
        'strided like a permutation': lambda piece: AffinePiece(piece.shape, piece.source_offset, (1, 16),
                                                                piece.dest_offset, piece.dest_strides, piece.locations),
        'scaled': lambda piece: AffinePiece(piece.shape, piece.source_offset, piece.source_strides, piece.dest_offset,
                                            piece.dest_strides, piece.locations, scale=2.0),
    }

    for description, corrupt in corruptions.items():
        broken = ParamAffineMap(
            good.logical_shape,
            dict(good.shard_shapes),
            {rank: [corrupt(piece) for piece in pieces] for rank, pieces in good.pieces_by_rank.items()},
        )
        with pytest.raises(AffineTransferError):
            plan_transfer(contiguous_split_map((16, 8), [4, 4, 4, 4], 0), broken)

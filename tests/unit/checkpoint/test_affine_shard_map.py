# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Check that AutoTP's fused and shared-QK layouts are describable as affine views.

These are the layouts `AUTOTP_UNSUPPORTED_PARAMETER_PATTERNS` currently refuses to
convert, on the grounds that the parameter "cannot be reassembled from the shards". The
tests below show the shards do cover the full tensor, and that affine pieces reproduce
them exactly, so what is missing is a way to describe the layout rather than the data.

Pieces are recovered by running the real partition function on a marker tensor and then
checked against *independent random data*. That second step is what makes this a test of
geometry: if pieces derived from markers reproduce a random tensor's shard bit-exactly,
the layout is a pure view and not something that depends on the values.

The partition functions used here take an explicit rank, so nothing in this file needs a
process group or an accelerator.
"""

import pytest
import torch

from deepspeed.checkpoint.affine import (AffinePiece, ParamAffineMap, replicated_map, contiguous_split_map,
                                         sub_param_map)
from deepspeed.module_inject.fusedqkv_utils import prepare_tp_fused_qkvw, shard_value_with_share_qk
from deepspeed.module_inject.tp_shard import AutoTPMeta


class _NamedModule(torch.nn.Module):
    """`get_fused_qkv_type` picks a layout by matching the module's class name as text."""

    def __init__(self, name):
        super().__init__()
        self._name = name

    def __str__(self):
        return self._name


def _row_markers(rows, cols):
    return torch.arange(rows, dtype=torch.float64).reshape(rows, 1).repeat(1, cols)


def _col_markers(rows, cols):
    return torch.arange(cols, dtype=torch.float64).reshape(1, cols).repeat(rows, 1)


def _index_runs(indices):
    """Group consecutive indices into (start_index, position_in_shard, length) runs.

    Compressing runs is what bounds the number of pieces: a contiguous block of the full
    tensor needs one piece however long it is.
    """
    runs = []
    start = 0
    for j in range(1, len(indices) + 1):
        if j == len(indices) or indices[j] != indices[j - 1] + 1:
            runs.append((indices[start], start, j - start))
            start = j
    return runs


def _source_indices(markers, shard, axis):
    """Recover which row (or column) of the full tensor each row of the shard came from."""
    if axis == 'row':
        lookup = {tuple(markers[i].tolist()): i for i in range(markers.shape[0])}
        keys = [tuple(shard[j].tolist()) for j in range(shard.shape[0])]
    else:
        lookup = {tuple(markers[:, i].tolist()): i for i in range(markers.shape[1])}
        keys = [tuple(shard[:, j].tolist()) for j in range(shard.shape[1])]

    indices = []
    for key in keys:
        assert key in lookup, 'a shard slice is not a slice of the full tensor, so the layout is not a view'
        indices.append(lookup[key])
    return indices


def _split_by_holders(indices, holders):
    """Break runs wherever the set of ranks holding the data changes.

    Merging across such a boundary would produce a piece whose own `locations` is wrong for
    part of it. GPTBigCode's last rank is the case that forces this: its q slice ends exactly
    where the replicated kv block begins, so unconstrained merging fuses a rank-private block
    onto a fully replicated one.
    """
    split = []
    for source_index, dest_index, length in _index_runs(indices):
        start = 0
        for step in range(1, length + 1):
            at_end = step == length
            if at_end or holders[source_index + step] != holders[source_index + start]:
                split.append((source_index + start, dest_index + start, step - start))
                start = step
    return split


def _pieces_for_rank(markers, shard, rank, axis, cols, holders):
    """Turn one rank's shard into affine pieces of the full tensor."""
    pieces = []
    indices = _source_indices(markers, shard, axis)
    for source_index, dest_index, length in _split_by_holders(indices, holders):
        if axis == 'row':
            shape = (length, cols)
            source_offset = source_index * cols
            dest_offset = dest_index * cols
            dest_strides = (cols, 1)
        else:
            shard_cols = shard.shape[1]
            shape = (shard.shape[0], length)
            source_offset = source_index
            dest_offset = dest_index
            dest_strides = (shard_cols, 1)
        pieces.append(
            AffinePiece(shape=shape,
                        source_offset=source_offset,
                        source_strides=(cols, 1),
                        dest_offset=dest_offset,
                        dest_strides=dest_strides,
                        locations=sorted(holders[source_index])))
    return pieces


def _build_map(shard_fn, rows, cols, mp_size, axis):
    """Derive the affine map of a layout by probing the real partition function."""
    markers = _row_markers(rows, cols) if axis == 'row' else _col_markers(rows, cols)
    shards = {rank: shard_fn(markers.clone(), rank) for rank in range(mp_size)}

    # Which ranks hold each slice of the full tensor. Needed before pieces can be cut,
    # because a piece may not span slices with different holders.
    holders = {}
    for rank, shard in shards.items():
        for index in _source_indices(markers, shard, axis):
            holders.setdefault(index, set()).add(rank)

    pieces_by_rank = {}
    shard_shapes = {}
    for rank, shard in shards.items():
        shard_shapes[rank] = tuple(shard.shape)
        pieces_by_rank[rank] = _pieces_for_rank(markers, shard, rank, axis, cols, holders)
    return ParamAffineMap(logical_shape=(rows, cols), shard_shapes=shard_shapes, pieces_by_rank=pieces_by_rank)


def _fused_qkv_shard_fn(layout_name, mp_size, meta):
    module = _NamedModule(layout_name)

    def shard_fn(full_param, rank):
        return prepare_tp_fused_qkvw(module, full_param, mp_size, rank, meta)

    return shard_fn


def _shared_qk_shard_fn(shard_value, meta):

    def shard_fn(full_param, rank):
        return shard_value_with_share_qk(full_param, None, rank, 2, shard_value, meta)[0].data

    return shard_fn


def _meta(num_kv_heads, n_embd=None, num_attention_heads=None):
    """The per-model shard metadata AutoTP derives from a model config."""
    return AutoTPMeta(num_kv_heads=num_kv_heads, n_embd=n_embd, num_attention_heads=num_attention_heads)


# (id, rows, cols, mp_size, axis, make_meta, build_shard_fn)
LAYOUTS = [
    ('bigcode', 48, 8, 4, 'row', lambda: _meta(4, n_embd=32, num_attention_heads=4),
     lambda mp, meta: _fused_qkv_shard_fn('GPTBigCodeBlock', mp, meta)),
    ('codegen', 24, 8, 2, 'row', lambda: _meta(8, n_embd=8, num_attention_heads=8),
     lambda mp, meta: _fused_qkv_shard_fn('CodeGenBlock', mp, meta)),
    ('yuan_value', 32, 8, 2, 'row', lambda: _meta(8), lambda mp, meta: _shared_qk_shard_fn(True, meta)),
    ('yuan_oproj', 8, 32, 2, 'col', lambda: _meta(8), lambda mp, meta: _shared_qk_shard_fn(False, meta)),
]


@pytest.mark.parametrize('name, rows, cols, mp_size, axis, make_meta, build_shard_fn',
                         LAYOUTS,
                         ids=[layout[0] for layout in LAYOUTS])
class TestAffineShardMap:

    def test_shards_cover_the_full_parameter(self, name, rows, cols, mp_size, axis, make_meta, build_shard_fn):
        """Every element of the full tensor is held by some rank, so it can be rebuilt."""
        affine_map = _build_map(build_shard_fn(mp_size, make_meta()), rows, cols, mp_size, axis)
        assert affine_map.uncovered_offsets() == []

    def test_every_piece_is_homogeneous(self, name, rows, cols, mp_size, axis, make_meta, build_shard_fn):
        """No piece spans elements held by different sets of ranks, so locations is exact."""
        affine_map = _build_map(build_shard_fn(mp_size, make_meta()), rows, cols, mp_size, axis)
        affine_map.validate_coverage()

    def test_pieces_reproduce_shards_of_unseen_data(self, name, rows, cols, mp_size, axis, make_meta, build_shard_fn):
        """Pieces derived from markers must reproduce shards of data they were not derived from."""
        shard_fn = build_shard_fn(mp_size, make_meta())
        affine_map = _build_map(shard_fn, rows, cols, mp_size, axis)

        torch.manual_seed(0)
        full_param = torch.randn(rows, cols, dtype=torch.float64)
        for rank in range(mp_size):
            expected = shard_fn(full_param.clone(), rank)
            assert torch.equal(affine_map.extract(full_param, rank), expected)

    def test_rebuild_inverts_extract(self, name, rows, cols, mp_size, axis, make_meta, build_shard_fn):
        """Rebuilding from the shards returns the parameter the shards were cut from."""
        shard_fn = build_shard_fn(mp_size, make_meta())
        affine_map = _build_map(shard_fn, rows, cols, mp_size, axis)

        torch.manual_seed(1)
        full_param = torch.randn(rows, cols, dtype=torch.float64)
        shards = {rank: shard_fn(full_param.clone(), rank) for rank in range(mp_size)}
        assert torch.equal(affine_map.rebuild(shards), full_param)


def test_piece_count_does_not_grow_with_model_size():
    """Piece count follows the block structure of the layout, not the size of the tensor.

    This is what keeps the description small: if piece count tracked the tensor it would
    be no cheaper than storing an index per element.
    """
    piece_counts = set()
    for hidden in (8, 16, 32, 64, 128):
        meta = _meta(8, n_embd=hidden, num_attention_heads=8)
        rows = 3 * hidden
        affine_map = _build_map(_fused_qkv_shard_fn('CodeGenBlock', 2, meta), rows, 8, 2, 'row')
        piece_counts.add(len(affine_map.pieces_by_rank[0]))

    assert len(piece_counts) == 1, f'piece count varied with model size: {sorted(piece_counts)}'


def test_replicated_piece_is_shared_by_every_rank():
    """GPTBigCode gives every rank the same kv block, which one owner per parameter cannot say.

    The kv rows appear in every rank's shard, so describing this layout needs replication
    to be expressible for part of a parameter rather than all of it.
    """
    meta = _meta(4, n_embd=32, num_attention_heads=4)
    mp_size = 4
    affine_map = _build_map(_fused_qkv_shard_fn('GPTBigCodeBlock', mp_size, meta), 48, 8, mp_size, 'row')

    kv_offsets = set(range(32 * 8, 48 * 8))
    for rank in range(mp_size):
        held = set()
        for piece in affine_map.pieces_by_rank[rank]:
            held.update(piece.source_offsets())
        assert kv_offsets <= held, f'rank {rank} does not hold the whole kv block'


def test_pieces_never_span_a_replication_boundary():
    """GPTBigCode's last rank is where an unconstrained merge would cross one.

    Its q slice ends exactly where the replicated kv block begins, so merging by adjacency
    alone yields one piece that is rank-private in its first half and replicated in its
    second. Splitting at the boundary costs one extra piece and keeps locations honest.
    """
    meta = _meta(4, n_embd=32, num_attention_heads=4)
    mp_size = 4
    affine_map = _build_map(_fused_qkv_shard_fn('GPTBigCodeBlock', mp_size, meta), 48, 8, mp_size, 'row')

    last_rank_pieces = affine_map.pieces_by_rank[mp_size - 1]
    assert len(last_rank_pieces) == 2
    assert {len(piece.locations) for piece in last_rank_pieces} == {1, mp_size}
    affine_map.validate_coverage()


def test_scale_round_trips_through_both_directions():
    """A row-parallel bias is replicated pre-divided by the world size, so a piece scales it.

    `shard == full * scale`, so extracting multiplies and rebuilding divides. Recording the
    factor keeps the layout describable without a rule that names which parameters are biases.
    """
    world_size = 4
    full_bias = torch.randn(16, dtype=torch.float64)
    pieces_by_rank = {
        rank: [
            AffinePiece(shape=(16, ),
                        source_offset=0,
                        source_strides=(1, ),
                        dest_offset=0,
                        dest_strides=(1, ),
                        locations=range(world_size),
                        scale=1.0 / world_size)
        ]
        for rank in range(world_size)
    }
    affine_map = ParamAffineMap(logical_shape=(16, ),
                                shard_shapes={rank: (16, )
                                              for rank in range(world_size)},
                                pieces_by_rank=pieces_by_rank)
    affine_map.validate_coverage()

    shards = {rank: affine_map.extract(full_bias, rank) for rank in range(world_size)}
    assert torch.equal(shards[0], full_bias / world_size)
    assert torch.equal(affine_map.rebuild(shards), full_bias)


def test_scale_round_trip_is_exact_for_power_of_two_world_sizes():
    """Dividing by a power of two only shifts the exponent, so no bias precision is lost."""
    full_bias = torch.randn(1024, dtype=torch.float32)
    for world_size in (2, 4, 8, 16):
        assert torch.equal(full_bias / world_size * world_size, full_bias)


# The universal-checkpoint converter merges tp slices with a chain of category-specific
# `torch.cat` arithmetic. The tests below rebuild the same parameter through the affine map
# and require the two to agree exactly, so the map can replace a branch without changing
# what any checkpoint converts to.


def _random_slices(shapes):
    torch.manual_seed(0)
    return [torch.randn(shape, dtype=torch.float64) for shape in shapes]


def test_parity_replicated():
    """`merge_tp_slices` takes slices[0] after asserting every rank matches."""
    tp_degree = 4
    full = torch.randn(6, 8, dtype=torch.float64)
    slices = [full.clone() for _ in range(tp_degree)]

    expected = slices[0]
    affine_map = replicated_map((6, 8), tp_degree)
    assert torch.equal(affine_map.rebuild(dict(enumerate(slices))), expected)


@pytest.mark.parametrize('cat_dim', [0, 1])
def test_parity_contiguous_split(cat_dim):
    """The default branch: `cat(slices, dim=1)` for row parallelism, `dim=0` otherwise."""
    per_rank = [3, 3, 2, 2]  # deliberately uneven; `chunk` would disagree here
    shapes = [(size, 8) if cat_dim == 0 else (8, size) for size in per_rank]
    slices = _random_slices(shapes)

    expected = torch.cat(slices, dim=cat_dim)
    affine_map = contiguous_split_map(tuple(expected.shape), per_rank, cat_dim)
    assert torch.equal(affine_map.rebuild(dict(enumerate(slices))), expected)


def test_parity_two_sub_params_cat_dim_0():
    """The 2-sub-param branch chunks each slice, merges each half, then concatenates."""
    tp_degree, half = 4, 3
    slices = _random_slices([(2 * half, 8)] * tp_degree)

    chunked = [torch.chunk(tp_slice, 2, dim=0) for tp_slice in slices]
    expected = torch.cat([
        torch.cat([chunk[0] for chunk in chunked], dim=0),
        torch.cat([chunk[1] for chunk in chunked], dim=0),
    ],
                         dim=0)

    total = half * tp_degree
    affine_map = sub_param_map(shape=(2 * total, 8),
                               sub_dim_sizes=(total, total),
                               shard_widths=[[half] * tp_degree, [half] * tp_degree],
                               partition_dim=0)
    assert torch.equal(affine_map.rebuild(dict(enumerate(slices))), expected)


def test_parity_sub_params_with_uneven_widths():
    """The sub-parameter branch, with the per-rank widths #8185 added for uneven splits.

    Q, K and V are different sizes and none divides evenly by the tp degree, which is the
    case the pre-0.4 metadata could not describe at all.
    """
    tp_degree = 3
    shard_widths = [[3, 2, 2], [2, 2, 1], [1, 1, 1]]
    sub_dim_sizes = [sum(widths) for widths in shard_widths]
    rows_per_rank = [sum(widths[rank] for widths in shard_widths) for rank in range(tp_degree)]
    slices = _random_slices([(rows, 8) for rows in rows_per_rank])

    # Exactly the arithmetic in `merge_tp_slices`: for each sub-parameter, take every
    # rank's block of it in turn, then concatenate the sub-parameters.
    offsets = [0] * tp_degree
    merged_chunks = []
    for widths in shard_widths:
        blocks = []
        for rank, tp_slice in enumerate(slices):
            blocks.append(tp_slice.narrow(0, offsets[rank], widths[rank]))
            offsets[rank] += widths[rank]
        merged_chunks.append(torch.cat(blocks, dim=0))
    expected = torch.cat(merged_chunks, dim=0)

    affine_map = sub_param_map(shape=(sum(sub_dim_sizes), 8),
                               sub_dim_sizes=sub_dim_sizes,
                               shard_widths=shard_widths,
                               partition_dim=0)
    affine_map.validate_coverage()
    assert torch.equal(affine_map.rebuild(dict(enumerate(slices))), expected)


def test_parity_round_trips_back_to_the_original_slices():
    """Extracting from the merged parameter returns the slices it was built from."""
    per_rank = [3, 3, 2, 2]
    slices = _random_slices([(size, 8) for size in per_rank])
    affine_map = contiguous_split_map((10, 8), per_rank, 0)

    full = affine_map.rebuild(dict(enumerate(slices)))
    for rank, tp_slice in enumerate(slices):
        assert torch.equal(affine_map.extract(full, rank), tp_slice)


def test_serialisation_round_trips():
    """The stored form must rebuild the same parameter as the map it came from."""
    affine_map = sub_param_map(shape=(18, 8),
                               sub_dim_sizes=[9, 9],
                               shard_widths=[[4, 3, 2], [3, 3, 3]],
                               partition_dim=0)
    restored = ParamAffineMap.from_dict(affine_map.to_dict())

    torch.manual_seed(0)
    slices = {rank: torch.randn(shape, dtype=torch.float64) for rank, shape in affine_map.shard_shapes.items()}
    assert torch.equal(restored.rebuild(slices), affine_map.rebuild(slices))


def test_serialisation_holds_only_plain_scalars():
    """Nothing torch-specific may reach the file, so the map stays readable on its own."""
    stored = contiguous_split_map((10, 8), [3, 3, 2, 2], 0).to_dict()

    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert isinstance(key, (str, int)), f'unexpected key type {type(key)}'
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        else:
            assert isinstance(value, (int, float, str)), f'unexpected value type {type(value)}'

    check(stored)


def test_serialisation_preserves_scale():
    """A scaled piece must survive the file, or a restored bias would be off by the divisor."""
    piece = AffinePiece(shape=(4, ),
                        source_offset=0,
                        source_strides=(1, ),
                        dest_offset=0,
                        dest_strides=(1, ),
                        locations=[0, 1],
                        scale=0.25)
    affine_map = ParamAffineMap(logical_shape=(4, ),
                                shard_shapes={
                                    0: (4, ),
                                    1: (4, )
                                },
                                pieces_by_rank={
                                    0: [piece],
                                    1: [piece]
                                })
    restored = ParamAffineMap.from_dict(affine_map.to_dict())
    assert restored.pieces_by_rank[0][0] == piece

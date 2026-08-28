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

from deepspeed.checkpoint.affine import AffinePiece, ParamAffineMap
from deepspeed.module_inject.fusedqkv_utils import prepare_tp_fused_qkvw, shard_value_with_share_qk
from deepspeed.module_inject.tp_shard import set_num_kv_heads, set_n_embd, set_num_attention_heads


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


def _pieces_for_rank(markers, shard, rank, axis, cols):
    """Turn one rank's shard into affine pieces of the full tensor."""
    pieces = []
    for source_index, dest_index, length in _index_runs(_source_indices(markers, shard, axis)):
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
                        locations=[rank]))
    return pieces


def _build_map(shard_fn, rows, cols, mp_size, axis):
    """Derive the affine map of a layout by probing the real partition function."""
    markers = _row_markers(rows, cols) if axis == 'row' else _col_markers(rows, cols)
    pieces_by_rank = {}
    shard_shapes = {}
    for rank in range(mp_size):
        shard = shard_fn(markers.clone(), rank)
        shard_shapes[rank] = tuple(shard.shape)
        pieces_by_rank[rank] = _pieces_for_rank(markers, shard, rank, axis, cols)
    return ParamAffineMap(logical_shape=(rows, cols), shard_shapes=shard_shapes, pieces_by_rank=pieces_by_rank)


def _fused_qkv_shard_fn(layout_name, mp_size):
    module = _NamedModule(layout_name)

    def shard_fn(full_param, rank):
        return prepare_tp_fused_qkvw(module, full_param, mp_size, rank)

    return shard_fn


def _shared_qk_shard_fn(shard_value):

    def shard_fn(full_param, rank):
        return shard_value_with_share_qk(full_param, None, rank, 2, shard_value)[0].data

    return shard_fn


def _configure_heads(num_kv_heads, n_embd=None, num_attention_heads=None):
    set_num_kv_heads(num_kv_heads)
    if n_embd is not None:
        set_n_embd(n_embd)
    if num_attention_heads is not None:
        set_num_attention_heads(num_attention_heads)


# (id, rows, cols, mp_size, axis, configure, build_shard_fn)
LAYOUTS = [
    ('bigcode', 48, 8, 4, 'row', lambda: _configure_heads(4, n_embd=32, num_attention_heads=4),
     lambda mp: _fused_qkv_shard_fn('GPTBigCodeBlock', mp)),
    ('codegen', 24, 8, 2, 'row', lambda: _configure_heads(8, n_embd=8, num_attention_heads=8),
     lambda mp: _fused_qkv_shard_fn('CodeGenBlock', mp)),
    ('yuan_value', 32, 8, 2, 'row', lambda: _configure_heads(8), lambda mp: _shared_qk_shard_fn(True)),
    ('yuan_oproj', 8, 32, 2, 'col', lambda: _configure_heads(8), lambda mp: _shared_qk_shard_fn(False)),
]


@pytest.mark.parametrize('name, rows, cols, mp_size, axis, configure, build_shard_fn',
                         LAYOUTS,
                         ids=[layout[0] for layout in LAYOUTS])
class TestAffineShardMap:

    def test_shards_cover_the_full_parameter(self, name, rows, cols, mp_size, axis, configure, build_shard_fn):
        """Every element of the full tensor is held by some rank, so it can be rebuilt."""
        configure()
        affine_map = _build_map(build_shard_fn(mp_size), rows, cols, mp_size, axis)
        assert affine_map.uncovered_offsets() == []

    def test_pieces_reproduce_shards_of_unseen_data(self, name, rows, cols, mp_size, axis, configure, build_shard_fn):
        """Pieces derived from markers must reproduce shards of data they were not derived from."""
        configure()
        shard_fn = build_shard_fn(mp_size)
        affine_map = _build_map(shard_fn, rows, cols, mp_size, axis)

        torch.manual_seed(0)
        full_param = torch.randn(rows, cols, dtype=torch.float64)
        for rank in range(mp_size):
            expected = shard_fn(full_param.clone(), rank)
            assert torch.equal(affine_map.extract(full_param, rank), expected)

    def test_rebuild_inverts_extract(self, name, rows, cols, mp_size, axis, configure, build_shard_fn):
        """Rebuilding from the shards returns the parameter the shards were cut from."""
        configure()
        shard_fn = build_shard_fn(mp_size)
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
        _configure_heads(8, n_embd=hidden, num_attention_heads=8)
        rows = 3 * hidden
        affine_map = _build_map(_fused_qkv_shard_fn('CodeGenBlock', 2), rows, 8, 2, 'row')
        piece_counts.add(len(affine_map.pieces_by_rank[0]))

    assert len(piece_counts) == 1, f'piece count varied with model size: {sorted(piece_counts)}'


def test_replicated_piece_is_shared_by_every_rank():
    """GPTBigCode gives every rank the same kv block, which one owner per parameter cannot say.

    The kv rows appear in every rank's shard, so describing this layout needs replication
    to be expressible for part of a parameter rather than all of it.
    """
    _configure_heads(4, n_embd=32, num_attention_heads=4)
    mp_size = 4
    affine_map = _build_map(_fused_qkv_shard_fn('GPTBigCodeBlock', mp_size), 48, 8, mp_size, 'row')

    kv_offsets = set(range(32 * 8, 48 * 8))
    for rank in range(mp_size):
        held = set()
        for piece in affine_map.pieces_by_rank[rank]:
            held.update(piece.source_offsets())
        assert kv_offsets <= held, f'rank {rank} does not hold the whole kv block'

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Affine description of how a parameter is sharded.

Universal checkpoint currently decides how to merge a parameter by matching its *name*
against regex categories (vocabulary, row-parallel, fused sub-parameters, ...), so a
layout that no category describes cannot be converted at all. This module describes a
shard geometrically instead: each rank holds a list of affine views of the full tensor,
and the views alone say how to rebuild it.

A piece may carry an invertible elementwise map, but never a reduction. Scaling a block by
a constant is reversible; averaging several elements into one is not, and would leave the
full tensor unrecoverable from the shards. That line is what keeps conversion invertible in
both directions.

See ``deepspeed/checkpoint/affine_ir_spec.md`` for the full specification.
"""

import torch

__all__ = [
    'AffinePiece', 'ParamAffineMap', 'AFFINE_MAP_FORMAT_VERSION', 'row_major_strides', 'replicated_map',
    'contiguous_split_map', 'sub_param_map'
]

# Encoding version of the stored map, independent of the universal checkpoint version so
# the two can move separately. A reader refuses a version it predates rather than
# misreading fields it does not know about.
AFFINE_MAP_FORMAT_VERSION = 1


class AffinePiece:
    """A block of elements, described where it lives in the full tensor and in the shard.

    A piece holds the same elements in the same arrangement on both sides, so ``shape`` is
    shared and only the offset and strides differ. Describing the shard side explicitly is
    what lets a piece land somewhere other than the end of the shard: a shard split along
    a column interleaves its pieces row by row, so "append each piece in turn" is not
    enough to say where the elements go.

    The offsets and strides are the arguments of ``torch.as_strided``, so a piece can be
    applied to a tensor without any interpretation step.

    ``locations`` is a set of ranks rather than a single owner because a piece may be
    replicated. Naming one owner here would be a scheduling decision, and the cheapest
    source depends on the topology, which this description does not know about.

    ``scale`` is the factor the shard holds the block by: ``shard == full * scale``. Row
    parallel layers pre-divide a replicated bias by the world size so that summing the
    all-reduced outputs adds the bias exactly once, and the divisor changes with the world
    size. Recording it as a number keeps that describable without a rule naming which
    parameters are biases.

    A piece must be *homogeneous*: every element it covers is held by the same set of ranks
    and carries the same scale. That is what makes ``locations`` exact rather than a hint,
    and it is why merging two adjacent blocks is only allowed when both agree.
    """

    __slots__ = ('shape', 'source_offset', 'source_strides', 'dest_offset', 'dest_strides', 'locations', 'scale')

    def __init__(self, shape, source_offset, source_strides, dest_offset, dest_strides, locations, scale=1.0):
        self.shape = tuple(int(dim) for dim in shape)
        self.source_offset = int(source_offset)
        self.source_strides = tuple(int(stride) for stride in source_strides)
        self.dest_offset = int(dest_offset)
        self.dest_strides = tuple(int(stride) for stride in dest_strides)
        self.locations = frozenset(int(rank) for rank in locations)
        self.scale = float(scale)
        assert self.scale != 0.0, 'A zero scale is not invertible, so the full tensor could not be rebuilt.'

    @property
    def numel(self):
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    def source_view(self, full_param):
        """This piece's elements as a view of the full parameter, without copying."""
        return torch.as_strided(full_param,
                                size=self.shape,
                                stride=self.source_strides,
                                storage_offset=self.source_offset)

    def dest_view(self, shard):
        """This piece's elements as a view of the shard that holds it."""
        return torch.as_strided(shard, size=self.shape, stride=self.dest_strides, storage_offset=self.dest_offset)

    def source_offsets(self):
        """Yield the offset into the full tensor of every element this piece covers.

        Used to check coverage. Pieces are small relative to the parameter, so walking
        them elementwise is affordable and avoids assuming anything about their shape.
        """
        return self._offsets(self.source_offset, self.source_strides)

    def _offsets(self, base, strides):
        if not self.shape:
            yield base
            return

        index = [0] * len(self.shape)
        while True:
            offset = base
            for axis, position in enumerate(index):
                offset += position * strides[axis]
            yield offset

            axis = len(self.shape) - 1
            while axis >= 0:
                index[axis] += 1
                if index[axis] < self.shape[axis]:
                    break
                index[axis] = 0
                axis -= 1
            if axis < 0:
                return

    def __repr__(self):
        return (f'AffinePiece(shape={self.shape}, source=({self.source_offset}, {self.source_strides}), '
                f'dest=({self.dest_offset}, {self.dest_strides}), locations={sorted(self.locations)}, '
                f'scale={self.scale})')

    def __eq__(self, other):
        if not isinstance(other, AffinePiece):
            return NotImplemented
        return (self.shape == other.shape and self.source_offset == other.source_offset
                and self.source_strides == other.source_strides and self.dest_offset == other.dest_offset
                and self.dest_strides == other.dest_strides and self.locations == other.locations
                and self.scale == other.scale)

    def __hash__(self):
        return hash((self.shape, self.source_offset, self.source_strides, self.dest_offset, self.dest_strides,
                     self.locations, self.scale))


class ParamAffineMap:
    """How one parameter is spread over a tensor-parallel group.

    ``shard_shapes`` gives each rank the shape of the tensor it holds, and
    ``pieces_by_rank`` says which blocks of the full parameter make it up.
    """

    def __init__(self, logical_shape, shard_shapes, pieces_by_rank):
        self.logical_shape = tuple(int(dim) for dim in logical_shape)
        self.shard_shapes = {int(rank): tuple(int(dim) for dim in shape) for rank, shape in shard_shapes.items()}
        self.pieces_by_rank = {int(rank): list(pieces) for rank, pieces in pieces_by_rank.items()}

    @property
    def numel(self):
        return _product(self.logical_shape)

    def uncovered_offsets(self):
        """Return the offsets of the full tensor that no rank holds.

        A non-empty result means the parameter cannot be rebuilt from its shards, so this
        is the check that has to pass before a map is used for conversion.
        """
        covered = set()
        for pieces in self.pieces_by_rank.values():
            for piece in pieces:
                covered.update(piece.source_offsets())
        return sorted(set(range(self.numel)) - covered)

    def holders(self):
        """Map each element offset of the full tensor to the set of ranks holding it."""
        holders = {}
        for rank, pieces in self.pieces_by_rank.items():
            for piece in pieces:
                for offset in piece.source_offsets():
                    holders.setdefault(offset, set()).add(rank)
        return holders

    def validate(self):
        """Cheap structural check: every rank's pieces account for exactly its shard."""
        for rank, pieces in self.pieces_by_rank.items():
            held = sum(piece.numel for piece in pieces)
            expected = _product(self.shard_shapes[rank])
            assert held == expected, (f'Rank {rank} holds a shard of {expected} elements but its pieces '
                                      f'account for {held}.')

    def validate_coverage(self):
        """Full check: the shards cover the parameter, and every piece is homogeneous.

        This walks the map element by element, so it costs O(numel) and is meant for tests
        and for validating a newly built map, not for every conversion of a large tensor.
        """
        self.validate()

        missing = self.uncovered_offsets()
        assert not missing, (f'Affine map for a parameter of shape {self.logical_shape} leaves '
                             f'{len(missing)} element(s) uncovered, starting at offset {missing[0]}, '
                             'so the parameter cannot be rebuilt from its shards.')

        # A piece whose elements are not all held by the same ranks would make its own
        # `locations` a lie, and a reader trusting that field would miss a replica.
        holders = self.holders()
        for rank, pieces in self.pieces_by_rank.items():
            for piece in pieces:
                for offset in piece.source_offsets():
                    assert holders[offset] == set(
                        piece.locations), (f'Piece {piece} on rank {rank} covers offset {offset}, which is held by '
                                           f'{sorted(holders[offset])}. A piece must not span elements with different '
                                           'holders, or its locations cannot be trusted.')

    def rebuild(self, shards):
        """Rebuild the full parameter from per-rank shards, using the pieces as the plan.

        ``shards`` maps a rank to its shard. Where pieces overlap the data is identical by
        construction, so writing them in any order gives the same result.
        """
        self.validate()
        any_shard = next(iter(shards.values()))
        full_param = torch.empty(self.numel, dtype=any_shard.dtype, device=any_shard.device)

        for rank, pieces in self.pieces_by_rank.items():
            flat_shard = _flat_buffer(shards[rank])
            for piece in pieces:
                target = piece.source_view(full_param)
                target.copy_(piece.dest_view(flat_shard))
                if piece.scale != 1.0:
                    target.div_(piece.scale)

        return full_param.view(self.logical_shape)

    def extract(self, full_param, rank):
        """Produce one rank's shard from the full parameter. The inverse of ``rebuild``."""
        flat_param = _flat_buffer(full_param)
        shard_shape = self.shard_shapes[rank]
        shard = torch.empty(_product(shard_shape), dtype=full_param.dtype, device=full_param.device)

        for piece in self.pieces_by_rank[rank]:
            target = piece.dest_view(shard)
            target.copy_(piece.source_view(flat_param))
            if piece.scale != 1.0:
                target.mul_(piece.scale)

        return shard.view(shard_shape)

    def to_dict(self):
        """Serialise to plain scalars, so the map can be read without importing torch."""
        return {
            'logical_shape': list(self.logical_shape),
            'ranks': {
                rank: {
                    'shard_shape': list(self.shard_shapes[rank]),
                    'pieces': [_piece_to_dict(piece) for piece in pieces],
                }
                for rank, pieces in sorted(self.pieces_by_rank.items())
            },
        }

    @classmethod
    def from_dict(cls, entry):
        ranks = entry['ranks']
        return cls(logical_shape=entry['logical_shape'],
                   shard_shapes={
                       int(rank): value['shard_shape']
                       for rank, value in ranks.items()
                   },
                   pieces_by_rank={
                       int(rank): [_piece_from_dict(piece) for piece in value['pieces']]
                       for rank, value in ranks.items()
                   })

    def __repr__(self):
        counts = {rank: len(pieces) for rank, pieces in sorted(self.pieces_by_rank.items())}
        return f'ParamAffineMap(logical_shape={self.logical_shape}, pieces_per_rank={counts})'


def _flat_buffer(tensor):
    """Flatten ``tensor`` into a buffer whose storage starts at its first element.

    Piece offsets are storage offsets, because that is what ``torch.as_strided`` takes.
    A tensor that is a view into a larger buffer starts partway into that storage, so
    applying a piece to it directly would read from the wrong place. Shards routinely
    arrive this way, since splitting a tensor produces views that share one buffer.
    """
    flat = tensor.reshape(-1)
    if flat.storage_offset() != 0:
        flat = flat.clone()
    return flat


def _product(shape):
    count = 1
    for dim in shape:
        count *= dim
    return count


def row_major_strides(shape):
    """Element stride per axis for a densely packed tensor of this shape."""
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)


def replicated_map(shape, tp_degree):
    """Every rank holds the whole parameter.

    One piece, named by every rank, which is what lets a converter read it from whichever
    rank is cheapest rather than from a designated owner.
    """
    shape = tuple(shape)
    strides = row_major_strides(shape)
    ranks = list(range(tp_degree))
    pieces = [
        AffinePiece(shape=shape,
                    source_offset=0,
                    source_strides=strides,
                    dest_offset=0,
                    dest_strides=strides,
                    locations=ranks)
    ]
    return ParamAffineMap(logical_shape=shape,
                          shard_shapes={rank: shape
                                        for rank in ranks},
                          pieces_by_rank={rank: list(pieces)
                                          for rank in ranks})


def contiguous_split_map(shape, per_rank_sizes, partition_dim, scale=1.0):
    """Each rank holds one contiguous block along ``partition_dim``.

    Covers row-parallel and column-parallel layers alike: they differ only in which axis
    the block is cut on, and therefore only in strides. ``per_rank_sizes`` may be uneven.
    """
    shape = tuple(shape)
    source_strides = row_major_strides(shape)
    pieces_by_rank = {}
    shard_shapes = {}
    start = 0
    for rank, size in enumerate(per_rank_sizes):
        shard_shape = list(shape)
        shard_shape[partition_dim] = size
        shard_shape = tuple(shard_shape)
        shard_shapes[rank] = shard_shape
        pieces_by_rank[rank] = [
            AffinePiece(shape=shard_shape,
                        source_offset=start * source_strides[partition_dim],
                        source_strides=source_strides,
                        dest_offset=0,
                        dest_strides=row_major_strides(shard_shape),
                        locations=[rank],
                        scale=scale)
        ]
        start += size
    return ParamAffineMap(logical_shape=shape, shard_shapes=shard_shapes, pieces_by_rank=pieces_by_rank)


def sub_param_map(shape, sub_dim_sizes, shard_widths, partition_dim):
    """A parameter that is several sub-parameters concatenated, each split across ranks.

    Fused QKV is the motivating case: the full parameter is Q then K then V along
    ``partition_dim``, and a rank's shard holds its slice of each in turn. ``shard_widths[i]``
    gives the per-rank widths of sub-parameter ``i``, so the sub-parameters may be split
    unevenly and need not be the same size as each other.
    """
    shape = tuple(shape)
    source_strides = row_major_strides(shape)
    tp_degree = len(shard_widths[0])

    shard_shapes = {}
    for rank in range(tp_degree):
        shard_shape = list(shape)
        shard_shape[partition_dim] = sum(widths[rank] for widths in shard_widths)
        shard_shapes[rank] = tuple(shard_shape)

    pieces_by_rank = {rank: [] for rank in range(tp_degree)}
    dest_starts = {rank: 0 for rank in range(tp_degree)}
    sub_param_start = 0
    for sub_index, sub_size in enumerate(sub_dim_sizes):
        widths = shard_widths[sub_index]
        source_start = sub_param_start
        for rank in range(tp_degree):
            piece_shape = list(shape)
            piece_shape[partition_dim] = widths[rank]
            dest_strides = row_major_strides(shard_shapes[rank])
            pieces_by_rank[rank].append(
                AffinePiece(shape=tuple(piece_shape),
                            source_offset=source_start * source_strides[partition_dim],
                            source_strides=source_strides,
                            dest_offset=dest_starts[rank] * dest_strides[partition_dim],
                            dest_strides=dest_strides,
                            locations=[rank]))
            source_start += widths[rank]
            dest_starts[rank] += widths[rank]
        sub_param_start += sub_size

    return ParamAffineMap(logical_shape=shape, shard_shapes=shard_shapes, pieces_by_rank=pieces_by_rank)


def _piece_to_dict(piece):
    entry = {
        'shape': list(piece.shape),
        'source': [piece.source_offset, list(piece.source_strides)],
        'dest': [piece.dest_offset, list(piece.dest_strides)],
        'locations': sorted(piece.locations),
    }
    # Most pieces are unscaled, so leaving the default out keeps the stored map smaller
    # and lets a reader that predates scaling still make sense of one that does not use it.
    if piece.scale != 1.0:
        entry['scale'] = piece.scale
    return entry


def _piece_from_dict(entry):
    source_offset, source_strides = entry['source']
    dest_offset, dest_strides = entry['dest']
    return AffinePiece(shape=entry['shape'],
                       source_offset=source_offset,
                       source_strides=source_strides,
                       dest_offset=dest_offset,
                       dest_strides=dest_strides,
                       locations=entry['locations'],
                       scale=entry.get('scale', 1.0))

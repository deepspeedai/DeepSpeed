# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Affine description of how a parameter is sharded.

Universal checkpoint currently decides how to merge a parameter by matching its *name*
against regex categories (vocabulary, row-parallel, fused sub-parameters, ...), so a
layout that no category describes cannot be converted at all. This module describes a
shard geometrically instead: each rank holds a list of affine views of the full tensor,
and the views alone say how to rebuild it.

The representation is deliberately view-only. A piece never combines several elements of
the full tensor, which is what keeps the mapping invertible: given the full tensor a rank's
shard can be produced, and given the shards the full tensor can be rebuilt.
"""

import torch

__all__ = ['AffinePiece', 'ParamAffineMap']


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
    """

    __slots__ = ('shape', 'source_offset', 'source_strides', 'dest_offset', 'dest_strides', 'locations')

    def __init__(self, shape, source_offset, source_strides, dest_offset, dest_strides, locations):
        self.shape = tuple(int(dim) for dim in shape)
        self.source_offset = int(source_offset)
        self.source_strides = tuple(int(stride) for stride in source_strides)
        self.dest_offset = int(dest_offset)
        self.dest_strides = tuple(int(stride) for stride in dest_strides)
        self.locations = frozenset(int(rank) for rank in locations)

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
                f'dest=({self.dest_offset}, {self.dest_strides}), locations={sorted(self.locations)})')

    def __eq__(self, other):
        if not isinstance(other, AffinePiece):
            return NotImplemented
        return (self.shape == other.shape and self.source_offset == other.source_offset
                and self.source_strides == other.source_strides and self.dest_offset == other.dest_offset
                and self.dest_strides == other.dest_strides and self.locations == other.locations)

    def __hash__(self):
        return hash(
            (self.shape, self.source_offset, self.source_strides, self.dest_offset, self.dest_strides, self.locations))


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

    def validate(self):
        missing = self.uncovered_offsets()
        assert not missing, (f'Affine map for a parameter of shape {self.logical_shape} leaves '
                             f'{len(missing)} element(s) uncovered, starting at offset {missing[0]}, '
                             'so the parameter cannot be rebuilt from its shards.')

        for rank, pieces in self.pieces_by_rank.items():
            held = sum(piece.numel for piece in pieces)
            expected = _product(self.shard_shapes[rank])
            assert held == expected, (f'Rank {rank} holds a shard of {expected} elements but its pieces '
                                      f'account for {held}.')

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
                piece.source_view(full_param).copy_(piece.dest_view(flat_shard))

        return full_param.view(self.logical_shape)

    def extract(self, full_param, rank):
        """Produce one rank's shard from the full parameter. The inverse of ``rebuild``."""
        flat_param = _flat_buffer(full_param)
        shard_shape = self.shard_shapes[rank]
        shard = torch.empty(_product(shard_shape), dtype=full_param.dtype, device=full_param.device)

        for piece in self.pieces_by_rank[rank]:
            piece.dest_view(shard).copy_(piece.source_view(flat_param))

        return shard.view(shard_shape)

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

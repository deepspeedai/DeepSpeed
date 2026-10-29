# Copyright (c) DeepSpeed Team.
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
    'AffinePiece', 'ParamAffineMap', 'AFFINE_MAP_FORMAT_VERSION', 'SCALE_POWER_BY_STATE', 'replicated_map',
    'contiguous_split_map', 'sub_param_map', 'segmented_map', 'block_gather_map', 'nest_affine_maps'
]

# Encoding version of the stored map, independent of the universal checkpoint version so
# the two can move separately. A reader refuses a version it predates rather than
# misreading fields it does not know about.
AFFINE_MAP_FORMAT_VERSION = 1

# How a piece's scale applies to each thing a checkpoint stores for a parameter. Scaling a
# parameter by s scales its gradient by 1/s, so Adam's first moment carries the inverse and
# its second moment the inverse square. Shared by conversion and restore so the two cannot
# disagree about which power a state needs.
SCALE_POWER_BY_STATE = {'fp32': 1, 'exp_avg': -1, 'exp_avg_sq': -2}


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
        if self.scale == 0.0:
            raise ValueError('A zero scale is not invertible, so the full tensor could not be rebuilt.')

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

    def dest_offsets(self):
        """Yield the offset into the shard of every element this piece covers."""
        return self._offsets(self.dest_offset, self.dest_strides)

    def _offsets(self, base, strides):
        if any(dim == 0 for dim in self.shape):
            # An empty piece covers no elements. Walking it anyway would report offsets it
            # does not hold, and a coverage check would accept a map that is missing data.
            return
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


def _dest_interval(piece):
    """Return a dense piece's destination interval, or None for a strided or empty piece."""
    if piece.numel == 0 or len(piece.dest_strides) != len(piece.shape):
        return None
    if any(size > 1 and got != want
           for size, got, want in zip(piece.shape, piece.dest_strides, _row_major_strides(piece.shape))):
        return None
    return piece.dest_offset, piece.dest_offset + piece.numel


def _dest_box(piece, shard_shape):
    """Return a piece's destination as per-axis ``(starts, stops)`` in the shard, or None.

    A column split writes each piece into a few columns of every shard row, so its
    destination is not one interval. It is still a box in the shard's own grid, because its
    strides are the shard's row-major strides, and a box can be checked for bounds and
    overlap from its corners without walking any elements.
    """
    if piece.numel == 0 or len(piece.shape) != len(shard_shape) or len(piece.dest_strides) != len(shard_shape):
        return None
    shard_strides = _row_major_strides(shard_shape)
    if any(size > 1 and got != want for size, got, want in zip(piece.shape, piece.dest_strides, shard_strides)):
        return None

    starts = []
    remainder = piece.dest_offset
    for stride in shard_strides:
        start, remainder = divmod(remainder, stride)
        starts.append(start)
    stops = [start + size for start, size in zip(starts, piece.shape)]
    return starts, stops


def _dest_address_range(piece):
    """Return the lowest and highest shard address a piece writes."""
    low = piece.dest_offset
    high = piece.dest_offset
    for size, stride in zip(piece.shape, piece.dest_strides):
        reach = (size - 1) * stride
        if reach < 0:
            low += reach
        else:
            high += reach
    return low, high


def _first_shared_address(box, other_box, shard_shape):
    """Return the first shard address two boxes both write, or None when they are disjoint."""
    corner = []
    for start, stop, other_start, other_stop in zip(box[0], box[1], other_box[0], other_box[1]):
        if max(start, other_start) >= min(stop, other_stop):
            return None
        corner.append(max(start, other_start))
    return sum(position * stride for position, stride in zip(corner, _row_major_strides(shard_shape)))


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
        """Check shard element counts, and destination bounds and overlap from piece metadata."""
        for rank, pieces in self.pieces_by_rank.items():
            held = sum(piece.numel for piece in pieces)
            expected = _product(self.shard_shapes[rank])
            if held != expected:
                raise ValueError(f'Rank {rank} holds a shard of {expected} elements but its pieces '
                                 f'account for {held}.')
            self._validate_destinations(rank, pieces, self.shard_shapes[rank])

    def _validate_destinations(self, rank, pieces, shard_shape):
        """Check that dense intervals and strided boxes stay within the shard and do not overlap."""
        shard_numel = _product(shard_shape)
        intervals = []
        has_strided_piece = False
        for piece in pieces:
            span = _dest_interval(piece)
            if span is None:
                if piece.numel:
                    has_strided_piece = True
                continue
            start, stop = span
            if start < 0 or stop > shard_numel:
                raise ValueError(f'Rank {rank} piece {piece} writes shard addresses [{start}, {stop}), '
                                 f'outside the shard of {shard_numel} elements.')
            intervals.append(span)

        intervals.sort()
        cursor = 0
        for start, stop in intervals:
            if start < cursor:
                raise ValueError(f'Rank {rank} writes shard address {start} twice, so one piece would '
                                 'silently overwrite another.')
            cursor = stop

        # A column split's pieces have no interval, so the check above skips them. Check them as
        # boxes against every other box-shaped piece of the rank instead. Ranks whose pieces are
        # all dense are fully covered above and skip this pairwise pass.
        if not has_strided_piece:
            return
        boxes = []
        for piece in pieces:
            if piece.numel and _dest_interval(piece) is None:
                low, high = _dest_address_range(piece)
                if low < 0 or high >= shard_numel:
                    raise ValueError(f'Rank {rank} piece {piece} writes shard addresses [{low}, {high}], '
                                     f'outside the shard of {shard_numel} elements.')
            box = _dest_box(piece, shard_shape)
            if box is None:
                continue
            starts, stops = box
            fits = all(start >= 0 and stop <= dim for start, stop, dim in zip(starts, stops, shard_shape))
            if not fits:
                # A view may wrap past the end of a shard row while staying inside the shard. It is
                # not a box of the shard's grid, so the walk in validate_coverage() checks it instead.
                continue
            boxes.append(box)
        for index, box in enumerate(boxes):
            for other_box in boxes[index + 1:]:
                shared = _first_shared_address(box, other_box, shard_shape)
                if shared is not None:
                    raise ValueError(f'Rank {rank} writes shard address {shared} twice, so one piece would '
                                     'silently overwrite another.')

    def validate_coverage(self):
        """Full check: each shard address is written once, the shards cover the parameter,
        and every piece is homogeneous.

        This walks the map element by element, so it costs O(numel) and is meant for tests
        and for validating a newly built map, not for every conversion of a large tensor.
        """
        self.validate()

        # validate() reads destinations from piece metadata, which cannot place every strided
        # layout. Walking each shard address makes this check complete: every address of every
        # shard is written by exactly one element.
        for rank, pieces in self.pieces_by_rank.items():
            shard_numel = _product(self.shard_shapes[rank])
            written = set()
            for piece in pieces:
                for offset in piece.dest_offsets():
                    if offset < 0 or offset >= shard_numel:
                        raise ValueError(f'Rank {rank} piece {piece} writes shard address {offset}, outside '
                                         f'the shard of {shard_numel} elements.')
                    if offset in written:
                        raise ValueError(f'Rank {rank} writes shard address {offset} twice, so one piece would '
                                         'silently overwrite another.')
                    written.add(offset)

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

    def rebuild(self, shards, scale_power=1):
        """Rebuild the full parameter from per-rank shards, using the pieces as the plan.

        ``shards`` maps a rank to its shard.

        ``scale_power`` says how a piece's ``scale`` applies to the tensor being moved:
        the shard holds ``full * scale ** scale_power``. A parameter uses 1. Optimizer
        moments do not, because scaling a parameter by ``s`` scales its gradient by
        ``1 / s``: Adam's first moment needs -1 and its second moment -2. The map records
        the geometry; the caller knows which tensor it is moving.
        """
        self.validate()
        self._reject_scaled_optimizer_state(scale_power)
        for rank, shard in shards.items():
            expected = _product(self.shard_shapes[rank])
            if shard.numel() != expected:
                raise ValueError(f'Rank {rank} supplied a shard of {shard.numel()} elements, but the map '
                                 f'describes {expected}. Applying the pieces would silently read a prefix '
                                 'of it and drop the rest.')
        any_shard = next(iter(shards.values()))
        full_param = torch.empty(self.numel, dtype=any_shard.dtype, device=any_shard.device)

        written = {}
        for rank, pieces in self.pieces_by_rank.items():
            if not pieces:
                continue
            flat_shard = _flat_buffer(shards[rank])
            for piece in pieces:
                target = piece.source_view(full_param)
                source = piece.dest_view(flat_shard)
                if len(piece.locations) > 1 and piece.source_offset in written:
                    # Every rank holding a replicated piece must agree. Letting the last
                    # writer win would hide rank drift or a corrupt shard, which the
                    # category-based converter guards against explicitly.
                    if not torch.equal(target, _scaled(source, piece.scale, scale_power)):
                        raise ValueError(f'Ranks {sorted(piece.locations)} hold different data for the same '
                                         f'block of a replicated parameter, starting at offset '
                                         f'{piece.source_offset}.')
                    continue
                target.copy_(_scaled(source, piece.scale, scale_power))
                written[piece.source_offset] = True

        return full_param.view(self.logical_shape)

    def _reject_scaled_optimizer_state(self, scale_power):
        """Refuse to move an optimizer state through a scaled piece.

        The moments themselves convert correctly under their powers, but Adam's update also
        depends on ``lr`` and ``eps``, and those live in the coordinate the optimizer was
        training in. Restoring a rescaled parameter without rescaling them
        (``lr / scale``, ``eps * scale``) resumes on a different trajectory, with an error
        that grows step by step and no failure to show for it. Describing that transform is
        outside this module, which knows about parameter geometry and not about optimizers,
        so refuse until the checkpoint contract covers it.
        """
        if scale_power == 1:
            return
        scaled = [piece for pieces in self.pieces_by_rank.values() for piece in pieces if piece.scale != 1.0]
        if scaled:
            raise NotImplementedError(
                'Converting an optimizer state through a scaled piece is not supported: the moments transform '
                'correctly, but the optimizer group hyperparameters would still be in the source coordinate, '
                'so a resumed run would diverge.')

    def extract(self, full_param, rank, scale_power=1):
        """Produce one rank's shard from the full parameter. The inverse of ``rebuild``."""
        self._reject_scaled_optimizer_state(scale_power)
        flat_param = _flat_buffer(full_param)
        shard_shape = self.shard_shapes[rank]
        shard = torch.empty(_product(shard_shape), dtype=full_param.dtype, device=full_param.device)

        for piece in self.pieces_by_rank[rank]:
            target = piece.dest_view(shard)
            target.copy_(piece.source_view(flat_param))
            factor = piece.scale**scale_power
            if factor != 1.0:
                target.mul_(factor)

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


def _scaled(tensor, scale, scale_power):
    """Undo the shard's scaling, giving the block as it appears in the full tensor."""
    factor = scale**scale_power
    if factor == 1.0:
        return tensor
    return tensor / factor


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


def _row_major_strides(shape):
    """Element stride per axis for a densely packed tensor of this shape."""
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)


def replicated_map(shape, tp_degree, scale=1.0):
    """Every rank holds the whole parameter.

    One piece, named by every rank, which is what lets a converter read it from whichever
    rank is cheapest rather than from a designated owner.

    ``scale`` belongs here rather than on a split, because the layouts that pre-divide a
    value hold it whole on every rank: a row-parallel layer replicates its bias divided by
    the world size so that summing the all-reduced outputs adds it exactly once. The weight
    beside it is split and unscaled.
    """
    shape = tuple(shape)
    strides = _row_major_strides(shape)
    ranks = list(range(tp_degree))
    pieces = [
        AffinePiece(shape=shape,
                    source_offset=0,
                    source_strides=strides,
                    dest_offset=0,
                    dest_strides=strides,
                    locations=ranks,
                    scale=scale)
    ]
    return ParamAffineMap(logical_shape=shape,
                          shard_shapes={rank: shape
                                        for rank in ranks},
                          pieces_by_rank={rank: list(pieces)
                                          for rank in ranks})


def contiguous_split_map(shape, per_rank_sizes, partition_dim):
    """Each rank holds one contiguous block along ``partition_dim``.

    Covers row-parallel and column-parallel layers alike: they differ only in which axis
    the block is cut on, and therefore only in strides. ``per_rank_sizes`` may be uneven.
    """
    shape = tuple(shape)
    source_strides = _row_major_strides(shape)
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
                        dest_strides=_row_major_strides(shard_shape),
                        locations=[rank])
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
    source_strides = _row_major_strides(shape)
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
            dest_strides = _row_major_strides(shard_shapes[rank])
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


def segmented_map(shape, segments, partition_dim, tp_degree, split_widths):
    """A parameter whose blocks are split or replicated independently along one axis.

    ``segments`` is an ordered list of ``(size, replicated)`` pairs covering ``partition_dim``.
    ``split_widths`` gives the per-rank widths of each split segment, in the same order. It is
    required rather than optional because the sizes a layer splits to are not always an even
    division -- they can be aligned to head counts or to a grain size -- and a map built by
    dividing reads the wrong rows while still covering the tensor, so nothing downstream
    notices. Callers pass the widths the partition itself computed.
    A fused QKV weight that shards its query rows but hands every rank the whole key/value
    block is two segments, and the resulting pieces differ in `locations` rather than in
    kind -- which is what the schema could not say before.

    Each segment becomes its own piece precisely because the pieces must stay homogeneous:
    a replicated block sitting next to a rank-private one cannot share a `locations`, even
    where the two happen to be adjacent in the full tensor.
    """
    shape = tuple(shape)
    source_strides = _row_major_strides(shape)
    ranks = list(range(tp_degree))

    split_index = 0
    widths_for_segment = []
    for size, replicated in segments:
        if replicated:
            widths_for_segment.append(None)
            continue
        widths = list(split_widths[split_index])
        if len(widths) != tp_degree or sum(widths) != size:
            raise ValueError(f'Split widths {widths} do not account for a segment of {size} '
                             f'elements across {tp_degree} ranks.')
        widths_for_segment.append(widths)
        split_index += 1

    shard_extents = {}
    for rank in ranks:
        extent = 0
        for (size, replicated), widths in zip(segments, widths_for_segment):
            extent += size if replicated else widths[rank]
        shard_extents[rank] = extent

    shard_shapes = {}
    for rank in ranks:
        shard_shape = list(shape)
        shard_shape[partition_dim] = shard_extents[rank]
        shard_shapes[rank] = tuple(shard_shape)

    pieces_by_rank = {rank: [] for rank in ranks}
    dest_starts = {rank: 0 for rank in ranks}
    source_start = 0
    for (size, replicated), per_rank in zip(segments, widths_for_segment):
        for rank in ranks:
            extent = size if replicated else per_rank[rank]
            offset = source_start if replicated else source_start + sum(per_rank[:rank])
            piece_shape = list(shape)
            piece_shape[partition_dim] = extent
            dest_strides = _row_major_strides(shard_shapes[rank])
            pieces_by_rank[rank].append(
                AffinePiece(shape=tuple(piece_shape),
                            source_offset=offset * source_strides[partition_dim],
                            source_strides=source_strides,
                            dest_offset=dest_starts[rank] * dest_strides[partition_dim],
                            dest_strides=dest_strides,
                            locations=ranks if replicated else [rank]))
            dest_starts[rank] += extent
        source_start += size

    return ParamAffineMap(logical_shape=shape, shard_shapes=shard_shapes, pieces_by_rank=pieces_by_rank)


def block_gather_map(shape, block_ids_by_rank, block_size, partition_dim):
    """Each rank holds a chosen set of equal-sized blocks, not necessarily adjacent ones.

    Yuan's shared-QK attention is the motivating case: a rank takes the value heads that
    pair with its query heads, and those sit in two separate runs rather than one span.
    Consecutive ids merge into a single piece, so the count follows the block structure of
    the selection rather than the number of blocks.
    """
    shape = tuple(shape)
    source_strides = _row_major_strides(shape)

    shard_shapes = {}
    for rank, block_ids in block_ids_by_rank.items():
        shard_shape = list(shape)
        shard_shape[partition_dim] = len(block_ids) * block_size
        shard_shapes[rank] = tuple(shard_shape)

    pieces_by_rank = {}
    for rank, block_ids in block_ids_by_rank.items():
        dest_strides = _row_major_strides(shard_shapes[rank])
        pieces = []
        dest_block = 0
        index = 0
        while index < len(block_ids):
            run = 1
            while index + run < len(block_ids) and block_ids[index + run] == block_ids[index + run - 1] + 1:
                run += 1
            piece_shape = list(shape)
            piece_shape[partition_dim] = run * block_size
            pieces.append(
                AffinePiece(shape=tuple(piece_shape),
                            source_offset=block_ids[index] * block_size * source_strides[partition_dim],
                            source_strides=source_strides,
                            dest_offset=dest_block * block_size * dest_strides[partition_dim],
                            dest_strides=dest_strides,
                            locations=[rank]))
            dest_block += run
            index += run
        pieces_by_rank[rank] = pieces

    return ParamAffineMap(logical_shape=shape, shard_shapes=shard_shapes, pieces_by_rank=pieces_by_rank)


def nest_affine_maps(outer, inner_by_outer_rank):
    """Compose a layout map with a description of how each of its shards is stored.

    ``outer`` describes the parameter's layout: the shard each layout rank (a tensor- or
    expert-parallel rank) holds. ``inner_by_outer_rank[r]`` describes how layout shard ``r``,
    flattened to ``(numel,)``, is spread over the processes that store it, as
    ``deepspeed.checkpoint.zero_affine.zero_storage_affine_map`` does for ZeRO replicas and
    partitions. Its pieces must be contiguous intervals.

    The result describes the parameter in its own shape, keyed by the inner ranks, and each shard is
    what a process stores: a flat run of its layout shard. A stored run is generally not a box of the
    parameter, since a ZeRO partition may begin or end inside a row, so it is described by a few
    boxes, each contiguous in storage: a partial row at either end and the whole rows between. A
    block that is not contiguous in its layout shard, such as one split along a trailing axis inside
    the shard, is cut further until each box is. An element listed on several layout ranks is held by
    the union of their holders, so pieces are cut wherever any of those holders' storage begins or
    ends.
    """
    if set(inner_by_outer_rank) != set(outer.shard_shapes):
        raise ValueError(f'Storage is described for layout ranks {sorted(inner_by_outer_rank)}, but the layout has '
                         f'ranks {sorted(outer.shard_shapes)}.')
    storage_shapes = {}
    for outer_rank, inner in inner_by_outer_rank.items():
        shard_numel = _product(outer.shard_shapes[outer_rank])
        if inner.logical_shape != (shard_numel, ):
            raise ValueError(f'Layout rank {outer_rank} holds {shard_numel} elements, but its storage map describes '
                             f'shape {inner.logical_shape}.')
        for storage_rank, shape in inner.shard_shapes.items():
            if storage_rank in storage_shapes:
                # One shard per process: a process storing two layout shards would need two buffers.
                raise ValueError(f'Rank {storage_rank} stores more than one layout shard.')
            storage_shapes[storage_rank] = shape
    intervals_by_outer_rank = {rank: _storage_intervals(inner) for rank, inner in inner_by_outer_rank.items()}

    # Replicated layout pieces describe the same block on each of their holders.
    replicas = {}
    for outer_rank, pieces in outer.pieces_by_rank.items():
        for piece in pieces:
            if piece.numel == 0:
                continue
            key = (piece.shape, piece.source_offset, piece.source_strides)
            replicas.setdefault(key, []).append((outer_rank, piece))

    pieces_by_rank = {rank: [] for rank in storage_shapes}
    for copies in replicas.values():
        listed = {outer_rank for outer_rank, _ in copies}
        for _, piece in copies:
            if piece.locations != listed:
                raise ValueError(f'Layout piece {piece} lists holders {sorted(piece.locations)}, but layout ranks '
                                 f'{sorted(listed)} hold it.')
        _nest_replicated_block(copies, intervals_by_outer_rank, pieces_by_rank)

    affine_map = ParamAffineMap(logical_shape=outer.logical_shape,
                                shard_shapes=storage_shapes,
                                pieces_by_rank=pieces_by_rank)
    affine_map.validate()
    return affine_map


def _storage_intervals(inner):
    # (start, stop, storage rank, offset in that rank's shard, scale) for every stored interval.
    intervals = []
    for storage_rank, pieces in inner.pieces_by_rank.items():
        for piece in pieces:
            if piece.numel == 0:
                continue
            if piece.source_strides != (1, ) or piece.dest_strides != (1, ):
                raise ValueError(f'Storage piece {piece} of rank {storage_rank} is not a contiguous interval.')
            intervals.append(
                (piece.source_offset, piece.source_offset + piece.numel, storage_rank, piece.dest_offset, piece.scale))
    return sorted(intervals)


def _nest_replicated_block(copies, intervals_by_outer_rank, pieces_by_rank):
    # Cuts are row-major element indices of the block. Between two cuts, every copy lies inside one
    # stored interval of every holder; each range between cuts is then split into boxes that are
    # contiguous in every copy's shard.
    block = copies[0][1]
    cuts = {0, block.numel}
    for outer_rank, piece in copies:
        _check_monotone_dest(piece)
        for start, stop, _, _, _ in intervals_by_outer_rank[outer_rank]:
            for boundary in (start, stop):
                cuts.add(_first_index_at_or_after(piece, boundary))
    cuts = sorted(cuts)

    for low, high in zip(cuts, cuts[1:]):
        if low == high:
            continue
        stored = []
        for outer_rank, piece in copies:
            address = _strided_address(piece.shape, piece.dest_offset, piece.dest_strides, low)
            covering = [
                interval for interval in intervals_by_outer_rank[outer_rank] if interval[0] <= address < interval[1]
            ]
            if not covering:
                raise ValueError(f'Layout rank {outer_rank} holds shard element {address}, but its storage map '
                                 'stores it on no rank.')
            stored.extend((piece, address, interval) for interval in covering)
        locations = {interval[2] for _, _, interval in stored}
        # A box must be one run of addresses in every copy, so the most fragmented copy decides.
        dense_rows = max(_dense_rows(piece) for piece, _, _ in stored)
        for origin, extent in _row_major_range_boxes(block.shape, low, high, dense_rows):
            source_offset = block.source_offset + _dot(origin, block.source_strides)
            for piece, _, (start, _, storage_rank, storage_offset, storage_scale) in stored:
                address = piece.dest_offset + _dot(origin, piece.dest_strides)
                pieces_by_rank[storage_rank].append(
                    AffinePiece(shape=extent,
                                source_offset=source_offset,
                                source_strides=block.source_strides,
                                dest_offset=storage_offset + address - start,
                                dest_strides=_row_major_strides(extent),
                                locations=locations,
                                scale=piece.scale * storage_scale))


def _dot(index, strides):
    return sum(position * stride for position, stride in zip(index, strides))


def _dense_rows(piece):
    # The number of leading axes after which the piece is contiguous in its shard: a box of the piece
    # that spans every later axis in full, and only those, is one run of shard addresses.
    run = 1
    for axis in range(len(piece.shape) - 1, -1, -1):
        size, stride = piece.shape[axis], piece.dest_strides[axis]
        if size > 1 and stride != run:
            return axis + 1
        run *= size
    return 0


def _row_major_range_boxes(shape, low, high, dense_rows=0):
    """Split the row-major index range ``[low, high)`` of ``shape`` into ``(origin, extent)`` boxes.

    At most two boxes per axis: a partial run at either end and whole runs between. The first
    ``dense_rows`` axes are never merged, so a box spans more than one position of such an axis only
    where the storage is contiguous across it.
    """
    if low >= high:
        return []
    if not shape:
        return [((), ())]
    if len(shape) == 1:
        return [((low, ), (high - low, ))]
    inner = _product(shape[1:])
    first, low_rest = divmod(low, inner)
    last, high_rest = divmod(high, inner)

    def within(row, start, stop):
        return [((row, ) + origin, (1, ) + extent)
                for origin, extent in _row_major_range_boxes(shape[1:], start, stop, dense_rows - 1)]

    if first == last:
        return within(first, low_rest, high_rest)
    boxes = []
    if low_rest:
        boxes.extend(within(first, low_rest, inner))
        first += 1
    if dense_rows > 0:
        for row in range(first, last):
            boxes.extend(within(row, 0, inner))
    elif last > first:
        boxes.append(((first, ) + (0, ) * (len(shape) - 1), (last - first, ) + tuple(shape[1:])))
    if high_rest:
        boxes.extend(within(last, 0, high_rest))
    return boxes


def _check_monotone_dest(piece):
    # Storage intervals are ranges of shard addresses. A piece whose addresses grow with its
    # row-major element index meets such a range in one run of indices; any other piece could
    # interleave with it.
    later_span = 0
    for size, stride in reversed(list(zip(piece.shape, piece.dest_strides))):
        if size > 1:
            if stride <= later_span:
                raise NotImplementedError(f'Layout piece {piece} does not store its elements in row-major order, so '
                                          'its storage cannot be described.')
            later_span += stride * (size - 1)


def _strided_address(shape, offset, strides, index):
    address = offset
    for size, stride in reversed(list(zip(shape, strides))):
        index, position = divmod(index, size)
        address += position * stride
    return address


def _first_index_at_or_after(piece, address):
    # The smallest row-major index whose shard address is at least ``address``; ``numel`` if none.
    remaining = address - piece.dest_offset
    if remaining <= 0:
        return 0
    spans = []
    later_span = 0
    for size, stride in reversed(list(zip(piece.shape, piece.dest_strides))):
        spans.append(later_span)
        later_span += stride * (size - 1)
    spans.reverse()
    if remaining > later_span:
        return piece.numel

    index = 0
    for axis, (size, stride) in enumerate(zip(piece.shape, piece.dest_strides)):
        position = 0
        if size > 1:
            # The first position from which the later axes can still reach ``remaining``.
            position = max(0, -(-(remaining - spans[axis]) // stride))
        index = index * size + position
        remaining -= position * stride
        if remaining <= 0:
            for later_size in piece.shape[axis + 1:]:
                index *= later_size
            return index
    return index

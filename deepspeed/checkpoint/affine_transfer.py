# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Plan a parameter move between two affine shardings without rebuilding the parameter.

``affine.py`` describes where one rank's block of a parameter lives on both sides of a shard, and
``rebuild``/``extract`` act on that description through the full tensor. This module composes two
descriptions -- the map the bytes were sharded by and the map they are being sharded into -- into
copies that go straight from a source shard to a target shard, so no rank has to address the full
parameter at any point.

Direction is the easy thing to get wrong here. A piece's ``source_*`` addresses the full logical
tensor and its ``dest_*`` addresses the shard holding it, so the read address of a source piece is
its ``dest_*``, not its ``source_*``. Every address this module emits points into a shard, and
offsets and strides are elements, as ``torch.as_strided`` takes them. Converting to bytes belongs to
whatever moves them.

Only ``scale == 1`` is planned. A scaled piece is invertible on its own, but an Adam resume under a
scale also needs its learning rate and epsilon carried into the other coordinate, which is geometry
this file does not own -- so it is refused here for the same reason ``affine.py`` refuses it.

Nothing here is allocated per element, which is why ``ParamAffineMap.validate_coverage`` is never
called: it walks every element of the parameter by design. Coverage is argued over pieces instead,
and ``_covers_exactly`` states what that argument does and does not buy.
"""

import collections.abc
import json
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union

from .affine import AffinePiece, ParamAffineMap

__all__ = [
    'AffineTransferError', 'InvalidMapError', 'InvalidReplicaPolicyError', 'UnsupportedLayoutError', 'PlanBudget',
    'PlanStatistics', 'ReplicaCandidate', 'ReplicaSelectionRequest', 'ReplicaRequest', 'ReplicaPart',
    'TransferSegment', 'TransferPlan', 'plan_transfer', 'plan_transfers'
]


class AffineTransferError(ValueError):
    """A composed transfer that must not be handed to a transport."""


class InvalidMapError(AffineTransferError):
    """A map contradicts itself, or the two maps contradict each other.

    Kept distinct from an unsupported layout because falling back would hide a corrupt checkpoint
    rather than route around a limitation.
    """


class UnsupportedLayoutError(AffineTransferError):
    """Legitimate geometry that this planner does not express.

    A caller may rebuild the parameter and re-shard it instead. That is slow, not wrong, and it is
    the honest reading of a layout this file never claimed to cover.
    """


class InvalidReplicaPolicyError(AffineTransferError):
    """A selector chose a rank outside the validated source candidates."""


@dataclass(frozen=True)
class PlanBudget:
    """Limits on the plan, each checked before the work it would bound.

    Policy rather than correctness: a plan twice this size is still a correct plan, and a caller that
    can carry it should be allowed to. They exist so that a map whose piece count is itself the bug
    cannot force an unbounded enumeration, and so that exhausting one is a failure the caller can see
    instead of a shortened plan it cannot.
    """
    max_ndim: int = 8
    max_pieces_per_map: int = 1 << 12
    max_pair_checks: int = 1 << 18
    max_segments: int = 1 << 16
    max_merge_passes: int = 8
    max_plan_bytes: int = 1 << 24
    max_merge_checks: int = 1 << 18


@dataclass(frozen=True)
class PlanStatistics:
    """What the planner looked at, so a caller can tell a big model from a fragmented layout."""
    source_pieces: int
    target_pieces: int
    pair_checks: int
    segments_before_merge: int
    segments: int
    plan_bytes: int


@dataclass(frozen=True)
class LogicalBox:
    """An axis-aligned block of the logical parameter: its first coordinate and its extents."""
    origin: Tuple[int, ...]
    extent: Tuple[int, ...]

    @property
    def volume(self) -> int:
        return _product(self.extent)

    @property
    def is_empty(self) -> bool:
        return any(size == 0 for size in self.extent)

    def upper(self) -> Tuple[int, ...]:
        return tuple(low + size for low, size in zip(self.origin, self.extent))

    def intersection(self, other: 'LogicalBox') -> 'LogicalBox':
        low = tuple(max(a, b) for a, b in zip(self.origin, other.origin))
        high = tuple(min(a, b) for a, b in zip(self.upper(), other.upper()))
        return LogicalBox(low, tuple(max(0, h - l) for h, l in zip(high, low)))

    def contains(self, other: 'LogicalBox') -> bool:
        return all(other_low >= low and other_high <= high
                   for low, high, other_low, other_high in zip(self.origin, self.upper(), other.origin, other.upper()))

    def __repr__(self) -> str:
        return f'Box({self.origin}+{self.extent})'


@dataclass(frozen=True)
class _Frame:
    """One piece: where it sits in the logical parameter and where it sits in one rank's shard."""
    rank: int
    locations: FrozenSet[int]
    box: LogicalBox
    local: LogicalBox
    offset: int
    strides: Tuple[int, ...]
    shard_numel: int

    def address(self, box: LogicalBox) -> int:
        """Shard offset of a sub-box of this frame's region, in this shard's own addressing."""
        return self.offset + sum(
            (low - own) * stride for low, own, stride in zip(box.origin, self.box.origin, self.strides))


@dataclass(frozen=True)
class ReplicaCandidate:
    source_rank: int
    source_offset: int
    source_strides: Tuple[int, ...]


@dataclass(frozen=True)
class ReplicaSelectionRequest:
    target_rank: int
    logical_origin: Tuple[int, ...]
    shape: Tuple[int, ...]
    target_offset: int
    target_strides: Tuple[int, ...]
    candidates: Tuple[ReplicaCandidate, ...]


@dataclass(frozen=True)
class ReplicaRequest:
    """One region a target piece needs, with every validated holder that can supply it.

    ``target_location`` and each candidate's ``source_rank`` are keys of the target and source maps,
    not processes: the same number names different processes on the two sides. ``target_process``
    and ``candidate_processes`` are processes, and are ``None`` unless the caller passed the
    location-to-process mappings. ``bytes`` is ``None`` unless the caller passed element sizes.
    """
    parameter: str
    target_location: int
    target_process: Optional[int]
    logical_origin: Tuple[int, ...]
    shape: Tuple[int, ...]
    target_offset: int
    target_strides: Tuple[int, ...]
    candidates: Tuple[ReplicaCandidate, ...]
    candidate_processes: Tuple[Optional[int], ...]
    bytes: Optional[int]

    @property
    def numel(self) -> int:
        return _product(self.shape)


@dataclass(frozen=True)
class ReplicaPart:
    """A cut of a request's region along axis 0, and the holder location it is read from."""
    logical_origin: Tuple[int, ...]
    shape: Tuple[int, ...]
    source_location: int


ReplicaAnswer = Union[int, Sequence[ReplicaPart]]
ReplicaPolicy = Callable[[Sequence[ReplicaRequest]], Sequence[ReplicaAnswer]]


@dataclass(frozen=True)
class TransferSegment:
    """One directly executable strided copy, from a source shard into a target shard.

    ``shape`` is shared by both sides, as in ``AffinePiece``. The two stride vectors generally
    differ: a block contiguous in one shard can stride through the other, and reading it as though
    both sides were packed is the mistake this split exists to prevent.
    """
    source_rank: int
    source_offset: int
    source_strides: Tuple[int, ...]
    target_rank: int
    target_offset: int
    target_strides: Tuple[int, ...]
    shape: Tuple[int, ...]

    @property
    def numel(self) -> int:
        return _product(self.shape)

    def to_dict(self) -> Dict[str, List[object]]:
        """Plain scalars, so a plan can cross a process boundary without these classes."""
        return {
            'shape': list(self.shape),
            'source': [self.source_rank, self.source_offset,
                       list(self.source_strides)],
            'target': [self.target_rank, self.target_offset,
                       list(self.target_strides)],
        }

    @classmethod
    def from_dict(cls, entry: Dict[str, List[object]]) -> 'TransferSegment':
        source_rank, source_offset, source_strides = entry['source']
        target_rank, target_offset, target_strides = entry['target']
        return cls(source_rank=int(source_rank),
                   source_offset=int(source_offset),
                   source_strides=tuple(int(stride) for stride in source_strides),
                   target_rank=int(target_rank),
                   target_offset=int(target_offset),
                   target_strides=tuple(int(stride) for stride in target_strides),
                   shape=tuple(int(size) for size in entry['shape']))


@dataclass(frozen=True)
class TransferPlan:
    """The ordered copies, the parameter they describe, and what it took to produce them."""
    segments: Tuple[TransferSegment, ...]
    logical_shape: Tuple[int, ...]
    moved_elements: int
    statistics: PlanStatistics

    def to_dict(self) -> Dict[str, List[object]]:
        return {
            'logical_shape': list(self.logical_shape),
            'segments': [segment.to_dict() for segment in self.segments],
        }


def _product(shape: Sequence[int]) -> int:
    count = 1
    for size in shape:
        count *= size
    return count


def _row_major_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)


def _coordinate(offset: int, strides: Tuple[int, ...], shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """The grid coordinate a storage offset names, or None when it names nothing."""
    coordinates = []
    rest = offset
    for stride, size in zip(strides, shape):
        coordinate, remainder = divmod(rest, stride)
        if coordinate >= size:
            return None
        coordinates.append(coordinate)
        rest = remainder
    return tuple(coordinates) if rest == 0 else None


def _frame_of(piece: AffinePiece, rank: int, side: str, logical_shape: Tuple[int, ...],
              shard_shape: Tuple[int, ...]) -> Optional[_Frame]:
    """Accept a piece only as an axis-aligned box, and refuse whatever else it might be.

    Both stride vectors must be the row-major strides of the tensor they address, which is what makes
    the box arithmetic below valid and what every builder in ``affine.py`` already emits. An axis the
    piece does not span is absent from storage, so its stride is not compared -- that is also where a
    permutation would hide, and a permutation is not one box. A flat shard is the one exception: it may
    hold a multi-axis box as a dense run, which is how ZeRO stores a tensor-parallel shard.
    """
    where = f'{side} rank {rank} piece {piece!r}'
    if len(piece.shape) != len(logical_shape):
        raise InvalidMapError(f'{where}: spans {len(piece.shape)} axes of a {len(logical_shape)}-axis parameter')
    if any(size < 0 for size in piece.shape):
        raise InvalidMapError(f'{where}: negative extent')
    if not piece.locations:
        raise InvalidMapError(f'{where}: names no holder')
    if piece.scale != 1.0:
        raise UnsupportedLayoutError(f'{where}: scale {piece.scale}, which would leave a resumed optimizer '
                                     'in the source coordinate')
    if len(piece.source_strides) != len(piece.shape) or len(piece.dest_strides) != len(piece.shape):
        raise InvalidMapError(f'{where}: stride vector length must match the piece rank')
    if piece.numel == 0:
        # A rank may hold none of one sub-parameter, and the builders give such a piece the running
        # offset where it would have started, which is legitimately one past the end of the tensor.
        # There is nothing to place, so there is nothing to address and nothing to refuse.
        return None

    logical_strides = _row_major_strides(logical_shape)
    shard_strides = _row_major_strides(shard_shape)
    for axis, (got, want, size) in enumerate(zip(piece.source_strides, logical_strides, piece.shape)):
        if size > 1 and got != want:
            raise UnsupportedLayoutError(f'{where}: source axis {axis} strides {got}, expected {want}')
    flat_storage = len(shard_shape) == 1 and len(piece.shape) != 1
    if flat_storage:
        # A flat shard, as ZeRO stores one, holds a box of a multi-axis parameter as one run of
        # addresses: the box's own row-major strides. Its place in the shard is that run.
        shard_strides = _row_major_strides(tuple(piece.shape))
    for axis, (got, want, size) in enumerate(zip(piece.dest_strides, shard_strides, piece.shape)):
        if size > 1 and got != want:
            raise UnsupportedLayoutError(f'{where}: dest axis {axis} strides {got}, expected {want}')

    origin = _origin(piece.source_offset, logical_strides, logical_shape, where, 'source')
    box = LogicalBox(origin, tuple(piece.shape))
    if not LogicalBox(tuple([0] * len(logical_shape)), logical_shape).contains(box):
        raise InvalidMapError(f'{where}: covers {box!r}, outside the logical parameter of {logical_shape}')
    if flat_storage:
        local = LogicalBox((piece.dest_offset, ), (piece.numel, ))
    else:
        local = LogicalBox(_origin(piece.dest_offset, shard_strides, shard_shape, where, 'dest'), tuple(piece.shape))
    if not LogicalBox(tuple([0] * len(shard_shape)), shard_shape).contains(local):
        raise InvalidMapError(f'{where}: writes {local!r}, outside the {shard_shape} shard')
    return _Frame(rank=rank,
                  locations=frozenset(piece.locations),
                  box=box,
                  local=local,
                  offset=piece.dest_offset,
                  strides=shard_strides,
                  shard_numel=_product(shard_shape))


def _origin(offset: int, strides: Tuple[int, ...], shape: Tuple[int, ...], where: str, which: str) -> Tuple[int, ...]:
    coordinate = _coordinate(offset, strides, shape)
    if coordinate is None:
        raise InvalidMapError(f'{where}: {which} offset {offset} is not a coordinate of a {shape} tensor')
    return coordinate


def _normalized(map_: ParamAffineMap, side: str, limits: PlanBudget) -> List[_Frame]:
    logical_shape = tuple(map_.logical_shape)
    if len(logical_shape) > limits.max_ndim:
        raise UnsupportedLayoutError(f'{side}: {len(logical_shape)} axes exceeds max_ndim {limits.max_ndim}')
    # A flat shard is flat storage, such as ZeRO's; its pieces carry the axis correspondence.
    if any(len(tuple(shape)) not in (1, len(logical_shape)) for shape in map_.shard_shapes.values()):
        raise UnsupportedLayoutError(
            f'{side}: a shard is described with a different number of axes than the logical parameter, so '
            'no axis correspondence exists to plan against')
    piece_count = sum(len(pieces) for pieces in map_.pieces_by_rank.values())
    if piece_count > limits.max_pieces_per_map:
        raise AffineTransferError(f'{side}: {piece_count} pieces exceeds max_pieces_per_map '
                                  f'{limits.max_pieces_per_map}')

    frames = []
    for rank, shape in map_.shard_shapes.items():
        if _product(shape) and not any(piece.numel for piece in map_.pieces_by_rank.get(rank, [])):
            raise InvalidMapError(f'{side}: rank {rank} declares a nonempty shard but has no pieces')
    for rank, pieces in sorted(map_.pieces_by_rank.items()):
        if rank not in map_.shard_shapes:
            raise InvalidMapError(f'{side}: rank {rank} holds pieces but declares no shard shape')
        shard_shape = tuple(map_.shard_shapes[rank])
        if _product(shard_shape) == 0:
            if any(piece.numel for piece in pieces):
                raise InvalidMapError(f'{side}: rank {rank} declares an empty shard and holds a full piece')
            continue
        for piece in pieces:
            frame = _frame_of(piece, rank, side, logical_shape, shard_shape)
            if frame is not None:
                frames.append(frame)
    _shards_tile(frames, side, limits)
    _validate_holders(frames, side, limits)
    return frames


def _shards_tile(frames: List[_Frame], side: str, limits: PlanBudget) -> None:
    """Prove each rank's pieces partition its shard by address, not merely by element count.

    This is the premise the rest of the proof borrows, and it does not follow from the logical boxes
    being disjoint: three pieces of one rank can cover three disjoint runs of the parameter while two
    of them land on the same addresses of the shard, and their numels still sum to the shard size. A
    shard that two pieces claim cannot be read or written by address, so a plan built on such a map is
    wrong in whichever direction it is used.
    """
    by_rank: Dict[int, List[_Frame]] = {}
    for frame in frames:
        by_rank.setdefault(frame.rank, []).append(frame)

    for rank, held in sorted(by_rank.items()):
        if len(held)**2 > limits.max_pair_checks:
            raise AffineTransferError(f'{side}: rank {rank} holds {len(held)} pieces, over the budget for '
                                      'address checks')
        volume = 0
        for index, frame in enumerate(held):
            volume += frame.local.volume
            for other in held[index + 1:]:
                if not frame.local.intersection(other.local).is_empty:
                    raise InvalidMapError(f'{side}: rank {rank} pieces {frame.box!r} and {other.box!r} both '
                                          'occupy part of the same shard addresses')
        if volume != held[0].shard_numel:
            raise InvalidMapError(f'{side}: rank {rank} pieces account for {volume} of its {held[0].shard_numel} '
                                  'shard elements')


def _validate_holders(frames: List[_Frame], side: str, limits: PlanBudget) -> None:
    """A piece's declared holders must cover exactly its logical box on every named rank."""
    if len(frames)**2 > limits.max_pair_checks:
        raise AffineTransferError(f'{side}: holder checks exceed max_pair_checks {limits.max_pair_checks}')
    by_rank: Dict[int, List[_Frame]] = {}
    for frame in frames:
        by_rank.setdefault(frame.rank, []).append(frame)

    for rank, held in by_rank.items():
        for index, frame in enumerate(held):
            for other in held[index + 1:]:
                if not frame.box.intersection(other.box).is_empty:
                    raise InvalidMapError(f'{side}: rank {rank} holds overlapping logical pieces')

    for frame in frames:
        actual = set()
        for rank, held in by_rank.items():
            covered = sum(frame.box.intersection(other.box).volume for other in held)
            if covered:
                actual.add(rank)
            if rank in frame.locations and covered != frame.box.volume:
                raise InvalidMapError(f'{side}: rank {rank} does not hold all of {frame.box!r}')
        if actual != set(frame.locations):
            raise InvalidMapError(f'{side}: {frame.box!r} has different locations: '
                                  f'declares {sorted(frame.locations)}, held by {sorted(actual)}')


def _sources(frames: List[_Frame], limits: PlanBudget) -> List[Tuple[LogicalBox, Tuple[_Frame, ...]]]:
    """Keep the validated holders of each disjoint source region.

    Every rank holding a replicated piece lists the same region in its own shard at its own offset,
    so a region appears once per holder. Selection happens per target intersection because different
    targets can use different holders, whose local offsets need not agree.

    Two regions that overlap without being equal mean one element has two unrelated owners, which no
    fallback can repair. Establishing that the regions are disjoint is also what lets
    ``_covers_exactly`` skip re-checking overlaps on every target.
    """
    by_box: Dict[LogicalBox, List[_Frame]] = {}
    for frame in frames:
        if not frame.box.is_empty:
            by_box.setdefault(frame.box, []).append(frame)

    regions = sorted(by_box, key=lambda box: (box.origin, box.extent))
    if len(regions)**2 > limits.max_pair_checks:
        raise AffineTransferError(f'{len(regions)} source regions would take {len(regions)**2} checks, over '
                                  f'max_pair_checks {limits.max_pair_checks}')
    grouped = []
    for index, box in enumerate(regions):
        holders = by_box[box]
        for other in regions[index + 1:]:
            if not box.intersection(other).is_empty:
                raise UnsupportedLayoutError(f'source: replicas of {box!r} and {other!r} have different '
                                             'piece boundaries')
        grouped.append((box, tuple(sorted(holders, key=lambda frame: frame.rank))))
    return grouped


def _covers_exactly(region: LogicalBox, parts: Sequence[LogicalBox]) -> bool:
    """Whether ``parts`` fill ``region`` once and only once.

    They are intersections of ``region`` with source regions, so each is inside it, and the source
    regions were made disjoint by ``_sources`` -- so the parts are disjoint without further checks.
    Disjoint parts inside a region account for it exactly when their volumes sum to it. This is the
    whole argument, and it is why comparing some element count of the *plan* against some element
    count of the parameter would prove nothing: only tiling per target region catches a gap, and a
    gap and an overlap of equal size cancel in any total.
    """
    return sum(part.volume for part in parts) == region.volume


def _subtract(region: LogicalBox, cut: LogicalBox) -> List[LogicalBox]:
    """Return the disjoint parts of a box outside another box."""
    overlap = region.intersection(cut)
    if overlap.is_empty:
        return [region]
    origin = list(region.origin)
    extent = list(region.extent)
    pieces = []
    for axis, (low, high) in enumerate(zip(overlap.origin, overlap.upper())):
        if origin[axis] < low:
            shape = extent.copy()
            shape[axis] = low - origin[axis]
            pieces.append(LogicalBox(tuple(origin), tuple(shape)))
            extent[axis] -= shape[axis]
            origin[axis] = low
        end = origin[axis] + extent[axis]
        if high < end:
            start = origin.copy()
            shape = extent.copy()
            start[axis] = high
            shape[axis] = end - high
            pieces.append(LogicalBox(tuple(start), tuple(shape)))
            extent[axis] = high - origin[axis]
    return pieces


def _covers_union(region: LogicalBox, boxes: Sequence[LogicalBox], limits: PlanBudget) -> bool:
    """Check a union without enumerating parameter elements or counting replica overlap twice."""
    uncovered = [] if region.is_empty else [region]
    checks = 0
    for box in boxes:
        remaining = []
        for gap in uncovered:
            checks += 1
            if checks > limits.max_pair_checks:
                raise AffineTransferError(f'coverage checks exceed max_pair_checks {limits.max_pair_checks}')
            remaining.extend(_subtract(gap, box))
        uncovered = remaining
        if not uncovered:
            return True
    return not uncovered


@dataclass(frozen=True)
class _Intersection:
    """One target piece's share of one source region, before a holder is chosen for it."""
    target: _Frame
    shared: LogicalBox
    holders: Tuple[_Frame, ...]


@dataclass(frozen=True)
class _Prepared:
    """A validated pair of maps, cut into intersections that only lack a choice of holder."""
    logical_shape: Tuple[int, ...]
    intersections: Tuple[_Intersection, ...]
    source_pieces: int
    target_pieces: int
    pair_checks: int


def plan_transfer(target_map: ParamAffineMap,
                  source_map: ParamAffineMap,
                  limits: Optional[PlanBudget] = None,
                  *,
                  replica_selector: Optional[Callable[[ReplicaSelectionRequest], int]] = None) -> TransferPlan:
    """Plan copies from shards laid out by ``source_map`` into shards laid out by ``target_map``.

    Both maps must describe the same logical parameter. Whether two parameters *are* the same
    parameter is the caller's question -- equal shapes are not evidence of it, and fused parameters
    have matched by byte count before -- so this checks only that the two geometries compose.
    """
    budget = limits or PlanBudget()
    prepared = _prepare(target_map, source_map, budget)

    choices: List[Tuple[Tuple[LogicalBox, _Frame], ...]] = []
    for item in prepared.intersections:
        target, shared, holders = item.target, item.shared, item.holders
        source = holders[0]
        if replica_selector is not None and len(holders) > 1:
            request = ReplicaSelectionRequest(target_rank=target.rank,
                                              logical_origin=shared.origin,
                                              shape=shared.extent,
                                              target_offset=target.address(shared),
                                              target_strides=target.strides,
                                              candidates=tuple(
                                                  ReplicaCandidate(holder.rank, holder.address(shared), holder.strides)
                                                  for holder in holders))
            rank = replica_selector(request)
            if type(rank) is not int or rank not in {holder.rank for holder in holders}:
                raise InvalidReplicaPolicyError(f'selector returned invalid source rank {rank!r} '
                                                f'for target rank {target.rank}')
            source = next(holder for holder in holders if holder.rank == rank)
        choices.append(((shared, source), ))
    return _emit(prepared, choices, budget)


def plan_transfers(pairs: Mapping[str, Tuple[ParamAffineMap, ParamAffineMap]],
                   limits: Optional[PlanBudget] = None,
                   *,
                   source_processes: Optional[Mapping[int, int]] = None,
                   target_processes: Optional[Mapping[int, int]] = None,
                   element_size: Optional[Mapping[str, int]] = None,
                   min_split_bytes: Optional[int] = None,
                   replica_policy: Optional[ReplicaPolicy] = None) -> Dict[str, TransferPlan]:
    """Plan every parameter of one sync, with one replica policy that sees every request at once.

    ``pairs`` maps a parameter name to its ``(target_map, source_map)``. Each pair is validated
    exactly as ``plan_transfer`` validates it, and ``limits`` applies to each parameter on its own.

    ``replica_policy`` is then called once, with a ``ReplicaRequest`` for every intersection of a
    target piece with a source region, in a fixed order: parameter, target location, origin. Regions
    with a single holder are included, because their load is load the policy cannot move. The policy
    returns one answer per request: the location of one validated holder for the whole region, or
    ``ReplicaPart``\\ s that cut the region along axis 0 and name a validated holder for each cut.

    Every answer is checked before any segment is emitted, and a failure raises
    ``InvalidReplicaPolicyError``. There must be one answer per request, every location must be a
    validated holder of that request, and the parts must tile the request exactly along axis 0. When
    ``min_split_bytes`` is given, each part of a split must also be at least that large. So a policy
    decides where data is read from but not what arrives: when the replicas agree, every accepted
    answer copies the same data.

    Plan locations are keys of the two maps, not processes. ``source_processes`` and
    ``target_processes`` say which process holds each location. When given, the requests carry them,
    so a policy can tell a local copy from a transfer. ``element_size`` gives bytes per element for
    each parameter, which the maps do not know, and ``min_split_bytes`` requires it. Without a
    policy, ``affine_replica_policy.local_first_least_loaded`` is used.
    """
    budget = limits or PlanBudget()
    names = sorted(pairs)
    for which, mapping in (('source_processes', source_processes), ('target_processes', target_processes)):
        if mapping is not None and any(
                type(location) is not int or type(process) is not int for location, process in mapping.items()):
            raise AffineTransferError(f'{which} must map int locations to int processes')
    if element_size is not None:
        for name in names:
            size = element_size.get(name)
            if type(size) is not int or size <= 0:
                raise AffineTransferError(f'element_size for {name!r} is {size!r}, not a positive int')
    if min_split_bytes is not None:
        if type(min_split_bytes) is not int or min_split_bytes <= 0:
            raise AffineTransferError(f'min_split_bytes is {min_split_bytes!r}, not a positive int')
        if element_size is None:
            raise AffineTransferError('min_split_bytes needs element_size: the maps count elements, not bytes')

    prepared = {}
    for name in names:
        target_map, source_map = pairs[name]
        prepared[name] = _prepare(target_map, source_map, budget)

    entries: List[Tuple[Tuple[object, ...], str, int, _Intersection]] = []
    for name in names:
        for index, item in enumerate(prepared[name].intersections):
            entries.append(((name, item.target.rank, item.shared.origin, item.shared.extent), name, index, item))
    entries.sort(key=lambda entry: entry[0])

    requests = []
    for _, name, _, item in entries:
        requests.append(
            ReplicaRequest(parameter=name,
                           target_location=item.target.rank,
                           target_process=_process(target_processes, item.target.rank, 'target'),
                           logical_origin=item.shared.origin,
                           shape=item.shared.extent,
                           target_offset=item.target.address(item.shared),
                           target_strides=item.target.strides,
                           candidates=tuple(
                               ReplicaCandidate(holder.rank, holder.address(item.shared), holder.strides)
                               for holder in item.holders),
                           candidate_processes=tuple(
                               _process(source_processes, holder.rank, 'source') for holder in item.holders),
                           bytes=None if element_size is None else item.shared.volume * element_size[name]))
    requests = tuple(requests)

    if replica_policy is None:
        from .affine_replica_policy import local_first_least_loaded
        replica_policy = local_first_least_loaded
    answers = replica_policy(requests)
    if not _is_sequence(answers) or len(answers) != len(requests):
        count = len(answers) if _is_sequence(answers) else type(answers).__name__
        raise InvalidReplicaPolicyError(f'the policy must return one answer per request: {len(requests)} '
                                        f'requests, got {count}')

    choices: Dict[str, List[Tuple[Tuple[LogicalBox, _Frame], ...]]] = {
        name: [()] * len(prepared[name].intersections)
        for name in names
    }
    parts_per_name = {name: 0 for name in names}
    for (_, name, index, item), request, answer in zip(entries, requests, answers):
        parts = _resolve(request, item, answer, min_split_bytes, element_size)
        choices[name][index] = parts
        parts_per_name[name] += len(parts)
        if parts_per_name[name] > budget.max_pair_checks:
            raise AffineTransferError(f'{name!r}: the policy split regions into more than max_pair_checks '
                                      f'{budget.max_pair_checks} parts')
    return {name: _emit(prepared[name], choices[name], budget) for name in names}


def _is_sequence(value: object) -> bool:
    """A real sequence: indexable and sized, which a generator is not and a string should not count as."""
    return isinstance(value, collections.abc.Sequence) and not isinstance(value, (str, bytes))


def _process(mapping: Optional[Mapping[int, int]], location: int, side: str) -> Optional[int]:
    if mapping is None:
        return None
    if location not in mapping:
        raise AffineTransferError(f'{side} location {location} has no process in {side}_processes')
    return mapping[location]


def _resolve(request: ReplicaRequest, item: _Intersection, answer: object, min_split_bytes: Optional[int],
             element_size: Optional[Mapping[str, int]]) -> Tuple[Tuple[LogicalBox, _Frame], ...]:
    """Turn one policy answer into (region, holder) pairs, or refuse it.

    The parts are checked against the request, never against each other's claims: each is inside the
    request's box by construction (it spans every axis but the first in full), and walking them in
    axis-0 order from the request's first row to its last proves they are disjoint and leave no gap.
    """
    where = (f'answer for {request.parameter!r} at {request.logical_origin} on target location '
             f'{request.target_location}')
    holders = {holder.rank: holder for holder in item.holders}
    box = item.shared
    if type(answer) is int:
        if answer not in holders:
            raise InvalidReplicaPolicyError(f'{where}: location {answer} is not a validated holder '
                                            f'{sorted(holders)}')
        return ((box, holders[answer]), )
    if not _is_sequence(answer) or not answer:
        raise InvalidReplicaPolicyError(f'{where}: expected a holder location or a non-empty sequence of '
                                        f'ReplicaPart, got {answer!r}')

    ndim = len(box.extent)
    parts = []
    for part in answer:
        if not isinstance(part, ReplicaPart):
            raise InvalidReplicaPolicyError(f'{where}: {part!r} is not a ReplicaPart')
        if type(part.source_location) is not int or part.source_location not in holders:
            raise InvalidReplicaPolicyError(f'{where}: location {part.source_location!r} is not a validated '
                                            f'holder {sorted(holders)}')
        origin, shape = tuple(part.logical_origin), tuple(part.shape)
        if len(origin) != ndim or len(shape) != ndim or any(type(value) is not int for value in origin + shape):
            raise InvalidReplicaPolicyError(f'{where}: part {origin}+{shape} is not a {ndim}-axis box of ints')
        if origin[1:] != box.origin[1:] or shape[1:] != box.extent[1:]:
            raise InvalidReplicaPolicyError(f'{where}: part {origin}+{shape} does not span every axis but the '
                                            f'first of {box!r}; only cuts along axis 0 are supported')
        if any(size <= 0 for size in shape):
            raise InvalidReplicaPolicyError(f'{where}: part {origin}+{shape} is empty')
        parts.append((LogicalBox(origin, shape), holders[part.source_location]))

    if ndim == 0:
        if len(parts) != 1:
            raise InvalidReplicaPolicyError(f'{where}: a scalar cannot be split')
        return tuple(parts)
    parts.sort(key=lambda entry: entry[0].origin[0])
    row = box.origin[0]
    for part_box, _ in parts:
        if part_box.origin[0] != row:
            kind = 'overlap' if part_box.origin[0] < row else 'gap'
            raise InvalidReplicaPolicyError(f'{where}: parts leave a {kind} at row {min(row, part_box.origin[0])} '
                                            f'of {box!r}')
        row += part_box.extent[0]
    if row != box.origin[0] + box.extent[0]:
        raise InvalidReplicaPolicyError(f'{where}: parts stop at row {row}, inside or past {box!r}')
    if len(parts) > 1 and min_split_bytes is not None:
        for part_box, _ in parts:
            size = part_box.volume * element_size[request.parameter]
            if size < min_split_bytes:
                raise InvalidReplicaPolicyError(f'{where}: part {part_box!r} is {size} bytes, under '
                                                f'min_split_bytes {min_split_bytes}')
    return tuple(parts)


def _prepare(target_map: ParamAffineMap, source_map: ParamAffineMap, budget: PlanBudget) -> _Prepared:
    """Validate both maps and cut the target into intersections with the source regions."""
    if tuple(source_map.logical_shape) != tuple(target_map.logical_shape):
        raise InvalidMapError(f'source describes {tuple(source_map.logical_shape)} and target describes '
                              f'{tuple(target_map.logical_shape)}')

    sources = _sources(_normalized(source_map, 'source', budget), budget)
    targets = _normalized(target_map, 'target', budget)
    logical_shape = tuple(target_map.logical_shape)
    logical_box = LogicalBox((0, ) * len(logical_shape), logical_shape)
    source_volume = sum(box.volume for box, _ in sources)
    if source_volume != logical_box.volume:
        raise InvalidMapError(f'source pieces cover {source_volume} of its {logical_box.volume} logical elements')
    if not _covers_union(logical_box, [frame.box for frame in targets], budget):
        raise InvalidMapError('target pieces do not cover the full logical parameter')

    pair_checks = len(sources) * len(targets)
    if pair_checks > budget.max_pair_checks:
        raise AffineTransferError(f'{pair_checks} candidate pairs exceeds max_pair_checks '
                                  f'{budget.max_pair_checks}')

    intersections: List[_Intersection] = []
    for frame in sorted(targets, key=lambda frame: (frame.rank, frame.box.origin, frame.box.extent)):
        filled: List[LogicalBox] = []
        for box, holders in sources:
            shared = frame.box.intersection(box)
            if shared.is_empty:
                continue
            filled.append(shared)
            intersections.append(_Intersection(frame, shared, holders))
        if not _covers_exactly(frame.box, filled):
            raise InvalidMapError(f'target rank {frame.rank} region {frame.box!r} would receive '
                                  f'{sum(part.volume for part in filled)} of its {frame.box.volume} elements')
    return _Prepared(logical_shape=logical_shape,
                     intersections=tuple(intersections),
                     source_pieces=len(sources),
                     target_pieces=len(targets),
                     pair_checks=pair_checks)


def _emit(prepared: _Prepared, choices: Sequence[Tuple[Tuple[LogicalBox, _Frame], ...]],
          budget: PlanBudget) -> TransferPlan:
    """Turn chosen (region, holder) pairs into merged segments and check the plan's budget."""
    candidates: List[TransferSegment] = []
    for item, parts in zip(prepared.intersections, choices):
        target = item.target
        for region, source in parts:
            candidates.append(
                TransferSegment(source_rank=source.rank,
                                source_offset=source.address(region),
                                source_strides=source.strides,
                                target_rank=target.rank,
                                target_offset=target.address(region),
                                target_strides=target.strides,
                                shape=region.extent))

    segments = _coalesce(candidates, budget)
    if len(segments) > budget.max_segments:
        raise AffineTransferError(f'{len(segments)} segments exceeds max_segments {budget.max_segments}')
    plan_bytes = _plan_bytes(segments, prepared.logical_shape)
    if plan_bytes > budget.max_plan_bytes:
        raise AffineTransferError(f'plan metadata is {plan_bytes} bytes, over max_plan_bytes '
                                  f'{budget.max_plan_bytes}')

    return TransferPlan(segments=segments,
                        logical_shape=prepared.logical_shape,
                        moved_elements=sum(segment.numel for segment in segments),
                        statistics=PlanStatistics(source_pieces=prepared.source_pieces,
                                                  target_pieces=prepared.target_pieces,
                                                  pair_checks=prepared.pair_checks,
                                                  segments_before_merge=len(candidates),
                                                  segments=len(segments),
                                                  plan_bytes=plan_bytes))


def _plan_bytes(segments: Tuple[TransferSegment, ...], logical_shape: Tuple[int, ...]) -> int:
    """Size of the serialized plan, so the budget is spent on what a caller actually ships."""
    payload = {'logical_shape': list(logical_shape), 'segments': [segment.to_dict() for segment in segments]}
    return len(json.dumps(payload, separators=(',', ':'), sort_keys=True).encode('utf-8'))


def _merge(one: TransferSegment, other: TransferSegment) -> Optional[TransferSegment]:
    """Join copies only when both shard address mappings continue along one axis."""
    if (one.source_rank != other.source_rank or one.target_rank != other.target_rank
            or one.source_strides != other.source_strides or one.target_strides != other.target_strides
            or len(one.shape) != len(other.shape)):
        return None
    for axis in range(len(one.shape)):
        if one.shape[:axis] + one.shape[axis + 1:] != other.shape[:axis] + other.shape[axis + 1:]:
            continue
        for first, second in ((one, other), (other, one)):
            if first.source_offset + first.shape[axis] * first.source_strides[axis] != second.source_offset:
                continue
            if first.target_offset + first.shape[axis] * first.target_strides[axis] != second.target_offset:
                continue
            shape = list(first.shape)
            shape[axis] += second.shape[axis]
            return TransferSegment(source_rank=first.source_rank,
                                   source_offset=first.source_offset,
                                   source_strides=first.source_strides,
                                   target_rank=first.target_rank,
                                   target_offset=first.target_offset,
                                   target_strides=first.target_strides,
                                   shape=tuple(shape))
    return None


def _coalesce(segments: List[TransferSegment], limits: PlanBudget) -> Tuple[TransferSegment, ...]:
    """Merge repeatedly in a fixed order until nothing joins.

    Deterministic, and identical for identical geometry whatever order the pairs were generated in.
    It does not claim the fewest segments: a shortest plan needs a search that has no business being
    on this path, and the counts that matter here are bounded by topology either way.
    """
    groups: Dict[Tuple[object, ...], List[TransferSegment]] = {}
    for segment in segments:
        key = (segment.source_rank, segment.target_rank, segment.source_strides, segment.target_strides,
               len(segment.shape))
        groups.setdefault(key, []).append(segment)

    folded = []
    checks = 0
    for group in groups.values():
        merged = sorted(group, key=lambda segment: (segment.target_offset, segment.source_offset, segment.shape))
        for _ in range(limits.max_merge_passes):
            remaining: List[TransferSegment] = []
            joined = False
            for segment in merged:
                for index, held in enumerate(remaining):
                    checks += 1
                    if checks > limits.max_merge_checks:
                        raise AffineTransferError(f'merge checks exceed max_merge_checks {limits.max_merge_checks}')
                    candidate = _merge(held, segment)
                    if candidate is not None:
                        remaining[index] = candidate
                        joined = True
                        break
                else:
                    remaining.append(segment)
            merged = remaining
            if not joined:
                break
        folded.extend(merged)
    return tuple(
        sorted(folded,
               key=lambda segment: (segment.target_rank, segment.target_offset, segment.source_rank, segment.
                                    source_offset, segment.shape, segment.source_strides, segment.target_strides)))

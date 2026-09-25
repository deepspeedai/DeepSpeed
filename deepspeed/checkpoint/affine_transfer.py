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

import json
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from .affine import AffinePiece, ParamAffineMap

__all__ = [
    'AffineTransferError', 'InvalidMapError', 'InvalidReplicaPolicyError', 'UnsupportedLayoutError', 'PlanBudget',
    'PlanStatistics', 'ReplicaCandidate', 'ReplicaSelectionRequest', 'TransferSegment', 'TransferPlan', 'plan_transfer'
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
    permutation would hide, and a permutation is not one box.
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
    for axis, (got, want, size) in enumerate(zip(piece.dest_strides, shard_strides, piece.shape)):
        if size > 1 and got != want:
            raise UnsupportedLayoutError(f'{where}: dest axis {axis} strides {got}, expected {want}')

    origin = _origin(piece.source_offset, logical_strides, logical_shape, where, 'source')
    box = LogicalBox(origin, tuple(piece.shape))
    if not LogicalBox(tuple([0] * len(logical_shape)), logical_shape).contains(box):
        raise InvalidMapError(f'{where}: covers {box!r}, outside the logical parameter of {logical_shape}')
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
    if any(len(tuple(shape)) != len(logical_shape) for shape in map_.shard_shapes.values()):
        raise UnsupportedLayoutError(
            f'{side}: a shard is described with a different number of axes than the logical parameter, so '
            'no axis correspondence exists to plan against')

    frames = []
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
    if len(frames) > limits.max_pieces_per_map:
        raise AffineTransferError(f'{side}: {len(frames)} pieces exceeds max_pieces_per_map '
                                  f'{limits.max_pieces_per_map}')
    _shards_tile(frames, side, limits)
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
        if len({frozenset(holder.locations) for holder in holders}) > 1:
            raise InvalidMapError(f'source: {box!r} is claimed by ranks {[h.rank for h in holders]} '
                                  'with different locations')
        for other in regions[index + 1:]:
            if not box.intersection(other).is_empty:
                raise InvalidMapError(f'source: {box!r} and {other!r} overlap without agreeing, so part of '
                                      'the parameter has two owners')
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
    if tuple(source_map.logical_shape) != tuple(target_map.logical_shape):
        raise InvalidMapError(f'source describes {tuple(source_map.logical_shape)} and target describes '
                              f'{tuple(target_map.logical_shape)}')

    sources = _sources(_normalized(source_map, 'source', budget), budget)
    targets = _normalized(target_map, 'target', budget)

    pair_checks = len(sources) * len(targets)
    if pair_checks > budget.max_pair_checks:
        raise AffineTransferError(f'{pair_checks} candidate pairs exceeds max_pair_checks '
                                  f'{budget.max_pair_checks}')

    pending: List[Tuple[_Frame, LogicalBox, Tuple[_Frame, ...]]] = []
    for frame in sorted(targets, key=lambda frame: (frame.rank, frame.box.origin, frame.box.extent)):
        filled: List[LogicalBox] = []
        for box, holders in sources:
            shared = frame.box.intersection(box)
            if shared.is_empty:
                continue
            filled.append(shared)
            pending.append((frame, shared, holders))
        if not _covers_exactly(frame.box, filled):
            raise InvalidMapError(f'target rank {frame.rank} region {frame.box!r} would receive '
                                  f'{sum(part.volume for part in filled)} of its {frame.box.volume} elements')

    candidates: List[TransferSegment] = []
    for target, shared, holders in pending:
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
        candidates.append(
            TransferSegment(source_rank=source.rank,
                            source_offset=source.address(shared),
                            source_strides=source.strides,
                            target_rank=target.rank,
                            target_offset=target.address(shared),
                            target_strides=target.strides,
                            shape=shared.extent))

    segments = _coalesce(candidates, budget)
    if len(segments) > budget.max_segments:
        raise AffineTransferError(f'{len(segments)} segments exceeds max_segments {budget.max_segments}')
    plan_bytes = _plan_bytes(segments)
    if plan_bytes > budget.max_plan_bytes:
        raise AffineTransferError(f'plan metadata is {plan_bytes} bytes, over max_plan_bytes '
                                  f'{budget.max_plan_bytes}')

    return TransferPlan(segments=segments,
                        logical_shape=tuple(target_map.logical_shape),
                        moved_elements=sum(segment.numel for segment in segments),
                        statistics=PlanStatistics(source_pieces=len(sources),
                                                  target_pieces=len(targets),
                                                  pair_checks=pair_checks,
                                                  segments_before_merge=len(candidates),
                                                  segments=len(segments),
                                                  plan_bytes=plan_bytes))


def _plan_bytes(segments: Tuple[TransferSegment, ...]) -> int:
    """Size of the serialized plan, so the budget is spent on what a caller actually ships."""
    return len(json.dumps([segment.to_dict() for segment in segments], separators=(',', ':'), sort_keys=True))


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
    for group in groups.values():
        merged = sorted(group, key=lambda segment: (segment.target_offset, segment.source_offset, segment.shape))
        for _ in range(limits.max_merge_passes):
            remaining: List[TransferSegment] = []
            joined = False
            for segment in merged:
                for index, held in enumerate(remaining):
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

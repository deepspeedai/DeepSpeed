# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Move a parameter between two shardings by executing a plan from ``affine_transfer``.

The plan names, for each copy, a region of one source shard and the region of one target shard that
should receive it. This module turns those names into data movement: pack out of the source buffer,
across a transport group, through a staging buffer, and into the target buffer. The full logical
parameter is never formed anywhere, which was the point of planning instead of converting.

Three things are deliberately not decided here.

*Endpoints.* Which process actually holds a plan location is supplied by the caller. The plan's two
rank namespaces are not one world size: a source rank 0 has no claim to be the process that is target
rank 0, and folding the two together is how a transfer reads one rank's shard as another's.

*Budget.* How many bytes may be in flight is the caller's, because what fits depends on what else is
resident on the device. A budget that is exhausted is an error, never a silently larger transfer.

*Dtype.* Passed in rather than read from the buffers, because a process that holds no buffer at all
still has to reach the same conclusion as the ones that do -- see ``plan_digest``.

The protocol is: agree on a digest of the schedule before any communication, then work in rounds. In
each round every rank arms all of its receives first and only then posts its sends, so two ranks that
each have something for the other cannot both be waiting on the other's send. That ordering is the
entire deadlock argument, and it only holds because the digest made every rank compute the same
rounds. Returns only once every payload it posted has landed, so a caller can read the target buffers
as soon as it is given control.

Staging is an artifact of the current transport, not of affine transfer. Torch point-to-point
operations move one contiguous tensor per peer per message, so a round whose regions are strided or come
from several segments is packed through a per-peer arena, and ``StagingBudget`` bounds that arena; a
round that is one region already contiguous in its buffer is sent or received in place.
A zero-copy transport such as GPUDirect RDMA would instead take each round's ``(offset, length)`` pairs
as a scatter/gather list over registered memory: that removes the arena copy, but needs a registration
lifetime this API has no place for. Expect the wire contract -- one packed message per peer per round --
to change when such a path lands.

``TransferWindow`` is that registration for peers that can address each other's memory (one node over
PCIe, NVLink or Xe Link). Buffers carved from a window are published to the group, and an operation whose
source or target lies in a window is copied in place by one process -- the target pulls, or the source
pushes -- with no staging, no peer message and no budget. Everything else keeps the protocol above.

One device per process, with explicit transport-group rank bindings, synchronous on return. No CUDA graphs, no overlap with compute,
and ``copy_`` for every region by default: graphs and overlap change the schedule, which is the last thing
a first version of a data mover should be perturbing. A prepared transfer may replace how a round's
copies run (``prepare_transfers(..., copier=affine_transfer_copy.fused_copies)``), never which regions it
copies or how rounds are ordered.
"""

import contextlib
import gc
import hashlib
import math
import sys
import weakref
import zlib
from bisect import bisect_right
from dataclasses import dataclass
from itertools import count, product
from typing import AbstractSet, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from deepspeed.accelerator import get_accelerator

from .affine_geometry import address_range, dot
from .affine_transfer import TransferPlan, TransferSegment

__all__ = [
    'AffineTransferExecutionError', 'TransferEndpoints', 'StagingBudget', 'TransferStats', 'TransferRequest',
    'plan_digest', 'execute_transfer', 'execute_transfers', 'prepare_transfers', 'PreparedTransfer', 'ScheduledRound',
    'ProcessSchedule', 'schedule_transfers', 'TransferWindow'
]


class AffineTransferExecutionError(ValueError):
    """Wrong enough that no data should have moved."""


@dataclass(frozen=True)
class TransferEndpoints:
    """Which process holds each location named by the plan. ``source`` covers source ranks, ``target``
    covers target ranks, and the two are separate mappings for the reason given above."""
    source: Mapping[int, int]
    target: Mapping[int, int]


@dataclass(frozen=True)
class StagingBudget:
    """Bytes a rank may stage per peer, in each direction, at one time.

    A property of the current torch point-to-point transport rather than of the plan; see the module
    docstring."""
    bytes_per_peer: int = 8 << 20


@dataclass(frozen=True)
class TransferStats:
    segments: int
    chunks: int
    rounds: int
    local_copy_bytes: int
    sent_bytes: int
    received_bytes: int
    mapped_bytes: int = 0


class TransferWindow:
    """Device memory that every process of the transfer group can address.

    Construction is collective over ``group``; processes may ask for different ``nbytes`` and every window
    gets the largest. Carve long-lived shards with ``tensor`` and pass the window to every call:
    registration is paid once, not per transfer. ``direction`` decides who copies when both the source
    and the target of an operation lie in a window; which one is faster depends on the interconnect, so
    it is the caller's.

    Pass the same window on every process or on none. Windows that disagree are refused, but a window
    passed on some processes only, or one whose construction fails on one process (out of memory, say),
    leaves the other processes waiting instead of refusing.
    """
    ALIGNMENT = 512
    # The newest window identity this process has seen; see __init__.
    _created = 0
    # Every window alive on this process, passed to a transfer or not; see _registered_locations.
    _live: 'weakref.WeakSet[TransferWindow]' = weakref.WeakSet()

    def __init__(self, nbytes: int, group: Optional[object] = None, direction: str = 'pull'):
        import deepspeed.comm as dist
        if direction not in ('pull', 'push'):
            raise AffineTransferExecutionError("window direction must be 'pull' or 'push'")
        self.group = group
        self.direction = direction
        # Symmetric memory needs one size on every process, while shard sizes differ between processes. The
        # same gather names the window's processes and gives it an identity the whole group shares, newer than
        # any window any of them made before, so that windows passed on different processes can be told apart.
        world_size = dist.get_world_size(group=group)
        local = torch.tensor([nbytes, TransferWindow._created, dist.get_rank()],
                             dtype=torch.int64,
                             device=get_accelerator().current_device_name())
        seen = torch.empty((world_size, 3), dtype=torch.int64, device=local.device)
        dist.all_gather(list(seen.unbind(0)), local, group=group)
        sizes, created, members = zip(*seen.cpu().tolist())
        self.identity = max(created) + 1
        TransferWindow._created = self.identity
        self.members = tuple(members)
        self.nbytes = -(-max(sizes) // self.ALIGNMENT) * self.ALIGNMENT
        self.pool, self._handle = dist.symm_mem_rendezvous(self.nbytes, torch.uint8, group)
        TransferWindow._live.add(self)
        self._used = 0
        self._peers: Dict[Tuple[int, torch.dtype], torch.Tensor] = {}

    def tensor(self, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        """A new shard inside the window; never freed before the window."""
        nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        start = self._used
        if start + nbytes > self.nbytes:
            raise AffineTransferExecutionError(f'window holds {self.nbytes - start} more bytes, {nbytes} requested')
        self._used = -(-(start + nbytes) // self.ALIGNMENT) * self.ALIGNMENT
        return self.pool[start:start + nbytes].view(dtype).view(tuple(shape))

    def offset_of(self, buffer: torch.Tensor) -> Optional[int]:
        """Element offset of ``buffer`` in this process's window, or None if it lies elsewhere."""
        if buffer.device != self.pool.device:
            return None
        start = buffer.data_ptr() - self.pool.data_ptr()
        size = buffer.element_size()
        if start < 0 or start % size or start + buffer.numel() * size > self.nbytes:
            return None
        return start // size

    def peer(self, process: int, dtype: torch.dtype) -> torch.Tensor:
        """``process``'s window as flat elements of ``dtype``, mapped into this process."""
        key = (process, dtype)
        if key not in self._peers:
            import deepspeed.comm as dist
            numel = self.nbytes // torch.empty((), dtype=dtype).element_size()
            self._peers[key] = dist.symm_mem_peer_tensor(self._handle, process, numel, dtype)
        return self._peers[key]


@dataclass(frozen=True)
class TransferRequest:
    """One parameter; ``key`` must identify it identically on every process."""
    key: str
    plan: Optional[TransferPlan]
    source_buffers: Mapping[int, torch.Tensor]
    target_buffers: Mapping[int, torch.Tensor]
    preflight_error: Optional[str] = None


@dataclass(frozen=True)
class _Operation:
    """One chunk of one segment, resolved to a pair of processes."""
    segment: TransferSegment
    chunk: int
    chunk_shape: Tuple[int, ...]
    source_offset: int
    target_offset: int
    from_process: int
    to_process: int
    elements: int

    @property
    def key(self) -> Tuple:
        return (self.segment.target_rank, self.segment.target_offset, self.segment.source_rank,
                self.segment.source_offset, self.chunk)

    @property
    def payload_bytes(self) -> int:
        return self.elements * self._element_size

    _element_size: int = 0


def _chunks(segment: TransferSegment, max_elements: int):
    """Tile a strided rectangle by its packed payload, retaining both address spaces."""
    if segment.numel <= max_elements:
        yield segment
        return
    tile = [1] * len(segment.shape)
    remaining = max_elements
    for axis in range(len(tile) - 1, -1, -1):
        tile[axis] = min(segment.shape[axis], remaining)
        remaining //= tile[axis]
    origins = [range(0, extent, step) for extent, step in zip(segment.shape, tile)]
    for origin in product(*origins):
        shape = tuple(min(step, extent - low) for step, extent, low in zip(tile, segment.shape, origin))
        yield TransferSegment(source_rank=segment.source_rank,
                              source_offset=segment.source_offset + dot(origin, segment.source_strides),
                              source_strides=segment.source_strides,
                              target_rank=segment.target_rank,
                              target_offset=segment.target_offset + dot(origin, segment.target_strides),
                              target_strides=segment.target_strides,
                              shape=shape)


def build_operations(plan: TransferPlan, endpoints: TransferEndpoints, budget: StagingBudget,
                     element_size: int) -> List[_Operation]:
    """The complete schedule, as a list every rank derives identically."""
    if element_size <= 0 or budget.bytes_per_peer < element_size:
        raise AffineTransferExecutionError('the staging budget must hold at least one element')
    operations: List[_Operation] = []
    for segment in plan.segments:
        _validate_segment(segment, len(plan.logical_shape))
        for index, chunk in enumerate(_chunks(segment, budget.bytes_per_peer // element_size)):
            operation = _Operation(segment=segment,
                                   chunk=index,
                                   chunk_shape=chunk.shape,
                                   source_offset=chunk.source_offset,
                                   target_offset=chunk.target_offset,
                                   from_process=endpoints.source[segment.source_rank],
                                   to_process=endpoints.target[segment.target_rank],
                                   elements=chunk.numel,
                                   _element_size=element_size)
            operations.append(operation)
    operations.sort(key=lambda operation: operation.key)
    return operations


def plan_digest(plan: TransferPlan, endpoints: TransferEndpoints, budget: StagingBudget, element_size: int) -> int:
    """A checksum over the schedule: geometry, addressing, endpoints and budget.

    Deliberately not over buffer contents -- those differ by design. This is the thing every rank must
    be able to compute before it owns any data, which is why dtype is an argument and not an
    inspection.
    """
    payload = [(segment.source_rank, segment.target_rank, segment.source_offset, segment.target_offset, segment.shape,
                segment.source_strides, segment.target_strides) for segment in plan.segments]
    framed = {
        'schema': 3,
        'segments': payload,
        'logical_shape': plan.logical_shape,
        'source_endpoints': dict(sorted(endpoints.source.items())),
        'target_endpoints': dict(sorted(endpoints.target.items())),
        'bytes_per_peer': budget.bytes_per_peer,
        'element_size': element_size,
    }
    return zlib.crc32(repr(sorted(framed.items(), key=lambda item: item[0])).encode())


def _validate_segment(segment: TransferSegment, ndim: int) -> None:
    if len(segment.shape) != ndim:
        raise AffineTransferExecutionError('segment dimensionality does not match its logical shape')
    for size in segment.shape:
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise AffineTransferExecutionError('segment extents must be positive integers')
    for offset, strides in ((segment.source_offset, segment.source_strides), (segment.target_offset,
                                                                              segment.target_strides)):
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise AffineTransferExecutionError('segment offsets must be non-negative integers')
        if len(strides) != ndim:
            raise AffineTransferExecutionError('segment stride dimensionality does not match its shape')
        for size, stride in zip(segment.shape, strides):
            if not isinstance(stride, int) or isinstance(stride, bool) or stride < 0 or (size > 1 and stride == 0):
                raise AffineTransferExecutionError('segment strides must be positive on non-singleton axes')


def _validate_buffers(plan: TransferPlan, source_buffers: Mapping[int, torch.Tensor],
                      target_buffers: Mapping[int, torch.Tensor], dtype: torch.dtype, device: torch.device,
                      endpoints: TransferEndpoints, rank: int) -> None:
    for which, buffers, bindings in (('source', source_buffers, endpoints.source), ('target', target_buffers,
                                                                                    endpoints.target)):
        for location, buffer in buffers.items():
            if bindings.get(location) != rank:
                raise AffineTransferExecutionError(f'{which} location {location} is not owned by this process')
            _flat(location, buffers, dtype)
            if buffer.device != device:
                raise AffineTransferExecutionError(
                    f'{which} location {location}: buffer is on {buffer.device}, transport is on {device}')
    for segment in plan.segments:
        _validate_segment(segment, len(plan.logical_shape))
        for which, location, offset, strides, buffers, bindings in (('source', segment.source_rank,
                                                                     segment.source_offset, segment.source_strides,
                                                                     source_buffers, endpoints.source),
                                                                    ('target', segment.target_rank,
                                                                     segment.target_offset, segment.target_strides,
                                                                     target_buffers, endpoints.target)):
            if location not in bindings:
                raise AffineTransferExecutionError(f'{which} location {location} has no endpoint binding')
            if bindings[location] != rank:
                continue
            if location not in buffers:
                raise AffineTransferExecutionError(f'{which} location {location} is owned here but holds no buffer')
            _, reach = address_range(offset, segment.shape, strides)
            if reach >= buffers[location].numel():
                raise AffineTransferExecutionError(
                    f'{which} of {segment!r} reaches element {reach} of a {buffers[location].numel()}-element buffer')


def _buffers_alias(requests: Sequence[TransferRequest]) -> bool:
    """Whether a target buffer shares storage with a source or with another target of the batch.

    Usually none does, and then no region can alias. Otherwise ``_validate_aliases`` decides from the
    regions this process actually reads and writes."""
    sources = [
        _data_range(buffer) for request in requests for buffer in request.source_buffers.values() if buffer.numel()
    ]
    targets = [
        _data_range(buffer) for request in requests for buffer in request.target_buffers.values() if buffer.numel()
    ]
    for low, high in targets:
        if any(low < source_high and source_low < high for source_low, source_high in sources):
            return True
    for index, (low, high) in enumerate(targets):
        if any(low < other_high and other_low < high for other_low, other_high in targets[:index]):
            return True
    return False


def _validate_aliases(
    requests: Sequence[TransferRequest],
    lane: '_Lane',
    element_size: int,
    rank: int,
    window_operations: Sequence[Tuple[int, '_Operation']] = ()) -> List[Tuple[int, '_Operation']]:
    """Refuse a region this process writes that it also reads, or writes twice; return the local copies to run.

    A local copy whose source and target are the same memory at the same strides moves nothing. That is
    how a relay rank holds its band: as a view of its own destination, which it sends to its peers and
    keeps in place. Such a copy is dropped and its region counts as read, so any other write into it is
    still refused. Regions are compared by the bytes they span, so strided regions that interleave
    without sharing an element are refused as well.

    ``window_operations`` are the window copies between this process and a peer. They are not in the
    lane, but they run while this process is in its rounds, whichever side copies: a source region here
    counts as read and a target region here as written.
    """

    def extent(buffers, location, offset, shape, strides) -> Tuple[int, int]:
        start = buffers[location].data_ptr()
        low, high = address_range(offset, shape, strides)
        return start + low * element_size, start + (high + 1) * element_size

    def source_extent(item, operation):
        return extent(requests[item].source_buffers, operation.segment.source_rank, operation.source_offset,
                      operation.chunk_shape, operation.segment.source_strides)

    def target_extent(item, operation):
        return extent(requests[item].target_buffers, operation.segment.target_rank, operation.target_offset,
                      operation.chunk_shape, operation.segment.target_strides)

    reads, writes, copies = [], [], []
    for item, operation in lane.local_copies:
        read, write = source_extent(item, operation), target_extent(item, operation)
        reads.append(read)
        in_place = read[0] == write[0] and all(size <= 1 or source == target for size, source, target in zip(
            operation.chunk_shape, operation.segment.source_strides, operation.segment.target_strides))
        if not in_place:
            writes.append(write)
            copies.append((item, operation))
    for rounds, side, regions in ((lane.send_rounds, source_extent, reads), (lane.recv_rounds, target_extent, writes)):
        for peer_rounds in rounds.values():
            for entries, _ in peer_rounds:
                regions.extend(side(item, operation) for item, operation, _ in entries)
    for item, operation in window_operations:
        if operation.from_process == rank:
            reads.append(source_extent(item, operation))
        if operation.to_process == rank:
            writes.append(target_extent(item, operation))

    writes.sort()
    for (_, high), (low, _) in zip(writes, writes[1:]):
        if low < high:
            raise AffineTransferExecutionError('two targets overlap within the transfer batch')
    highs = [high for _, high in writes]
    for low, high in reads:
        index = bisect_right(highs, low)
        if index < len(writes) and writes[index][0] < high:
            raise AffineTransferExecutionError('a source and a target overlap within the transfer batch')
    return copies


def execute_transfer(plan: TransferPlan,
                     source_buffers: Mapping[int, torch.Tensor],
                     target_buffers: Mapping[int, torch.Tensor],
                     endpoints: TransferEndpoints,
                     dtype: torch.dtype,
                     budget: StagingBudget = StagingBudget(),
                     group: Optional[object] = None,
                     window: Optional[TransferWindow] = None) -> TransferStats:
    """Execute one plan with the same preflight and completion contract as a batch.

    Bindings use transport-group ranks. Buffers must be packed views on the owning process;
    their storage offsets are retained. Rejection on any participant precedes all target writes.
    Transport failures after payload begins may leave partially written targets.
    """
    request = TransferRequest('parameter', plan, source_buffers, target_buffers)
    return execute_transfers([request], endpoints, dtype, budget, group, window)


def _packed_rounds(indexed_operations: Sequence[Tuple[int, _Operation]], max_elements: int):
    rounds = []
    for item, operation in indexed_operations:
        if not rounds or rounds[-1][1] + operation.elements > max_elements:
            rounds.append(([], 0))
        entries, used = rounds[-1]
        entries.append((item, operation, used))
        rounds[-1] = (entries, used + operation.elements)
    return rounds


def _is_packed(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """Whether a region is one run of consecutive elements in row-major order.

    This is ``Tensor.is_contiguous`` for the ``as_strided`` view ``_region`` builds over a packed
    shard, decided from geometry alone: axes of extent 1 place no constraint on their stride.
    """
    if 0 in shape:
        return True
    expected = 1
    for size, stride in zip(reversed(shape), reversed(strides)):
        if size != 1:
            if stride != expected:
                return False
            expected *= size
    return True


def _round_is_direct(entries, source: bool, registered: Optional[AbstractSet] = None) -> bool:
    # A view can replace staging only when the transport sees the same packed element order, and only
    # when it is not window memory; see _registered_locations.
    if len(entries) != 1:
        return False
    item, operation, _ = entries[0]
    location = operation.segment.source_rank if source else operation.segment.target_rank
    if registered is not None and (item, source, location) in registered:
        return False
    return _is_packed(operation.chunk_shape,
                      operation.segment.source_strides if source else operation.segment.target_strides)


def _direct_payload(entries, requests, dtype, source):
    item, operation, _ = entries[0]
    request, segment = requests[item], operation.segment
    if source:
        flat = _flat(segment.source_rank, request.source_buffers, dtype)
        offset, strides = operation.source_offset, segment.source_strides
    else:
        flat = _flat(segment.target_rank, request.target_buffers, dtype)
        offset, strides = operation.target_offset, segment.target_strides
    return _region(flat, offset, operation.chunk_shape, strides).view(-1)


def _indexed_operations(plans: Sequence[TransferPlan], endpoints: TransferEndpoints, budget: StagingBudget,
                        element_size: int) -> List[Tuple[int, _Operation]]:
    """Every chunk of a batch in execution order, tagged with the index of its plan.

    Many parameters share one plan object; its geometry is built once per call.
    """
    built: Dict[int, List[_Operation]] = {}
    indexed: List[Tuple[int, _Operation]] = []
    for item, plan in enumerate(plans):
        if id(plan) not in built:
            built[id(plan)] = build_operations(plan, endpoints, budget, element_size)
        indexed.extend((item, operation) for operation in built[id(plan)])
    return indexed


# Per request or plan, the source and the target locations whose shards lie in a window.
_WindowLocations = Sequence[Tuple[AbstractSet[int], AbstractSet[int]]]


def _window_offsets(requests: Sequence[TransferRequest], window: TransferWindow) -> Dict[Tuple, int]:
    """This process's shards that lie in the window, keyed so that every process can look them up."""
    offsets = {}
    for request in requests:
        for side, buffers in (('source', request.source_buffers), ('target', request.target_buffers)):
            for location, buffer in buffers.items():
                if isinstance(buffer, torch.Tensor) and buffer.is_contiguous():
                    offset = window.offset_of(buffer)
                    if offset is not None:
                        offsets[(request.key, side, location)] = offset
    return offsets


def _registered_locations(requests: Sequence[TransferRequest]) -> Optional[AbstractSet[Tuple[int, bool, int]]]:
    """``(item, source?, location)`` of this process's buffers that lie in any window, or None if none do.

    NCCL leaves a send or receive undefined when one side's buffer is registered with the communicator
    and the other side's is not, and hangs on it between hosts. A peer cannot know whether this process's
    buffer is registered, so a message never starts or ends in window memory: such a region is staged
    through the arena on this side, which is plain memory, whatever the peer does with its own side.
    """
    spans = [(window.pool.device, window.pool.data_ptr(), window.nbytes) for window in list(TransferWindow._live)]
    if not spans:
        return None
    found = set()
    for item, request in enumerate(requests):
        for source, buffers in ((True, request.source_buffers), (False, request.target_buffers)):
            for location, buffer in buffers.items():
                if not isinstance(buffer, torch.Tensor):
                    continue
                address = buffer.data_ptr()
                if any(buffer.device == device and start <= address < start + nbytes
                       for device, start, nbytes in spans):
                    found.add((item, source, location))
    return found or None


def _window_locations(requests: Sequence[TransferRequest], offsets: Mapping[Tuple, int]) -> _WindowLocations:
    """Per request, the source and the target locations whose shards lie in a window on any process."""
    sides = {request.key: (set(), set()) for request in requests}
    for key, side, location in offsets:
        if key in sides:
            sides[key][0 if side == 'source' else 1].add(location)
    return [sides[request.key] for request in requests]


def _mapped_direction(item: int, operation: _Operation, window_locations: Optional[_WindowLocations],
                      both: Optional[str]) -> Optional[str]:
    """``'pull'``, ``'push'`` or None for a cross-process operation, identically on every process.

    ``both`` is the window's direction, for an operation whose source and target both lie in a window.
    """
    if window_locations is None or operation.from_process == operation.to_process:
        return None
    sources, targets = window_locations[item]
    pull = operation.segment.source_rank in sources
    push = operation.segment.target_rank in targets
    if pull and push:
        return both
    if pull:
        return 'pull'
    return 'push' if push else None


def _staggered(mapped_copies, rank: int):
    """This process's window copies, starting at a different peer than its neighbours.

    Every copier walking its peers in the same order makes them all read from (or write to) one peer at a
    time, which serializes on that peer's link while the others idle. Rotating the start by rank spreads
    the copiers over the peers; within a peer the plan order is kept.
    """

    def peer_of(entry):
        _, operation, direction = entry
        return operation.from_process if direction == 'pull' else operation.to_process

    peers = sorted({peer_of(entry) for entry in mapped_copies})
    if not peers:
        return []
    start = rank % len(peers)
    position = {peer: (index - start) % len(peers) for index, peer in enumerate(peers)}
    return sorted(mapped_copies, key=lambda entry: position[peer_of(entry)])


@dataclass
class _Lane:
    """One process's share of a batch schedule, before any buffer is touched."""
    local_copies: List[Tuple[int, _Operation]]
    recv_rounds: Dict[int, list]
    send_rounds: Dict[int, list]
    recv_direct: Dict[int, List[bool]]
    send_direct: Dict[int, List[bool]]
    # Elements of the staging arena for each peer direction that stages at least one round.
    recv_arena: Dict[int, int]
    send_arena: Dict[int, int]


def _lanes(indexed: Sequence[Tuple[int, _Operation]],
           processes: Sequence[int],
           max_elements: int,
           window_locations: Optional[_WindowLocations],
           registered: Optional[Dict[int, AbstractSet]] = None) -> Dict[int, _Lane]:
    """Each process's local copies and staged rounds; ``registered`` gives, per process asked for, the
    regions of that process's buffers in window memory, which only its own process knows; see
    _registered_locations. ``schedule_transfers`` is only asked for this process's own, and so passes
    one entry; a single set passed for the whole group works unchanged."""
    local = {process: [] for process in processes}
    incoming: Dict[int, Dict[int, List[Tuple[int, _Operation]]]] = {process: {} for process in processes}
    outgoing: Dict[int, Dict[int, List[Tuple[int, _Operation]]]] = {process: {} for process in processes}
    for item, operation in indexed:
        if operation.from_process == operation.to_process:
            if operation.from_process in local:
                local[operation.from_process].append((item, operation))
            continue
        # Copied in place through the window, so it is neither a message nor staged; who copies does not matter.
        if _mapped_direction(item, operation, window_locations, 'pull') is not None:
            continue
        if operation.to_process in incoming:
            incoming[operation.to_process].setdefault(operation.from_process, []).append((item, operation))
        if operation.from_process in outgoing:
            outgoing[operation.from_process].setdefault(operation.to_process, []).append((item, operation))

    def arenas(rounds_by_peer, direct_by_peer):
        sizes = {}
        for peer, rounds in rounds_by_peer.items():
            staged = [count for (_, count), direct in zip(rounds, direct_by_peer[peer]) if not direct]
            if staged:
                sizes[peer] = max(staged)
        return sizes

    lanes = {}
    for process in processes:
        # A window buffer is only its owner's business, so each process reads its own entry, or, for a
        # flat set, whatever the caller supplied for every process.
        own = None if registered is None else registered.get(process) if isinstance(registered, dict) else registered
        recv_rounds = {peer: _packed_rounds(ops, max_elements) for peer, ops in incoming[process].items()}
        send_rounds = {peer: _packed_rounds(ops, max_elements) for peer, ops in outgoing[process].items()}
        recv_direct = {
            peer: [_round_is_direct(entries, False, own) for entries, _ in rounds]
            for peer, rounds in recv_rounds.items()
        }
        send_direct = {
            peer: [_round_is_direct(entries, True, own) for entries, _ in rounds]
            for peer, rounds in send_rounds.items()
        }
        lanes[process] = _Lane(local[process], recv_rounds, send_rounds, recv_direct, send_direct,
                               arenas(recv_rounds, recv_direct), arenas(send_rounds, send_direct))
    return lanes


@dataclass(frozen=True)
class ScheduledRound:
    """One message to or from a peer: ``elements`` packed from ``chunks`` plan chunks.

    ``direct`` means this side sends or receives the region in place rather than through its arena.
    """
    elements: int
    chunks: int
    direct: bool


@dataclass(frozen=True)
class ProcessSchedule:
    """What one process does for a batch, exactly as ``execute_transfers`` or a prepared transfer runs it.

    ``sends`` and ``receives`` map each peer to its rounds in order. ``send_arena_elements`` and
    ``receive_arena_elements`` give the staging arena, in elements, that each peer direction
    allocates; peers whose rounds are all direct allocate none. ``rounds`` is the most rounds this
    process has with any one peer, which ``TransferStats.rounds`` reports.
    """
    process: int
    sends: Mapping[int, Tuple[ScheduledRound, ...]]
    receives: Mapping[int, Tuple[ScheduledRound, ...]]
    send_arena_elements: Mapping[int, int]
    receive_arena_elements: Mapping[int, int]
    local_copy_chunks: int
    local_copy_elements: int
    rounds: int


def _scheduled_rounds(rounds_by_peer, direct_by_peer) -> Dict[int, Tuple[ScheduledRound, ...]]:
    return {
        peer:
        tuple(
            ScheduledRound(count, len(entries), direct)
            for (entries, count), direct in zip(rounds, direct_by_peer[peer]))
        for peer, rounds in sorted(rounds_by_peer.items())
    }


def schedule_transfers(plans: Sequence[TransferPlan],
                       endpoints: TransferEndpoints,
                       budget: StagingBudget,
                       element_size: int,
                       process: Optional[int] = None,
                       window_locations: Optional[_WindowLocations] = None,
                       window_registered: Optional[Dict[int, AbstractSet]] = None) -> Dict[int, ProcessSchedule]:
    """The schedule ``execute_transfers`` would run for a batch, without buffers or communication.

    ``plans`` are the batch's plans in request order, since that order fixes how chunks are packed
    into rounds. Returns a ``ProcessSchedule`` for ``process``, or for every process the endpoints
    name. ``execute_transfers`` builds its own rounds and arenas from the same computation, so a
    cost model or policy reading this sees the messages and staging memory the executor will use.
    It sees no buffers, so a local copy that ``execute_transfers`` drops as in place (source and
    target the same memory at the same strides) still counts in ``local_copy_chunks`` and
    ``local_copy_elements``; ``TransferStats.local_copy_bytes`` does not count it.

    A batch run with a ``TransferWindow`` copies some operations in place instead, which only the
    buffers can tell. ``window_locations`` gives, per plan, the source and the target locations
    whose shards lie in the window on any process; operations between two processes that touch one
    are left out of the rounds, as the executor leaves them out. They are not messages, so this
    schedule does not report them.

    ``window_registered`` is, per process asked for, the set of ``(plan index, source side?,
    location)`` of that process's buffers that lie in a window -- pass ``_registered_locations``
    with the same requests and process. A message that starts or ends in such memory is staged
    through the arena on that side, exactly as the executor stages it, so this schedule reports
    ``direct=False`` and the arena the executor allocates for it.
    """
    if not isinstance(element_size, int) or isinstance(element_size, bool) or element_size <= 0:
        raise AffineTransferExecutionError('element_size must be a positive integer')
    if (not isinstance(budget.bytes_per_peer, int) or isinstance(budget.bytes_per_peer, bool)
            or budget.bytes_per_peer < element_size):
        raise AffineTransferExecutionError('the staging budget must hold at least one element')
    for plan in plans:
        for segment in plan.segments:
            if segment.source_rank not in endpoints.source or segment.target_rank not in endpoints.target:
                raise AffineTransferExecutionError(f'{segment!r} names a location with no endpoint binding')
    if window_locations is not None and len(window_locations) != len(plans):
        raise AffineTransferExecutionError('window_locations must name one pair of location sets per plan')
    indexed = _indexed_operations(plans, endpoints, budget, element_size)
    named = set(endpoints.source.values()) | set(endpoints.target.values())
    if process is not None and process not in named:
        raise AffineTransferExecutionError(f'process {process} is not named by the endpoints')
    processes = sorted(named) if process is None else [process]
    registered = None if window_registered is None else {
        process: window_registered.get(process)
        for process in processes
    }
    schedules = {}
    lanes = _lanes(indexed, processes, budget.bytes_per_peer // element_size, window_locations, registered)
    for member, lane in lanes.items():
        queues = list(lane.recv_rounds.values()) + list(lane.send_rounds.values())
        schedules[member] = ProcessSchedule(process=member,
                                            sends=_scheduled_rounds(lane.send_rounds, lane.send_direct),
                                            receives=_scheduled_rounds(lane.recv_rounds, lane.recv_direct),
                                            send_arena_elements=dict(sorted(lane.send_arena.items())),
                                            receive_arena_elements=dict(sorted(lane.recv_arena.items())),
                                            local_copy_chunks=len(lane.local_copies),
                                            local_copy_elements=sum(operation.elements
                                                                    for _, operation in lane.local_copies),
                                            rounds=max((len(queue) for queue in queues), default=0))
    return schedules


@torch.no_grad()
def execute_transfers(requests: Sequence[TransferRequest],
                      endpoints: TransferEndpoints,
                      dtype: torch.dtype,
                      budget: StagingBudget = StagingBudget(),
                      group: Optional[object] = None,
                      window: Optional[TransferWindow] = None) -> TransferStats:
    """Move a frozen parameter batch with collective preflight and bounded peer messages.

    Each direction stages at most ``bytes_per_peer`` per active peer. All requests and staging
    allocations are checked before payload, including read/write aliases between parameters.
    The caller stream's final unpack and local copies complete before this function returns.
    With a ``window`` (the same one on every process), operations touching window shards are copied
    in place and the call ends with a group barrier, so no peer reads or writes them after return.
    A batch moved repeatedly can be prepared once with ``prepare_transfers`` instead.
    """
    agreed = _agree(requests, endpoints, dtype, budget, group, keep=False, window=window)
    lane = agreed.lane
    # The local and the mapped copy views were resolved inside the agreement, so a map into a peer's
    # window that fails on one process after everyone else agreed refused the batch before a byte
    # moved, as the prepare path does; what is built here is each round's views, made when the move
    # reaches them, on top of the copy views the round before may still read.
    rounds = _round_views(lane, agreed.recv_arenas, agreed.send_arenas, requests, dtype)
    sent, received, copied = _run_rounds(rounds, _round_count(lane), agreed.local_copies, group, agreed.element_size,
                                         agreed.stream_ordered)
    _run_mapped(agreed.mapped_copies, None, agreed.mapped_total, group, agreed.stream_ordered)
    return TransferStats(agreed.segments, agreed.chunks, _round_count(lane), copied, sent, received,
                         agreed.mapped_bytes)


@dataclass(frozen=True)
class _StagedRound:
    """One peer's message in one round, and the copies that fill or empty it."""
    peer: int
    # What goes on the wire: a slice of the peer's arena, or the region itself when it is sent in place.
    payload: torch.Tensor
    # ``(destination, source)`` view pairs: packs into the arena before a send, unpacks after a receive.
    # Empty when the round is sent or received in place.
    copies: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


# Turns a list of ``(destination, source)`` region copies into the callable a prepared run invokes for it.
Copier = Callable[[Sequence[Tuple[torch.Tensor, torch.Tensor]]], Callable[[], None]]


@dataclass
class _Agreement:
    """A batch every process agreed to move, and this process's share of it."""
    lane: Optional[_Lane]
    # The lane's local copies, less those dropped as in place.
    # The lane's local copies, less those dropped as in place; used to resolve the views below for both paths.
    copies: List[Tuple[int, _Operation]]
    recv_arenas: Dict[int, torch.Tensor]
    send_arenas: Dict[int, torch.Tensor]
    # Built before the agreement only when a prepared handle keeps them.
    rounds: List[Tuple[List[_StagedRound], List[_StagedRound]]]
    # The copy views in both paths, resolved in the shared check, for the run or kept by the handle.
    local_copies: Tuple[Tuple[torch.Tensor, torch.Tensor, int], ...]
    layout: Tuple
    segments: int
    chunks: int
    element_size: int
    identity: int
    world_size: int
    device: torch.device
    stream_ordered: bool
    # What the copier built from ``rounds`` and ``local_copies``; empty unless kept.
    packs: List[Callable[[], None]]
    unpacks: List[Callable[[], None]]
    local_copy: Optional[Callable[[], None]]
    # This process's window copies, as ``(item, operation, direction)`` in the order it makes them.
    mapped: Tuple
    # Every process's shards in the window, by ``(request key, side, location)``.
    offsets: Dict[Tuple, int]
    mapped_bytes: int
    # Window copies over the whole group, so that every process decides the same about barriers.
    mapped_total: int
    # ``(target region, source region)`` for each window copy, in both paths, and what the copier built
    # from them, kept only.
    mapped_copies: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    mapped_copy: Optional[Callable[[], None]]
    copier: Optional[Copier]


# This process's preparations so far. Two batches prepared alike share a digest, so what tells their
# handles apart is which call prepared them on each process.
_preparations = count()


def _agree(requests: Sequence[TransferRequest],
           endpoints: TransferEndpoints,
           dtype: torch.dtype,
           budget: StagingBudget,
           group: Optional[object],
           keep: bool,
           copier: Optional[Copier] = None,
           window: Optional[TransferWindow] = None) -> _Agreement:
    """Validate a batch, lay out this process's share and agree on it with every process.

    Collective. With ``keep``, the views a prepared handle keeps are built here too, before the
    agreement, so a process that cannot build them refuses on every process.
    """
    import deepspeed.comm as dist

    preparation = next(_preparations)
    initialized = dist.is_initialized()
    if initialized and group is not None and callable(getattr(group, 'rank', None)):
        # Ask the group itself: a weight-update group built outside the default world (as trainer <-> engine
        # groups are) maps every member's default-world rank 0 to group rank 0 in torch's bookkeeping.
        rank, world_size = group.rank(), group.size()
    else:
        rank = dist.get_rank(group=group) if initialized else 0
        world_size = dist.get_world_size(group=group) if initialized else 1
    device = torch.device(get_accelerator().current_device_name()) if initialized else torch.device('cpu')
    # Without a process group every buffer is on the host, whatever the accelerator is.
    stream_ordered = initialized and not get_accelerator().is_synchronized_device()
    lane, copies, recv_arenas, send_arenas = None, [], {}, {}
    indexed, rounds, local_copies, layout = [], [], (), ()
    packs, unpacks, local_copy = [], [], None
    element_size, digest, rejected = 0, 0, 0
    error = None
    mapped, mapped_bytes, mapped_total, mapped_copies, mapped_copy = (), 0, 0, (), None
    offsets: Dict[Tuple, int] = {}
    if window is not None and world_size > 1:
        # Collective, so it runs before anything that may reject; a bad buffer simply publishes nothing.
        try:
            local_offsets = _window_offsets(requests, window)
        except (AttributeError, RuntimeError, TypeError):
            local_offsets = {}
        published = [None] * world_size
        dist.all_gather_object(published, local_offsets, group=group)
        for table in published:
            offsets.update(table)
    try:
        element_size = torch.empty((), dtype=dtype).element_size()
        if (not isinstance(budget.bytes_per_peer, int) or isinstance(budget.bytes_per_peer, bool)
                or budget.bytes_per_peer < element_size):
            raise AffineTransferExecutionError('the staging budget must hold at least one element')
        for bindings in (endpoints.source, endpoints.target):
            for location, process in bindings.items():
                if (not isinstance(location, int) or isinstance(location, bool) or location < 0
                        or not isinstance(process, int) or isinstance(process, bool) or not 0 <= process < world_size):
                    raise AffineTransferExecutionError('endpoint bindings must name valid transport-group ranks')
        keys = [request.key for request in requests]
        if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
            raise AffineTransferExecutionError('transfer batch keys must be distinct non-empty strings')
        for request in requests:
            if request.preflight_error is not None:
                raise AffineTransferExecutionError(request.preflight_error)
            if request.plan is None:
                raise AffineTransferExecutionError('a transfer request has no plan')
            _validate_buffers(request.plan, request.source_buffers, request.target_buffers, dtype, device, endpoints,
                              rank)
        aliased = _buffers_alias(requests)
        # Many parameters share the same plan. Reuse only geometry within this call;
        # every request still validates its own buffers and participates in the digest.
        plans = [request.plan for request in requests]
        indexed = _indexed_operations(plans, endpoints, budget, element_size)
        digests = {}
        for plan in plans:
            if id(plan) not in digests:
                digests[id(plan)] = plan_digest(plan, endpoints, budget, element_size)
        if window is not None and initialized and (len(window.members) != world_size
                                                   or window.members[rank] != dist.get_rank()):
            raise AffineTransferExecutionError('the window was created over a different group than the transfer')
        # Which window, which shards lie in it and who copies decide what each process does, so ranks must
        # agree on them.
        mapping = None if window is None else (window.identity, window.members, window.direction,
                                               tuple(sorted(offsets.items())))
        fingerprint = (3, str(dtype), budget.bytes_per_peer, tuple(sorted(endpoints.source.items())),
                       tuple(sorted(endpoints.target.items())),
                       tuple((request.key, digests[id(request.plan)]) for request in requests), mapping)
        digest = zlib.crc32(repr(fingerprint).encode())
        window_locations = None if window is None else _window_locations(requests, offsets)
        # The same computation as schedule_transfers, so what it reports is what runs here.
        paused = _collector_paused() if keep else contextlib.nullcontext()
        with paused:
            lane = _lanes(indexed, [rank], budget.bytes_per_peer // element_size, window_locations,
                          _registered_locations(requests))[rank]
            # A local copy whose source and target are the same memory moves nothing. It is dropped here,
            # before the agreement, for a one-shot move and for every run of a prepared handle.
            if aliased:
                window_operations = [] if window_locations is None else [
                    (item, operation) for item, operation in indexed
                    if rank in (operation.from_process, operation.to_process)
                    and _mapped_direction(item, operation, window_locations, window.direction) is not None
                ]
                copies = _validate_aliases(requests, lane, element_size, rank, window_operations)
            else:
                copies = lane.local_copies
            recv_arenas = {
                peer: torch.empty(size, dtype=dtype, device=device)
                for peer, size in lane.recv_arena.items()
            }
            send_arenas = {
                peer: torch.empty(size, dtype=dtype, device=device)
                for peer, size in lane.send_arena.items()
            }
            mapped, mapped_bytes, mapped_total = _mapped_share(indexed, window_locations, window, rank)
            # Both paths resolve the copy views here, inside the shared check below: the keep path feeds
            # them to the copier, the one-shot path runs them, and a map into a peer's window can still
            # fail on one process after everyone agreed, so either way a failure must refuse every
            # process before data moves, as the prepare path already does.
            local_copies = tuple(_local_copy_views(copies, requests, dtype))
            mapped_copies = tuple(_mapped_copy_views(mapped, requests, offsets, window, dtype))
            if keep:
                rounds = list(_round_views(lane, recv_arenas, send_arenas, requests, dtype))
                layout = _buffer_layout(requests)
                if copier is None:
                    from .affine_transfer_copy import sequential_copies as copier
                # Copier build failures, of any kind, refuse the batch on every process: they take the
                # shared exit below instead of raising on one process while its peers wait.
                copier = _safe_copier(copier)
                packs, unpacks = _round_copiers(rounds, copier)
                local_copy = copier([(target, source) for target, source, _ in local_copies])
                # A peer's window is device memory mapped into this process, so the same table format covers
                # copies whose destination or source lies in it.
                if mapped_copies:
                    mapped_copy = copier(mapped_copies)
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, OverflowError, MemoryError) as caught:
        # A copier build failure arrives here as an AffineTransferExecutionError from _safe_copier.
        error = caught
        rejected = 1
        recv_arenas, send_arenas, rounds = {}, {}, []
        packs, unpacks, local_copy = [], [], None
        local_copies, mapped, mapped_copies, mapped_copy = (), (), (), None
        print(f'[transfer] rank {rank} rejected the batch: {caught}', file=sys.stderr, flush=True)

    preparations = [preparation]
    # Invalid metadata must not be inspected again while constructing the agreement payload.
    if world_size > 1:
        local = torch.tensor([digest, rejected, preparation], dtype=torch.int64, device=device)
        seen = torch.empty((world_size, 3), dtype=torch.int64, device=device)
        dist.all_gather(list(seen.unbind(0)), local, group=group)
        # Read the complete decision once instead of synchronizing each rank's scalar separately.
        decisions = seen.cpu().tolist()
        if any(row[1] for row in decisions):
            raise AffineTransferExecutionError(
                'a rank rejected the transfer batch before it began; no data moved') from error
        if len({row[0] for row in decisions}) != 1:
            raise AffineTransferExecutionError('ranks disagree about the transfer schedule (batch schedule)')
        preparations = [row[2] for row in decisions]
    elif rejected:
        raise AffineTransferExecutionError(f'the transfer batch was rejected before data moved: {error}') from error

    return _Agreement(lane=lane,
                      copies=copies,
                      recv_arenas=recv_arenas,
                      send_arenas=send_arenas,
                      rounds=rounds,
                      local_copies=local_copies,
                      layout=layout,
                      segments=sum(len(request.plan.segments) for request in requests),
                      chunks=len(indexed),
                      element_size=element_size,
                      identity=_identity(digest, preparations),
                      world_size=world_size,
                      device=device,
                      stream_ordered=stream_ordered,
                      packs=packs,
                      unpacks=unpacks,
                      local_copy=local_copy,
                      mapped=mapped,
                      offsets=offsets,
                      mapped_bytes=mapped_bytes,
                      mapped_total=mapped_total,
                      mapped_copies=mapped_copies,
                      mapped_copy=mapped_copy,
                      copier=copier)


def _round_copiers(rounds, copier: Copier) -> Tuple[List[Callable[[], None]], List[Callable[[], None]]]:
    """Per round, the callable for its packs and the callable for its unpacks."""
    packs = [copier([pair for staged in sends for pair in staged.copies]) for _, sends in rounds]
    unpacks = [copier([pair for staged in recvs for pair in staged.copies]) for recvs, _ in rounds]
    return packs, unpacks


def _safe_copier(copier: Copier) -> Copier:
    """A caller's copier behind the shared refusal: any build failure is the error prepare refuses on."""

    def build(pairs):
        try:
            return copier(pairs)
        except Exception as build_failure:
            # Without this fold, a copier that raises anything but the checked value errors on one
            # process alone, and its peers wait in the move or the final barrier instead of refusing.
            raise AffineTransferExecutionError(f'the copier failed to build a copy list: {build_failure}') \
                from build_failure

    return build


def _mapped_share(indexed, window_locations: Optional[_WindowLocations], window: Optional[TransferWindow],
                  rank: int) -> Tuple[Tuple, int, int]:
    """This process's window copies in the order it makes them, their bytes, and the group's count."""
    if window is None:
        return (), 0, 0
    mine, total = [], 0
    for item, operation in indexed:
        direction = _mapped_direction(item, operation, window_locations, window.direction)
        if direction is None:
            continue
        total += 1
        copier = operation.to_process if direction == 'pull' else operation.from_process
        if copier == rank:
            mine.append((item, operation, direction))
    return tuple(_staggered(mine, rank)), sum(operation.payload_bytes for _, operation, _ in mine), total


def _identity(digest: int, preparations: Sequence[int]) -> int:
    """One preparation, named the same way on every process: its digest and each process's call count."""
    hashed = hashlib.blake2b(repr((digest, tuple(preparations))).encode(), digest_size=8).digest()
    # Shifted to fit the signed 64-bit element that carries it in run's check.
    return int.from_bytes(hashed, 'little') >> 1


@torch.no_grad()
def prepare_transfers(requests: Sequence[TransferRequest],
                      endpoints: TransferEndpoints,
                      dtype: torch.dtype,
                      budget: StagingBudget = StagingBudget(),
                      group: Optional[object] = None,
                      copier: Optional[Copier] = None,
                      window: Optional[TransferWindow] = None) -> 'PreparedTransfer':
    """Everything ``execute_transfers`` does before data moves, done once for repeated runs.

    Collective, with the same preflight and refusal contract as ``execute_transfers``. The handle keeps
    the requests' buffers, the staging arenas and a view of every region each round packs, unpacks or
    copies locally, so ``PreparedTransfer.run`` only moves data. The buffers must stay where they are,
    and every process must run the handle the same number of times; ``run`` checks both by default.

    ``copier`` turns each list of ``(destination, source)`` region copies -- a round's packs, a round's
    unpacks, the copies that stay on this process, and the copies it makes through a ``window`` -- into
    the callable ``run`` invokes for it.
    ``affine_transfer_copy.sequential_copies`` (one ``copy_`` per region) is the default;
    ``affine_transfer_copy.fused_copies`` copies each list with one kernel launch where the device allows.
    Processes may pass different copiers, since it does not change what moves.

    With a ``window``, created over ``group``, every process must pass the same one; the handle also
    keeps a view of every region this process copies in place through it, peer memory included.
    """
    agreed = _agree(requests, endpoints, dtype, budget, group, keep=True, copier=copier, window=window)
    return PreparedTransfer(_requests=tuple(requests),
                            _layout=agreed.layout,
                            _rounds=agreed.rounds,
                            _local_copies=agreed.local_copies,
                            _packs=agreed.packs,
                            _unpacks=agreed.unpacks,
                            _local_copy=agreed.local_copy,
                            _mapped_copies=agreed.mapped_copies,
                            _mapped_copy=agreed.mapped_copy,
                            _mapped_bytes=agreed.mapped_bytes,
                            _mapped_total=agreed.mapped_total,
                            _lane=agreed.lane,
                            _dtype=dtype,
                            _copier=agreed.copier,
                            _segments=agreed.segments,
                            _chunks=agreed.chunks,
                            _element_size=agreed.element_size,
                            _identity=agreed.identity,
                            _group=group,
                            _world_size=agreed.world_size,
                            _device=agreed.device,
                            _stream_ordered=agreed.stream_ordered,
                            _window=window)


@dataclass(eq=False)
class PreparedTransfer:
    """A batch validated, agreed and laid out by ``prepare_transfers``; ``run`` only moves data.

    The handle holds the requests' buffers and views into them, so their storage stays allocated while
    the handle lives: drop it before ZeRO-3 ``offload_states`` if that memory should be freed, and
    prepare again after ``reload_states``. A buffer that moves to new storage would leave the views
    writing into the old one, so ``run`` checks first by default, but only the tensors it is shown. A
    view such as ``zero_param_storage_shard`` returns keeps its old storage when the partition behind it
    is re-pointed, so pass views like it rebuilt from the live state, as ``run(requests)``.
    """
    _requests: Tuple[TransferRequest, ...]
    _layout: Tuple
    # Per round: the receives, then the sends, each ordered by peer.
    _rounds: List[Tuple[List[_StagedRound], List[_StagedRound]]]
    # ``(target region, source region, bytes)`` for each copy that stays on this process.
    _local_copies: Tuple[Tuple[torch.Tensor, torch.Tensor, int], ...]
    # What ``copier`` built from the copy lists above: per round, the packs and the unpacks; then the local copies.
    _packs: List[Callable[[], None]]
    _unpacks: List[Callable[[], None]]
    _local_copy: Callable[[], None]
    # ``(target region, source region)`` for each window copy this process makes; one side is peer memory.
    _mapped_copies: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    _mapped_copy: Callable[[], None]
    _mapped_bytes: int
    # Window copies over the whole group, so that every process decides the same about barriers.
    _mapped_total: int
    # What rebuilding the staging after ``release_staging`` needs.
    _lane: '_Lane'
    _dtype: torch.dtype
    _copier: Copier
    _segments: int
    _chunks: int
    _element_size: int
    # Differs between any two prepare_transfers calls, even for the same batch; see _preparations.
    _identity: int
    _group: Optional[object]
    _world_size: int
    _device: torch.device
    _stream_ordered: bool
    _runs: int = 0
    _released: bool = False
    # Held so that the window, and the peer memory its copies address, live as long as the handle.
    _window: Optional[TransferWindow] = None

    @torch.no_grad()
    def run(self, requests: Optional[Sequence[TransferRequest]] = None, *, check: bool = True) -> TransferStats:
        """Move the batch again from the buffers' current contents. Collective; synchronous on return.

        With ``check`` (the default), every process first compares its buffers with what it prepared:
        ``requests``, rebuilt from the live state such as fresh ``zero_param_storage_shard`` views, or
        else the tensors the handle holds, which catches a buffer re-pointed in place (``.data = ...``).
        One small all-gather then refuses, on every process and before any data moves, if a buffer
        moved or changed shape on any process, or if the processes are not running the same handle for
        the same time. Prepare the batch again after a refusal. ``check=False`` skips that collective,
        for a caller that already guarantees all three; every process must pass the same ``check``.
        The aliases between buffers, and which local copies are dropped because their source and target
        are the same memory, were decided at prepare too, so ``check=False`` cannot see that a moved
        buffer changed them.

        A run after ``release_staging`` first allocates the staging again and rebuilds its views. With
        ``check``, a process that cannot refuses on every process; without it, that process raises alone.

        A run with window copies anywhere in the group ends with a group barrier, and without ``check``
        every run also starts with one, so owners and copiers never overlap on a window shard.
        """
        failed = 0
        # Views of a moved buffer may not even build, and the check refuses the run anyway: skip restaging.
        moved = self._moved(requests) if check else 0
        if self._released and not moved:
            try:
                self._restage()
            except (ValueError, TypeError, RuntimeError, MemoryError) as caught:
                if not check:
                    raise AffineTransferExecutionError(f'could not rebuild the staging: {caught}') from caught
                failed = 1
                print(f'[transfer] could not rebuild the staging: {caught}', file=sys.stderr, flush=True)
        if check:
            self._check(moved, failed)
        elif self._mapped_total:
            import deepspeed.comm as dist
            # Owners may have rewritten their window shards since prepare or the last run, and nothing else
            # orders those writes before a peer's copies; the check covers that case when it runs.
            dist.barrier(group=self._group)
        return self._move()

    def release_staging(self) -> None:
        """Free the staging arenas between runs; the next ``run`` allocates them again.

        For a caller that needs the memory between syncs more than it needs the next run to be fast. The
        buffers, the local copies and the window copies stay prepared, since they hold no staging. Local
        and idempotent: a process may release on its own, because the next run rebuilds the same rounds and
        messages. If allocating them again may fail, run with ``check``: a process that cannot then refuses
        the run on every process, while without it that process raises alone and its peers wait for it.
        """
        self._rounds, self._packs, self._unpacks = [], [], []
        self._released = True

    def _restage(self) -> None:
        lane, dtype = self._lane, self._dtype
        recv_arenas = {
            peer: torch.empty(size, dtype=dtype, device=self._device)
            for peer, size in lane.recv_arena.items()
        }
        send_arenas = {
            peer: torch.empty(size, dtype=dtype, device=self._device)
            for peer, size in lane.send_arena.items()
        }
        with _collector_paused():
            rounds = list(_round_views(lane, recv_arenas, send_arenas, self._requests, dtype))
        self._packs, self._unpacks = _round_copiers(rounds, self._copier)
        self._rounds = rounds
        self._released = False

    def _moved(self, requests: Optional[Sequence[TransferRequest]]) -> int:
        try:
            return int(_buffer_layout(self._requests if requests is None else requests) != self._layout)
        except (AttributeError, TypeError, RuntimeError):
            return 1

    def _check(self, moved: int, failed: int = 0) -> None:
        import deepspeed.comm as dist
        if self._world_size > 1:
            # One process raising alone would leave its peers waiting in the transfer.
            local = torch.tensor([self._identity, self._runs, moved, failed], dtype=torch.int64, device=self._device)
            seen = torch.empty((self._world_size, 4), dtype=torch.int64, device=self._device)
            dist.all_gather(list(seen.unbind(0)), local, group=self._group)
            decisions = seen.cpu().tolist()
            if any(row[3] for row in decisions):
                failed = 1
            elif any(row[2] for row in decisions):
                moved = 1
            elif len({(row[0], row[1]) for row in decisions}) != 1:
                raise AffineTransferExecutionError('processes are not running the same prepared transfer for the '
                                                   'same time; no data moved')
        if failed:
            raise AffineTransferExecutionError('a process could not rebuild the released staging; '
                                               'no data moved')
        if moved:
            raise AffineTransferExecutionError('a buffer of the prepared transfer moved or changed since '
                                               'prepare_transfers; prepare the batch again. No data moved.')

    def _move(self) -> TransferStats:
        sent, received, copied = _run_rounds(self._rounds,
                                             len(self._rounds),
                                             self._local_copies,
                                             self._group,
                                             self._element_size,
                                             self._stream_ordered,
                                             packs=self._packs,
                                             unpacks=self._unpacks,
                                             local_copy=self._local_copy)
        _run_mapped(self._mapped_copies, self._mapped_copy, self._mapped_total, self._group, self._stream_ordered)
        self._runs += 1
        return TransferStats(self._segments, self._chunks, len(self._rounds), copied, sent, received,
                             self._mapped_bytes)


def _run_mapped(mapped_copies, mapped_copy: Optional[Callable[[], None]], mapped_total: int, group: Optional[object],
                stream_ordered: bool) -> None:
    """This process's window copies, then the barrier that hands every window shard back to its owner."""
    if not mapped_total:
        return
    import deepspeed.comm as dist
    if mapped_copy is not None:
        mapped_copy()
    else:
        for target, source in mapped_copies:
            target.copy_(source)
    if stream_ordered:
        get_accelerator().current_stream().synchronize()
    # A pulled source or pushed target belongs to another process: neither may be reused until every
    # copier is done.
    dist.barrier(group=group)


def _run_rounds(rounds,
                round_count: int,
                local_copies,
                group: Optional[object],
                element_size: int,
                stream_ordered: bool,
                packs: Optional[Sequence[Callable[[], None]]] = None,
                unpacks: Optional[Sequence[Callable[[], None]]] = None,
                local_copy: Optional[Callable[[], None]] = None) -> Tuple[int, int, int]:
    """Move the rounds in order, then the local copies; returns the bytes sent, received and copied.

    ``packs``, ``unpacks`` and ``local_copy`` are a prepared handle's copier callables; without them,
    each region is copied with ``copy_``.
    """
    import deepspeed.comm as dist
    sent = received = copied = 0
    pending = []
    for round_index, (recvs, sends) in enumerate(rounds):
        if packs is not None:
            packs[round_index]()
        else:
            for staged in sends:
                for destination, source in staged.copies:
                    destination.copy_(source)
        p2p = [(dist.irecv, staged) for staged in recvs] + [(dist.isend, staged) for staged in sends]
        operations = [
            dist.P2POp(direction,
                       staged.payload,
                       dist.get_global_rank(group, staged.peer) if group is not None else staged.peer,
                       group=group) for direction, staged in p2p
        ]
        if operations:
            for handle in dist.batch_isend_irecv(operations):
                block_stream = getattr(handle, 'block_current_stream', None)
                if stream_ordered and callable(block_stream):
                    # Order arena reuse on the caller stream; transport errors surface at the bounded drain below.
                    block_stream()
                    pending.append(handle)
                else:
                    handle.wait()
        # Stream ordering already protects arena reuse. Bound retained handles while avoiding
        # a host polling delay after every small group of rounds; all errors are observed on return.
        if (round_index + 1) % 32 == 0 or round_index + 1 == round_count:
            for handle in pending:
                handle.wait()
            pending.clear()
        sent += sum(staged.payload.numel() for staged in sends) * element_size
        received += sum(staged.payload.numel() for staged in recvs) * element_size
        if unpacks is not None:
            unpacks[round_index]()
        else:
            for staged in recvs:
                for destination, source in staged.copies:
                    destination.copy_(source)

    if local_copy is not None:
        local_copy()
    for target, source, size in local_copies:
        if local_copy is None:
            target.copy_(source)
        copied += size

    if stream_ordered:
        get_accelerator().current_stream().synchronize()
    return sent, received, copied


@contextlib.contextmanager
def _collector_paused():
    """Pause Python's cyclic garbage collector while a batch's views are built.

    Tens of thousands of views and tuples stay alive together there, and every collection their
    allocation triggers walks them all again: at 8 processes on CPU that was a third of a one-shot
    ``execute_transfers``. They hold no reference cycles, so there is nothing for it to collect.
    """
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _operation_region(requests, item, operation, dtype, source: bool) -> torch.Tensor:
    """One operation's region in its source or target buffer, as a view."""
    segment = operation.segment
    if source:
        flat = _flat(segment.source_rank, requests[item].source_buffers, dtype)
        return _region(flat, operation.source_offset, operation.chunk_shape, segment.source_strides)
    flat = _flat(segment.target_rank, requests[item].target_buffers, dtype)
    return _region(flat, operation.target_offset, operation.chunk_shape, segment.target_strides)


def _mapped_copy_views(mapped, requests, offsets, window, dtype):
    """Each window copy as ``(target, source)`` views; the peer's side is addressed through the window."""
    for item, operation, direction in mapped:
        key, segment = requests[item].key, operation.segment
        if direction == 'pull':
            peer = window.peer(operation.from_process, dtype)
            base = offsets[(key, 'source', segment.source_rank)]
            source = _region(peer, base + operation.source_offset, operation.chunk_shape, segment.source_strides)
            yield _operation_region(requests, item, operation, dtype, source=False), source
        else:
            peer = window.peer(operation.to_process, dtype)
            base = offsets[(key, 'target', segment.target_rank)]
            target = _region(peer, base + operation.target_offset, operation.chunk_shape, segment.target_strides)
            yield target, _operation_region(requests, item, operation, dtype, source=True)


def _round_count(lane: _Lane) -> int:
    return max((len(rounds) for rounds in list(lane.recv_rounds.values()) + list(lane.send_rounds.values())),
               default=0)


def _round_views(lane: _Lane, recv_arenas, send_arenas, requests, dtype):
    """Each round's receives, then its sends, ordered by peer, as wire payloads and the copies around them."""
    for index in range(_round_count(lane)):
        recvs, sends = [], []
        for peer, peer_rounds in sorted(lane.recv_rounds.items()):
            if index < len(peer_rounds):
                direct, arena = lane.recv_direct[peer][index], recv_arenas.get(peer)
                recvs.append(_staged_round(peer, peer_rounds[index], direct, arena, requests, dtype, source=False))
        for peer, peer_rounds in sorted(lane.send_rounds.items()):
            if index < len(peer_rounds):
                direct, arena = lane.send_direct[peer][index], send_arenas.get(peer)
                sends.append(_staged_round(peer, peer_rounds[index], direct, arena, requests, dtype, source=True))
        yield recvs, sends


def _staged_round(peer, peer_round, direct, arena, requests, dtype, source: bool) -> _StagedRound:
    """One peer's message in one direction, resolved to its wire payload and the copies around it."""
    entries, elements = peer_round
    if direct:
        return _StagedRound(peer, _direct_payload(entries, requests, dtype, source), ())
    copies = []
    for item, operation, offset in entries:
        slot = arena[offset:offset + operation.elements].view(operation.chunk_shape)
        region = _operation_region(requests, item, operation, dtype, source)
        copies.append((slot, region) if source else (region, slot))
    return _StagedRound(peer, arena[:elements], tuple(copies))


def _local_copy_views(copies: Sequence[Tuple[int, _Operation]], requests, dtype):
    """``(target region, source region, bytes)`` for each copy that stays on this process."""
    for item, operation in copies:
        target = _operation_region(requests, item, operation, dtype, source=False)
        source = _operation_region(requests, item, operation, dtype, source=True)
        yield target, source, operation.payload_bytes


def _buffer_layout(requests: Sequence[TransferRequest]) -> Tuple:
    """Where each request's local buffers are, and under which plan; a moved buffer changes it."""
    layout = []
    for request in requests:
        for side, buffers in (('source', request.source_buffers), ('target', request.target_buffers)):
            for location in sorted(buffers):
                buffer = buffers[location]
                layout.append((request.key, request.plan, side, location, buffer.data_ptr(), tuple(buffer.shape),
                               buffer.stride(), buffer.dtype, buffer.device, buffer.is_conj(), buffer.is_neg()))
    return tuple(layout)


def _data_range(buffer: torch.Tensor) -> Tuple[int, int]:
    """Byte extent a tensor's elements occupy in storage."""
    start = buffer.data_ptr()
    span = buffer.numel() * buffer.element_size()
    return start, start + span


def _flat(location: int, buffers: Mapping[int, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    """The buffer for ``location`` as its own elements, without copying it."""
    try:
        buffer = buffers[location]
    except KeyError:
        raise AffineTransferExecutionError(f'location {location} is bound to this process but holds no buffer')
    if buffer.layout != torch.strided or not buffer.is_contiguous():
        raise AffineTransferExecutionError(f'location {location}: shard is not contiguous in its own storage')
    if buffer.is_conj() or buffer.is_neg():
        # Point-to-point transport does not carry the lazy bit, and such a view of a destination is not
        # the destination itself. Refusing it here refuses it wherever the copy would run.
        raise AffineTransferExecutionError(f'location {location}: shard is a conjugate or negative view; '
                                           'pass it resolved (resolve_conj, resolve_neg)')
    if buffer.dtype != dtype:
        raise AffineTransferExecutionError(f'location {location}: {buffer.dtype} is not {dtype}')
    return buffer.view(-1)


def _region(flat: torch.Tensor, offset: int, shape: Tuple[int, ...], strides: Tuple[int, ...]) -> torch.Tensor:
    """``flat`` viewed as one planned copy region.

    Plan offsets count elements from the first element of the shard; ``as_strided`` counts from the
    start of the storage. Adding the buffer's own storage offset is the explicit form of that, and the
    alternative -- making the buffer contiguous -- would copy the whole shard to fix arithmetic.
    """
    return torch.as_strided(flat, size=shape, stride=strides, storage_offset=flat.storage_offset() + offset)

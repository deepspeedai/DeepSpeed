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

One device per process, with explicit transport-group rank bindings, synchronous on return. No CUDA graphs, no overlap with compute,
no custom kernels: those all change the schedule, which is the last thing a first version of a data
mover should be perturbing.
"""

import sys
import zlib
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from deepspeed.accelerator import get_accelerator

from .affine_transfer import TransferPlan, TransferSegment

__all__ = [
    'AffineTransferExecutionError', 'TransferEndpoints', 'StagingBudget', 'TransferStats', 'TransferRequest',
    'plan_digest', 'execute_transfer', 'execute_transfers'
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
    """Bytes a rank may stage per peer, in each direction, at one time."""
    bytes_per_peer: int = 8 << 20


@dataclass(frozen=True)
class TransferStats:
    segments: int
    chunks: int
    rounds: int
    local_copy_bytes: int
    sent_bytes: int
    received_bytes: int


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
        yield TransferSegment(
            source_rank=segment.source_rank,
            source_offset=segment.source_offset + sum(low * stride
                                                      for low, stride in zip(origin, segment.source_strides)),
            source_strides=segment.source_strides,
            target_rank=segment.target_rank,
            target_offset=segment.target_offset + sum(low * stride
                                                      for low, stride in zip(origin, segment.target_strides)),
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
            reach = offset + sum((size - 1) * stride for size, stride in zip(segment.shape, strides))
            if reach >= buffers[location].numel():
                raise AffineTransferExecutionError(
                    f'{which} of {segment!r} reaches element {reach} of a {buffers[location].numel()}-element buffer')


def _validate_aliases(requests: Sequence[TransferRequest]) -> None:
    sources = [
        _data_range(buffer) for request in requests for buffer in request.source_buffers.values() if buffer.numel()
    ]
    targets = [
        _data_range(buffer) for request in requests for buffer in request.target_buffers.values() if buffer.numel()
    ]
    for low, high in targets:
        if any(low < source_high and source_low < high for source_low, source_high in sources):
            raise AffineTransferExecutionError('a source and a target overlap within the transfer batch')
    for index, (low, high) in enumerate(targets):
        if any(low < other_high and other_low < high for other_low, other_high in targets[:index]):
            raise AffineTransferExecutionError('two targets overlap within the transfer batch')


def execute_transfer(plan: TransferPlan,
                     source_buffers: Mapping[int, torch.Tensor],
                     target_buffers: Mapping[int, torch.Tensor],
                     endpoints: TransferEndpoints,
                     dtype: torch.dtype,
                     budget: StagingBudget = StagingBudget(),
                     group: Optional[object] = None) -> TransferStats:
    """Execute one plan with the same preflight and completion contract as a batch.

    Bindings use transport-group ranks. Buffers must be packed views on the owning process;
    their storage offsets are retained. Rejection on any participant precedes all target writes.
    Transport failures after payload begins may leave partially written targets.
    """
    request = TransferRequest('parameter', plan, source_buffers, target_buffers)
    return execute_transfers([request], endpoints, dtype, budget, group)


def _packed_rounds(indexed_operations: Sequence[Tuple[int, _Operation]], max_elements: int):
    rounds = []
    for item, operation in indexed_operations:
        if not rounds or rounds[-1][1] + operation.elements > max_elements:
            rounds.append(([], 0))
        entries, used = rounds[-1]
        entries.append((item, operation, used))
        rounds[-1] = (entries, used + operation.elements)
    return rounds


def _direct_payload(entries, requests, dtype, source):
    if len(entries) != 1:
        return None
    item, operation, _ = entries[0]
    request, segment = requests[item], operation.segment
    if source:
        flat = _flat(segment.source_rank, request.source_buffers, dtype)
        offset, strides = operation.source_offset, segment.source_strides
    else:
        flat = _flat(segment.target_rank, request.target_buffers, dtype)
        offset, strides = operation.target_offset, segment.target_strides
    region = _region(flat, offset, operation.chunk_shape, strides)
    # A view can replace staging only when the transport sees the same packed element order.
    return region.view(-1) if region.is_contiguous() else None


@torch.no_grad()
def execute_transfers(requests: Sequence[TransferRequest],
                      endpoints: TransferEndpoints,
                      dtype: torch.dtype,
                      budget: StagingBudget = StagingBudget(),
                      group: Optional[object] = None) -> TransferStats:
    """Move a frozen parameter batch with collective preflight and bounded peer messages.

    Each direction stages at most ``bytes_per_peer`` per active peer. All requests and staging
    allocations are checked before payload, including read/write aliases between parameters.
    The caller stream's final unpack and local copies complete before this function returns.
    """
    import deepspeed.comm as dist

    initialized = dist.is_initialized()
    rank = dist.get_rank(group=group) if initialized else 0
    world_size = dist.get_world_size(group=group) if initialized else 1
    device = torch.device(get_accelerator().current_device_name()) if initialized else torch.device('cpu')
    indexed, local_copies = [], []
    recv_rounds, send_rounds, recv_arenas, send_arenas = {}, {}, {}, {}
    recv_direct, send_direct = {}, {}
    digest, rejected = 0, 0
    error = None
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
        schedules = {}
        for item, request in enumerate(requests):
            if request.preflight_error is not None:
                raise AffineTransferExecutionError(request.preflight_error)
            if request.plan is None:
                raise AffineTransferExecutionError('a transfer request has no plan')
            _validate_buffers(request.plan, request.source_buffers, request.target_buffers, dtype, device, endpoints,
                              rank)
            # Many parameters share the same plan. Reuse only geometry within this call;
            # every request still validates its own buffers and participates in the digest.
            identity = id(request.plan)
            if identity not in schedules:
                schedules[identity] = (build_operations(request.plan, endpoints, budget, element_size),
                                       plan_digest(request.plan, endpoints, budget, element_size))
            indexed.extend((item, operation) for operation in schedules[identity][0])
        _validate_aliases(requests)
        fingerprint = (3, str(dtype), budget.bytes_per_peer, tuple(sorted(endpoints.source.items())),
                       tuple(sorted(endpoints.target.items())),
                       tuple((request.key, schedules[id(request.plan)][1]) for request in requests))
        digest = zlib.crc32(repr(fingerprint).encode())
        incoming: Dict[int, List[Tuple[int, _Operation]]] = {}
        outgoing: Dict[int, List[Tuple[int, _Operation]]] = {}
        for item, operation in indexed:
            if operation.from_process == operation.to_process == rank:
                local_copies.append((item, operation))
            elif operation.to_process == rank:
                incoming.setdefault(operation.from_process, []).append((item, operation))
            elif operation.from_process == rank:
                outgoing.setdefault(operation.to_process, []).append((item, operation))
        max_elements = budget.bytes_per_peer // element_size
        recv_rounds = {peer: _packed_rounds(ops, max_elements) for peer, ops in incoming.items()}
        send_rounds = {peer: _packed_rounds(ops, max_elements) for peer, ops in outgoing.items()}
        recv_direct = {
            peer: [_direct_payload(entries, requests, dtype, False) for entries, _ in rounds]
            for peer, rounds in recv_rounds.items()
        }
        send_direct = {
            peer: [_direct_payload(entries, requests, dtype, True) for entries, _ in rounds]
            for peer, rounds in send_rounds.items()
        }
        recv_arenas = {
            peer:
            torch.empty(max(count for index, (_, count) in enumerate(rounds) if recv_direct[peer][index] is None),
                        dtype=dtype,
                        device=device)
            for peer, rounds in recv_rounds.items() if any(payload is None for payload in recv_direct[peer])
        }
        send_arenas = {
            peer:
            torch.empty(max(count for index, (_, count) in enumerate(rounds) if send_direct[peer][index] is None),
                        dtype=dtype,
                        device=device)
            for peer, rounds in send_rounds.items() if any(payload is None for payload in send_direct[peer])
        }
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, OverflowError, MemoryError) as caught:
        error = caught
        rejected = 1
        recv_arenas, send_arenas = {}, {}
        print(f'[transfer] rank {rank} rejected the batch: {caught}', file=sys.stderr, flush=True)

    # Invalid metadata must not be inspected again while constructing the agreement payload.
    if world_size > 1:
        local = torch.tensor([digest, rejected], dtype=torch.int64, device=device)
        seen = torch.empty((world_size, 2), dtype=torch.int64, device=device)
        dist.all_gather(list(seen.unbind(0)), local, group=group)
        # Read the complete decision once instead of synchronizing each rank's scalar separately.
        decisions = seen.cpu().tolist()
        if any(pair[1] for pair in decisions):
            raise AffineTransferExecutionError(
                'a rank rejected the transfer batch before it began; no data moved') from error
        if len({pair[0] for pair in decisions}) != 1:
            raise AffineTransferExecutionError('ranks disagree about the transfer schedule (batch schedule)')
    elif rejected:
        raise AffineTransferExecutionError(f'the transfer batch was rejected before data moved: {error}') from error

    rounds = max((len(value) for value in list(recv_rounds.values()) + list(send_rounds.values())), default=0)
    sent = received = copied = 0
    pending = []

    for round_index in range(rounds):
        p2p = []
        for peer, peer_rounds in sorted(recv_rounds.items()):
            if round_index < len(peer_rounds):
                payload = recv_direct[peer][round_index]
                if payload is None:
                    payload = recv_arenas[peer][:peer_rounds[round_index][1]]
                p2p.append(('recv', payload, peer))
        for peer, peer_rounds in sorted(send_rounds.items()):
            if round_index >= len(peer_rounds):
                continue
            entries, count = peer_rounds[round_index]
            payload = send_direct[peer][round_index]
            if payload is None:
                arena = send_arenas[peer]
                for item, operation, offset in entries:
                    source = _flat(operation.segment.source_rank, requests[item].source_buffers, dtype)
                    arena[offset:offset + operation.elements].view(operation.chunk_shape).copy_(
                        _region(source, operation.source_offset, operation.chunk_shape,
                                operation.segment.source_strides))
                payload = arena[:count]
            sent += count * element_size
            p2p.append(('send', payload, peer))
        operations = [
            dist.P2POp(dist.irecv if direction == 'recv' else dist.isend,
                       buffer,
                       dist.get_global_rank(group, peer) if group is not None else peer,
                       group=group) for direction, buffer, peer in p2p
        ]
        if operations:
            for handle in dist.batch_isend_irecv(operations):
                block_stream = getattr(handle, 'block_current_stream', None)
                if device.type == 'cuda' and callable(block_stream):
                    # Order arena reuse on the caller stream without a host polling delay per round.
                    block_stream()
                    pending.append(handle)
                else:
                    handle.wait()
        # Stream ordering already protects arena reuse. Bound retained handles while avoiding
        # a host polling delay after every small group of rounds; all errors are observed on return.
        if (round_index + 1) % 32 == 0 or round_index + 1 == rounds:
            for handle in pending:
                handle.wait()
            pending.clear()
        for peer, peer_rounds in recv_rounds.items():
            if round_index >= len(peer_rounds):
                continue
            received += peer_rounds[round_index][1] * element_size
            if recv_direct[peer][round_index] is not None:
                continue
            for item, operation, offset in peer_rounds[round_index][0]:
                target = _flat(operation.segment.target_rank, requests[item].target_buffers, dtype)
                payload = recv_arenas[peer][offset:offset + operation.elements].view(operation.chunk_shape)
                _region(target, operation.target_offset, operation.chunk_shape,
                        operation.segment.target_strides).copy_(payload)

    for item, operation in local_copies:
        source = _flat(operation.segment.source_rank, requests[item].source_buffers, dtype)
        target = _flat(operation.segment.target_rank, requests[item].target_buffers, dtype)
        _region(target, operation.target_offset, operation.chunk_shape, operation.segment.target_strides).copy_(
            _region(source, operation.source_offset, operation.chunk_shape, operation.segment.source_strides))
        copied += operation.payload_bytes

    if device.type != 'cpu':
        get_accelerator().current_stream().synchronize()
    return TransferStats(sum(len(request.plan.segments) for request in requests), len(indexed), rounds, copied, sent,
                         received)


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

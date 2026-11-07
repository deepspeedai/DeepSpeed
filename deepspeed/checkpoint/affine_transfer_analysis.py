# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Describe what a transfer batch will do before it runs, without deciding anything.

The planner chooses segments and replicas, and the executor chooses chunks, rounds and staging.
This module only reads those outputs. It reports volume, granularity, per-process send and receive
load, the executor's schedule and staging under a ``StagingBudget``, and bytes per path class, so
that a plan can be compared with another, or priced with ``fit_links`` results, before it runs.

Nothing here feeds back into a plan, ranks alternatives or models time. Processes are those of
``TransferEndpoints``; messages are labelled with the same ``path_of(from_process, to_process)``
hook as ``calibrate_links``.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

from .affine_transfer import TransferPlan
from .affine_transfer_calibration import LOCAL, default_path_of
from .affine_transfer_executor import StagingBudget, TransferEndpoints, _packed_rounds, build_operations

__all__ = [
    'TransferMessage', 'ProcessReport', 'PlanReport', 'analyze_transfers', 'expected_stats', 'report_to_dict',
    'distribution'
]

REPORT_SCHEMA = 1


@dataclass(frozen=True)
class TransferMessage:
    """One point-to-point payload the executor posts in one round."""
    round: int
    from_process: int
    to_process: int
    nbytes: int
    entries: int
    packed_on_send: bool
    unpacked_on_receive: bool
    path: str


@dataclass(frozen=True)
class ProcessReport:
    process: int
    sent_bytes: int = 0
    received_bytes: int = 0
    local_copy_bytes: int = 0
    send_messages: int = 0
    receive_messages: int = 0
    rounds: int = 0
    send_peers: int = 0
    receive_peers: int = 0
    pack_bytes: int = 0
    unpack_bytes: int = 0
    staging_bytes: int = 0


@dataclass(frozen=True)
class PlanReport:
    element_size: int
    bytes_per_peer: int
    parameters: int
    segments: int
    chunks: int
    logical_bytes: int
    moved_bytes: int
    local_copy_bytes: int
    remote_bytes: int
    rounds: int
    processes: Tuple[ProcessReport, ...]
    messages: Tuple[TransferMessage, ...]
    segment_bytes: Tuple[int, ...]
    chunk_bytes: Tuple[int, ...]
    planner: Dict[str, int] = field(default_factory=dict)

    def process(self, process: int) -> ProcessReport:
        for entry in self.processes:
            if entry.process == process:
                return entry
        return ProcessReport(process)


def _is_packed(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """Whether a strided region is one run of elements in row-major order, as torch judges it."""
    expected = 1
    for extent, stride in zip(reversed(tuple(shape)), reversed(tuple(strides))):
        if extent == 1:
            continue
        if stride != expected:
            return False
        expected *= extent
    return True


def _chunk_is_packed(operation, source_side: bool) -> bool:
    strides = operation.segment.source_strides if source_side else operation.segment.target_strides
    return _is_packed(operation.chunk_shape, strides)


def analyze_transfers(plans: Sequence[TransferPlan],
                      endpoints: TransferEndpoints,
                      element_size: int,
                      budget: StagingBudget = StagingBudget(),
                      path_of: Optional[Callable[[int, int], str]] = None) -> PlanReport:
    """Report one ``execute_transfers`` batch, given its plans in request order.

    The schedule comes from the executor's own ``build_operations`` and round packing, so the
    counts match what each process will return in ``TransferStats``.
    """
    if path_of is None:
        path_of = default_path_of

    indexed = []
    schedules = {}
    for item, plan in enumerate(plans):
        identity = id(plan)
        if identity not in schedules:
            schedules[identity] = build_operations(plan, endpoints, budget, element_size)
        indexed.extend((item, operation) for operation in schedules[identity])

    local_copy = {}
    by_pair: Dict[Tuple[int, int], list] = {}
    for item, operation in indexed:
        if operation.from_process == operation.to_process:
            process = operation.from_process
            local_copy[process] = local_copy.get(process, 0) + operation.payload_bytes
        else:
            pair = (operation.from_process, operation.to_process)
            by_pair.setdefault(pair, []).append((item, operation))

    max_elements = budget.bytes_per_peer // element_size
    messages = []
    sent, received, send_count, receive_count = {}, {}, {}, {}
    pack, unpack, send_peers, receive_peers = {}, {}, {}, {}
    rounds_of, staging = {}, {}
    for pair in sorted(by_pair):
        from_process, to_process = pair
        rounds = _packed_rounds(by_pair[pair], max_elements)
        largest_packed_send = 0
        largest_unpacked_receive = 0
        for round_index, (entries, count) in enumerate(rounds):
            nbytes = count * element_size
            # The executor sends a view only when the round holds one packed region; otherwise it stages.
            single = len(entries) == 1
            send_direct = single and _chunk_is_packed(entries[0][1], source_side=True)
            receive_direct = single and _chunk_is_packed(entries[0][1], source_side=False)
            messages.append(
                TransferMessage(round=round_index,
                                from_process=from_process,
                                to_process=to_process,
                                nbytes=nbytes,
                                entries=len(entries),
                                packed_on_send=not send_direct,
                                unpacked_on_receive=not receive_direct,
                                path=path_of(from_process, to_process)))
            if not send_direct:
                pack[from_process] = pack.get(from_process, 0) + nbytes
                largest_packed_send = max(largest_packed_send, nbytes)
            if not receive_direct:
                unpack[to_process] = unpack.get(to_process, 0) + nbytes
                largest_unpacked_receive = max(largest_unpacked_receive, nbytes)
            sent[from_process] = sent.get(from_process, 0) + nbytes
            received[to_process] = received.get(to_process, 0) + nbytes
        send_count[from_process] = send_count.get(from_process, 0) + len(rounds)
        receive_count[to_process] = receive_count.get(to_process, 0) + len(rounds)
        send_peers[from_process] = send_peers.get(from_process, 0) + 1
        receive_peers[to_process] = receive_peers.get(to_process, 0) + 1
        # A process runs as many rounds as its busiest peer queue, in either direction.
        rounds_of[from_process] = max(rounds_of.get(from_process, 0), len(rounds))
        rounds_of[to_process] = max(rounds_of.get(to_process, 0), len(rounds))
        # One arena per peer and direction, sized by its largest staged round.
        staging[from_process] = staging.get(from_process, 0) + largest_packed_send
        staging[to_process] = staging.get(to_process, 0) + largest_unpacked_receive

    processes = set(endpoints.source.values()) | set(endpoints.target.values())
    process_reports = tuple(
        ProcessReport(process=process,
                      sent_bytes=sent.get(process, 0),
                      received_bytes=received.get(process, 0),
                      local_copy_bytes=local_copy.get(process, 0),
                      send_messages=send_count.get(process, 0),
                      receive_messages=receive_count.get(process, 0),
                      rounds=rounds_of.get(process, 0),
                      send_peers=send_peers.get(process, 0),
                      receive_peers=receive_peers.get(process, 0),
                      pack_bytes=pack.get(process, 0),
                      unpack_bytes=unpack.get(process, 0),
                      staging_bytes=staging.get(process, 0)) for process in sorted(processes))

    planner = {}
    for plan in plans:
        for name, value in asdict(plan.statistics).items():
            planner[name] = planner.get(name, 0) + value

    segment_bytes = tuple(segment.numel * element_size for plan in plans for segment in plan.segments)
    remote_bytes = sum(sent.values())
    local_bytes = sum(local_copy.values())
    return PlanReport(element_size=element_size,
                      bytes_per_peer=budget.bytes_per_peer,
                      parameters=len(plans),
                      segments=len(segment_bytes),
                      chunks=len(indexed),
                      logical_bytes=sum(math.prod(plan.logical_shape) * element_size for plan in plans),
                      moved_bytes=sum(segment_bytes),
                      local_copy_bytes=local_bytes,
                      remote_bytes=remote_bytes,
                      rounds=max(rounds_of.values(), default=0),
                      processes=process_reports,
                      messages=tuple(messages),
                      segment_bytes=segment_bytes,
                      chunk_bytes=tuple(operation.payload_bytes for _, operation in indexed),
                      planner=planner)


def expected_stats(report: PlanReport, process: int) -> Dict[str, int]:
    """The ``TransferStats`` fields ``process`` should return for the analyzed batch."""
    entry = report.process(process)
    return {
        'segments': report.segments,
        'chunks': report.chunks,
        'rounds': entry.rounds,
        'local_copy_bytes': entry.local_copy_bytes,
        'sent_bytes': entry.sent_bytes,
        'received_bytes': entry.received_bytes,
    }


def distribution(values: Sequence[float]) -> Dict[str, float]:
    """Count, sum and order statistics, with nearest-rank percentiles."""
    ordered = sorted(values)
    if not ordered:
        return {'count': 0, 'sum': 0}

    def percentile(fraction):
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    return {
        'count': len(ordered),
        'sum': sum(ordered),
        'min': ordered[0],
        'p50': percentile(0.5),
        'p90': percentile(0.9),
        'max': ordered[-1],
        'mean': sum(ordered) / len(ordered),
    }


def _imbalance(values: Sequence[int]) -> Optional[float]:
    """Busiest over mean; 1.0 is perfectly even. None when nothing moves."""
    total = sum(values)
    if total == 0:
        return None
    return max(values) / (total / len(values))


def report_to_dict(report: PlanReport, include_messages: bool = False) -> Dict[str, object]:
    """A JSON-ready summary; ``include_messages`` adds the full per-round message list."""
    pairs: Dict[str, Dict[str, int]] = {}
    paths: Dict[str, Dict[str, int]] = {}
    for message in report.messages:
        pair = pairs.setdefault(f'{message.from_process}->{message.to_process}', {'messages': 0, 'bytes': 0})
        pair['messages'] += 1
        pair['bytes'] += message.nbytes
        path = paths.setdefault(message.path, {'messages': 0, 'bytes': 0})
        path['messages'] += 1
        path['bytes'] += message.nbytes
    local_bytes = report.local_copy_bytes
    if local_bytes:
        local = paths.setdefault(LOCAL, {'messages': 0, 'bytes': 0})
        local['bytes'] += local_bytes

    sent = [entry.sent_bytes for entry in report.processes]
    received = [entry.received_bytes for entry in report.processes]
    staged_messages = sum(1 for message in report.messages if message.packed_on_send or message.unpacked_on_receive)
    summary = {
        'schema': REPORT_SCHEMA,
        'element_size': report.element_size,
        'bytes_per_peer': report.bytes_per_peer,
        'volume': {
            'parameters': report.parameters,
            'logical_bytes': report.logical_bytes,
            'moved_bytes': report.moved_bytes,
            'local_copy_bytes': report.local_copy_bytes,
            'remote_bytes': report.remote_bytes,
        },
        'granularity': {
            'segments': distribution(report.segment_bytes),
            'chunks': distribution(report.chunk_bytes),
            'messages': distribution([message.nbytes for message in report.messages]),
            'staged_messages': staged_messages,
            'pack_bytes': sum(entry.pack_bytes for entry in report.processes),
            'unpack_bytes': sum(entry.unpack_bytes for entry in report.processes),
        },
        'schedule': {
            'rounds': report.rounds,
            'staging_bytes_max': max((entry.staging_bytes for entry in report.processes), default=0),
        },
        'balance': {
            'sent_imbalance': _imbalance(sent),
            'received_imbalance': _imbalance(received),
            'active_pairs': len(pairs),
        },
        'processes': [asdict(entry) for entry in report.processes],
        'pairs': pairs,
        'paths': paths,
        'planner': report.planner,
    }
    if include_messages:
        summary['messages'] = [asdict(message) for message in report.messages]
    # Round-trip once so callers get exactly what json.dump would write (string keys, plain types).
    return json.loads(json.dumps(summary))

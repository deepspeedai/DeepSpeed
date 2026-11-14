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
from .affine_transfer_executor import StagingBudget, TransferEndpoints, schedule_transfers

__all__ = [
    'TransferMessage', 'ProcessReport', 'PlanReport', 'analyze_transfers', 'expected_stats', 'report_to_dict',
    'distribution'
]

REPORT_SCHEMA = 1
# Processes are the transport-group ranks TransferEndpoints binds, as in execute_transfers.
PROCESS_NUMBERING = 'transport_group_rank'


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
    planner: Dict[str, int] = field(default_factory=dict)

    def process(self, process: int) -> ProcessReport:
        for entry in self.processes:
            if entry.process == process:
                return entry
        return ProcessReport(process)


def analyze_transfers(plans: Sequence[TransferPlan],
                      endpoints: TransferEndpoints,
                      element_size: int,
                      budget: StagingBudget = StagingBudget(),
                      path_of: Optional[Callable[[int, int], str]] = None) -> PlanReport:
    """Report one ``execute_transfers`` batch, given its plans in request order.

    The schedule comes from the executor's ``schedule_transfers``, so the counts match what each
    process will return in ``TransferStats`` and the staging is what it will allocate.
    """
    if path_of is None:
        path_of = default_path_of

    schedules = schedule_transfers(plans, endpoints, budget, element_size)
    messages = []
    for from_process, schedule in schedules.items():
        for to_process, sent_rounds in schedule.sends.items():
            received_rounds = schedules[to_process].receives[from_process]
            for round_index, (sent, received) in enumerate(zip(sent_rounds, received_rounds)):
                messages.append(
                    TransferMessage(round=round_index,
                                    from_process=from_process,
                                    to_process=to_process,
                                    nbytes=sent.elements * element_size,
                                    entries=sent.chunks,
                                    packed_on_send=not sent.direct,
                                    unpacked_on_receive=not received.direct,
                                    path=path_of(from_process, to_process)))

    def total_bytes(rounds_by_peer, staged_only=False):
        elements = 0
        for rounds in rounds_by_peer.values():
            elements += sum(entry.elements for entry in rounds if not (staged_only and entry.direct))
        return elements * element_size

    process_reports = []
    chunks = 0
    for process, schedule in schedules.items():
        arena_elements = sum(schedule.send_arena_elements.values()) + sum(schedule.receive_arena_elements.values())
        process_reports.append(
            ProcessReport(process=process,
                          sent_bytes=total_bytes(schedule.sends),
                          received_bytes=total_bytes(schedule.receives),
                          local_copy_bytes=schedule.local_copy_elements * element_size,
                          send_messages=sum(len(rounds) for rounds in schedule.sends.values()),
                          receive_messages=sum(len(rounds) for rounds in schedule.receives.values()),
                          rounds=schedule.rounds,
                          send_peers=len(schedule.sends),
                          receive_peers=len(schedule.receives),
                          pack_bytes=total_bytes(schedule.sends, staged_only=True),
                          unpack_bytes=total_bytes(schedule.receives, staged_only=True),
                          staging_bytes=arena_elements * element_size))
        # Count each remote chunk once, on its sending side.
        chunks += schedule.local_copy_chunks
        chunks += sum(entry.chunks for rounds in schedule.sends.values() for entry in rounds)

    planner = {}
    for plan in plans:
        for name, value in asdict(plan.statistics).items():
            planner[name] = planner.get(name, 0) + value

    segment_bytes = tuple(segment.numel * element_size for plan in plans for segment in plan.segments)
    return PlanReport(element_size=element_size,
                      bytes_per_peer=budget.bytes_per_peer,
                      parameters=len(plans),
                      segments=len(segment_bytes),
                      chunks=chunks,
                      logical_bytes=sum(math.prod(plan.logical_shape) * element_size for plan in plans),
                      moved_bytes=sum(segment_bytes),
                      local_copy_bytes=sum(entry.local_copy_bytes for entry in process_reports),
                      remote_bytes=sum(entry.sent_bytes for entry in process_reports),
                      rounds=max((entry.rounds for entry in process_reports), default=0),
                      processes=tuple(process_reports),
                      messages=tuple(messages),
                      segment_bytes=segment_bytes,
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
        'process_numbering': PROCESS_NUMBERING,
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
            'chunks': report.chunks,
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

# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Measure the links an affine transfer uses, as raw times by message size.

A transfer plan says how many bytes go between which processes; this module measures how long
that takes on the machine at hand. Spec-sheet bandwidth is not a usable substitute: what a
transfer sees depends on the communication library, on how many links share a switch, root port
or receiver, and on fixed per-message costs, none of which a datasheet states.

``calibrate_links`` records raw samples, each labeled with a path class by ``path_of``;
``fit_links`` summarizes them per path as a fixed cost plus a bandwidth, with the fit error
alongside so a consumer can see where that form does not hold. Modeling contention beyond the
measured patterns, and choosing plans, belong to the replica policy above the planner.

Processes are ranks of the default process group. Plan locations are not processes; when the
transfer group is the default group, ``TransferEndpoints`` maps locations to these ranks.
"""

import statistics
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

__all__ = [
    'CALIBRATION_SCHEMA', 'DEFAULT_SIZES', 'LOCAL', 'PEER', 'INTRA_NODE', 'INTER_NODE', 'default_path_of',
    'node_path_of', 'calibrate_links', 'fit_links'
]

CALIBRATION_SCHEMA = 1

# 4 KiB to 256 MiB: small messages fix the per-message cost, and fast links reach line rate only
# from about 64 MiB up.
DEFAULT_SIZES = tuple(1 << shift for shift in range(12, 29, 2))

LOCAL = 'local'
PEER = 'peer'
INTRA_NODE = 'intra_node'
INTER_NODE = 'inter_node'


def default_path_of(from_process: int, to_process: int) -> str:
    """Tell a copy within one process from a transfer between two."""
    return LOCAL if from_process == to_process else PEER


def node_path_of(processes_per_node: int) -> Callable[[int, int], str]:
    """Classify by node, for processes numbered node by node."""
    if processes_per_node <= 0:
        raise ValueError('processes_per_node must be positive')

    def path_of(from_process: int, to_process: int) -> str:
        if from_process == to_process:
            return LOCAL
        if from_process // processes_per_node == to_process // processes_per_node:
            return INTRA_NODE
        return INTER_NODE

    return path_of


def _synchronize() -> None:
    get_accelerator().synchronize()


def _exchange(send: torch.Tensor, to_processes: Sequence[int], receives: Sequence[Tuple[torch.Tensor, int]]) -> None:
    # oneCCL hangs on a group call that starts with a receive, so sends always go first.
    ops = [dist.P2POp(dist.isend, send, process) for process in to_processes]
    ops += [dist.P2POp(dist.irecv, buffer, process) for buffer, process in receives]
    if ops:
        for handle in dist.batch_isend_irecv(ops):
            handle.wait()


def _timed_exchange(send, to_processes, receives) -> float:
    dist.barrier()
    _synchronize()
    started = time.perf_counter()
    _exchange(send, to_processes, receives)
    _synchronize()
    return time.perf_counter() - started


def _timed_copy(send: torch.Tensor, recv: torch.Tensor) -> float:
    _synchronize()
    started = time.perf_counter()
    recv.copy_(send)
    _synchronize()
    return time.perf_counter() - started


def calibrate_links(sizes: Sequence[int] = DEFAULT_SIZES,
                    repeats: int = 10,
                    warmup: int = 2,
                    path_of: Callable[[int, int], str] = default_path_of,
                    fan_width: Optional[int] = None) -> Dict:
    """Time local copies, every ordered process pair alone, all processes at once, and fans.

    Every process of the default group must call this. It returns the same dictionary on every
    process, with ``repeats`` samples per entry and no aggregation:

    - ``local``: a device-local copy on each process.
    - ``peer``: each ordered pair with every other process idle, so that links through different
      switches and sockets are each sampled. A sample is the slower of its two endpoints.
    - ``concurrent``: every process sends to ``process + shift`` and receives from
      ``process - shift`` at the same time, for every shift, so links are sampled while sharing.
    - ``fan_out``: for every root, the root sends one message to each of its ``fan_width`` peers,
      ``root + 1`` onwards, at once (1 to N), while every other process is idle.
    - ``fan_in``: for every root, each of its peers sends one message to the root at once (N to 1),
      so the root's receive side is shared.

    ``bytes`` is the size of one message. ``path`` on each sample is
    ``path_of(from_process, to_process)``, so measured times can be joined to a plan's messages by
    the same classification. Fan entries keep the root's time apart from its peers' times, and
    label each peer with the path of its own message.

    ``fan_width`` defaults to every other process; 0 skips the fans. The pair sweep makes
    ``world_size * (world_size - 1) * len(sizes) * (warmup + repeats)`` barriers, so keep ``sizes``
    short on large jobs. Each process allocates ``(2 + fan_width) * max(sizes)`` bytes of device
    memory at most.
    """
    if repeats <= 0 or warmup < 0:
        raise ValueError('repeats must be positive and warmup non-negative')
    if any(size <= 0 for size in sizes):
        raise ValueError('sizes must be positive byte counts')

    rank = dist.get_rank()
    world = dist.get_world_size()
    if fan_width is None:
        fan_width = world - 1
    if not 0 <= fan_width < world:
        raise ValueError(f'fan_width must be between 0 and {world - 1}, got {fan_width}')
    fan_peers = {root: [(root + step) % world for step in range(1, fan_width + 1)] for root in range(world)}
    device = torch.device(get_accelerator().current_device_name())
    local_times: Dict[int, List[float]] = {}
    peer_times: Dict[tuple, List[float]] = {}
    concurrent_times: Dict[tuple, List[float]] = {}
    fan_times: Dict[tuple, List[float]] = {}
    for size in sizes:
        send = torch.ones(size, dtype=torch.uint8, device=device)
        recv = torch.empty(size, dtype=torch.uint8, device=device)

        times = [_timed_copy(send, recv) for _ in range(warmup + repeats)]
        local_times[size] = times[warmup:]

        for source in range(world):
            for target in range(world):
                if source == target:
                    continue
                to_processes = [target] if rank == source else []
                receives = [(recv, source)] if rank == target else []
                times = [_timed_exchange(send, to_processes, receives) for _ in range(warmup + repeats)]
                if rank in (source, target):
                    peer_times[(size, source, target)] = times[warmup:]

        for shift in range(1, world):
            to_processes = [(rank + shift) % world]
            receives = [(recv, (rank - shift) % world)]
            times = [_timed_exchange(send, to_processes, receives) for _ in range(warmup + repeats)]
            concurrent_times[(size, shift)] = times[warmup:]

        for root in range(world if fan_width else 0):
            peers = fan_peers[root]
            if rank == root:
                to_processes, receives = peers, []
            else:
                to_processes, receives = [], ([(recv, root)] if rank in peers else [])
            times = [_timed_exchange(send, to_processes, receives) for _ in range(warmup + repeats)]
            fan_times[('fan_out', size, root)] = times[warmup:]

            if rank == root:
                receives = [(torch.empty(size, dtype=torch.uint8, device=device), peer) for peer in peers]
                to_processes = []
            else:
                receives, to_processes = [], ([root] if rank in peers else [])
            times = [_timed_exchange(send, to_processes, receives) for _ in range(warmup + repeats)]
            fan_times[('fan_in', size, root)] = times[warmup:]

    # Samples stay on their own process until every timing is done, so no collective runs inside a timed region.
    gathered = [None] * world
    dist.all_gather_object(gathered, (local_times, peer_times, concurrent_times, fan_times))

    local, peer, concurrent = [], [], []
    fans = {'fan_out': [], 'fan_in': []}
    for size in sizes:
        for process in range(world):
            local.append({'bytes': size, 'process': process, 'path': LOCAL, 'seconds': gathered[process][0][size]})
        for source in range(world):
            for target in range(world):
                if source == target:
                    continue
                key = (size, source, target)
                seconds = [max(pair) for pair in zip(gathered[source][1][key], gathered[target][1][key])]
                peer.append({
                    'bytes': size,
                    'from_process': source,
                    'to_process': target,
                    'path': path_of(source, target),
                    'seconds': seconds
                })
        for shift in range(1, world):
            concurrent.append({
                'bytes': size,
                'shift': shift,
                'paths': [path_of(process, (process + shift) % world) for process in range(world)],
                'seconds_by_process': [gathered[process][2][(size, shift)] for process in range(world)],
            })
        for root in range(world if fan_width else 0):
            peers = fan_peers[root]
            for kind, fans_of_kind in fans.items():
                key = (kind, size, root)
                paths = [path_of(root, peer) if kind == 'fan_out' else path_of(peer, root) for peer in peers]
                fans_of_kind.append({
                    'bytes': size,
                    'root': root,
                    'peers': peers,
                    'paths': paths,
                    'root_seconds': gathered[root][3][key],
                    'seconds_by_peer': [gathered[peer][3][key] for peer in peers],
                })
    return {
        'schema': CALIBRATION_SCHEMA,
        'world_size': world,
        'communication_backend': get_accelerator().communication_backend_name(),
        'repeats': repeats,
        'warmup': warmup,
        'local': local,
        'peer': peer,
        'concurrent': concurrent,
        'fan_width': fan_width,
        'fan_out': fans['fan_out'],
        'fan_in': fans['fan_in'],
    }


def _fit(points: List[tuple]) -> Dict:
    """Fit ``seconds = alpha + bytes / bandwidth`` to (bytes, seconds) points.

    Residuals are weighted by 1/seconds, so the fit minimizes relative error. Unweighted least
    squares would let the largest messages decide everything and leave alpha meaningless.
    """
    weights = [1.0 / (seconds * seconds) for _, seconds in points]
    total = sum(weights)
    mean_x = sum(w * x for w, (x, _) in zip(weights, points)) / total
    mean_y = sum(w * y for w, (_, y) in zip(weights, points)) / total
    spread = sum(w * (x - mean_x)**2 for w, (x, _) in zip(weights, points))
    if spread == 0:
        raise ValueError('a fit needs at least two distinct message sizes')
    slope = sum(w * (x - mean_x) * (y - mean_y) for w, (x, y) in zip(weights, points)) / spread
    alpha = mean_y - slope * mean_x
    predicted = [alpha + slope * x for x, _ in points]
    return {
        'alpha_seconds': alpha,
        # A non-positive slope means time did not grow with size, so no bandwidth is implied.
        'bytes_per_second': 1.0 / slope if slope > 0 else None,
        'max_relative_error': max(abs(p - y) / y for p, (_, y) in zip(predicted, points)),
        'sizes': sorted({x
                         for x, _ in points}),
    }


def _fit_groups(groups: Dict[str, Dict[int, List[float]]]) -> Dict[str, Dict]:
    fits = {}
    for path, seconds_by_size in groups.items():
        points = [(size, statistics.median(seconds)) for size, seconds in sorted(seconds_by_size.items())]
        fits[path] = _fit(points)
    return fits


def fit_links(calibration: Dict) -> Dict:
    """Summarize a ``calibrate_links`` result as ``alpha + bytes / bandwidth`` per path.

    Samples are pooled by path label and by size, and the median at each size is fitted, so a
    single slow repeat does not move the line. Returns ``{'schema', 'local', 'peer',
    'concurrent', 'fan_out', 'fan_in'}``; each but the first maps a path label to
    ``alpha_seconds``, ``bytes_per_second``, ``max_relative_error`` and the ``sizes`` fitted.
    ``concurrent`` is fitted from what each process saw while every process was sending, i.e.
    per-link cost under the shift sharing pattern, not in isolation. ``fan_out`` and ``fan_in``
    are fitted from the peers' times, each under its own message's path, so they give per-link
    cost while one sender or one receiver is shared; the root's times stay in the raw result.
    """
    if calibration.get('schema') != CALIBRATION_SCHEMA:
        raise ValueError(f'unsupported calibration schema {calibration.get("schema")!r}')

    local: Dict[str, Dict[int, List[float]]] = {}
    for sample in calibration['local']:
        local.setdefault(sample['path'], {}).setdefault(sample['bytes'], []).extend(sample['seconds'])

    peer: Dict[str, Dict[int, List[float]]] = {}
    for sample in calibration['peer']:
        peer.setdefault(sample['path'], {}).setdefault(sample['bytes'], []).extend(sample['seconds'])

    concurrent: Dict[str, Dict[int, List[float]]] = {}
    for sample in calibration['concurrent']:
        for path, seconds in zip(sample['paths'], sample['seconds_by_process']):
            concurrent.setdefault(path, {}).setdefault(sample['bytes'], []).extend(seconds)

    fits = {
        'schema': CALIBRATION_SCHEMA,
        'local': _fit_groups(local),
        'peer': _fit_groups(peer),
        'concurrent': _fit_groups(concurrent),
    }
    for kind in ('fan_out', 'fan_in'):
        fanned: Dict[str, Dict[int, List[float]]] = {}
        for sample in calibration[kind]:
            for path, seconds in zip(sample['paths'], sample['seconds_by_peer']):
                fanned.setdefault(path, {}).setdefault(sample['bytes'], []).extend(seconds)
        fits[kind] = _fit_groups(fanned)
    return fits

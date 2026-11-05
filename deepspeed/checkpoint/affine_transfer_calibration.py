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

Processes are ranks of the measured process group, the default group unless ``group`` is given.
Plan locations are not processes; when the transfer runs on the measured group,
``TransferEndpoints`` maps locations to these ranks.
"""

import socket
import statistics
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

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


def _global_rank(group: Optional[object], process: int) -> int:
    return dist.get_global_rank(group, process) if group is not None else process


def _exchange(send: torch.Tensor,
              to_processes: Sequence[int],
              receives: Sequence[Tuple[torch.Tensor, int]],
              group: Optional[object] = None) -> None:
    # oneCCL hangs on a group call that starts with a receive, so sends always go first.
    ops = [dist.P2POp(dist.isend, send, _global_rank(group, process), group=group) for process in to_processes]
    ops += [dist.P2POp(dist.irecv, buffer, _global_rank(group, process), group=group) for buffer, process in receives]
    if ops:
        for handle in dist.batch_isend_irecv(ops):
            handle.wait()


def _timed_exchange(send, to_processes, receives, group: Optional[object] = None) -> float:
    dist.barrier(group=group)
    _synchronize()
    started = time.perf_counter()
    _exchange(send, to_processes, receives, group)
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
                    fan_width: Optional[Union[int, Sequence[int]]] = None,
                    fan_peers: Optional[Callable[[int, int], Sequence[int]]] = None,
                    group: Optional[object] = None) -> Dict:
    """Time local copies, every ordered process pair alone, and fans.

    Every process of ``group`` (the default group unless given) must call this. It returns the
    same dictionary on every process, with ``repeats`` samples per entry and no aggregation:

    - ``local``: a device-local copy on each process.
    - ``peer``: each ordered pair with every other process idle, so that links through different
      switches and sockets are each sampled. A sample is the slower of its two endpoints.
    - ``fan_out``: for every root and fan width, the root sends one message to each of its
      ``width`` peers at once (1 to N), while every other process is idle.
    - ``fan_in``: for every root and fan width, each of its peers sends one message to the root
      at once (N to 1), so the root's receive side is shared.

    ``bytes`` is the size of one message. ``path`` on each sample is
    ``path_of(from_process, to_process)``, so measured times can be joined to a plan's messages by
    the same classification. Fan entries keep the root's time apart from its peers' times, and
    label each peer with the path of its own message.

    ``fan_width`` is one width or several; 0 skips the fans. It defaults to one less than the
    fewest processes on any node, and at least 1, so default fans fit in a node as jobs grow.
    Measuring more than one width shows how a process's send or receive capacity changes
    with the number of peers sharing it, which a single width cannot. Each fan sample records its
    ``width``. By default a root's peers are taken from ``root + 1`` onwards, those on its node
    first, unless ``fan_peers(root, width)`` returns ``width`` other processes, e.g. the senders
    behind one PCIe switch or across a socket; it must return the same peers on every process.
    Nodes are told apart by host name.

    The pair sweep makes ``world_size * (world_size - 1) * len(sizes) * (warmup + repeats)``
    barriers, so keep ``sizes`` short on large jobs. Each process allocates
    ``(2 + max(widths)) * max(sizes)`` bytes of device memory at most, 2.25 GiB with the default
    width at 8 processes per node and 256 MiB messages.
    """
    if repeats <= 0 or warmup < 0:
        raise ValueError('repeats must be positive and warmup non-negative')
    if any(size <= 0 for size in sizes):
        raise ValueError('sizes must be positive byte counts')

    if group is not None and callable(getattr(group, 'rank', None)):
        # Ask the group itself, as the executor does: torch misnumbers groups built outside the default world.
        rank, world = group.rank(), group.size()
    else:
        rank = dist.get_rank(group=group)
        world = dist.get_world_size(group=group)
    hostnames = [None] * world
    dist.all_gather_object(hostnames, socket.gethostname(), group=group)
    if fan_width is None:
        fewest_per_node = min(hostnames.count(host) for host in set(hostnames))
        fan_width = min(world - 1, max(1, fewest_per_node - 1))
    requested = [fan_width] if isinstance(fan_width, int) else list(fan_width)
    for width in requested:
        if not isinstance(width, int) or not 0 <= width < world:
            raise ValueError(f'fan widths must be integers between 0 and {world - 1}, got {width!r}')
    widths = sorted(set(requested) - {0})
    peer_sets = {}
    for root in range(world):
        ring = [(root + step) % world for step in range(1, world)]
        # Peers on the root's node come first so default fans stay in it; the sort is stable, so ring order holds.
        nearest = sorted(ring, key=lambda process: hostnames[process] != hostnames[root])
        for width in widths:
            peers = list(fan_peers(root, width)) if fan_peers is not None else nearest[:width]
            distinct = len(set(peers)) == len(peers) == width
            in_range = all(0 <= peer < world and peer != root for peer in peers)
            if not (distinct and in_range):
                raise ValueError(f'fan_peers({root}, {width}) must return {width} distinct processes other than root')
            peer_sets[(root, width)] = peers
    device = torch.device(get_accelerator().current_device_name())
    local_times: Dict[int, List[float]] = {}
    peer_times: Dict[tuple, List[float]] = {}
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
                times = [_timed_exchange(send, to_processes, receives, group) for _ in range(warmup + repeats)]
                if rank in (source, target):
                    peer_times[(size, source, target)] = times[warmup:]

        for width in widths:
            for root in range(world):
                peers = peer_sets[(root, width)]
                if rank == root:
                    to_processes, receives = peers, []
                else:
                    to_processes, receives = [], ([(recv, root)] if rank in peers else [])
                times = [_timed_exchange(send, to_processes, receives, group) for _ in range(warmup + repeats)]
                fan_times[('fan_out', size, width, root)] = times[warmup:]

                if rank == root:
                    receives = [(torch.empty(size, dtype=torch.uint8, device=device), peer) for peer in peers]
                    to_processes = []
                else:
                    receives, to_processes = [], ([root] if rank in peers else [])
                times = [_timed_exchange(send, to_processes, receives, group) for _ in range(warmup + repeats)]
                fan_times[('fan_in', size, width, root)] = times[warmup:]

    # Samples stay on their own process until every timing is done, so no collective runs inside a timed region.
    gathered = [None] * world
    dist.all_gather_object(gathered, (local_times, peer_times, fan_times), group=group)

    local, peer = [], []
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
        for width in widths:
            for root in range(world):
                peers = peer_sets[(root, width)]
                for kind, fans_of_kind in fans.items():
                    key = (kind, size, width, root)
                    paths = [path_of(root, peer) if kind == 'fan_out' else path_of(peer, root) for peer in peers]
                    fans_of_kind.append({
                        'bytes': size,
                        'width': width,
                        'root': root,
                        'peers': peers,
                        'paths': paths,
                        'root_seconds': gathered[root][2][key],
                        'seconds_by_peer': [gathered[peer][2][key] for peer in peers],
                    })
    return {
        'schema': CALIBRATION_SCHEMA,
        'world_size': world,
        'communication_backend': get_accelerator().communication_backend_name(),
        'repeats': repeats,
        'warmup': warmup,
        'local': local,
        'peer': peer,
        'fan_widths': widths,
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


def _fit_groups(groups: Dict[object, Dict[int, List[float]]]) -> Dict[object, Dict]:
    fits = {}
    for key, seconds_by_size in groups.items():
        points = [(size, statistics.median(seconds)) for size, seconds in sorted(seconds_by_size.items())]
        fits[key] = _fit(points)
    return fits


def fit_links(calibration: Dict) -> Dict:
    """Summarize a ``calibrate_links`` result as ``alpha + bytes / bandwidth`` per path.

    Samples are pooled by path label and by size, and the median at each size is fitted, so a
    single slow repeat does not move the line. Every fit holds ``alpha_seconds``,
    ``bytes_per_second``, ``max_relative_error`` and the ``sizes`` fitted. Returns:

    - ``local`` and ``peer``: a fit per path label.
    - ``peer_pairs``: a fit per ordered ``(from_process, to_process)`` pair, from the ``peer``
      samples. It keeps the sender and direction asymmetry that pooling by path label averages
      away, as between cards behind PCIe switches.
    - ``fan_out`` and ``fan_in``: per fan, keyed by ``(root, peers)`` with the peers sorted, a fit
      per path label, from the peers' times, each under its own message's path. They give
      per-link cost while those peers share one sender or one receiver.
    - ``send`` and ``receive``: per fan, keyed the same way, a fit from the root's time against all
      the bytes it sent (fan-out) or received (fan-in). That is the root's aggregate rate with
      those peers, which is what pricing a plan's per-process sent and received bytes needs. A
      fan finishes when the busiest resource it crosses does: a peer's link, the root, or a link
      the peers share. Behind PCIe switches that depends on *which* peers share the root, not only
      on how many, so fans are keyed by their peers; the width is ``len(peers)``.

    The returned fits use tuple keys and are not directly JSON-serializable. Persist the raw
    ``calibrate_links`` result instead, then call ``fit_links`` again after loading it.
    """
    if calibration.get('schema') != CALIBRATION_SCHEMA:
        raise ValueError(f'unsupported calibration schema {calibration.get("schema")!r}')

    local: Dict[str, Dict[int, List[float]]] = {}
    for sample in calibration['local']:
        local.setdefault(sample['path'], {}).setdefault(sample['bytes'], []).extend(sample['seconds'])

    peer: Dict[str, Dict[int, List[float]]] = {}
    peer_pairs: Dict[Tuple[int, int], Dict[int, List[float]]] = {}
    for sample in calibration['peer']:
        peer.setdefault(sample['path'], {}).setdefault(sample['bytes'], []).extend(sample['seconds'])
        pair = (sample['from_process'], sample['to_process'])
        peer_pairs.setdefault(pair, {}).setdefault(sample['bytes'], []).extend(sample['seconds'])

    fits = {
        'schema': CALIBRATION_SCHEMA,
        'local': _fit_groups(local),
        'peer': _fit_groups(peer),
        'peer_pairs': _fit_groups(peer_pairs),
    }
    for kind, direction in (('fan_out', 'send'), ('fan_in', 'receive')):
        fanned: Dict[tuple, Dict[str, Dict[int, List[float]]]] = {}
        rooted: Dict[tuple, Dict[int, List[float]]] = {}
        for sample in calibration[kind]:
            width = len(sample['peers'])
            fan = (sample['root'], tuple(sorted(sample['peers'])))
            for path, seconds in zip(sample['paths'], sample['seconds_by_peer']):
                fanned.setdefault(fan, {}).setdefault(path, {}).setdefault(sample['bytes'], []).extend(seconds)
            # The root moves one message per peer, so its time covers width times the message size.
            rooted.setdefault(fan, {}).setdefault(sample['bytes'] * width, []).extend(sample['root_seconds'])
        fits[kind] = {fan: _fit_groups(groups) for fan, groups in fanned.items()}
        fits[direction] = _fit_groups(rooted)
    return fits

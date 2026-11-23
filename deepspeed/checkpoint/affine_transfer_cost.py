# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Estimate how long a set of affine transfers takes, and keep the cheapest way to run it.

Plans that move the same data can differ in time by more than 5x: which holder each target reads,
whether a node's copy crosses the network once and then spreads inside the node, how many rounds
the staging budget cuts each message into. A plan says what moves, and by the planner's guarantee
it is exact whichever holder it reads (``affine_transfer``). This module only prices plans, so
choosing between them never changes what arrives.

The model is the one of RFC #9026. ``execute_transfers`` runs a batch as rounds: in round ``k``,
each process posts its ``k``-th message to or from every peer in one ``batch_isend_irecv``, which
``schedule_transfers`` lists without running anything. Every byte of a message occupies resources
on its way, such as the sender's network send, the receiver's network receive, or the link between
two GPUs of a node. Each resource binds on its own, so a round takes its most loaded resource's
bytes over that resource's capacity, plus a fixed cost per round::

    rounds = sum over rounds k of (seconds_per_round + max over resources r of B[r, k] / C[r])

The host also launches one copy per staged or local region. Those launches overlap the bytes
rather than adding to them, so a call takes ``seconds_per_call + max(rounds, launches)``, where
``launches`` is the region count of the process with the most regions times ``seconds_per_region``.

The capacities and costs depend on the machine, the transport configuration and the executor, so
they are calibrated for each. Every rank must reach the same choice, because each rank runs its own
half of the chosen plan, so the model must be built from data every rank shares, never from a
rank-local measurement. A plan's one-time preparation is not priced here.
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Mapping, Sequence, Tuple

from .affine_transfer import AffineTransferError, TransferPlan
from .affine_transfer_executor import StagingBudget, TransferEndpoints, schedule_transfers

__all__ = ['TransferCostModel', 'TransferStage', 'PlanChoice', 'node_route', 'estimate_seconds', 'cheapest']

# A resource is a tuple whose first entry names its kind, such as ('network_send', 3).
Resource = Tuple[Hashable, ...]
Route = Callable[[int, int], Sequence[Resource]]

_INTRA_NODE_KINDS = ('device_send', 'device_receive', 'device_pair')


@dataclass(frozen=True)
class TransferCostModel:
    """The calibrated costs of one machine, transport configuration and executor.

    ``route(sender, receiver)`` names the resources that a message between two processes occupies.
    ``capacity`` gives bytes per second for every kind of resource a route names; a resource that no
    route names does not limit anything. The other fields are in seconds.
    """
    route: Route
    capacity: Mapping[Hashable, float]
    seconds_per_call: float = 0.0
    seconds_per_round: float = 0.0
    seconds_per_region: float = 0.0

    def __post_init__(self):
        for kind, value in self.capacity.items():
            if not _is_number(value) or not value > 0 or math.isinf(value):
                raise AffineTransferError(f'capacity of {kind!r} is {value!r}, not a positive finite number')
        for name in ('seconds_per_call', 'seconds_per_round', 'seconds_per_region'):
            value = getattr(self, name)
            if not _is_number(value) or not value >= 0 or math.isinf(value):
                raise AffineTransferError(f'{name} is {value!r}, not a non-negative finite number')


@dataclass(frozen=True)
class TransferStage:
    """What one ``execute_transfers`` call is given, apart from the buffers.

    ``plans`` are in request order, since that order fixes how chunks are packed into rounds.
    """
    plans: Sequence[TransferPlan]
    endpoints: TransferEndpoints
    element_size: int
    budget: StagingBudget = StagingBudget()


@dataclass(frozen=True)
class PlanChoice:
    """The name of the cheapest candidate, and the estimate in seconds of every candidate."""
    name: str
    seconds: Mapping[str, float]


def node_route(nodes: Mapping[int, int], intra_node: Sequence[str] = ('device_send', 'device_receive')) -> Route:
    """Route each message by the nodes of its two processes.

    Across nodes, a message occupies the sender's ``network_send`` and the receiver's
    ``network_receive``. Within a node, it occupies the kinds named in ``intra_node``:
    ``device_send`` and ``device_receive`` are each process's own link, as on NVSwitch, and
    ``device_pair`` is the link from the sender to the receiver, as with NVLink between two GPUs.
    """
    nodes = dict(nodes)
    intra_node = tuple(intra_node)
    for kind in intra_node:
        if kind not in _INTRA_NODE_KINDS:
            raise AffineTransferError(f'unknown intra-node resource {kind!r}; expected one of {_INTRA_NODE_KINDS}')

    def route(sender: int, receiver: int) -> Tuple[Resource, ...]:
        for process in (sender, receiver):
            if process not in nodes:
                raise AffineTransferError(f'process {process} has no node')
        if nodes[sender] != nodes[receiver]:
            return (('network_send', sender), ('network_receive', receiver))
        resources = []
        for kind in intra_node:
            if kind == 'device_send':
                resources.append((kind, sender))
            elif kind == 'device_receive':
                resources.append((kind, receiver))
            else:
                resources.append((kind, sender, receiver))
        return tuple(resources)

    return route


def estimate_seconds(stages: Sequence[TransferStage], model: TransferCostModel) -> float:
    """The model's time for running ``stages`` one after another, each as one ``execute_transfers`` call.

    A two-stage relay is two stages. The executor does not overlap them, so neither does the estimate.
    """
    return sum(_call_seconds(stage, model) for stage in stages)


def cheapest(candidates: Mapping[str, Sequence[TransferStage]], model: TransferCostModel) -> PlanChoice:
    """Estimate every candidate and keep the cheapest.

    Each candidate is a way to run the same move, as the stages ``estimate_seconds`` takes. A tie goes
    to the name that sorts first, so every rank that is given the same candidates chooses the same one.
    """
    if not candidates:
        raise AffineTransferError('there is no candidate to choose from')
    for name in candidates:
        if not isinstance(name, str):
            raise AffineTransferError(f'candidate names must be strings, got {name!r}')
    seconds = {name: estimate_seconds(candidates[name], model) for name in sorted(candidates)}
    name = min(seconds, key=lambda name: (seconds[name], name))
    return PlanChoice(name=name, seconds=seconds)


def _call_seconds(stage: TransferStage, model: TransferCostModel) -> float:
    schedules = schedule_transfers(stage.plans, stage.endpoints, stage.budget, stage.element_size)

    # Bytes through each resource in each round. Every message is listed once by its sender and once
    # by its receiver, with the same rounds, so counting the senders' side counts each byte once.
    loads: Dict[int, Dict[Resource, int]] = {}
    most_regions = 0
    for process, schedule in schedules.items():
        for peer, rounds in schedule.sends.items():
            resources = model.route(process, peer)
            for index, scheduled in enumerate(rounds):
                load = loads.setdefault(index, {})
                for resource in resources:
                    load[resource] = load.get(resource, 0) + scheduled.elements * stage.element_size
        # A round sent or received in place needs no copy; a staged one packs or unpacks each chunk.
        staged = [
            scheduled for rounds in list(schedule.sends.values()) + list(schedule.receives.values())
            for scheduled in rounds if not scheduled.direct
        ]
        regions = sum(scheduled.chunks for scheduled in staged) + schedule.local_copy_chunks
        most_regions = max(most_regions, regions)

    round_seconds = 0.0
    for index in sorted(loads):
        busiest = max((load / _capacity(model, resource) for resource, load in loads[index].items()), default=0.0)
        round_seconds += model.seconds_per_round + busiest
    launch_seconds = most_regions * model.seconds_per_region
    return model.seconds_per_call + max(round_seconds, launch_seconds)


def _capacity(model: TransferCostModel, resource: Resource) -> float:
    kind = resource[0]
    if kind not in model.capacity:
        raise AffineTransferError(f'resource {resource!r} is routed, but the model has no capacity for {kind!r}')
    return model.capacity[kind]


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

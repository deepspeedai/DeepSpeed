# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Replica policies for ``affine_transfer.plan_transfers``.

A policy takes every ``ReplicaRequest`` of one sync and returns one answer per request: the
location of a validated holder, or ``ReplicaPart``\\ s that cut the request along axis 0. The planner
checks every answer before it emits a segment, so a policy decides only where data is read from;
what arrives is fixed by the maps. Policies live here, not in the planner, so cost-aware ones can
grow without the planner growing with them.

A policy that keeps state across syncs can be any callable object; the planner holds none.
"""

from typing import Callable, Dict, Hashable, List, Sequence

from .affine_transfer import ReplicaAnswer, ReplicaPolicy, ReplicaRequest, ReplicaSelectionRequest

__all__ = ['first_holder', 'local_first_least_loaded', 'per_request']

ReplicaSelector = Callable[[ReplicaSelectionRequest], int]


def first_holder(requests: Sequence[ReplicaRequest]) -> List[ReplicaAnswer]:
    """The lowest holder location for every request, which is what ``plan_transfer`` does by default."""
    return [request.candidates[0].source_rank for request in requests]


def local_first_least_loaded(requests: Sequence[ReplicaRequest]) -> List[ReplicaAnswer]:
    """Read locally when the target's own process holds the region, otherwise from the least loaded holder.

    Load is what a sending process has been assigned so far in this batch: bytes when the requests
    carry them, elements otherwise. Requests with a single holder count, since that load cannot
    move, and local copies do not, since they use no link. Without process mappings a local copy
    cannot be recognised, so each holder location counts as its own sender. Ties go to the lowest
    location and requests arrive in the planner's fixed order, so the answer is deterministic. Nothing
    is split.
    """
    load: Dict[Hashable, int] = {}
    answers: List[ReplicaAnswer] = []
    for request in requests:
        local = [
            candidate.source_rank for candidate, process in zip(request.candidates, request.candidate_processes)
            if process is not None and process == request.target_process
        ]
        if local:
            answers.append(min(local))
            continue

        def sender(index: int) -> Hashable:
            process = request.candidate_processes[index]
            if process is not None:
                return ('process', process)
            return ('location', request.candidates[index].source_rank)

        chosen = min(range(len(request.candidates)),
                     key=lambda index: (load.get(sender(index), 0), request.candidates[index].source_rank))
        size = request.bytes if request.bytes is not None else request.numel
        load[sender(chosen)] = load.get(sender(chosen), 0) + size
        answers.append(request.candidates[chosen].source_rank)
    return answers


def per_request(selector: ReplicaSelector) -> ReplicaPolicy:
    """Run a ``plan_transfer``-style ``replica_selector`` as a batch policy.

    As in ``plan_transfer``, the selector is consulted only where a region has several holders, and
    it sees each request on its own.
    """

    def policy(requests: Sequence[ReplicaRequest]) -> List[ReplicaAnswer]:
        answers: List[ReplicaAnswer] = []
        for request in requests:
            if len(request.candidates) == 1:
                answers.append(request.candidates[0].source_rank)
                continue
            answers.append(
                selector(
                    ReplicaSelectionRequest(target_rank=request.target_location,
                                            logical_origin=request.logical_origin,
                                            shape=request.shape,
                                            target_offset=request.target_offset,
                                            target_strides=request.target_strides,
                                            candidates=request.candidates)))
        return answers

    return policy

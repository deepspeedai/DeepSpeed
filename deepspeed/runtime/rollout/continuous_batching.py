# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Request scheduling primitives for continuous-batching rollouts.

This module deliberately stops at the scheduler/cache boundary. The model
backend owns prompt prefill and decode; the scheduler reports which old cache
rows survive, which requests retire, and which pending requests can be
admitted into the newly free rows.
"""

from collections import deque
from dataclasses import dataclass
from typing import Hashable


@dataclass(frozen=True)
class ContinuousBatchRequest:
    """A request waiting for a slot in a continuous decode batch."""

    request_id: Hashable
    max_new_tokens: int

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass(frozen=True)
class ContinuousBatchUpdate:
    """Result of one scheduler transition.

    ``keep_slots`` indexes the previous active batch. The caller should
    compact the KV cache with these indices, then prefill ``admitted`` into
    the free rows at the end of the compacted batch.
    """

    active: tuple[ContinuousBatchRequest, ...]
    keep_slots: tuple[int, ...]
    retired: tuple[Hashable, ...]
    admitted: tuple[ContinuousBatchRequest, ...]

    @property
    def active_ids(self) -> tuple[Hashable, ...]:
        return tuple(request.request_id for request in self.active)

    @property
    def admitted_slots(self) -> tuple[int, ...]:
        """Rows available for prefilling the newly admitted requests."""
        start = len(self.active) - len(self.admitted)
        return tuple(range(start, len(self.active)))


class ContinuousBatchScheduler:
    """FIFO scheduler for bounded, slot-based continuous batching.

    ``schedule`` performs admission/retirement without advancing tokens.
    ``advance`` represents one decode step for every active request and also
    retires requests whose token budget has been consumed. A caller may pass
    explicit finished IDs when the model emits EOS before that budget.
    """

    def __init__(self, max_batch_size: int):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.max_batch_size = max_batch_size
        self._pending = deque()
        self._active = []
        self._generated = {}
        self._known_ids = set()

    @property
    def active(self) -> tuple[ContinuousBatchRequest, ...]:
        return tuple(request for request, _ in self._active)

    @property
    def pending(self) -> tuple[ContinuousBatchRequest, ...]:
        return tuple(self._pending)

    def submit(self, request: ContinuousBatchRequest) -> None:
        if not isinstance(request, ContinuousBatchRequest):
            raise TypeError("request must be a ContinuousBatchRequest")
        if request.request_id in self._known_ids:
            raise ValueError(f"duplicate request_id: {request.request_id!r}")
        self._known_ids.add(request.request_id)
        self._pending.append(request)

    def schedule(self, finished_ids=()) -> ContinuousBatchUpdate:
        finished_ids = tuple(finished_ids)
        active_by_id = {request.request_id: slot for slot, (request, _) in enumerate(self._active)}
        unknown = set(finished_ids) - active_by_id.keys()
        if unknown:
            raise ValueError(f"finished request is not active: {next(iter(unknown))!r}")

        finished = set(finished_ids)
        survivors = [(request, self._generated[request.request_id]) for request, _ in self._active
                     if request.request_id not in finished]
        keep_slots = tuple(slot for slot, (request, _) in enumerate(self._active)
                           if request.request_id not in finished)
        retired = tuple(request.request_id for request, _ in self._active if request.request_id in finished)

        free_slots = self.max_batch_size - len(survivors)
        admitted = []
        for _ in range(free_slots):
            if not self._pending:
                break
            request = self._pending.popleft()
            admitted.append(request)
            survivors.append((request, 0))
            self._generated[request.request_id] = 0

        self._active = survivors
        for request_id in retired:
            self._generated.pop(request_id, None)
            self._known_ids.discard(request_id)
        return ContinuousBatchUpdate(tuple(request for request, _ in survivors), keep_slots, retired, tuple(admitted))

    def advance(self, finished_ids=()) -> ContinuousBatchUpdate:
        explicit_finished = set(finished_ids)
        active_ids = {request.request_id for request, _ in self._active}
        unknown = explicit_finished - active_ids
        if unknown:
            raise ValueError(f"finished request is not active: {next(iter(unknown))!r}")
        finished = set(explicit_finished)
        updated = []
        for request, generated in self._active:
            generated += 1
            self._generated[request.request_id] = generated
            if generated >= request.max_new_tokens:
                finished.add(request.request_id)
            updated.append((request, generated))
        self._active = updated
        return self.schedule(finished)

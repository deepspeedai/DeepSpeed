# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Request scheduling primitives for continuous-batching rollouts.

This module deliberately stops at the scheduler/cache boundary. The model
backend owns prompt prefill and decode; the scheduler reports which old cache
rows survive, which requests retire, and which pending requests fill newly
free rows.
"""

import asyncio
import concurrent.futures
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Hashable, Optional

import torch

from deepspeed.runtime.rollout.base import RolloutRecord
from deepspeed.runtime.rollout.utils import update_sample_from_response


@dataclass(frozen=True)
class ContinuousBatchRequest:
    """A request waiting for a slot in a continuous decode batch."""

    request_id: Hashable
    # Tokens the request may generate; None uses the scheduler's max_new_tokens.
    max_new_tokens: Optional[int] = None


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

    A request generates up to its own ``max_new_tokens``, or the scheduler's when
    it has none; requests may still retire earlier when the model emits EOS.
    ``schedule`` performs admission/retirement without advancing tokens.
    ``advance`` represents one decode step for every active request and also
    retires requests whose token budget has been consumed. A caller may pass
    explicit finished IDs when the model emits EOS before that budget.
    """

    def __init__(self, max_batch_size: int, max_new_tokens: int):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens
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

    def submit(self, request: ContinuousBatchRequest, front: bool = False) -> None:
        """Queue a request, optionally ahead of those already waiting; a retired request may return."""
        if not isinstance(request, ContinuousBatchRequest):
            raise TypeError("request must be a ContinuousBatchRequest")
        if request.request_id in self._known_ids:
            raise ValueError(f"duplicate request_id: {request.request_id!r}")
        if request.max_new_tokens is not None and request.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self._known_ids.add(request.request_id)
        if front:
            self._pending.appendleft(request)
        else:
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
            limit = request.max_new_tokens if request.max_new_tokens is not None else self.max_new_tokens
            if generated >= limit:
                finished.add(request.request_id)
            updated.append((request, generated))
        self._active = updated
        return self.schedule(finished)


@dataclass
class ContinuousGraphState:
    signature: tuple
    cache: object
    static_input: torch.Tensor
    static_attention: torch.Tensor
    static_causal_attention: torch.Tensor
    static_position_ids: torch.Tensor
    static_write_positions: torch.Tensor
    static_cache_position: torch.Tensor
    graph: object
    static_logits: torch.Tensor


class ToolQueue:
    """The tool queue of a continuous-batching rollout.

    A request retired from the batch either leaves for good or waits here while the tools it called run on a
    background event loop, so the other rows keep decoding. Once they finish, it leaves for the front of the pending
    queue with its tool output appended.
    """

    def __init__(self, rollout, trajectories, max_new_tokens):
        self.rollout = rollout
        self.trajectories = trajectories
        self.max_new_tokens = max_new_tokens
        self.generated = [0] * len(trajectories)
        self.turns = [0] * len(trajectories)
        self.tool_ms = 0.0
        # Trajectory -> (response, log-probs, calls, future, submit time) while its tools run.
        self.in_flight = {}
        self._loop = None
        self._thread = None

    def __enter__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="rollout-tools", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc_info):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()
        return False

    @property
    def busy(self):
        return bool(self.in_flight)

    def retire(self, i, response, log_probs):
        """Take trajectory ``i`` from the batch: record its response, then end it or start the tools it called."""
        rollout = self.rollout
        sample = self.trajectories[i]
        turn = self.turns[i]
        self.turns[i] += 1
        self.generated[i] += len(response)
        calls = rollout._end_turn(sample, response, turn, self.generated[i], self.max_new_tokens)
        if calls is None:
            update_sample_from_response(sample, response, log_probs, [], rollout.tokenizer)
            return
        future = asyncio.run_coroutine_threadsafe(rollout._run_tool_calls({i: calls}), self._loop)
        self.in_flight[i] = (response, log_probs, calls, future, time.perf_counter())

    def pop_ready(self):
        """Append the output of finished tools, and return the trajectories that go back to the pending queue."""
        rollout = self.rollout
        ready = []
        for i in [i for i, entry in self.in_flight.items() if entry[3].done()]:
            response, log_probs, calls, future, submitted = self.in_flight.pop(i)
            self.tool_ms += (time.perf_counter() - submitted) * 1000.0
            tool_response = rollout._tool_response(calls, future.result()[i])
            sample = self.trajectories[i]
            if not rollout._tool_output_fits(sample, response, tool_response):
                sample.status = RolloutRecord.Status.TRUNCATED
                update_sample_from_response(sample, response, log_probs, [], rollout.tokenizer)
                continue
            update_sample_from_response(sample, response, log_probs, tool_response, rollout.tokenizer)
            ready.append(i)
        return ready

    def wait(self):
        """Block until one of the running tools finishes."""
        concurrent.futures.wait([entry[3] for entry in self.in_flight.values()],
                                return_when=concurrent.futures.FIRST_COMPLETED)

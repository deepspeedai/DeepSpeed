# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Continuous batching for HybridEngineRollout.

A scheduler assigns requests to decode rows, and two run classes carry out one
rollout call, with graph capture and without it.
"""

import asyncio
import concurrent.futures
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Hashable, Optional

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutRecord
from deepspeed.runtime.rollout.utils import get_pad_token_id, left_pad_samples, update_sample_from_response


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
class _PromptRow:
    """One left-padded prompt."""
    prompt_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor


class _ContinuousRunBase:
    """Turn and tool bookkeeping shared by both continuous-batching runs of a rollout call."""

    def __init__(self, rollout, trajectories, sampling):
        self.rollout = rollout
        self.trajectories = trajectories
        self.sampling = sampling
        self.generated = [0] * len(trajectories)
        self.turns_done = [0] * len(trajectories)
        # Trajectory -> (response, log-probs, calls, future, submit time) while its tools run.
        self.tools_in_flight = {}
        self.tool_ms = 0.0
        self._tool_loop = None
        self._tool_thread = None

    def __enter__(self):
        self._tool_loop = asyncio.new_event_loop()
        self._tool_thread = threading.Thread(target=self._tool_loop.run_forever, name="rollout-tools", daemon=True)
        self._tool_thread.start()
        return self

    def __exit__(self, *_exc_info):
        self._tool_loop.call_soon_threadsafe(self._tool_loop.stop)
        self._tool_thread.join()
        self._tool_loop.close()
        return False

    def requeue(self, i):
        raise NotImplementedError

    def finish_turn(self, i, response, log_probs):
        """Record a finished turn, or start the tools it called."""
        rollout = self.rollout
        sample = self.trajectories[i]
        turn = self.turns_done[i]
        self.turns_done[i] += 1
        self.generated[i] += len(response)
        calls = rollout._end_turn(sample, response, turn, self.generated[i], self.sampling.max_new_tokens)
        if calls is None:
            update_sample_from_response(sample, response, log_probs, [], rollout.tokenizer)
            return
        future = asyncio.run_coroutine_threadsafe(rollout._run_tool_calls({i: calls}), self._tool_loop)
        self.tools_in_flight[i] = (response, log_probs, calls, future, time.perf_counter())

    def collect_tool_results(self):
        """Add the output of finished tools and requeue their trajectories."""
        for i in [i for i, entry in self.tools_in_flight.items() if entry[3].done()]:
            response, log_probs, calls, future, submitted = self.tools_in_flight.pop(i)
            self.tool_ms += (time.perf_counter() - submitted) * 1000.0
            tool_response = self.rollout._tool_response(calls, future.result()[i])
            sample = self.trajectories[i]
            if not self.rollout._tool_output_fits(sample, response, tool_response):
                sample.status = RolloutRecord.Status.TRUNCATED
                update_sample_from_response(sample, response, log_probs, [], self.rollout.tokenizer)
                continue
            update_sample_from_response(sample, response, log_probs, tool_response, self.rollout.tokenizer)
            self.requeue(i)

    def wait_for_a_tool(self):
        concurrent.futures.wait([entry[3] for entry in self.tools_in_flight.values()],
                                return_when=concurrent.futures.FIRST_COMPLETED)


class _ContinuousRunGraph(_ContinuousRunBase):
    """Continuous batching on a captured graph, with each row decoding at its own position."""

    def __init__(self, rollout, trajectories, sampling, num_prompts, capacity, selector):
        super().__init__(rollout, trajectories, sampling)
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        self.module = module = rollout.engine.module
        self.num_prompts = num_prompts
        self.num_rows = num_rows = rollout.continuous_batch_size
        self.capacity = capacity
        self.selector = selector
        self.accelerator = get_accelerator()
        self.call_start = time.perf_counter()
        self.device = device = self.accelerator.current_device_name()
        self.model_dtype = model_dtype = next(module.parameters()).dtype
        self.logits_kwargs = rollout._last_logits_kwargs(module)
        self.stats = {"prefill_ms": 0.0, "decode_ms": 0.0, "steps": 0, "turns": 0}
        self.pending = deque(range(len(trajectories)))
        self.row_owner = [None] * num_rows

        self.row_indices = torch.arange(num_rows, device=device)
        self.slots = torch.arange(capacity, device=device)
        self.ds_cache = DeepSpeedStaticCache(module.config,
                                             batch_size=num_rows,
                                             max_cache_len=capacity,
                                             device=device,
                                             dtype=model_dtype)
        self.write_pos = torch.zeros(num_rows, dtype=torch.long, device=device)
        self.ds_cache.set_write_position(self.write_pos)
        self.static_token = torch.zeros(num_rows, 1, dtype=torch.long, device=device)
        self.static_attn = torch.zeros(num_rows, 1, 1, capacity, dtype=torch.bool, device=device)
        # A row that never held a turn still attends one slot, so its throwaway logits stay finite.
        self.static_attn[:, 0, 0, 0] = True
        # Eager attention adds its mask to the attention weights, so the model gets an additive copy of static_attn.
        self.decode_mask = torch.zeros(num_rows, 1, 1, capacity, dtype=model_dtype, device=device)
        self.static_cache_pos = self.write_pos[:1]
        self.static_pos_ids = self.write_pos.unsqueeze(1)
        # Each row's turn on the device, so decode steps never wait for the host.
        self.active = torch.zeros(num_rows, dtype=torch.bool, device=device)
        self.num_generated = torch.zeros(num_rows, dtype=torch.long, device=device)
        self.turn_limit = torch.zeros(num_rows, dtype=torch.long, device=device)
        max_new_tokens = sampling.max_new_tokens
        self.turn_len = max(1, min(capacity, max_new_tokens, sampling.max_new_tokens_per_turn or max_new_tokens))
        self.token_buf = torch.zeros(num_rows, self.turn_len, dtype=torch.long, device=device)
        self.log_prob_buf = None
        if rollout.return_log_probs:
            self.log_prob_buf = torch.zeros(num_rows, self.turn_len, dtype=torch.float32, device=device)
        eos_token_id = rollout.tokenizer.eos_token_id
        self.stop_token = eos_token_id if eos_token_id is not None else -1
        self.step_trace = [] if rollout.enable_profiling else None
        self.decoded = torch.zeros((), dtype=torch.long, device=device)
        self.graph = None
        self.static_logits = None

    def requeue(self, i):
        # A trajectory in progress goes before one not yet started.
        self.pending.appendleft(i)

    def turn_budget(self, i):
        sample = self.trajectories[i]
        budget = self.rollout._turn_budget(sample, self.generated[i], self.sampling)
        return min(budget, self.capacity - sum(sample.attention_mask))

    def admit(self):
        """Prefill queued turns into free rows, and report whether any was admitted."""
        free_rows = [row for row, owner in enumerate(self.row_owner) if owner is None]
        admitted = []
        while free_rows and self.pending:
            i = self.pending.popleft()
            budget = self.turn_budget(i)
            if budget <= 0:
                self.trajectories[i].status = RolloutRecord.Status.TRUNCATED
                continue
            admitted.append((free_rows.pop(0), i, budget))
        if not admitted:
            return False
        device = self.device
        prefill_start = self.rollout._sync_time(self.accelerator)
        target = torch.tensor([row for row, _, _ in admitted], device=device)
        prompt_ids, prompt_attn = left_pad_samples([self.trajectories[i] for _, i, _ in admitted],
                                                   get_pad_token_id(self.rollout.tokenizer))
        prompt_ids = prompt_ids.to(device)
        prompt_attn = prompt_attn.to(device)
        lengths = prompt_attn.sum(dim=1)
        logits = self.rollout._prefill_rows(self.module, prompt_ids, prompt_attn, self.ds_cache, self.logits_kwargs,
                                            self.model_dtype, device, target, self.num_rows)
        first, first_log_probs = self.selector.first(logits, prompt_ids, prompt_attn, target)
        limits = torch.tensor([budget for _, _, budget in admitted], device=device)
        self.static_attn[target, 0, 0, :] = self.slots.unsqueeze(0) <= lengths.unsqueeze(1)
        self.write_pos[target] = lengths
        self.static_token[target] = first
        self.token_buf[target, 0] = first[:, 0]
        if self.log_prob_buf is not None:
            self.log_prob_buf[target, 0] = first_log_probs[:, 0]
        self.num_generated[target] = 1
        self.turn_limit[target] = limits
        self.active[target] = (first[:, 0] != self.stop_token) & (limits > 1)
        for row, i, _ in admitted:
            self.row_owner[row] = i
        self.stats["turns"] += len(admitted)
        self.stats["prefill_ms"] += (self.rollout._sync_time(self.accelerator) - prefill_start) * 1000.0
        return True

    def forward(self):
        # Inside the captured forward, so every replay sees the rows' current slots.
        self.rollout._update_static_causal_attention(self.static_attn[:, 0, 0], self.decode_mask)
        return self.module(self.static_token,
                           attention_mask=self.decode_mask,
                           past_key_values=self.ds_cache,
                           use_cache=True,
                           cache_position=self.static_cache_pos,
                           position_ids=self.static_pos_ids)

    def capture(self):
        module = self.module
        accelerator = self.accelerator
        # The KV writes of the warmup and capture forwards land in the slots the first real step overwrites.
        # Forward hooks synchronize, which is illegal during graph capture.
        saved_pre = dict(module._forward_pre_hooks)
        saved_post = dict(module._forward_hooks)
        module._forward_pre_hooks.clear()
        module._forward_hooks.clear()
        try:
            stream = accelerator.Stream()
            stream.wait_stream(accelerator.current_stream())
            with accelerator.stream(stream):
                for _ in range(3):
                    self.forward()
            accelerator.current_stream().wait_stream(stream)
            self.graph = accelerator.create_graph()
            with accelerator.capture_to_graph(self.graph):
                self.static_logits = self.forward().logits
        finally:
            module._forward_pre_hooks.update(saved_pre)
            module._forward_hooks.update(saved_post)

    def decode_step(self):
        self.accelerator.replay_graph(self.graph)
        tokens, log_probs = self.selector(self.static_logits[:, -1, :])
        position = self.num_generated.clamp(max=self.turn_len - 1).unsqueeze(1)
        recording = self.active.unsqueeze(1)
        self.token_buf.scatter_(1, position, torch.where(recording, tokens, self.token_buf.gather(1, position)))
        if self.log_prob_buf is not None:
            self.log_prob_buf.scatter_(1, position,
                                       torch.where(recording, log_probs, self.log_prob_buf.gather(1, position)))
        self.num_generated.add_(self.active.long())
        if self.step_trace is not None:
            self.decoded.add_(self.active.sum())
        ended = (tokens[:, 0] == self.stop_token) | (self.num_generated >= self.turn_limit)
        self.active.logical_and_(~ended)
        self.static_token.copy_(tokens)
        self.write_pos.add_(self.active.long())
        self.static_attn[self.row_indices, 0, 0, self.write_pos] |= self.active

    def collect_finished_turns(self, still_active):
        finished = [row for row, owner in enumerate(self.row_owner) if owner is not None and not still_active[row]]
        if not finished:
            return
        finished_rows = torch.tensor(finished, device=self.device)
        counts = self.num_generated[finished_rows].tolist()
        width = max(counts)
        tokens = self.token_buf[finished_rows, :width].tolist()
        log_probs = [None] * len(finished)
        if self.log_prob_buf is not None:
            log_probs = [row[:count] for row, count in zip(self.log_prob_buf[finished_rows, :width].tolist(), counts)]
        for row, count, row_tokens, row_log_probs in zip(finished, counts, tokens, log_probs):
            i = self.row_owner[row]
            self.row_owner[row] = None
            self.finish_turn(i, row_tokens[:count], row_log_probs)

    def run(self):
        stats = self.stats
        with self:
            while any(owner is not None for owner in self.row_owner) or self.pending or self.tools_in_flight:
                if self.admit() and self.graph is None:
                    self.capture()
                rows_held = sum(owner is not None for owner in self.row_owner)
                if rows_held:
                    decode_start = time.perf_counter()
                    self.decode_step()
                    if self.step_trace is None:
                        still_active = self.active.tolist()
                    else:
                        # One read for both, so profiling adds no host sync.
                        *still_active, decoded_so_far = torch.cat((self.active.long(), self.decoded.view(1))).tolist()
                    stats["decode_ms"] += (time.perf_counter() - decode_start) * 1000.0
                    stats["steps"] += 1
                    self.collect_finished_turns(still_active)
                    if self.step_trace is not None:
                        self.step_trace.append((stats["steps"], round(time.perf_counter() - self.call_start,
                                                                      3), rows_held, sum(still_active), decoded_so_far,
                                                len(self.pending), len(self.tools_in_flight)))
                elif self.tools_in_flight and not self.pending:
                    self.wait_for_a_tool()
                self.collect_tool_results()
        if self.rollout.enable_profiling:
            self.rollout._last_profile = self.profile()

    def profile(self):
        stats = self.stats
        total_ms = (self.rollout._sync_time(self.accelerator) - self.call_start) * 1000.0
        num_generated_tokens = sum(self.generated)
        decode_slots = stats["steps"] * self.num_rows
        return {
            "generation_ms":
            stats["prefill_ms"] + stats["decode_ms"],
            "prefill_forward_ms":
            stats["prefill_ms"],
            "decode_forward_ms":
            stats["decode_ms"],
            "num_decode_forwards":
            stats["steps"],
            "tool_ms":
            self.tool_ms,
            "num_generated_tokens":
            num_generated_tokens,
            "num_turns":
            stats["turns"],
            "total_ms":
            total_ms,
            "tokens_per_second":
            num_generated_tokens / (total_ms / 1000.0) if total_ms > 0.0 else 0.0,
            "batch_size":
            self.num_prompts,
            "num_samples_per_prompt":
            self.sampling.n_samples_per_prompt,
            "continuous_rows":
            self.num_rows,
            # The share of decode slots that produced a token, not counting each turn's first, which its prefill produces.
            "continuous_decode_occupancy":
            ((num_generated_tokens - stats["turns"]) / decode_slots if decode_slots else 0.0),
            # Each entry holds the decode steps so far, seconds since the call started, rows holding a turn during the
            # step, rows still decoding after it, tokens decoded so far, turns queued and trajectories waiting on
            # tools.
            "continuous_step_trace":
            self.step_trace,
        }


class _ContinuousRun(_ContinuousRunBase):
    """Continuous batching without graph capture, with all rows writing one shared cache column."""

    def __init__(self, rollout, trajectories, sampling, num_prompts, prompt_len, max_cache_len, selector):
        super().__init__(rollout, trajectories, sampling)
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        self.module = module = rollout.engine.module
        self.num_prompts = num_prompts
        self.max_batch_size = max_batch_size = rollout.continuous_batch_size
        self.align = rollout.align_decode_fronts
        self.prompt_len = prompt_len
        self.max_cache_len = max_cache_len
        self.selector = selector
        self.device = device = next(module.parameters()).device
        self.model_dtype = model_dtype = next(module.parameters()).dtype
        self.profile = rollout._start_continuous_profile() if rollout.enable_profiling else None

        turn_cap = sampling.max_new_tokens
        if sampling.max_new_tokens_per_turn is not None:
            turn_cap = min(turn_cap, sampling.max_new_tokens_per_turn)
        scheduler_start = rollout._profile_start(self.profile)
        self.scheduler = ContinuousBatchScheduler(max_batch_size, turn_cap)
        self.request_by_id = {}
        self.prompt_lengths = {}
        self.responses = {}
        # Each response token's log-prob under the distribution it was picked from.
        self.response_log_probs = {}
        request_order = range(len(trajectories))
        if self.align:
            request_order = sorted(request_order, key=lambda i: -sum(trajectories[i].attention_mask))
        for i in request_order:
            self.submit(i)
        rollout._profile_end(self.profile, "scheduler_overhead_ms", scheduler_start)

        cache_start = rollout._profile_start(self.profile)
        self.cache = DeepSpeedStaticCache(module.config,
                                          batch_size=max_batch_size,
                                          max_cache_len=max_cache_len,
                                          device=device,
                                          dtype=model_dtype)
        self.write_positions = torch.full((max_batch_size, ), -1, dtype=torch.long, device=device)
        self.cache.set_write_position(self.write_positions)
        self.attention_mask = torch.zeros((max_batch_size, max_cache_len), dtype=torch.long, device=device)
        cache_bytes = sum(layer.keys.numel() * layer.keys.element_size() +
                          layer.values.numel() * layer.values.element_size() for layer in self.cache.layers)
        self.stats = {
            "cache_capacity":
            max_cache_len,
            "peak_cache_length":
            prompt_len,
            "cache_memory_bytes":
            cache_bytes + self.attention_mask.numel() * self.attention_mask.element_size() +
            self.write_positions.numel() * self.write_positions.element_size(),
            "trim_count":
            0,
            "trimmed_columns":
            0,
            "trim_latency_ms":
            0.0,
            "decode_steps":
            0,
        }
        self.next_tokens = {}
        self.next_log_probs = {}
        self.logical_positions = {}
        self.cache_position = prompt_len
        self.span_starts = [0] * max_batch_size
        rollout._profile_end(self.profile, "cache_management_overhead_ms", cache_start)

    def submit(self, i, front=False):
        """Queue a trajectory's next turn, or end the trajectory if it has no budget left."""
        sample = self.trajectories[i]
        budget = self.rollout._turn_budget(sample, self.generated[i], self.sampling)
        if budget <= 0:
            sample.status = RolloutRecord.Status.TRUNCATED
            return
        prompt_ids = torch.tensor([sample.tokens], dtype=torch.long, device=self.device)
        prompt_attention_mask = torch.tensor([sample.attention_mask], dtype=torch.long, device=self.device)
        self.request_by_id[i] = _PromptRow(prompt_ids, prompt_attention_mask)
        self.prompt_lengths[i] = sum(sample.attention_mask) if self.align else len(sample.tokens)
        self.responses[i] = []
        self.response_log_probs[i] = []
        self.scheduler.submit(ContinuousBatchRequest(i, max_new_tokens=budget), front=front)

    def requeue(self, i):
        # A trajectory in progress goes before one not yet started.
        self.submit(i, front=True)

    def finish_retired_turn(self, i):
        response = torch.cat(self.responses[i], dim=1)[0].tolist()
        log_probs = None
        if self.rollout.return_log_probs:
            log_probs = torch.cat(self.response_log_probs[i], dim=1)[0].tolist()
        self.finish_turn(i, response, log_probs)

    def trim(self, dead_prefix, survivor_count):
        profile_accelerator = self.profile["accelerator"] if self.profile is not None else None
        trim_start = None
        if profile_accelerator is not None:
            profile_accelerator.synchronize()
            trim_start = time.perf_counter()
        cache = self.cache
        attention_mask = self.attention_mask
        moved_bytes = sum((layer.keys[:, :, dead_prefix:, :].numel() * layer.keys.element_size() +
                           layer.values[:, :, dead_prefix:, :].numel() * layer.values.element_size())
                          for layer in cache.layers)
        moved_bytes += attention_mask[:, dead_prefix:].numel() * attention_mask.element_size()
        cache.trim_left(dead_prefix)
        remaining_columns = self.max_cache_len - dead_prefix
        if remaining_columns:
            attention_mask[:, :remaining_columns].copy_(attention_mask[:, dead_prefix:].clone())
        attention_mask[:, remaining_columns:].zero_()
        self.write_positions[:survivor_count].sub_(dead_prefix)
        self.cache_position -= dead_prefix
        stats = self.stats
        stats["trim_count"] += 1
        stats["trimmed_columns"] += dead_prefix
        stats["trim_bytes_moved"] = stats.get("trim_bytes_moved", 0) + moved_bytes
        if profile_accelerator is not None:
            profile_accelerator.synchronize()
            stats["trim_latency_ms"] += (time.perf_counter() - trim_start) * 1000.0
        if self.align:
            self.span_starts[:survivor_count] = [start - dead_prefix for start in self.span_starts[:survivor_count]]

    def manage_cache(self, update):
        """Prepare the cache for this step's admissions, and return the number of rows still decoding."""
        rollout = self.rollout
        device = self.device
        keep_slots = torch.tensor(update.keep_slots, dtype=torch.long, device=device)
        survivor_count = keep_slots.numel()
        if survivor_count:
            if update.retired:
                self.cache.compact(keep_slots)
                self.selector.keep(keep_slots)
                identity = torch.arange(survivor_count, dtype=torch.long, device=device)
                if not torch.equal(keep_slots, identity):
                    survivor_attention = self.attention_mask.index_select(0, keep_slots).clone()
                    self.attention_mask[:survivor_count].copy_(survivor_attention)
                self.attention_mask[survivor_count:].zero_()
                if self.align:
                    self.span_starts[:survivor_count] = [self.span_starts[index] for index in update.keep_slots]
                    self.span_starts[survivor_count:] = [0] * (self.max_batch_size - survivor_count)

            dead_prefix = 0
            if rollout.enable_cache_trimming:
                if self.align:
                    dead_prefix = min(self.span_starts[:survivor_count])
                else:
                    trim_threshold = max(1, self.prompt_len)
                    dead_prefix = rollout._continuous_dead_prefix(self.attention_mask, survivor_count)
                    if dead_prefix < trim_threshold and self.cache_position < self.max_cache_len - trim_threshold:
                        dead_prefix = 0
            elif self.cache_position >= self.max_cache_len:
                if self.align:
                    dead_prefix = min(self.span_starts[:survivor_count])
                else:
                    dead_prefix = rollout._continuous_dead_prefix(self.attention_mask, survivor_count)
            if dead_prefix:
                if update.admitted:
                    longest_admitted = max(self.prompt_lengths[request.request_id] for request in update.admitted)
                    dead_prefix = min(dead_prefix, max(0, self.cache_position - longest_admitted))
                self.trim(dead_prefix, survivor_count)
        else:
            self.cache.reset()
            self.write_positions.fill_(-1)
            self.attention_mask.zero_()
            self.cache_position = self.prompt_len
            self.span_starts = [0] * self.max_batch_size

        if update.admitted:
            # A returning turn's context can be longer than the column; the column moves forward to it, and the
            # live rows' masks skip the columns in between.
            longest_admitted = max(self.prompt_lengths[request.request_id] for request in update.admitted)
            self.cache_position = max(self.cache_position, longest_admitted)
        return survivor_count

    def decode(self, update, survivor_count):
        """Decode one step for the live rows."""
        rollout = self.rollout
        device = self.device
        survivor_ids = update.active_ids[:survivor_count]
        decode_input = torch.cat([self.next_tokens[request_id] for request_id in survivor_ids], dim=0)
        self.write_positions[:survivor_count].fill_(self.cache_position)
        if self.align:
            position_ids = torch.tensor([self.logical_positions[request_id] for request_id in survivor_ids],
                                        dtype=torch.long,
                                        device=device).unsqueeze(1)
        else:
            position_ids = self.attention_mask[:survivor_count, :self.cache_position].sum(dim=1, keepdim=True)
        self.attention_mask[:survivor_count, self.cache_position] = 1
        decode_start = rollout._profile_start(self.profile)
        output = rollout._call_model(
            self.module,
            decode_input,
            attention_mask=self.attention_mask[:survivor_count],
            past_key_values=self.cache,
            use_cache=True,
            cache_position=torch.tensor([self.cache_position], dtype=torch.long, device=device),
            position_ids=position_ids,
        )
        rollout._profile_end(self.profile, "decode_forward_ms", decode_start, count="num_decode_forwards")
        decoded, decoded_log_probs = self.selector(output.logits[:, -1, :])
        self.stats["decode_steps"] += 1
        for request_id in survivor_ids:
            self.logical_positions[request_id] += 1
        decoded_log_probs_by_id = {}
        if decoded_log_probs is not None:
            decoded_log_probs_by_id = dict(zip(survivor_ids, decoded_log_probs.split(1, dim=0)))
        return dict(zip(survivor_ids, decoded.split(1, dim=0))), decoded_log_probs_by_id

    def step(self, update):
        """Run one admission and decode step, and return the trajectories whose turns ended in EOS."""
        from transformers import StaticCache

        rollout = self.rollout
        if self.profile is not None:
            self.profile["active_batch_sizes"].append(len(update.active))
        cache_start = rollout._profile_start(self.profile)
        survivor_count = self.manage_cache(update)
        rollout._profile_end(self.profile, "cache_management_overhead_ms", cache_start)
        if self.cache_position > self.max_cache_len or (survivor_count and self.cache_position >= self.max_cache_len):
            raise rollout._continuous_cache_exhaustion_error(self.max_cache_len)
        admitted_tokens, admitted_log_probs = rollout._continuous_prefill(
            self.module,
            StaticCache,
            self.cache,
            update,
            self.request_by_id,
            self.attention_mask,
            self.write_positions,
            self.cache_position,
            self.prompt_lengths,
            self.model_dtype,
            self.device,
            self.sampling,
            self.selector,
            self.align,
            self.profile,
        )
        for admitted, target_row in zip(update.admitted, update.admitted_slots):
            self.logical_positions[admitted.request_id] = self.prompt_lengths[admitted.request_id]
            if self.align:
                self.span_starts[target_row] = self.cache_position - self.prompt_lengths[admitted.request_id]

        decoded_tokens, decoded_log_probs = {}, {}
        if survivor_count:
            decoded_tokens, decoded_log_probs = self.decode(update, survivor_count)

        self.next_tokens = decoded_tokens | admitted_tokens
        self.next_log_probs = decoded_log_probs | admitted_log_probs
        finished_ids = []
        for request_id in update.active_ids:
            token = self.next_tokens[request_id]
            self.responses[request_id].append(token)
            if rollout.return_log_probs:
                self.response_log_probs[request_id].append(self.next_log_probs[request_id])
            if rollout._is_eos(token):
                finished_ids.append(request_id)
        if survivor_count:
            self.cache_position += 1
        self.stats["peak_cache_length"] = max(self.stats["peak_cache_length"], self.cache_position)
        return finished_ids

    def run(self):
        rollout = self.rollout
        scheduler = self.scheduler
        with self:
            scheduler_start = rollout._profile_start(self.profile)
            update = scheduler.schedule()
            rollout._profile_end(self.profile, "scheduler_overhead_ms", scheduler_start)
            while update.active or self.tools_in_flight:
                if not update.active:
                    self.wait_for_a_tool()
                    self.collect_tool_results()
                    update = scheduler.schedule()
                    continue
                finished_ids = self.step(update)
                scheduler_start = rollout._profile_start(self.profile)
                update = scheduler.advance(finished_ids)
                rollout._profile_end(self.profile, "scheduler_overhead_ms", scheduler_start)
                for request_id in update.retired:
                    self.finish_retired_turn(request_id)
                self.collect_tool_results()
        self.finish()

    def finish(self):
        rollout = self.rollout
        stats = self.stats
        profile = self.profile
        generation_end = rollout._profile_start(profile)
        if profile is not None:
            profile["accelerator"].synchronize()
            total_ms = (time.perf_counter() - profile["start"]) * 1000.0
            stats["end_to_end_ms"] = total_ms
            stats["tokens_per_second"] = sum(self.generated) / (total_ms / 1000.0) if total_ms > 0.0 else 0.0
        else:
            stats["end_to_end_ms"] = None
            stats["tokens_per_second"] = None
        stats.setdefault("trim_bytes_moved", 0)
        stats["trim_frequency"] = (stats["trim_count"] / stats["decode_steps"] if stats["decode_steps"] else 0.0)
        rollout._last_continuous_stats = stats
        if profile is not None:
            rollout._finish_continuous_profile(profile, self.num_prompts, self.sampling.n_samples_per_prompt,
                                               self.generated,
                                               [sample.response_length for sample in self.trajectories],
                                               sum(self.turns_done), self.tool_ms, self.max_batch_size,
                                               self.prompt_len, generation_end, generation_end)

# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engine backed by DeepSpeed's hybrid engine.

Two generation paths:
  1. **model.generate()** (default): delegates to HuggingFace generate.
     Supports sampling (temperature, top_p) and greedy.
  2. **graph capture + DeepSpeedStaticCache**: greedy, or sampling with
     temperature, top_k and top_p. Pre-allocates a StaticCache, captures
     the decode forward pass with a CUDA graph, and replays it for each decode
     step.  Eliminates kernel launch overhead.
  3. **continuous batching (experimental)**: an opt-in bounded path, greedy or sampled like the graph path,
     selected through ``HybridEngineRolloutConfig.continuous_batch_size``.
"""

import asyncio
import concurrent.futures
import threading
import time
from collections import deque
from copy import copy
from dataclasses import dataclass, field, replace
from inspect import signature
from typing import Optional

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutEngine, RolloutRecord, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.continuous_batching import ContinuousBatchRequest, ContinuousBatchScheduler
from deepspeed.runtime.rollout.tool import TOOL_CALL_PARSERS, ToolSet
from deepspeed.runtime.rollout.utils import (
    expand_samples,
    left_pad_samples,
    masked_lists,
    render_prompts,
    tokenize_tool_responses,
    update_sample_from_response,
)


class _ForwardProfiler:
    """Measure top-level model forwards without synchronizing each call."""

    def __init__(self, accelerator):
        self.event_type = accelerator.Event
        self.use_host_timer = accelerator.use_host_timers()
        # Host intervals are only reliable when there is no asynchronous device
        # work, as on CPU. MPS exposes events but cannot time them, so its host
        # intervals would measure enqueue latency rather than model execution.
        self.is_available = not self.use_host_timer or self.event_type is None
        self.active_start = None
        self.measurements = []

    def register(self, module):
        if not self.is_available:
            return ()
        pre_handle = module.register_forward_pre_hook(self._start_forward)
        post_handle = module.register_forward_hook(self._end_forward)
        return pre_handle, post_handle

    def get_stage_times(self):
        if not self.measurements:
            return None, None, 0

        durations = [self._elapsed_time(start, end) for start, end in self.measurements]
        return durations[0], sum(durations[1:]), len(durations) - 1

    def _start_forward(self, _module, _args):
        self.active_start = self._record_time()

    def _end_forward(self, _module, _args, _output):
        end = self._record_time()
        self.measurements.append((self.active_start, end))
        self.active_start = None

    def _record_time(self):
        if self.use_host_timer:
            return time.perf_counter()
        event = self.event_type(enable_timing=True)
        event.record()
        return event

    def _elapsed_time(self, start, end):
        if self.use_host_timer:
            return (end - start) * 1000.0
        return start.elapsed_time(end)


def _sampled_log_prob_recorder(temperature: float = 1.0):
    """Return ``(processor, finish)`` that record each sampled token's log-probability."""
    steps = []
    previous = []

    def processor(input_ids, scores):
        if previous:
            steps.append(previous.pop().gather(1, input_ids[:, -1:]))
        previous.append(torch.log_softmax(scores.float() / temperature, dim=-1))
        return scores

    def finish(last_tokens):
        steps.append(previous.pop().gather(1, last_tokens))
        return torch.cat(steps, dim=1)

    return processor, finish


def _chosen_log_probs(logits, tokens):
    return torch.log_softmax(logits.float(), dim=-1).gather(1, tokens)


def _token_selector(sampling):
    """A function picking the next tokens ``[B, 1]`` from logits ``[B, V]``, returning them with their log-probs.

    Tokens are picked outside any captured graph. Sampling applies temperature, top-k and top-p in the order
    ``generate()`` applies them, in float32 as it does, so every path draws from the same distribution.
    """
    if sampling.temperature <= 0.0:

        def select_greedy(logits):
            tokens = logits.argmax(dim=-1, keepdim=True)
            return tokens, _chosen_log_probs(logits, tokens)

        return select_greedy

    warpers = []
    if sampling.top_k > 0 or sampling.top_p < 1.0:
        from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper
        if sampling.top_k > 0:
            warpers.append(TopKLogitsWarper(top_k=sampling.top_k))
        if sampling.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=sampling.top_p))

    def select_sampled(logits):
        scores = logits.float() / sampling.temperature
        for warper in warpers:
            # The warpers read only the scores, not the generated ids.
            scores = warper(None, scores)
        tokens = torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)
        return tokens, torch.log_softmax(scores, dim=-1).gather(1, tokens)

    return select_sampled


def _repetition_penalty(sampling, module):
    """``sampling``'s penalty, else the one in the model's ``generation_config``, as ``generate()`` does; else 1.0."""
    if sampling.repetition_penalty is not None:
        return sampling.repetition_penalty
    penalty = getattr(getattr(module, "generation_config", None), "repetition_penalty", None)
    return penalty if isinstance(penalty, (int, float)) else 1.0


def _penalize_repeats(logits, input_ids, repetition_penalty):
    """``logits`` with the tokens of ``input_ids`` penalized, in float32 and before any warper, as ``generate()`` does."""
    if repetition_penalty == 1.0:
        return logits
    logits = logits.float()
    scores = logits.gather(1, input_ids)
    scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
    return logits.scatter(1, input_ids, scores)


@dataclass
class _PromptRow:
    """One left-padded prompt."""
    prompt_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor


@dataclass
class HybridEngineRolloutConfig:
    """Configuration for HybridEngineRollout.

    ``align_decode_fronts`` is opt-in. When false, continuous batching keeps
    the existing equal-width padded layout. Requests may still have different
    effective lengths in their attention masks, but their physical decode
    fronts are not right-aligned or compacted. When true, the rollout uses
    effective prompt lengths to align physical decode fronts.

    Cache trimming is opt-in because it copies the live cache columns after
    each reclaim. Without it, the rollout trims only when the cache would
    otherwise exhaust and a dead prefix can be reclaimed.
    """
    use_graph_capture: bool = False
    enable_profiling: bool = False
    use_shared_prefill: bool = False
    align_decode_fronts: bool = False
    enable_cache_trimming: bool = False
    multi_turn: bool = False
    continuous_cache_capacity: Optional[int] = None
    max_turns: Optional[int] = None
    max_context_len: Optional[int] = None
    chat_template_kwargs: dict = field(default_factory=dict)
    # Format of the model's tool calls, a key of TOOL_CALL_PARSERS: "hermes" (Qwen2.5, Qwen3) or
    # "qwen3_coder" (Qwen3-Coder, Qwen3.5).
    tool_call_parser: str = "hermes"
    continuous_batch_size: Optional[int] = None
    return_log_probs: bool = False

    def __post_init__(self) -> None:
        if self.multi_turn and not self.max_turns:
            raise ValueError("Max turns not specified when using multi-turn rollout")
        if self.continuous_batch_size is not None and self.continuous_batch_size < 1:
            raise ValueError("continuous_batch_size must be positive")
        if self.tool_call_parser not in TOOL_CALL_PARSERS:
            raise ValueError(f"unknown tool_call_parser {self.tool_call_parser!r}; "
                             f"choose from {', '.join(TOOL_CALL_PARSERS)}")


@dataclass
class _ContinuousGraphState:
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


class HybridEngineRollout(RolloutEngine):
    """Rollout engine using DeepSpeed hybrid engine.

    Args:
        engine: DeepSpeed engine wrapping the model.
        tokenizer: HuggingFace tokenizer (must have pad_token_id or eos_token_id).
        cfg: Optional HybridEngineRolloutConfig.
        tools: Optional ToolSet the model may call in multi-turn rollout.
    """

    def __init__(self, engine, tokenizer, cfg=None, tools: Optional[ToolSet] = None):
        self.engine = engine
        self.tokenizer = tokenizer
        self.tools = tools if tools is not None else ToolSet([])
        self.use_graph_capture = getattr(cfg, 'use_graph_capture', False) if cfg else False
        self.enable_profiling = getattr(cfg, 'enable_profiling', False) if cfg else False
        self.use_shared_prefill = getattr(cfg, 'use_shared_prefill', False) if cfg else False
        self.align_decode_fronts = getattr(cfg, 'align_decode_fronts', False) if cfg else False
        self.enable_cache_trimming = getattr(cfg, 'enable_cache_trimming', False) if cfg else False
        self.continuous_cache_capacity = getattr(cfg, 'continuous_cache_capacity', None) if cfg else None
        self.continuous_batch_size = getattr(cfg, 'continuous_batch_size', None) if cfg else None
        self.return_log_probs = getattr(cfg, 'return_log_probs', False) if cfg else False
        self.multi_turn = getattr(cfg, 'multi_turn', False) if cfg else False
        self.max_turns = getattr(cfg, 'max_turns', None) if cfg else None
        self.max_context_len = getattr(cfg, 'max_context_len', None) if cfg else None
        self.chat_template_kwargs = (getattr(cfg, 'chat_template_kwargs', None) if cfg else None) or {}
        self.parse_tool_calls = TOOL_CALL_PARSERS[getattr(cfg, 'tool_call_parser', 'hermes') if cfg else 'hermes']
        self._last_profile = None
        self._continuous_graph_state = None
        self._last_continuous_stats = None

    @torch.no_grad()
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        if self.multi_turn:
            return self._generate_multi_turn(request, sampling)
        return self._generate_single_turn(request, sampling)

    @torch.no_grad()
    def _generate_single_turn(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        samples = request.samples
        if not samples:
            return RolloutBatch(samples=[])
        self._check_sampling(sampling)
        if sampling.max_new_tokens_per_turn is not None:
            max_new_tokens = min(sampling.max_new_tokens, sampling.max_new_tokens_per_turn)
            sampling = replace(sampling, max_new_tokens=max_new_tokens, max_new_tokens_per_turn=None)
        if self.continuous_batch_size is not None:
            return self._generate_continuous(samples, sampling, self.continuous_batch_size)
        device = get_accelerator().current_device_name()
        n = sampling.n_samples_per_prompt
        pad_token_id = self.tokenizer.pad_token_id
        module = self.engine.module

        if self.enable_profiling:
            accelerator = get_accelerator()
            accelerator.synchronize()
            profile_start = time.perf_counter()

        samples = render_prompts(samples, self.tokenizer, self.tools.specs, pad_token_id, self.chat_template_kwargs)
        # Copies of a prompt are adjacent, the layout shared prefill expects.
        trajectories = expand_samples(samples, n)
        prompt_ids, prompt_attn = left_pad_samples(trajectories, pad_token_id)
        prompt_len = prompt_ids.shape[1]
        prompt_ids = prompt_ids.to(device)
        prompt_attn = prompt_attn.to(device)

        if self.enable_profiling:
            accelerator.synchronize()
            expansion_end = time.perf_counter()

        forward_profiler = None
        forward_profile_handles = []
        if self.enable_profiling and not self._graph_capture_enabled():
            forward_profiler = _ForwardProfiler(accelerator)
            forward_profile_handles = forward_profiler.register(module)

        shared_prefill_handles = []
        try:
            if self.use_shared_prefill and n > 1:
                if self.use_graph_capture:
                    raise RuntimeError("Shared prefill does not support graph capture")
                self.engine.prepare_shared_prefill(len(samples), n, prompt_len)
                shared_prefill_handles = self._register_shared_prefill_hooks(module, len(samples), n)
            output_ids, log_probs = self._generate_tokens(prompt_ids, prompt_attn, sampling.max_new_tokens, sampling,
                                                          pad_token_id)
        finally:
            for handle in shared_prefill_handles:
                handle.remove()
            for handle in forward_profile_handles:
                handle.remove()

        if self.enable_profiling:
            accelerator.synchronize()
            generation_end = time.perf_counter()

        # Keep each response up to its first EOS: under ZeRO-3, generation
        # ran past it, and rows that stopped early are padded either way.
        output_ids, response_attn = self._pad_after_eos(
            output_ids,
            response_start=prompt_len,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
        )
        responses = masked_lists(output_ids[:, prompt_len:], response_attn)
        response_log_probs = masked_lists(log_probs, response_attn) if self.return_log_probs else None

        for row, sample in enumerate(trajectories):
            response = responses[row]
            # A response that did not reach EOS was cut by max_new_tokens.
            if response and response[-1] == self.tokenizer.eos_token_id:
                sample.status = RolloutRecord.Status.COMPLETED
            else:
                sample.status = RolloutRecord.Status.TRUNCATED
            row_log_probs = response_log_probs[row] if response_log_probs is not None else None
            update_sample_from_response(sample, response, row_log_probs, [], self.tokenizer)

        if self.enable_profiling:
            accelerator.synchronize()
            post_processing_end = time.perf_counter()
            prompt_expansion_ms = (expansion_end - profile_start) * 1000.0
            generation_ms = (generation_end - expansion_end) * 1000.0
            post_processing_ms = (post_processing_end - generation_end) * 1000.0
            total_ms = (post_processing_end - profile_start) * 1000.0
            prefill_forward_ms = None
            decode_forward_ms = None
            num_decode_forwards = 0
            if forward_profiler is not None:
                prefill_forward_ms, decode_forward_ms, num_decode_forwards = forward_profiler.get_stage_times()
            generation_overhead_ms = generation_ms
            if prefill_forward_ms is not None:
                generation_overhead_ms -= prefill_forward_ms + decode_forward_ms
                generation_overhead_ms = max(0.0, generation_overhead_ms)
            response_length = int(output_ids.shape[1] - prompt_len)
            num_generated_tokens = int(output_ids.shape[0] * response_length)
            tokens_per_second = 0.0
            if total_ms > 0.0:
                tokens_per_second = num_generated_tokens / (total_ms / 1000.0)
            self._last_profile = {
                "prompt_expansion_ms": prompt_expansion_ms,
                "generation_ms": generation_ms,
                "prefill_forward_ms": prefill_forward_ms,
                "decode_forward_ms": decode_forward_ms,
                "generation_overhead_ms": generation_overhead_ms,
                "num_decode_forwards": num_decode_forwards,
                "post_processing_ms": post_processing_ms,
                "total_ms": total_ms,
                "num_generated_tokens": num_generated_tokens,
                "tokens_per_second": tokens_per_second,
                "batch_size": len(samples),
                "num_samples_per_prompt": n,
                "prompt_length": prompt_len,
                "response_length": response_length,
            }

        return RolloutBatch(samples=trajectories)

    @torch.no_grad()
    def _generate_multi_turn(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        if self.continuous_batch_size is not None:
            return self._generate_multi_turn_continuous(request, sampling)

        samples = request.samples
        if not samples:
            return RolloutBatch(samples=[])
        self._check_sampling(sampling)
        device = get_accelerator().current_device_name()
        n = sampling.n_samples_per_prompt
        max_new_tokens = sampling.max_new_tokens
        return_log_probs = self.return_log_probs
        pad_token_id = self.tokenizer.pad_token_id
        module = self.engine.module

        if self.enable_profiling:
            accelerator = get_accelerator()
            accelerator.synchronize()
            call_start = time.perf_counter()
            # Summed over turns, so the profile covers the whole multi-turn rollout.
            profile = {
                "prompt_expansion_ms": 0.0,
                "generation_ms": 0.0,
                "prefill_forward_ms": 0.0,
                "decode_forward_ms": 0.0,
                "num_decode_forwards": 0,
                "tool_ms": 0.0,
                "post_processing_ms": 0.0,
                "num_generated_tokens": 0,
                "num_turns": 0,
            }

        samples = render_prompts(samples, self.tokenizer, self.tools.specs, pad_token_id, self.chat_template_kwargs)
        trajectories = expand_samples(samples, n)
        generated = [0] * len(trajectories)
        active = list(range(len(trajectories)))

        for _turn in range(self.max_turns):
            # Each trajectory may generate up to its remaining token budget, the
            # per-turn cap and the room left in its context; one with no room
            # stops before generating.
            room = {}
            for i in active:
                room[i] = max_new_tokens - generated[i]
                if sampling.max_new_tokens_per_turn is not None:
                    room[i] = min(room[i], sampling.max_new_tokens_per_turn)
                if self.max_context_len is not None:
                    room[i] = min(room[i], self.max_context_len - sum(trajectories[i].attention_mask))
                if room[i] <= 0:
                    trajectories[i].status = RolloutRecord.Status.TRUNCATED
            active = [i for i in active if room[i] > 0]
            if not active:
                break
            turn_samples = [trajectories[i] for i in active]
            budgets = [room[i] for i in active]

            if self.enable_profiling:
                accelerator.synchronize()
                turn_start = time.perf_counter()

            prompt_ids, prompt_attn = left_pad_samples(turn_samples, pad_token_id)
            prompt_len = prompt_ids.shape[1]
            prompt_ids = prompt_ids.to(device)
            prompt_attn = prompt_attn.to(device)

            if self.enable_profiling:
                accelerator.synchronize()
                expansion_end = time.perf_counter()

            forward_profiler = None
            forward_profile_handles = []
            if self.enable_profiling and not self._graph_capture_enabled():
                forward_profiler = _ForwardProfiler(accelerator)
                forward_profile_handles = forward_profiler.register(module)

            # One generate call serves all active trajectories, so it runs to the
            # largest remaining budget; each row is cut back to its own below.
            turn_max_new_tokens = max(budgets)
            try:
                output_ids, log_probs = self._generate_tokens(prompt_ids, prompt_attn, turn_max_new_tokens, sampling,
                                                              pad_token_id)
            finally:
                for handle in forward_profile_handles:
                    handle.remove()

            if self.enable_profiling:
                accelerator.synchronize()
                generation_end = time.perf_counter()

            # Keep each response up to its first EOS: under ZeRO-3, generation
            # ran past it, and rows that stopped early are padded either way.
            output_ids, response_attn = self._pad_after_eos(
                output_ids,
                response_start=prompt_len,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
            )
            responses = masked_lists(output_ids[:, prompt_len:], response_attn)
            response_log_probs = masked_lists(log_probs, response_attn) if return_log_probs else None

            # A trajectory continues only if it has budget left and asked for a
            # tool. The last turn runs no tools: no later turn would read them.
            turn_responses = []
            turn_log_probs = []
            turn_calls = {}
            for row, i in enumerate(active):
                response = responses[row][:budgets[row]]
                turn_responses.append(response)
                turn_log_probs.append(response_log_probs[row][:budgets[row]] if return_log_probs else None)
                generated[i] += len(response)
                # A turn that did not reach EOS was cut by the token or context budget.
                if not response or response[-1] != self.tokenizer.eos_token_id:
                    trajectories[i].status = RolloutRecord.Status.TRUNCATED
                    continue
                # Keep special tokens when decoding: tool-call tags can be special tokens.
                calls = self.parse_tool_calls(self.tokenizer.decode(response), self.tools)
                if not calls:
                    trajectories[i].status = RolloutRecord.Status.COMPLETED
                    continue
                # It asked for a tool but has no budget or turn left to read the result.
                if generated[i] >= max_new_tokens or _turn == self.max_turns - 1:
                    trajectories[i].status = RolloutRecord.Status.TRUNCATED
                    continue
                turn_calls[row] = calls
            tool_start = time.perf_counter()
            tool_outputs = asyncio.run(self._run_tool_calls(turn_calls))
            tool_ms = (time.perf_counter() - tool_start) * 1000.0

            tool_active = []
            for row, i in enumerate(active):
                tool_response = []
                if row in turn_calls:
                    tool_messages = [{
                        "role": "tool",
                        "name": call.name or "unknown",
                        "content": output
                    } for call, output in zip(turn_calls[row], tool_outputs[row])]
                    tool_response = tokenize_tool_responses(tool_messages, self.tokenizer, self.chat_template_kwargs)
                    tool_active.append(i)
                update_sample_from_response(trajectories[i], turn_responses[row], turn_log_probs[row], tool_response,
                                            self.tokenizer)
            active = tool_active

            if self.enable_profiling:
                accelerator.synchronize()
                turn_end = time.perf_counter()
                profile["prompt_expansion_ms"] += (expansion_end - turn_start) * 1000.0
                profile["generation_ms"] += (generation_end - expansion_end) * 1000.0
                profile["tool_ms"] += tool_ms
                profile["post_processing_ms"] += (turn_end - generation_end) * 1000.0 - tool_ms
                profile["num_generated_tokens"] += int(output_ids.shape[0] * (output_ids.shape[1] - prompt_len))
                profile["num_turns"] += 1
                prefill_ms, decode_ms, num_decodes = (None, None, 0)
                if forward_profiler is not None:
                    prefill_ms, decode_ms, num_decodes = forward_profiler.get_stage_times()
                # Forward times are reported only when every turn measured them.
                if prefill_ms is None or profile["prefill_forward_ms"] is None:
                    profile["prefill_forward_ms"] = None
                    profile["decode_forward_ms"] = None
                else:
                    profile["prefill_forward_ms"] += prefill_ms
                    profile["decode_forward_ms"] += decode_ms
                profile["num_decode_forwards"] += num_decodes

        if self.enable_profiling:
            accelerator.synchronize()
            total_ms = (time.perf_counter() - call_start) * 1000.0
            generation_overhead_ms = profile["generation_ms"]
            if profile["prefill_forward_ms"] is not None:
                forward_ms = profile["prefill_forward_ms"] + profile["decode_forward_ms"]
                generation_overhead_ms = max(0.0, generation_overhead_ms - forward_ms)
            tokens_per_second = 0.0
            if total_ms > 0.0:
                tokens_per_second = profile["num_generated_tokens"] / (total_ms / 1000.0)
            profile.update({
                "generation_overhead_ms": generation_overhead_ms,
                "total_ms": total_ms,
                "tokens_per_second": tokens_per_second,
                "batch_size": len(trajectories),
                "num_samples_per_prompt": n,
                "prompt_length": len(samples[0].tokens),
                "response_length": max(sample.response_length for sample in trajectories),
            })
            self._last_profile = profile

        return RolloutBatch(samples=trajectories)

    @torch.no_grad()
    def _generate_multi_turn_continuous(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        """Multi-turn rollout with continuous batching over ``continuous_batch_size`` decode rows.

        The scheduling unit is one turn of one trajectory. A turn waits for a free row, is prefilled into it
        with the whole conversation so far (unpadded, as in ``_prefill_rows``) and decodes there until
        EOS or its budget. The row is released as soon as the turn ends. If the trajectory called tools, they
        run on a background event loop while the row serves other turns, and the next turn joins the queue when
        they return. Every row decodes at its own position in one step (a captured graph with
        ``use_graph_capture``); rows without a turn decode throwaway tokens. Finished turns are collected after
        every step.

        Turn budgets, statuses and the returned samples follow ``_generate_multi_turn``.
        """
        samples = request.samples
        if not samples:
            return RolloutBatch(samples=[])
        self._check_sampling(sampling)
        module = self.engine.module
        num_rows = self.continuous_batch_size
        capacity = self.continuous_cache_capacity or self.max_context_len
        self._validate_multi_turn_continuous(module, num_rows, capacity)

        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        accelerator = get_accelerator()
        call_start = time.perf_counter()
        device = accelerator.current_device_name()
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        select_next_tokens = _token_selector(sampling)
        repetition_penalty = _repetition_penalty(sampling, module)
        # With a repetition penalty, which tokens each row's trajectory holds, so a decode step needs no host read.
        seen_tokens = None
        max_new_tokens = sampling.max_new_tokens
        stats = {"prefill_ms": 0.0, "decode_ms": 0.0, "tool_ms": 0.0, "steps": 0, "turns": 0}

        samples = render_prompts(samples, self.tokenizer, self.tools.specs, pad_token_id, self.chat_template_kwargs)
        trajectories = expand_samples(samples, sampling.n_samples_per_prompt)
        generated = [0] * len(trajectories)
        turns_done = [0] * len(trajectories)
        # Trajectories whose next turn waits for a row; a turn that follows a tool call goes first.
        pending = deque(range(len(trajectories)))
        row_owner = [None] * num_rows
        # Trajectory -> (response, log-probs, calls, future, submit time) while its tools run.
        tools_in_flight = {}

        def turn_budget(i):
            context_len = sum(trajectories[i].attention_mask)
            budget = max_new_tokens - generated[i]
            if sampling.max_new_tokens_per_turn is not None:
                budget = min(budget, sampling.max_new_tokens_per_turn)
            if self.max_context_len is not None:
                budget = min(budget, self.max_context_len - context_len)
            return min(budget, capacity - context_len)

        model_dtype = next(module.parameters()).dtype
        rows = torch.arange(num_rows, device=device)
        slots = torch.arange(capacity, device=device)
        ds_cache = DeepSpeedStaticCache(module.config,
                                        batch_size=num_rows,
                                        max_cache_len=capacity,
                                        device=device,
                                        dtype=model_dtype)
        write_pos = torch.zeros(num_rows, dtype=torch.long, device=device)
        ds_cache.set_write_position(write_pos)
        static_token = torch.zeros(num_rows, 1, dtype=torch.long, device=device)
        static_attn = torch.zeros(num_rows, 1, 1, capacity, dtype=torch.bool, device=device)
        # A row that never held a turn still attends one slot, so its throwaway logits stay finite.
        static_attn[:, 0, 0, 0] = True
        decode_mask = static_attn
        static_cache_pos = write_pos[:1]
        static_pos_ids = write_pos.unsqueeze(1)
        # Each row's turn on the device, so decode steps never wait for the host.
        active = torch.zeros(num_rows, dtype=torch.bool, device=device)
        num_generated = torch.zeros(num_rows, dtype=torch.long, device=device)
        turn_limit = torch.zeros(num_rows, dtype=torch.long, device=device)
        turn_len = max(1, min(capacity, max_new_tokens, sampling.max_new_tokens_per_turn or max_new_tokens))
        token_buf = torch.zeros(num_rows, turn_len, dtype=torch.long, device=device)
        log_prob_buf = torch.zeros(num_rows, turn_len, dtype=torch.float32, device=device)
        stop_token = eos_token_id if eos_token_id is not None else -1
        # With profiling, one entry per step shows how the rows were used over the call (the drain at its end).
        step_trace = [] if self.enable_profiling else None
        decoded = torch.zeros((), dtype=torch.long, device=device)
        logits_kwargs = self._last_logits_kwargs(module)

        def penalize_seen(logits, seen):
            logits = logits.float()
            penalized = torch.where(logits < 0, logits * repetition_penalty, logits / repetition_penalty)
            return torch.where(seen, penalized, logits)

        def admit():
            nonlocal seen_tokens
            free_rows = [row for row, owner in enumerate(row_owner) if owner is None]
            admitted = []
            while free_rows and pending:
                i = pending.popleft()
                budget = turn_budget(i)
                if budget <= 0:
                    # No room left to generate: the trajectory ends without another turn.
                    trajectories[i].status = RolloutRecord.Status.TRUNCATED
                    continue
                admitted.append((free_rows.pop(0), i, budget))
            if not admitted:
                return False
            prefill_start = self._sync_time(accelerator)
            target = torch.tensor([row for row, _, _ in admitted], device=device)
            prompt_ids, prompt_attn = left_pad_samples([trajectories[i] for _, i, _ in admitted], pad_token_id)
            lengths = prompt_attn.sum(dim=1).to(device)
            logits = self._prefill_rows(module, prompt_ids.to(device), lengths, ds_cache, logits_kwargs, model_dtype,
                                        device, target, num_rows)
            if repetition_penalty != 1.0:
                if seen_tokens is None:
                    seen_tokens = torch.zeros(num_rows, logits.shape[-1], dtype=torch.bool, device=device)
                seen_tokens[target] = False
                for row, i, _ in admitted:
                    sample = trajectories[i]
                    real = [token for token, keep in zip(sample.tokens, sample.attention_mask) if keep]
                    seen_tokens[row, torch.tensor(real, device=device)] = True
                logits = penalize_seen(logits, seen_tokens[target])
            first, first_log_probs = select_next_tokens(logits)
            if repetition_penalty != 1.0:
                seen_tokens[target] = seen_tokens[target].scatter(1, first, True)
            limits = torch.tensor([budget for _, _, budget in admitted], device=device)
            # The turn's context, and the slot its first decode input is written to.
            static_attn[target, 0, 0, :] = slots.unsqueeze(0) <= lengths.unsqueeze(1)
            write_pos[target] = lengths
            static_token[target] = first
            token_buf[target, 0] = first[:, 0]
            log_prob_buf[target, 0] = first_log_probs[:, 0]
            num_generated[target] = 1
            turn_limit[target] = limits
            active[target] = (first[:, 0] != stop_token) & (limits > 1)
            for row, i, _ in admitted:
                row_owner[row] = i
            stats["turns"] += len(admitted)
            stats["prefill_ms"] += (self._sync_time(accelerator) - prefill_start) * 1000.0
            return True

        def forward():
            return module(static_token,
                          attention_mask=decode_mask,
                          past_key_values=ds_cache,
                          use_cache=True,
                          cache_position=static_cache_pos,
                          position_ids=static_pos_ids)

        def capture():
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
                        forward()
                accelerator.current_stream().wait_stream(stream)
                graph = accelerator.create_graph()
                with accelerator.capture_to_graph(graph):
                    logits = forward().logits
            finally:
                module._forward_pre_hooks.update(saved_pre)
                module._forward_hooks.update(saved_post)
            return graph, logits

        graph = None
        static_logits = None

        def decode_step():
            if graph is not None:
                accelerator.replay_graph(graph)
                logits = static_logits[:, -1, :]
            else:
                logits = forward().logits[:, -1, :]
            if repetition_penalty != 1.0:
                logits = penalize_seen(logits, seen_tokens)
            tokens, log_probs = select_next_tokens(logits)
            if repetition_penalty != 1.0:
                seen_tokens.scatter_(1, tokens, True)
            # Record each decoding row's token at its next position; other rows keep their buffers.
            position = num_generated.clamp(max=turn_len - 1).unsqueeze(1)
            recording = active.unsqueeze(1)
            token_buf.scatter_(1, position, torch.where(recording, tokens, token_buf.gather(1, position)))
            log_prob_buf.scatter_(1, position, torch.where(recording, log_probs, log_prob_buf.gather(1, position)))
            num_generated.add_(active.long())
            if step_trace is not None:
                decoded.add_(active.sum())
            ended = (tokens[:, 0] == stop_token) | (num_generated >= turn_limit)
            active.logical_and_(~ended)
            # The sampled token is each row's next input, written to the following slot.
            static_token.copy_(tokens)
            write_pos.add_(active.long())
            static_attn[rows, 0, 0, write_pos] |= active

        def finish_turn(i, response, response_log_probs):
            sample = trajectories[i]
            turn = turns_done[i]
            turns_done[i] += 1
            generated[i] += len(response)
            log_probs = response_log_probs if self.return_log_probs else None
            # A turn that did not reach EOS was cut by the token or context budget.
            if not response or response[-1] != eos_token_id:
                sample.status = RolloutRecord.Status.TRUNCATED
                update_sample_from_response(sample, response, log_probs, [], self.tokenizer)
                return
            # Keep special tokens when decoding: tool-call tags can be special tokens.
            calls = self.parse_tool_calls(self.tokenizer.decode(response), self.tools)
            if not calls:
                sample.status = RolloutRecord.Status.COMPLETED
                update_sample_from_response(sample, response, log_probs, [], self.tokenizer)
                return
            # It asked for a tool but has no budget or turn left to read the result.
            if generated[i] >= max_new_tokens or turn == self.max_turns - 1:
                sample.status = RolloutRecord.Status.TRUNCATED
                update_sample_from_response(sample, response, log_probs, [], self.tokenizer)
                return
            future = asyncio.run_coroutine_threadsafe(self._run_tool_calls({i: calls}), tool_loop)
            tools_in_flight[i] = (response, log_probs, calls, future, time.perf_counter())

        def collect_finished_turns(still_active):
            finished = [row for row, owner in enumerate(row_owner) if owner is not None and not still_active[row]]
            if not finished:
                return
            finished_rows = torch.tensor(finished, device=device)
            counts = num_generated[finished_rows].tolist()
            width = max(counts)
            tokens = token_buf[finished_rows, :width].tolist()
            log_probs = log_prob_buf[finished_rows, :width].tolist()
            for row, count, row_tokens, row_log_probs in zip(finished, counts, tokens, log_probs):
                i = row_owner[row]
                row_owner[row] = None
                finish_turn(i, row_tokens[:count], row_log_probs[:count])

        def collect_tool_results():
            for i in [i for i, entry in tools_in_flight.items() if entry[3].done()]:
                response, log_probs, calls, future, submitted = tools_in_flight.pop(i)
                stats["tool_ms"] += (time.perf_counter() - submitted) * 1000.0
                outputs = future.result()[i]
                tool_messages = [{
                    "role": "tool",
                    "name": call.name or "unknown",
                    "content": output
                } for call, output in zip(calls, outputs)]
                tool_response = tokenize_tool_responses(tool_messages, self.tokenizer, self.chat_template_kwargs)
                update_sample_from_response(trajectories[i], response, log_probs, tool_response, self.tokenizer)
                pending.appendleft(i)

        use_graph_capture = self._graph_capture_enabled()
        tool_loop = asyncio.new_event_loop()
        tool_thread = threading.Thread(target=tool_loop.run_forever, name="rollout-tools", daemon=True)
        tool_thread.start()
        try:
            while any(owner is not None for owner in row_owner) or pending or tools_in_flight:
                if admit() and use_graph_capture and graph is None:
                    graph, static_logits = capture()
                rows_held = sum(owner is not None for owner in row_owner)
                if rows_held:
                    decode_start = time.perf_counter()
                    decode_step()
                    if step_trace is None:
                        still_active = active.tolist()
                    else:
                        # One read for both, so profiling adds no host sync.
                        *still_active, decoded_so_far = torch.cat((active.long(), decoded.view(1))).tolist()
                    stats["decode_ms"] += (time.perf_counter() - decode_start) * 1000.0
                    stats["steps"] += 1
                    collect_finished_turns(still_active)
                    if step_trace is not None:
                        step_trace.append((stats["steps"], round(time.perf_counter() - call_start, 3), rows_held,
                                           sum(still_active), decoded_so_far, len(pending), len(tools_in_flight)))
                elif tools_in_flight and not pending:
                    # Nothing to decode or admit until a tool returns.
                    concurrent.futures.wait([entry[3] for entry in tools_in_flight.values()],
                                            return_when=concurrent.futures.FIRST_COMPLETED)
                collect_tool_results()
        finally:
            tool_loop.call_soon_threadsafe(tool_loop.stop)
            tool_thread.join()
            tool_loop.close()

        if self.enable_profiling:
            total_ms = (self._sync_time(accelerator) - call_start) * 1000.0
            num_generated_tokens = sum(generated)
            self._last_profile = {
                "generation_ms":
                stats["prefill_ms"] + stats["decode_ms"],
                "prefill_forward_ms":
                stats["prefill_ms"],
                "decode_forward_ms":
                stats["decode_ms"],
                "num_decode_forwards":
                stats["steps"],
                "tool_ms":
                stats["tool_ms"],
                "num_generated_tokens":
                num_generated_tokens,
                "num_turns":
                stats["turns"],
                "total_ms":
                total_ms,
                "tokens_per_second":
                num_generated_tokens / (total_ms / 1000.0) if total_ms > 0.0 else 0.0,
                "batch_size":
                len(trajectories),
                "num_samples_per_prompt":
                sampling.n_samples_per_prompt,
                "continuous_rows":
                num_rows,
                # Decode slots used: tokens after each turn's first (which comes from its prefill) per row-step.
                "continuous_decode_occupancy":
                ((num_generated_tokens - stats["turns"]) / (stats["steps"] * num_rows) if stats["steps"] else 0.0),
                # Per step: (decode steps so far, seconds since the call started, rows holding a turn during the
                # step, rows still decoding after it, tokens decoded so far, turns queued, trajectories waiting on
                # tools).
                "continuous_step_trace":
                step_trace,
            }
        return RolloutBatch(samples=trajectories)

    @staticmethod
    def _sync_time(accelerator):
        accelerator.synchronize()
        return time.perf_counter()

    def _validate_multi_turn_continuous(self, module, num_rows, capacity):
        if num_rows <= 0:
            raise ValueError("continuous_batch_size must be positive")
        if capacity is None:
            raise ValueError("multi-turn continuous batching sizes its rows by max_context_len or "
                             "continuous_cache_capacity; set one of them")
        max_positions = getattr(module.config, "max_position_embeddings", None)
        if max_positions is not None and capacity > max_positions:
            raise ValueError("continuous batching cache exceeds the model maximum position embeddings")
        if self.use_shared_prefill:
            raise ValueError("continuous batching does not support shared prompt prefill")
        partitions_weights = getattr(self.engine, "zero_optimization_partition_weights", None)
        if partitions_weights is not None and partitions_weights():
            # ZeRO-3 gathers parameters inside each forward, which neither a captured graph nor rows that
            # finish at different steps allow.
            raise ValueError("multi-turn continuous batching does not support ZeRO-3")
        if getattr(module, "_supports_cache_class", None) is False:
            raise ValueError("continuous batching requires a model with cache-class support")
        if getattr(module.config, "_attn_implementation", None) == "flash_attention_2":
            raise ValueError("multi-turn continuous batching does not support flash_attention_2")

    def _check_sampling(self, sampling: SamplingConfig) -> None:
        if sampling.max_new_tokens_per_turn is not None and sampling.max_new_tokens_per_turn <= 0:
            raise ValueError("max_new_tokens_per_turn must be positive")
        return_log_probs = self.return_log_probs
        if return_log_probs and sampling.temperature > 0.0 and (sampling.top_p < 1.0 or sampling.top_k > 0):
            raise ValueError("return_log_probs requires top_p=1 and top_k<=0 when sampling")
        if sampling.repetition_penalty is not None and sampling.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        repetition_penalty = _repetition_penalty(sampling, self.engine.module)
        if return_log_probs and repetition_penalty != 1.0:
            # The penalty depends on the context and is applied before the
            # log-probs are recorded; a training forward pass cannot reproduce it.
            logger.warning_once(f"return_log_probs with repetition_penalty={repetition_penalty}: a training forward "
                                "pass cannot reproduce the recorded log-probs")

    def _graph_capture_enabled(self):
        """Whether to decode with a captured graph; a backend that only exposes a graph API has not opted in."""
        if not self.use_graph_capture:
            return False
        if get_accelerator().is_graph_capture_supported():
            return True
        logger.warning_once("HybridEngineRollout: graph capture is not supported on the current accelerator; "
                            "falling back to eager generation.")
        return False

    def _generate_tokens(self, prompt_ids, prompt_attn, max_new_tokens, sampling, pad_token_id):
        module = self.engine.module
        if self._graph_capture_enabled():
            return self._generate_graph(prompt_ids, prompt_attn, max_new_tokens, pad_token_id, sampling, module,
                                        prompt_ids.device)

        # ZeRO-3 gathers parameters during each decode forward, so every
        # data-parallel rank must run the same number of decode steps and
        # generation cannot stop at EOS. Other stages stop once every row ends.
        stop_at_eos = not self.engine.zero_optimization_partition_weights()
        temperature = max(sampling.temperature, 1e-8)
        do_sample = sampling.temperature > 0.0
        generate_kwargs = {
            "attention_mask": prompt_attn,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id if stop_at_eos else None,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else 1.0,
            "top_p": sampling.top_p if do_sample else 1.0,
            "pad_token_id": pad_token_id,
        }
        if do_sample:
            generate_kwargs["top_k"] = max(sampling.top_k, 0)
        if sampling.repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = sampling.repetition_penalty
        finish_log_probs = None
        if self.return_log_probs:
            from transformers import LogitsProcessorList
            # Greedy decoding picks from the raw logits, i.e. temperature 1.
            processor, finish_log_probs = _sampled_log_prob_recorder(temperature if do_sample else 1.0)
            generate_kwargs["logits_processor"] = LogitsProcessorList([processor])
        output_ids = module.generate(prompt_ids, **generate_kwargs)
        log_probs = finish_log_probs(output_ids[:, -1:]) if finish_log_probs is not None else None
        return output_ids, log_probs

    async def _run_tool_calls(self, calls_by_row):
        """Run every row's calls concurrently; returns each row's outputs in call order."""
        rows = list(calls_by_row)
        outputs = await asyncio.gather(*(asyncio.gather(*(self.tools.run(call) for call in calls_by_row[row]))
                                         for row in rows))
        return dict(zip(rows, outputs))

    @torch.no_grad()
    def _generate_continuous(self, samples, sampling, max_batch_size):
        """Generate independent responses for ``samples`` in an experimental continuous batch.

        Continuous batching currently requires one sample
        per prompt. Completed rows retire immediately and pending prompts
        prefill into the released rows before the next decode step.
        """
        self._validate_continuous_inputs(sampling, max_batch_size)
        pad_token_id = self.tokenizer.pad_token_id
        samples = render_prompts(samples, self.tokenizer, self.tools.specs, pad_token_id, self.chat_template_kwargs)
        trajectories = expand_samples(samples, 1)
        if not all(any(sample.attention_mask) for sample in trajectories):
            raise ValueError("continuous batching requires at least one prompt token per request")
        prompt_ids, prompt_attention_mask = left_pad_samples(trajectories, pad_token_id)
        module = self.engine.module
        device = next(module.parameters()).device
        prompt_ids = prompt_ids.to(device)
        prompt_attention_mask = prompt_attention_mask.to(device)
        requests = tuple(
            _PromptRow(prompt_ids[index:index + 1], prompt_attention_mask[index:index + 1])
            for index in range(len(trajectories)))
        if self._graph_capture_enabled():
            return self._generate_continuous_graph(trajectories, requests, sampling, max_batch_size)

        profile = self._start_continuous_profile() if self.enable_profiling else None
        profile_accelerator = profile["accelerator"] if profile is not None else None
        profile_start = profile["start"] if profile is not None else None
        prompt_lengths = {
            request_id: (int(request.prompt_attention_mask.sum().item())
                         if self.align_decode_fronts else request.prompt_ids.shape[1])
            for request_id, request in enumerate(requests)
        }
        prompt_len = max(prompt_lengths.values())
        max_positions = getattr(module.config, "max_position_embeddings", None)
        if max_positions is not None:
            if any(length + sampling.max_new_tokens > max_positions for length in prompt_lengths.values()):
                raise ValueError("continuous batching request exceeds the model maximum position embeddings")
        if self.continuous_cache_capacity is not None:
            if self.continuous_cache_capacity < prompt_len:
                raise ValueError("continuous_cache_capacity must be at least the padded prompt length")
            max_cache_len = self.continuous_cache_capacity
        else:
            max_cache_len = (prompt_len + sampling.max_new_tokens if self.align_decode_fronts else
                             self._estimate_continuous_cache_len(prompt_len, [sampling.max_new_tokens] *
                                                                 len(requests), max_batch_size))
        if max_positions is not None and max_cache_len > max_positions:
            raise ValueError("continuous batching cache exceeds the model maximum position embeddings")
        if getattr(module, "_supports_cache_class", None) is False:
            raise ValueError("continuous batching requires a model with cache-class support; use the default "
                             "generate() path or upgrade transformers")

        from transformers import StaticCache
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        model_dtype = next(module.parameters()).dtype

        scheduler_start = self._profile_start(profile)
        scheduler = ContinuousBatchScheduler(max_batch_size, sampling.max_new_tokens)
        request_by_id = {}
        responses = {}
        select_next_tokens = _token_selector(sampling)
        # Each response token's log-prob under the distribution it was picked from.
        response_log_probs = {}
        request_order = (sorted(range(len(requests)), key=lambda request_id: -prompt_lengths[request_id])
                         if self.align_decode_fronts else range(len(requests)))
        for request_id in request_order:
            request = requests[request_id]
            scheduler.submit(ContinuousBatchRequest(request_id))
            request_by_id[request_id] = request
            responses[request_id] = []
            response_log_probs[request_id] = []
        self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)

        cache_start = self._profile_start(profile)
        cache = DeepSpeedStaticCache(
            module.config,
            batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=model_dtype,
        )
        write_positions = torch.full((max_batch_size, ), -1, dtype=torch.long, device=device)
        cache.set_write_position(write_positions)
        attention_mask = torch.zeros((max_batch_size, max_cache_len), dtype=torch.long, device=device)
        stats = {
            "cache_capacity":
            max_cache_len,
            "peak_cache_length":
            prompt_len,
            "cache_memory_bytes":
            sum(layer.keys.numel() * layer.keys.element_size() + layer.values.numel() * layer.values.element_size()
                for layer in cache.layers) + attention_mask.numel() * attention_mask.element_size() +
            write_positions.numel() * write_positions.element_size(),
            "trim_count":
            0,
            "trimmed_columns":
            0,
            "trim_latency_ms":
            0.0,
            "decode_steps":
            0,
        }
        next_tokens = {}
        next_log_probs = {}
        logical_positions = dict(prompt_lengths)
        cache_position = prompt_len
        span_starts = [0] * max_batch_size

        self._profile_end(profile, "cache_management_overhead_ms", cache_start)
        scheduler_start = self._profile_start(profile)
        update = scheduler.schedule()
        self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)

        while update.active:
            if profile is not None:
                profile["active_batch_sizes"].append(len(update.active))
            cache_start = self._profile_start(profile)
            keep_slots = torch.tensor(update.keep_slots, dtype=torch.long, device=device)
            survivor_count = keep_slots.numel()
            if survivor_count:
                if update.retired:
                    cache.compact(keep_slots)
                    identity = torch.arange(survivor_count, dtype=torch.long, device=device)
                    if not torch.equal(keep_slots, identity):
                        survivor_attention = attention_mask.index_select(0, keep_slots).clone()
                        attention_mask[:survivor_count].copy_(survivor_attention)
                    attention_mask[survivor_count:].zero_()
                    if self.align_decode_fronts:
                        span_starts[:survivor_count] = [span_starts[index] for index in update.keep_slots]
                        span_starts[survivor_count:] = [0] * (max_batch_size - survivor_count)

                dead_prefix = 0
                if self.enable_cache_trimming:
                    if self.align_decode_fronts:
                        dead_prefix = min(span_starts[:survivor_count])
                    else:
                        trim_threshold = max(1, prompt_len)
                        dead_prefix = self._continuous_dead_prefix(attention_mask, survivor_count)
                        if dead_prefix < trim_threshold and cache_position < max_cache_len - trim_threshold:
                            dead_prefix = 0
                elif cache_position >= max_cache_len:
                    if self.align_decode_fronts:
                        dead_prefix = min(span_starts[:survivor_count])
                    else:
                        dead_prefix = self._continuous_dead_prefix(attention_mask, survivor_count)
                if dead_prefix:
                    if update.admitted:
                        longest_admitted = max(prompt_lengths[request.request_id] for request in update.admitted)
                        dead_prefix = min(dead_prefix, max(0, cache_position - longest_admitted))
                    trim_start = None
                    if profile_accelerator is not None:
                        profile_accelerator.synchronize()
                        trim_start = time.perf_counter()
                    moved_bytes = sum((layer.keys[:, :, dead_prefix:, :].numel() * layer.keys.element_size() +
                                       layer.values[:, :, dead_prefix:, :].numel() * layer.values.element_size())
                                      for layer in cache.layers)
                    moved_bytes += attention_mask[:, dead_prefix:].numel() * attention_mask.element_size()
                    cache.trim_left(dead_prefix)
                    remaining_columns = max_cache_len - dead_prefix
                    if remaining_columns:
                        attention_mask[:, :remaining_columns].copy_(attention_mask[:, dead_prefix:].clone())
                    attention_mask[:, remaining_columns:].zero_()
                    write_positions[:survivor_count].sub_(dead_prefix)
                    cache_position -= dead_prefix
                    stats["trim_count"] += 1
                    stats["trimmed_columns"] += dead_prefix
                    stats["trim_bytes_moved"] = stats.get("trim_bytes_moved", 0) + moved_bytes
                    if profile_accelerator is not None:
                        profile_accelerator.synchronize()
                        stats["trim_latency_ms"] += (time.perf_counter() - trim_start) * 1000.0
                    if self.align_decode_fronts:
                        span_starts[:survivor_count] = [start - dead_prefix for start in span_starts[:survivor_count]]
            else:
                cache.reset()
                write_positions.fill_(-1)
                attention_mask.zero_()
                cache_position = prompt_len
                span_starts = [0] * max_batch_size

            self._profile_end(profile, "cache_management_overhead_ms", cache_start)
            if survivor_count and cache_position >= max_cache_len:
                raise self._continuous_cache_exhaustion_error(max_cache_len)
            admitted_tokens, admitted_log_probs = self._continuous_prefill(
                module,
                StaticCache,
                cache,
                update,
                request_by_id,
                attention_mask,
                write_positions,
                cache_position,
                prompt_lengths,
                model_dtype,
                device,
                sampling,
                self.align_decode_fronts,
                profile,
            )
            if self.align_decode_fronts:
                for admitted, target_row in zip(update.admitted, update.admitted_slots):
                    span_starts[target_row] = cache_position - prompt_lengths[admitted.request_id]

            decoded_tokens = {}
            decoded_log_probs = {}
            if survivor_count:
                survivor_ids = update.active_ids[:survivor_count]
                decode_input = torch.cat([next_tokens[request_id] for request_id in survivor_ids], dim=0)
                if cache_position >= max_cache_len:
                    raise self._continuous_cache_exhaustion_error(max_cache_len)
                write_positions[:survivor_count].fill_(cache_position)
                if self.align_decode_fronts:
                    position_ids = torch.tensor([logical_positions[request_id] for request_id in survivor_ids],
                                                dtype=torch.long,
                                                device=device).unsqueeze(1)
                else:
                    position_ids = attention_mask[:survivor_count, :cache_position].sum(dim=1, keepdim=True)
                attention_mask[:survivor_count, cache_position] = 1
                decode_start = self._profile_start(profile)
                output = self._call_model(
                    module,
                    decode_input,
                    attention_mask=attention_mask[:survivor_count],
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=torch.tensor([cache_position], dtype=torch.long, device=device),
                    position_ids=position_ids,
                )
                self._profile_end(profile, "decode_forward_ms", decode_start, count="num_decode_forwards")
                decoded, decoded_log_probs_all = self._continuous_next_tokens(
                    output.logits[:, -1, :],
                    survivor_ids,
                    request_by_id,
                    responses,
                    _repetition_penalty(sampling, module),
                    select_next_tokens,
                )
                stats["decode_steps"] += 1
                decoded_tokens = dict(zip(survivor_ids, decoded.split(1, dim=0)))
                decoded_log_prob_rows = decoded_log_probs_all.split(1, dim=0)
                decoded_log_probs = dict(zip(survivor_ids, decoded_log_prob_rows))
                for request_id in survivor_ids:
                    logical_positions[request_id] += 1

            next_tokens = decoded_tokens | admitted_tokens
            next_log_probs = decoded_log_probs | admitted_log_probs
            finished_ids = []
            for request_id in update.active_ids:
                token = next_tokens[request_id]
                responses[request_id].append(token)
                response_log_probs[request_id].append(next_log_probs[request_id])
                if self._is_eos(token):
                    finished_ids.append(request_id)

            if survivor_count:
                cache_position += 1
            stats["peak_cache_length"] = max(stats["peak_cache_length"], cache_position)
            scheduler_start = self._profile_start(profile)
            update = scheduler.advance(finished_ids)
            self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)

        generation_end = self._profile_start(profile)
        post_processing_start = generation_end
        output = self._build_continuous_batch(trajectories, responses, response_log_probs, self.return_log_probs)
        post_processing_end = self._profile_end(profile, None, post_processing_start)
        if profile_accelerator is not None:
            profile_accelerator.synchronize()
            total_ms = (time.perf_counter() - profile_start) * 1000.0
            generated_tokens = sum(len(response) for response in responses.values())
            stats["end_to_end_ms"] = total_ms
            stats["tokens_per_second"] = generated_tokens / (total_ms / 1000.0) if total_ms > 0.0 else 0.0
        else:
            stats["end_to_end_ms"] = None
            stats["tokens_per_second"] = None
        stats.setdefault("trim_bytes_moved", 0)
        stats["trim_frequency"] = (stats["trim_count"] / stats["decode_steps"] if stats["decode_steps"] else 0.0)
        self._last_continuous_stats = stats
        if profile is not None:
            self._finish_continuous_profile(profile, len(trajectories), responses, max_batch_size, prompt_len,
                                            generation_end, post_processing_end)
        return output

    @torch.no_grad()
    def _generate_continuous_graph(self, trajectories, requests, sampling, max_batch_size):
        """Run continuous decoding with a fixed-capacity CUDA graph.

        Scheduler slots stay attached to physical cache rows for their whole
        lifetime. Retired rows are masked until a new prompt is prefetched into
        them, so the captured decode always receives the same batch shape and
        static tensor addresses.
        """
        module = self.engine.module
        prompt_len = requests[0].prompt_ids.shape[1]
        self._validate_continuous_graph_options(prompt_len, sampling)
        if getattr(module, "_supports_cache_class", None) is False:
            raise ValueError("continuous batching requires a model with cache-class support; use the default "
                             "generate() path or upgrade transformers")
        if getattr(module.config, "_attn_implementation", None) == "flash_attention_2":
            raise ValueError("continuous graph capture does not support flash_attention_2")
        device = requests[0].prompt_ids.device

        from transformers import StaticCache
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        profile = self._start_continuous_profile() if self.enable_profiling else None
        profile_accelerator = profile["accelerator"] if profile is not None else None
        profile_start = profile["start"] if profile is not None else None
        minimum_cache_len = prompt_len + sampling.max_new_tokens
        max_cache_len = (self.continuous_cache_capacity
                         if self.continuous_cache_capacity is not None else minimum_cache_len)
        if max_cache_len < minimum_cache_len:
            raise ValueError("continuous_cache_capacity must fit the prompt and all generated tokens")
        max_positions = getattr(module.config, "max_position_embeddings", None)
        if max_positions is not None and max_cache_len > max_positions:
            raise ValueError("continuous batching cache exceeds the model maximum position embeddings")
        model_dtype = next(module.parameters()).dtype
        scheduler = ContinuousBatchScheduler(max_batch_size, sampling.max_new_tokens)
        prompt_lengths = {request_id: prompt_len for request_id in range(len(requests))}
        request_by_id = {}
        responses = {}
        select_next_tokens = _token_selector(sampling)
        # Each response token's log-prob under the distribution it was picked from.
        response_log_probs = {}
        for request_id, request in enumerate(requests):
            scheduler.submit(ContinuousBatchRequest(request_id))
            request_by_id[request_id] = request
            responses[request_id] = []
            response_log_probs[request_id] = []

        state = self._get_continuous_graph_state(module, max_batch_size, prompt_len, max_cache_len, device,
                                                 model_dtype, DeepSpeedStaticCache)
        cache = state.cache
        static_input = state.static_input
        static_attention = state.static_attention
        static_causal_attention = state.static_causal_attention
        static_position_ids = state.static_position_ids
        static_write_positions = state.static_write_positions
        graph = state.graph
        static_logits = state.static_logits
        stats = {
            "cache_capacity":
            max_cache_len,
            "peak_cache_length":
            prompt_len,
            "cache_memory_bytes":
            sum(layer.keys.numel() * layer.keys.element_size() + layer.values.numel() * layer.values.element_size()
                for layer in cache.layers) + static_input.numel() * static_input.element_size() +
            static_attention.numel() * static_attention.element_size() +
            static_causal_attention.numel() * static_causal_attention.element_size() +
            static_position_ids.numel() * static_position_ids.element_size() +
            static_write_positions.numel() * static_write_positions.element_size(),
            "trim_count":
            0,
            "trimmed_columns":
            0,
            "trim_latency_ms":
            0.0,
            "trim_bytes_moved":
            0,
            "decode_steps":
            0,
        }

        slot_by_request = {}
        write_positions = {}
        next_tokens = {}
        next_log_probs = {}
        update = scheduler.schedule()
        while update.active:
            if profile is not None:
                profile["active_batch_sizes"].append(len(update.active))

            for request_id in update.retired:
                static_attention[slot_by_request[request_id]].zero_()
                slot_by_request.pop(request_id, None)
                write_positions.pop(request_id, None)
                next_tokens.pop(request_id, None)
                next_log_probs.pop(request_id, None)

            free_slots = [slot for slot in range(max_batch_size) if slot not in slot_by_request.values()]
            admitted_ids = tuple(request.request_id for request in update.admitted)
            admitted_slots = []
            for request_id in admitted_ids:
                slot = free_slots.pop(0)
                slot_by_request[request_id] = slot
                write_positions[request_id] = prompt_len
                admitted_slots.append(slot)

            cache_start = self._profile_start(profile)
            if admitted_ids:
                admitted_tokens, admitted_log_probs = self._continuous_prefill(
                    module,
                    StaticCache,
                    cache,
                    update,
                    request_by_id,
                    static_attention,
                    static_write_positions,
                    prompt_len,
                    prompt_lengths,
                    model_dtype,
                    device,
                    sampling,
                    profile=profile,
                    target_slots=tuple(admitted_slots),
                )
                next_tokens.update(admitted_tokens)
                next_log_probs.update(admitted_log_probs)
            self._profile_end(profile, "cache_management_overhead_ms", cache_start)

            decode_ids = tuple(request_id for request_id in update.active_ids if request_id not in admitted_ids)
            static_input.zero_()
            static_position_ids.zero_()
            # Captured replay always executes every physical row. Keep non-decoding
            # rows away from their prefetched prompt KV until they become active.
            static_write_positions.fill_(max_cache_len - 1)
            for request_id in decode_ids:
                slot = slot_by_request[request_id]
                position = write_positions[request_id]
                static_input[slot:slot + 1].copy_(next_tokens[request_id])
                static_position_ids[slot, 0] = static_attention[slot, :position].sum()
                static_attention[slot, position] = 1
                static_write_positions[slot] = position

            decoded_tokens = {}
            if decode_ids:
                self._update_static_causal_attention(static_attention, static_causal_attention)
                decode_start = self._profile_start(profile)
                get_accelerator().replay_graph(graph)
                self._profile_end(profile, "decode_forward_ms", decode_start, count="num_decode_forwards")
                decode_logits = torch.cat(
                    [static_logits[slot_by_request[request_id], -1, :].unsqueeze(0) for request_id in decode_ids],
                    dim=0)
                decoded, decoded_log_probs_all = self._continuous_next_tokens(
                    decode_logits,
                    decode_ids,
                    request_by_id,
                    responses,
                    _repetition_penalty(sampling, module),
                    select_next_tokens,
                )
                decoded_log_prob_rows = decoded_log_probs_all.split(1, dim=0)
                for request_id, token, log_prob in zip(decode_ids, decoded.split(1, dim=0), decoded_log_prob_rows):
                    decoded_tokens[request_id] = token
                    next_log_probs[request_id] = log_prob
                    write_positions[request_id] += 1
                stats["decode_steps"] += 1
                next_tokens.update(decoded_tokens)

            finished_ids = []
            for request_id in update.active_ids:
                token = next_tokens[request_id]
                responses[request_id].append(token)
                response_log_probs[request_id].append(next_log_probs[request_id])
                if self._is_eos(token):
                    finished_ids.append(request_id)

            scheduler_start = self._profile_start(profile)
            update = scheduler.advance(finished_ids)
            self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)
            if write_positions:
                stats["peak_cache_length"] = max(stats["peak_cache_length"], max(write_positions.values()))

        generation_end = self._profile_start(profile)
        output = self._build_continuous_batch(trajectories, responses, response_log_probs, self.return_log_probs)
        post_processing_end = self._profile_end(profile, None, generation_end)
        if profile_accelerator is not None:
            profile_accelerator.synchronize()
            total_ms = (time.perf_counter() - profile_start) * 1000.0
            generated_tokens = sum(len(response) for response in responses.values())
            stats["end_to_end_ms"] = total_ms
            stats["tokens_per_second"] = generated_tokens / (total_ms / 1000.0) if total_ms > 0.0 else 0.0
        else:
            stats["end_to_end_ms"] = None
            stats["tokens_per_second"] = None
        stats["trim_frequency"] = 0.0
        self._last_continuous_stats = stats
        if profile is not None:
            self._finish_continuous_profile(profile, len(trajectories), responses, max_batch_size, prompt_len,
                                            generation_end, post_processing_end)
        return output

    def _validate_continuous_graph_options(self, prompt_len, sampling):
        if self.align_decode_fronts:
            raise ValueError("continuous graph capture does not support aligned decode fronts")
        if self.enable_cache_trimming:
            raise ValueError("continuous graph capture does not support cache trimming")
        if (self.continuous_cache_capacity is not None
                and self.continuous_cache_capacity < prompt_len + sampling.max_new_tokens):
            raise ValueError("continuous_cache_capacity must fit the prompt and all generated tokens")

    def _get_continuous_graph_state(self, module, max_batch_size, prompt_len, max_cache_len, device, model_dtype,
                                    cache_type):
        signature = self._continuous_graph_signature(module, max_batch_size, prompt_len, max_cache_len, device,
                                                     model_dtype)
        state = self._continuous_graph_state
        if state is None or state.signature != signature:
            cache = cache_type(
                module.config,
                batch_size=max_batch_size,
                max_cache_len=max_cache_len,
                device=device,
                dtype=model_dtype,
            )
            static_input = torch.zeros((max_batch_size, 1), dtype=torch.long, device=device)
            static_attention = torch.zeros((max_batch_size, max_cache_len), dtype=torch.long, device=device)
            static_causal_attention = torch.zeros((max_batch_size, 1, 1, max_cache_len),
                                                  dtype=model_dtype,
                                                  device=device)
            static_position_ids = torch.zeros((max_batch_size, 1), dtype=torch.long, device=device)
            static_write_positions = torch.zeros(max_batch_size, dtype=torch.long, device=device)
            cache.set_write_position(static_write_positions)
            # Transformers uses cache_position to build the causal mask. Per-row
            # validity is represented by static_attention, so a fixed upper bound
            # lets one captured graph serve requests at different decode lengths.
            static_cache_position = torch.tensor([max_cache_len - 1], dtype=torch.long, device=device)
            graph, static_logits = self._capture_continuous_graph(
                module,
                cache,
                static_input,
                static_causal_attention,
                static_write_positions,
                static_cache_position,
                static_position_ids,
            )
            state = _ContinuousGraphState(
                signature,
                cache,
                static_input,
                static_attention,
                static_causal_attention,
                static_position_ids,
                static_write_positions,
                static_cache_position,
                graph,
                static_logits,
            )
            self._continuous_graph_state = state

        state.cache.reset()
        state.static_input.zero_()
        state.static_attention.zero_()
        state.static_causal_attention.zero_()
        state.static_position_ids.zero_()
        state.static_write_positions.zero_()
        return state

    @staticmethod
    def _continuous_graph_signature(module, max_batch_size, prompt_len, max_cache_len, device, model_dtype):
        parameter_signature = tuple((parameter.data_ptr(), parameter._version) for parameter in module.parameters())
        return (id(module), device.type, device.index, model_dtype, max_batch_size, prompt_len, max_cache_len,
                parameter_signature)

    @staticmethod
    def _update_static_causal_attention(static_attention, static_causal_attention):
        static_causal_attention.copy_(static_attention[:, None, None, :])
        static_causal_attention.neg_().add_(1).mul_(torch.finfo(static_causal_attention.dtype).min)

    @staticmethod
    def _capture_continuous_graph(module, cache, static_input, static_attention, static_write_positions,
                                  static_cache_position, static_position_ids):
        """Capture one fixed-capacity decode forward and return its static logits."""
        accelerator = get_accelerator()
        saved_pre = dict(module._forward_pre_hooks)
        saved_post = dict(module._forward_hooks)
        module._forward_pre_hooks.clear()
        module._forward_hooks.clear()
        try:
            warmup_stream = accelerator.Stream()
            warmup_stream.wait_stream(accelerator.current_stream())
            with accelerator.stream(warmup_stream):
                for _ in range(3):
                    module(
                        static_input,
                        attention_mask=static_attention,
                        past_key_values=cache,
                        use_cache=True,
                        cache_position=static_cache_position,
                        position_ids=static_position_ids,
                    )
            accelerator.current_stream().wait_stream(warmup_stream)
            graph = accelerator.create_graph()
            with accelerator.capture_to_graph(graph):
                output = module(
                    static_input,
                    attention_mask=static_attention,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=static_cache_position,
                    position_ids=static_position_ids,
                )
        finally:
            module._forward_pre_hooks.update(saved_pre)
            module._forward_hooks.update(saved_post)
        return graph, output.logits

    @staticmethod
    def _continuous_cache_exhaustion_error(max_cache_len):
        return ValueError(
            f"continuous batching cache capacity ({max_cache_len}) exhausted before active requests retired; "
            "increase HybridEngineRolloutConfig.continuous_cache_capacity or enable cache trimming with "
            "HybridEngineRolloutConfig(enable_cache_trimming=True)")

    @staticmethod
    def _estimate_continuous_cache_len(prompt_len, max_new_tokens, max_batch_size):
        """Estimate the largest cache span before the scheduler becomes empty."""
        pending = list(max_new_tokens)
        active = []
        max_cache_len = prompt_len
        busy_span = 0

        while active or pending:
            if not active:
                active = pending[:max_batch_size]
                pending = pending[max_batch_size:]
                busy_span = 0

            busy_span += 1
            max_cache_len = max(max_cache_len, prompt_len + busy_span)
            survivors = [remaining - 1 for remaining in active if remaining > 1]
            free_slots = max_batch_size - len(survivors)
            admitted = pending[:free_slots]
            pending = pending[free_slots:]

            if survivors:
                active = survivors + admitted
                continue

            if admitted:
                active = admitted
                # All previous rows retired, so the implementation resets the cache
                # before prefilling the newly admitted requests.
                busy_span = 0
            else:
                active = []

        return max_cache_len

    def _validate_continuous_inputs(self, sampling, max_batch_size):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.use_shared_prefill:
            raise ValueError("continuous batching does not support shared prompt prefill")
        if sampling.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if sampling.n_samples_per_prompt != 1:
            raise ValueError("continuous batching currently supports one sample per prompt")

    @staticmethod
    def _continuous_dead_prefix(attention_mask, active_count):
        if active_count == 0:
            return 0
        occupied = attention_mask[:active_count].any(dim=0)
        if not occupied.any():
            return 0
        return int(occupied.to(dtype=torch.int32).argmax().item())

    def _continuous_prefill(self,
                            module,
                            static_cache_type,
                            cache,
                            update,
                            request_by_id,
                            attention_mask,
                            write_positions,
                            cache_position,
                            prompt_lengths,
                            model_dtype,
                            device,
                            sampling,
                            align_decode_fronts=False,
                            profile=None,
                            target_slots=None):
        if not update.admitted:
            return {}, {}

        if not align_decode_fronts:
            admitted_ids = tuple(request.request_id for request in update.admitted)
            if target_slots is None:
                target_slots = update.admitted_slots
            if len(target_slots) != len(admitted_ids):
                raise ValueError("continuous prefill needs one target slot per admitted request")
            prompt_ids = torch.cat([request_by_id[request_id].prompt_ids for request_id in admitted_ids], dim=0)
            prompt_attention = torch.cat(
                [request_by_id[request_id].prompt_attention_mask for request_id in admitted_ids], dim=0)
            prompt_len = prompt_ids.shape[1]
            prefill_start = self._profile_start(profile)
            prefill_tokens, prefill_log_probs, prefill_cache = self._continuous_prefill_cache(
                module,
                static_cache_type,
                prompt_ids,
                prompt_attention,
                model_dtype,
                device,
                sampling,
            )
            self._profile_end(profile, "prefill_forward_ms", prefill_start, count="num_prefill_forwards")
            cache_copy_start = self._profile_start(profile)
            cache_start = cache_position - prompt_len
            if cache_start < 0:
                raise ValueError("continuous batching prompt does not fit behind the decode front")
            for layer_idx in range(self._cache_layer_count(prefill_cache)):
                prefill_keys, prefill_values = self._cache_layer_tensors(prefill_cache, layer_idx)
                target_layer = cache.layers[layer_idx]
                for source_row, target_row in enumerate(target_slots):
                    target_layer.keys[target_row, :, cache_start:cache_position].copy_(prefill_keys[source_row])
                    target_layer.values[target_row, :, cache_start:cache_position].copy_(prefill_values[source_row])
            for source_row, target_row in enumerate(target_slots):
                attention_mask[target_row, cache_start:cache_position].copy_(prompt_attention[source_row])
                write_positions[target_row] = cache_position
            self._profile_end(profile, "cache_management_overhead_ms", cache_copy_start)
            admitted_tokens = dict(zip(admitted_ids, prefill_tokens.split(1, dim=0)))
            admitted_log_probs = dict(zip(admitted_ids, prefill_log_probs.split(1, dim=0)))
            return admitted_tokens, admitted_log_probs

        admitted_tokens = {}
        admitted_log_probs = {}
        for admitted, target_row in zip(update.admitted, update.admitted_slots):
            request_id = admitted.request_id
            request = request_by_id[request_id]
            prompt_len = prompt_lengths[request_id]
            valid = request.prompt_attention_mask[0].bool()
            prompt_ids = request.prompt_ids[:, valid]
            prompt_attention = torch.ones_like(prompt_ids)
            prefill_start = self._profile_start(profile)
            prefill_tokens, prefill_log_probs, prefill_cache = self._continuous_prefill_cache(
                module,
                static_cache_type,
                prompt_ids,
                prompt_attention,
                model_dtype,
                device,
                sampling,
            )
            self._profile_end(profile, "prefill_forward_ms", prefill_start, count="num_prefill_forwards")
            cache_copy_start = self._profile_start(profile)
            admitted_tokens[request_id] = prefill_tokens
            admitted_log_probs[request_id] = prefill_log_probs
            cache_start = cache_position - prompt_len
            if cache_start < 0:
                raise ValueError("continuous batching prompt does not fit behind the decode front")
            for layer_idx in range(self._cache_layer_count(prefill_cache)):
                prefill_keys, prefill_values = self._cache_layer_tensors(prefill_cache, layer_idx)
                target_layer = cache.layers[layer_idx]
                target_layer.keys[target_row, :, cache_start:cache_position].copy_(prefill_keys[0])
                target_layer.values[target_row, :, cache_start:cache_position].copy_(prefill_values[0])
            attention_mask[target_row, cache_start:cache_position].fill_(1)
            write_positions[target_row] = cache_position
            self._profile_end(profile, "cache_management_overhead_ms", cache_copy_start)
        return admitted_tokens, admitted_log_probs

    def _start_continuous_profile(self):
        accelerator = get_accelerator()
        accelerator.synchronize()
        return {
            "accelerator": accelerator,
            "start": time.perf_counter(),
            "prefill_forward_ms": 0.0,
            "decode_forward_ms": 0.0,
            "scheduler_overhead_ms": 0.0,
            "cache_management_overhead_ms": 0.0,
            "num_prefill_forwards": 0,
            "num_decode_forwards": 0,
            "active_batch_sizes": [],
        }

    @staticmethod
    def _profile_start(profile):
        if profile is None:
            return None
        profile["accelerator"].synchronize()
        return time.perf_counter()

    @staticmethod
    def _profile_end(profile, field, start, count=None):
        if profile is None or start is None:
            return None
        profile["accelerator"].synchronize()
        end = time.perf_counter()
        if field is not None:
            profile[field] += (end - start) * 1000.0
        if count is not None:
            profile[count] += 1
        return end

    def _finish_continuous_profile(self, profile, batch_size, responses, max_batch_size, prompt_len, generation_end,
                                   post_processing_end):
        total_ms = (post_processing_end - profile["start"]) * 1000.0
        generation_ms = (generation_end - profile["start"]) * 1000.0
        prefill_ms = profile["prefill_forward_ms"]
        decode_ms = profile["decode_forward_ms"]
        response_lengths = [len(response) for response in responses.values()]
        num_generated_tokens = sum(response_lengths)
        self._last_profile = {
            "prompt_expansion_ms": 0.0,
            "generation_ms": generation_ms,
            "prefill_forward_ms": prefill_ms,
            "decode_forward_ms": decode_ms,
            "generation_overhead_ms": max(0.0, generation_ms - prefill_ms - decode_ms),
            "num_prefill_forwards": profile["num_prefill_forwards"],
            "num_decode_forwards": profile["num_decode_forwards"],
            "scheduler_overhead_ms": profile["scheduler_overhead_ms"],
            "cache_management_overhead_ms": profile["cache_management_overhead_ms"],
            "post_processing_ms": (post_processing_end - generation_end) * 1000.0,
            "total_ms": total_ms,
            "num_generated_tokens": num_generated_tokens,
            "tokens_per_second": num_generated_tokens / (total_ms / 1000.0) if total_ms > 0.0 else 0.0,
            "batch_size": batch_size,
            "num_samples_per_prompt": 1,
            "prompt_length": prompt_len,
            "response_length": max(response_lengths, default=0),
            "active_batch_size": max(profile["active_batch_sizes"], default=0),
            "continuous_batch_size": max_batch_size,
        }

    def _continuous_prefill_cache(self, module, static_cache_type, prompt_ids, prompt_attention, model_dtype, device,
                                  sampling):
        position_ids = self._prefill_position_ids(prompt_attention)
        if getattr(module, "_supports_cache_class", None) is None:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            prefill_output = module.generate(
                prompt_ids,
                attention_mask=prompt_attention,
                position_ids=position_ids,
                max_new_tokens=1,
                eos_token_id=None,
                pad_token_id=pad_token_id,
                return_dict_in_generate=True,
                output_logits=True,
                **self._hf_sampling_kwargs(sampling),
            )
            tokens = prefill_output.sequences[:, -1:]
            # output_logits holds the raw logits; return_log_probs rules out every other setting that changes them.
            scale = sampling.temperature if sampling.temperature > 0.0 else 1.0
            log_probs = _chosen_log_probs(prefill_output.logits[-1] / scale, tokens)
            return tokens, log_probs, prefill_output.past_key_values

        prompt_len = prompt_ids.shape[1]
        prefill_cache = self._create_static_cache(static_cache_type, module.config, prompt_ids.shape[0], prompt_len,
                                                  device, model_dtype)
        prefill_output = self._call_model(
            module,
            prompt_ids,
            attention_mask=prompt_attention,
            past_key_values=prefill_cache,
            use_cache=True,
            cache_position=torch.arange(prompt_len, device=device),
            position_ids=position_ids,
            **self._last_logits_kwargs(module),
        )
        last_logits = _penalize_repeats(prefill_output.logits[:, -1, :], prompt_ids,
                                        _repetition_penalty(sampling, module))
        tokens, log_probs = _token_selector(sampling)(last_logits)
        return tokens, log_probs, prefill_cache

    @staticmethod
    def _continuous_next_tokens(logits, request_ids, request_by_id, responses, repetition_penalty, select_next_tokens):
        if repetition_penalty == 1.0:
            return select_next_tokens(logits)

        penalized = []
        for row, request_id in enumerate(request_ids):
            input_ids = torch.cat((request_by_id[request_id].prompt_ids, *responses[request_id]), dim=1)
            penalized.append(_penalize_repeats(logits[row:row + 1], input_ids, repetition_penalty))
        return select_next_tokens(torch.cat(penalized, dim=0))

    @staticmethod
    def _hf_sampling_kwargs(sampling):
        """Hugging Face ``generate()`` arguments for ``sampling``; ``temperature <= 0`` is greedy."""
        if sampling.temperature <= 0.0:
            kwargs = {"do_sample": False, "temperature": 1.0, "top_p": 1.0}
        else:
            kwargs = {
                "do_sample": True,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": max(sampling.top_k, 0),
            }
        if sampling.repetition_penalty is not None:
            kwargs["repetition_penalty"] = sampling.repetition_penalty
        return kwargs

    @staticmethod
    def _prefill_position_ids(attention_mask):
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        return position_ids.masked_fill(attention_mask == 0, 1)

    def _is_eos(self, token):
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            return False
        eos_ids = torch.as_tensor(eos_token_id, dtype=token.dtype, device=token.device).flatten()
        return bool((token == eos_ids).any().item())

    @staticmethod
    def _last_logits_kwargs(module):
        """Ask models that support it for last-position logits only, since prefill samples just one token."""
        # Without this the prefill materializes [batch, prompt_len, vocab] logits, the largest activation of the
        # rollout. Models that do not declare the argument may forward unknown kwargs to submodules, so skip them.
        if "logits_to_keep" in signature(module.forward).parameters:
            return {"logits_to_keep": 1}
        return {}

    @staticmethod
    def _call_model(module, input_ids, **kwargs):
        """Call models with only the cache arguments their HF version accepts."""
        parameters = signature(module.forward).parameters
        accepts_kwargs = any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values())
        if not accepts_kwargs:
            kwargs = {name: value for name, value in kwargs.items() if name in parameters}
        return module(input_ids, **kwargs)

    @staticmethod
    def _create_static_cache(static_cache_type, config, batch_size, max_cache_len, device, dtype):
        """Construct StaticCache across Transformers' batch-size API variants."""
        cache_config = config
        if not hasattr(config, "num_key_value_heads") or config.num_key_value_heads is None:
            cache_config = copy(config)
            cache_config.num_key_value_heads = config.num_attention_heads
        common_kwargs = {
            "config": cache_config,
            "max_cache_len": max_cache_len,
            "device": device,
            "dtype": dtype,
        }
        parameters = signature(static_cache_type).parameters
        if "batch_size" in parameters:
            common_kwargs["batch_size"] = batch_size
        elif "max_batch_size" in parameters:
            common_kwargs["max_batch_size"] = batch_size
        return static_cache_type(**common_kwargs)

    @staticmethod
    def _cache_layer_count(cache):
        if hasattr(cache, "layers"):
            return len(cache.layers)
        return len(cache.key_cache)

    @staticmethod
    def _cache_layer_tensors(cache, layer_idx):
        if hasattr(cache, "layers"):
            layer = cache.layers[layer_idx]
            return layer.keys, layer.values
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]

    def _build_continuous_batch(self, trajectories, responses, response_log_probs, return_log_probs):
        """Append each request's response, which stops at its EOS or ``max_new_tokens``, to its sample."""
        for request_id, sample in enumerate(trajectories):
            response = torch.cat(responses[request_id], dim=1)
            if self._is_eos(response[:, -1]):
                sample.status = RolloutRecord.Status.COMPLETED
            else:
                sample.status = RolloutRecord.Status.TRUNCATED
            log_probs = None
            if return_log_probs:
                log_probs = torch.cat(response_log_probs[request_id], dim=1)[0].tolist()
            update_sample_from_response(sample, response[0].tolist(), log_probs, [], self.tokenizer)
        return RolloutBatch(samples=trajectories)

    def get_last_profile(self):
        """Return the most recent profiling snapshot for this rollout instance."""
        return self._last_profile

    def get_last_continuous_stats(self):
        """Return cache and throughput statistics from the most recent continuous rollout."""
        return self._last_continuous_stats

    def _register_shared_prefill_hooks(self, module, batch_size, repeats):
        state = {"pending": True, "reduced": False}

        def reduce_prompt_batch(_module, args, kwargs):
            input_ids = kwargs.get("input_ids")
            if not state["pending"]:
                return args, kwargs
            if input_ids is None:
                raise RuntimeError("Shared prefill requires input_ids as a keyword argument")
            expected_batch_size = batch_size * repeats
            if input_ids.shape[0] != expected_batch_size:
                raise RuntimeError("Shared prefill input batch does not match the expanded rollout batch")
            if input_ids.shape[1] <= 1:
                raise RuntimeError("Shared prefill requires a prompt with more than one token")
            kwargs = dict(kwargs)
            kwargs["input_ids"] = input_ids[::repeats]
            for name in ("attention_mask", "position_ids", "token_type_ids"):
                value = kwargs.get(name)
                if isinstance(value, torch.Tensor) and value.shape[0] == expected_batch_size:
                    kwargs[name] = value[::repeats]
            state["reduced"] = True
            return args, kwargs

        def expand_prompt_output(_module, _args, _kwargs, output):
            if not state["pending"]:
                return output
            if not state["reduced"]:
                raise RuntimeError("Shared prefill did not reduce the prompt batch")
            state["pending"] = False
            output.past_key_values = self.engine.repeat_shared_prefill_cache(batch_size, repeats)
            output.logits = output.logits.repeat_interleave(repeats, dim=0)
            return output

        pre_handle = module.register_forward_pre_hook(reduce_prompt_batch, with_kwargs=True)
        post_handle = module.register_forward_hook(expand_prompt_output, with_kwargs=True)
        return pre_handle, post_handle

    @staticmethod
    def _pad_after_eos(output_ids, response_start, eos_token_id, pad_token_id):
        """Retain the first response EOS and pad every subsequent position.

        Only positions after a row's first EOS are padding: generation pads a
        row only once it has ended, so a token equal to ``pad_token_id`` before
        that was generated and stays part of the response.
        """
        response_ids = output_ids[:, response_start:]
        if eos_token_id is None or response_ids.shape[1] == 0:
            return output_ids, torch.ones_like(response_ids)

        eos_ids = torch.as_tensor(eos_token_id, device=response_ids.device, dtype=response_ids.dtype).flatten()
        is_eos = (response_ids.unsqueeze(-1) == eos_ids).any(dim=-1)
        has_eos = is_eos.any(dim=-1)
        first_eos_idx = is_eos.long().argmax(dim=-1)
        positions = torch.arange(response_ids.shape[1], device=response_ids.device).unsqueeze(0)
        after_first_eos = has_eos.unsqueeze(1) & (positions > first_eos_idx.unsqueeze(1))

        output_ids = output_ids.clone()
        output_ids[:, response_start:].masked_fill_(after_first_eos, pad_token_id)
        return output_ids, (~after_first_eos).long()

    # ------------------------------------------------------------------
    # Graph capture decode loop
    # ------------------------------------------------------------------

    def _prefill_rows(self, module, prompt_ids, row_lengths, ds_cache, logits_kwargs, model_dtype, device, target_rows,
                      cache_rows):
        """Prefill left-padded prompts without their padding; returns the last-position logits.

        Each group of equally long rows is prefilled on its own, unpadded, into a prompt-sized HF cache, and its KV
        lands in slots ``0..len-1`` of rows ``target_rows`` of ``ds_cache``, which has ``cache_rows`` rows. Newly
        admitted turns thereby fill free rows of a cache that outlives the call.
        """
        from transformers import StaticCache

        batch_size, prompt_len = prompt_ids.shape
        last_logits = None
        lengths = row_lengths.tolist()
        for length in sorted(set(lengths)):
            group = torch.tensor([row for row, n in enumerate(lengths) if n == length], device=device)
            targets = target_rows[group]
            group_cache = self._create_static_cache(StaticCache, module.config, group.numel(), length, device,
                                                    model_dtype)
            prefill_out = module(
                prompt_ids[group, prompt_len - length:],
                attention_mask=None,
                past_key_values=group_cache,
                use_cache=True,
                cache_position=torch.arange(length, device=device),
                **logits_kwargs,
            )
            if last_logits is None:
                last_logits = prefill_out.logits.new_empty(batch_size, prefill_out.logits.shape[-1])
            last_logits[group] = prefill_out.logits[:, -1, :]

            for layer_idx, ds_layer in enumerate(ds_cache.layers):
                hf_keys, hf_values = self._cache_layer_tensors(group_cache, layer_idx)
                if not ds_layer.is_initialized:
                    ds_layer.lazy_initialization(hf_keys.new_empty(cache_rows, *hf_keys.shape[1:]),
                                                 hf_values.new_empty(cache_rows, *hf_values.shape[1:]))
                ds_layer.keys[targets, :, :length, :] = hf_keys
                ds_layer.values[targets, :, :length, :] = hf_values
            # Its rows are copied out; dropping the group's cache frees its KV before the next group runs.
            del prefill_out, group_cache
        return last_logits

    def _generate_graph(self, prompt_ids, prompt_attn, max_new_tokens, pad_token_id, sampling, module, device):
        """Decode with DeepSpeedStaticCache + CUDA graph capture, greedy or sampled per ``sampling``.

        Only the decode forward is captured. Tokens are chosen eagerly from the
        graph's output logits after each replay, so sampling draws fresh random
        numbers every step instead of replaying the captured ones. Rows keep the
        left-padded layout and the mask hides their pad slots.
        """
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        max_len = prompt_len + max_new_tokens
        eos_token_id = self.tokenizer.eos_token_id
        model_dtype = next(module.parameters()).dtype
        # AutoTP wraps module.config as a dict after sharding; restore the
        # original config object so cache creation and mask logic work
        if isinstance(getattr(module, "config", None), dict):
            from types import SimpleNamespace
            module.config = SimpleNamespace(**module.config)
        if getattr(module.config, "_attn_implementation", None) == "flash_attention_2":
            raise ValueError("graph capture does not support flash_attention_2")
        rows = torch.arange(batch_size, device=device)

        ds_tp_config = module.config
        import deepspeed.comm as dist
        if dist.is_initialized() and dist.get_world_size() > 1:
            ds_tp_config = copy(module.config)
            ds_tp_config.num_key_value_heads = max(1, module.config.num_key_value_heads // dist.get_world_size())
            ds_tp_config.num_attention_heads = max(1, module.config.num_attention_heads // dist.get_world_size())
        ds_cache = DeepSpeedStaticCache(
            ds_tp_config,
            batch_size=batch_size,
            max_cache_len=max_len,
            device=device,
            dtype=model_dtype,
        )
        logits_kwargs = self._last_logits_kwargs(module)
        # --- Prefill straight into the cache the graph decodes from ---
        # A second full-length cache for the prefill alone would double the KV memory of the call.
        # The prompt fills slots 0..prompt_len-1.
        ds_cache.set_write_position(torch.tensor(0, dtype=torch.long, device=device))
        prefill_out = module(
            prompt_ids,
            attention_mask=prompt_attn,
            past_key_values=ds_cache,
            use_cache=True,
            # Explicit positions: HF otherwise derives them from the cache, which syncs with the host and
            # aborts stream capture on some model/cache combinations.
            cache_position=torch.arange(prompt_len, device=device),
            position_ids=torch.arange(prompt_len, device=device).expand(batch_size, -1),
            **logits_kwargs,
        )
        prefill_logits = prefill_out.logits[:, -1, :]
        # Left-padded rows all decode from the end of the padded prompt; the mask hides their pad slots.
        decode_start = torch.full((batch_size, ), prompt_len, dtype=torch.long, device=device)
        prompt_slots = prompt_attn.bool()
        select_next_tokens = _token_selector(sampling)
        repetition_penalty = _repetition_penalty(sampling, module)
        # generate() penalizes the sequence so far, prompt included.
        next_token, first_log_probs = select_next_tokens(
            _penalize_repeats(prefill_logits, prompt_ids, repetition_penalty))
        log_probs = [first_log_probs]

        output_ids = [prompt_ids, next_token]

        # Decoding writes one slot per row per step.
        write_pos = decode_start.clone()
        ds_cache.set_write_position(write_pos)

        # --- Static buffers for graph capture ---
        static_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        # Full-width mask, revealed one position per decode step by in-place writes (graph-replay safe).
        static_attn = torch.zeros(batch_size, 1, 1, max_len, dtype=torch.bool, device=device)
        static_attn[:, 0, 0, :prompt_len] = prompt_slots
        static_attn[rows, 0, 0, decode_start] = True
        # Views of write_pos, not copies: the captured forwards read this memory, so replays see the new position
        # without host-side updates.
        static_cache_pos = write_pos[:1]
        static_pos_ids = write_pos.unsqueeze(1)

        # A pre-built 4-D additive mask: Transformers checks a 2-D mask on the host during capture, and eager attention
        # adds its mask to the attention weights, so a boolean one would be wrong.
        static_causal_attn = torch.zeros((batch_size, 1, 1, max_len), dtype=model_dtype, device=device)
        self._update_static_causal_attention(static_attn[:, 0, 0], static_causal_attn)

        # Remove forward hooks (they synchronize — illegal during graph capture)
        saved_pre = dict(module._forward_pre_hooks)
        saved_post = dict(module._forward_hooks)
        module._forward_pre_hooks.clear()
        module._forward_hooks.clear()

        try:
            # Warmup on side stream
            static_token.copy_(next_token)
            s = get_accelerator().Stream()
            s.wait_stream(get_accelerator().current_stream())
            with get_accelerator().stream(s):
                for _ in range(3):
                    out = module(
                        static_token,
                        attention_mask=static_causal_attn,
                        past_key_values=ds_cache,
                        use_cache=True,
                        cache_position=static_cache_pos,
                        position_ids=static_pos_ids,
                    )
            get_accelerator().current_stream().wait_stream(s)

            # Capture
            graph = get_accelerator().create_graph()
            with get_accelerator().capture_to_graph(graph):
                out = module(
                    static_token,
                    attention_mask=static_causal_attn,
                    past_key_values=ds_cache,
                    use_cache=True,
                    cache_position=static_cache_pos,
                    position_ids=static_pos_ids,
                )
                static_logits = out.logits
            # The capture run consumed static_token: restore the write position, so decoding starts right after
            # prefill.
            write_pos.copy_(decode_start)
            static_token.copy_(next_token)
        finally:
            module._forward_pre_hooks.update(saved_pre)
            module._forward_hooks.update(saved_post)

        # --- Decode loop ---
        # Rows that ended keep decoding until every row has; callers cut each response at its first EOS.
        eos_mask = next_token.squeeze(1) == eos_token_id
        for step in range(max_new_tokens - 1):
            if bool(eos_mask.all().item()):
                break

            # Update static inputs. Only the current token's own slot is revealed: later slots still hold warmup
            # and capture leftovers.
            static_token.copy_(next_token)
            torch.add(decode_start, step, out=write_pos)
            static_attn[rows, 0, 0, write_pos] = True
            self._update_static_causal_attention(static_attn[:, 0, 0], static_causal_attn)

            # Replay
            get_accelerator().replay_graph(graph)
            step_logits = static_logits[:, -1, :]
            if repetition_penalty != 1.0:
                step_logits = _penalize_repeats(step_logits, torch.cat(output_ids, dim=1), repetition_penalty)
            next_token, step_log_probs = select_next_tokens(step_logits)
            output_ids.append(next_token)
            log_probs.append(step_log_probs)
            eos_mask |= (next_token.squeeze(1) == eos_token_id)

        # Every row ended: the rest of the response is padding.
        remaining = max_new_tokens - (len(output_ids) - 1)
        if remaining > 0:
            output_ids.append(torch.full((batch_size, remaining), pad_token_id, dtype=torch.long, device=device))
            log_probs.append(torch.zeros((batch_size, remaining), device=device))
        return torch.cat(output_ids, dim=1), torch.cat(log_probs, dim=1)

    def sync_weights(self, step: int) -> None:  # noqa: ARG002
        """No-op: hybrid engine reads model weights live."""
        return None

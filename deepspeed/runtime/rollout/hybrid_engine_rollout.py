# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engine backed by DeepSpeed's hybrid engine.

Two generation paths:
  1. **model.generate()** (default): delegates to HuggingFace generate.
     Supports sampling (temperature, top_p) and greedy.
  2. **graph capture + DeepSpeedStaticCache**: greedy and sampling
     (temperature, top_k, top_p). Pre-allocates a StaticCache, captures the
     decode forward pass into a graph, and replays it for each decode
     step.  Eliminates kernel launch overhead.
  3. **continuous batching (experimental)**: an opt-in bounded path
     selected through ``SamplingConfig.continuous_batch_size``; greedy or
     sampled like the graph path, with or without graph capture.
"""

import math
import time
from copy import copy
from dataclasses import dataclass, replace
from inspect import signature
from typing import Optional

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutEngine, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.continuous_batching import (ContinuousBatchRequest, ContinuousBatchScheduler,
                                                           plan_prefill_buckets)
from deepspeed.utils import logger


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


class GlobalWorkRemaining:
    """Rendezvous a local work flag over a distributed process group."""

    def __init__(self, process_group=None):
        self.process_group = process_group

    def __call__(self, local_work_remaining):
        work_remaining = torch.as_tensor(local_work_remaining, dtype=torch.int32).clone()
        if dist.is_initialized() and dist.get_world_size(group=self.process_group) > 1:
            dist.all_reduce(work_remaining, op=dist.ReduceOp.MAX, group=self.process_group)
        return bool(work_remaining.item())


class _GlobalEosStoppingCriteria:
    """Stop all local sequences only after every rank has observed EOS."""

    def __init__(self, response_start, eos_token_id, work_remaining):
        self.response_start = response_start
        self.eos_token_ids = torch.as_tensor(eos_token_id).flatten()
        self.work_remaining = work_remaining

    def __call__(self, input_ids, scores, **kwargs):  # noqa: ARG002
        response_ids = input_ids[:, self.response_start:]
        eos_token_ids = self.eos_token_ids.to(device=input_ids.device, dtype=input_ids.dtype)
        sequence_has_eos = (response_ids.unsqueeze(-1) == eos_token_ids).any(dim=-1).any(dim=-1)
        local_work_remaining = (~sequence_has_eos).any().to(dtype=torch.int32)
        global_work_remaining = self.work_remaining(local_work_remaining)
        # This is a rank-global decision, so a scalar preserves the same
        # behavior for every local sequence and remains compatible with older
        # Transformers releases that apply Python ``any`` to criterion results.
        return not global_work_remaining


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
    continuous_cache_capacity: Optional[int] = None
    adaptive_prefill: bool = False
    prefill_fixed_cost_ms: float = 25.0
    prefill_token_cost_ms: float = 0.1
    prefill_attention_cost_ms: float = 0.01
    prefill_kv_cost_ms: float = 0.04
    continuous_decode_cost_ms: float = 10.0
    prefill_max_tokens: Optional[int] = 65536


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
    """

    def __init__(self, engine, tokenizer, cfg=None):
        self.engine = engine
        self.tokenizer = tokenizer
        self.use_graph_capture = getattr(cfg, 'use_graph_capture', False) if cfg else False
        self.enable_profiling = getattr(cfg, 'enable_profiling', False) if cfg else False
        self.use_shared_prefill = getattr(cfg, 'use_shared_prefill', False) if cfg else False
        self.align_decode_fronts = getattr(cfg, 'align_decode_fronts', False) if cfg else False
        self.adaptive_prefill = getattr(cfg, 'adaptive_prefill', False) if cfg else False
        for name, default in (("prefill_fixed_cost_ms", 25.0), ("prefill_token_cost_ms", 0.1),
                              ("prefill_attention_cost_ms", 0.01), ("prefill_kv_cost_ms",
                                                                    0.04), ("continuous_decode_cost_ms", 10.0)):
            value = getattr(cfg, name, default) if cfg else default
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            setattr(self, name, value)
        self.prefill_max_tokens = getattr(cfg, 'prefill_max_tokens', 65536) if cfg else 65536
        if self.prefill_max_tokens is not None:
            if not isinstance(self.prefill_max_tokens, int) or self.prefill_max_tokens <= 0:
                raise ValueError("prefill_max_tokens must be a positive integer or None")
        self.enable_cache_trimming = getattr(cfg, 'enable_cache_trimming', False) if cfg else False
        self.continuous_cache_capacity = getattr(cfg, 'continuous_cache_capacity', None) if cfg else None
        self._last_profile = None
        self._continuous_graph_state = None
        self._last_continuous_stats = None
        self._generation_defaults = None

    @torch.no_grad()
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        if sampling.continuous_batch_size is not None:
            return self._generate_continuous(request, sampling, sampling.continuous_batch_size)
        return self._generate_batched(request, sampling)

    def _generation_kwargs(self, sampling, attention_mask, pad_token_id, module=None):
        zero3_sync = self._zero3_work_remaining()
        stopping_criteria = None
        if zero3_sync is not None and self.tokenizer.eos_token_id is not None:
            stopping_criteria = [
                _GlobalEosStoppingCriteria(attention_mask.shape[1], self.tokenizer.eos_token_id, zero3_sync)
            ]
        kwargs = {
            "attention_mask": attention_mask,
            "max_new_tokens": sampling.max_new_tokens,
            "eos_token_id": None if zero3_sync is not None else self.tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria,
            # The group-aware criterion already synchronizes ZeRO-3 parameter gathers.
            "synced_gpus": False if zero3_sync is not None else None,
            **self._hf_sampling_kwargs(sampling),
            "pad_token_id": pad_token_id,
        }
        if module is not None:
            prepare = getattr(module, "_prepare_generation_config", None)
            if callable(prepare) and "use_model_defaults" in signature(prepare).parameters:
                kwargs["use_model_defaults"] = False
        return kwargs

    def _generate_batched(self, request, sampling, generation_config=None, mask_padding=True):
        device = request.prompt_ids.device
        B = request.prompt_ids.shape[0]
        n = sampling.n_samples_per_prompt
        total = B * n
        prompt_len = request.prompt_ids.shape[1]
        max_new_tokens = sampling.max_new_tokens
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("The tokenizer must define pad_token_id or eos_token_id")

        module = self.engine.module

        if self.enable_profiling:
            accelerator = get_accelerator()
            accelerator.synchronize()
            profile_start = time.perf_counter()

        # Expand prompts for n samples per prompt
        if n > 1:
            prompt_ids = request.prompt_ids.repeat_interleave(n, dim=0)
            prompt_attn = request.prompt_attention_mask.repeat_interleave(n, dim=0)
        else:
            prompt_ids = request.prompt_ids
            prompt_attn = request.prompt_attention_mask

        if self.enable_profiling:
            accelerator.synchronize()
            expansion_end = time.perf_counter()

        use_graph_capture = self.use_graph_capture
        if use_graph_capture and not get_accelerator().is_graph_capture_supported():
            # Same opt-in as the HybridEngine graph cache and the continuous
            # rollout path: falling back keeps backends that only expose a
            # graph API on the eager path.
            logger.warning_once("HybridEngineRollout: graph capture is not supported on the current accelerator; "
                                "falling back to eager generation.")
            use_graph_capture = False

        forward_profiler = None
        forward_profile_handles = []
        if self.enable_profiling and not use_graph_capture:
            forward_profiler = _ForwardProfiler(accelerator)
            forward_profile_handles = forward_profiler.register(module)

        shared_prefill_handles = []
        try:
            if self.use_shared_prefill and n > 1:
                if self.use_graph_capture:
                    raise RuntimeError("Shared prefill does not support graph capture")
                self.engine.prepare_shared_prefill(B, n, prompt_len)
                shared_prefill_handles = self._register_shared_prefill_hooks(module, B, n)
            if use_graph_capture:
                output_ids = self._generate_graph(prompt_ids, prompt_attn, max_new_tokens, pad_token_id, module,
                                                  device, sampling)
            else:
                generate_kwargs = self._generation_kwargs(sampling, prompt_attn, pad_token_id,
                                                          module if generation_config is not None else None)
                if generation_config is not None:
                    generate_kwargs["generation_config"] = generation_config
                output_ids = module.generate(prompt_ids, **generate_kwargs)
        finally:
            for handle in shared_prefill_handles:
                handle.remove()
            for handle in forward_profile_handles:
                handle.remove()

        if self.enable_profiling:
            accelerator.synchronize()
            generation_end = time.perf_counter()

        # ZeRO-3 generation ignores EOS in the decode loop so every rank runs the same
        # number of parameter-gather collectives. Restore the usual generation semantics
        # before returning: retain the first EOS in each response and replace every later
        # token with padding.
        output_ids, response_attn = self._pad_after_eos(
            output_ids,
            response_start=prompt_len,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            mask_padding=mask_padding,
        )

        # Build attention mask: pad positions (both left padding from prompt
        # and right padding from EOS / shorter sequences) are 0.
        response_start = prompt_len
        attention_mask = (output_ids != pad_token_id).long()
        attention_mask[:, response_start:] = response_attn
        for i in range(total):
            prompt_valid = request.prompt_attention_mask[i // n if B > 1 else 0]
            attention_mask[i, :prompt_len] = prompt_valid

        rollout_batch = RolloutBatch(
            input_ids=output_ids,
            attention_mask=attention_mask,
            response_start_idx=torch.full((total, ), response_start, dtype=torch.long, device=device),
        )

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
                "batch_size": B,
                "num_samples_per_prompt": n,
                "prompt_length": prompt_len,
                "response_length": response_length,
            }

        return rollout_batch

    @torch.no_grad()
    def _generate_continuous(self, request, sampling, max_batch_size):
        """Generate independent requests in an experimental continuous batch.

        Continuous batching currently requires one prompt row per request.
        Completed rows retire immediately and pending prompts prefill into the
        released rows before the next decode step.
        """
        profile = self._start_continuous_profile() if self.adaptive_prefill and self.enable_profiling else None
        original_request = request
        batch_size = request.prompt_ids.shape[0]
        effective_lengths = self._validate_continuous_inputs(request, sampling, max_batch_size)
        use_graph_capture = self.use_graph_capture
        if use_graph_capture and not get_accelerator().is_graph_capture_supported():
            # The eager continuous path is a full fallback, so keep it instead
            # of aborting when the backend has not opted into graph capture.
            logger.warning_once("HybridEngineRollout: graph capture is not supported on the current accelerator; "
                                "falling back to eager continuous batching.")
            use_graph_capture = False
        module = self.engine.module
        select_next_tokens = self._token_selector(sampling)
        prepared_generation_config = None
        if self.adaptive_prefill and dist.is_initialized() and dist.get_world_size() > 1:
            raise ValueError("adaptive prefill currently supports single-process rollouts only")
        align_decode_fronts = self.adaptive_prefill or self.align_decode_fronts
        if self.adaptive_prefill:
            generation_config = getattr(module, "generation_config", None)
            if generation_config is not None:
                from transformers import GenerationConfig, GenerationMixin
                # HF can refresh untouched defaults from legacy model settings during generate().
                prepare_config = getattr(module, "_prepare_generation_config", None)
                if callable(prepare_config):
                    native_prepare = getattr(prepare_config, "__func__",
                                             None) is GenerationMixin._prepare_generation_config
                    needs_prepare = not native_prepare or getattr(generation_config, "cache_implementation",
                                                                  None) == "hybrid"
                    if getattr(generation_config, "_from_model_config", False):
                        legacy_settings = getattr(module.config, "_get_non_default_generation_parameters", None)
                        needs_prepare = needs_prepare or not callable(legacy_settings) or bool(legacy_settings())
                    if needs_prepare:
                        pad_token_id = self.tokenizer.pad_token_id
                        if pad_token_id is None:
                            pad_token_id = self.tokenizer.eos_token_id
                        overrides = self._generation_kwargs(sampling, request.prompt_attention_mask, pad_token_id,
                                                            module)
                        # generate() consumes these arguments before preparing its generation config.
                        overrides.pop("stopping_criteria")
                        overrides.pop("synced_gpus")
                        generation_config, model_kwargs = prepare_config(None, **overrides)
                        # generate() would use a replaced mask, while CB routes keep the request mask.
                        prepared_mask = model_kwargs.pop("attention_mask", overrides["attention_mask"])
                        if model_kwargs or prepared_mask is not overrides["attention_mask"]:
                            raise ValueError("adaptive prefill does not support custom generation model kwargs")
                        # CB routes read these settings from the request, while generate() reruns the preparer.
                        for name, value in overrides.items():
                            if hasattr(generation_config, name) and getattr(generation_config, name) != value:
                                raise ValueError(f"adaptive prefill does not support a generation preparer that "
                                                 f"changes {name}")
                    prepared_generation_config = generation_config
                if self._generation_defaults is None:
                    self._generation_defaults = GenerationConfig()
                defaults = self._generation_defaults
                supported_settings = {
                    "max_length", "max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "bos_token_id",
                    "eos_token_id", "pad_token_id", "repetition_penalty", "transformers_version"
                }
                # CB implements repetition penalty; other HF processors can change tokens across routes.
                for name, value in vars(generation_config).items():
                    if name.startswith("_") or name in supported_settings:
                        continue
                    if not hasattr(defaults, name) or value != getattr(defaults, name):
                        raise ValueError(f"adaptive prefill does not support non-default generation setting: {name}")
        if align_decode_fronts:
            prompt_lengths = dict(enumerate(map(int, effective_lengths)))
        else:
            prompt_lengths = dict.fromkeys(range(batch_size), request.prompt_ids.shape[1])
        prompt_len = max(prompt_lengths.values())
        if self.adaptive_prefill and self.prefill_max_tokens is not None and prompt_len > self.prefill_max_tokens:
            raise ValueError("a prompt exceeds the prefill token limit; increase the configured limit")
        if use_graph_capture:
            requests = tuple(
                RolloutRequest(request.prompt_ids[index:index + 1], request.prompt_attention_mask[index:index + 1])
                for index in range(batch_size))
            return self._generate_continuous_graph(original_request,
                                                   requests,
                                                   sampling,
                                                   max_batch_size,
                                                   prompt_lengths=prompt_lengths,
                                                   generation_config=prepared_generation_config,
                                                   profile=profile)

        max_positions = getattr(module.config, "max_position_embeddings", None)
        if max_positions is not None:
            if any(length + sampling.max_new_tokens > max_positions for length in prompt_lengths.values()):
                raise ValueError("continuous batching request exceeds the model maximum position embeddings")
        if self.continuous_cache_capacity is not None:
            if self.continuous_cache_capacity < prompt_len:
                raise ValueError("continuous_cache_capacity must be at least the padded prompt length")
            max_cache_len = self.continuous_cache_capacity
        else:
            max_cache_len = (prompt_len +
                             sampling.max_new_tokens if align_decode_fronts else self._estimate_continuous_cache_len(
                                 prompt_len, [sampling.max_new_tokens] * batch_size, max_batch_size))
        if max_positions is not None and max_cache_len > max_positions:
            raise ValueError("continuous batching cache exceeds the model maximum position embeddings")
        if getattr(module, "_supports_cache_class", None) is False:
            raise ValueError("continuous batching requires a model with cache-class support; use the default "
                             "generate() path or upgrade transformers")

        use_batched = False
        if self.adaptive_prefill:
            full_width = request.prompt_ids.shape[1]
            full_tokens_fit = self.prefill_max_tokens is None or request.prompt_ids.numel() <= self.prefill_max_tokens
            full_positions_fit = max_positions is None or full_width + sampling.max_new_tokens <= max_positions
            if (batch_size <= max_batch_size and full_tokens_fit and full_positions_fit
                    and self.continuous_cache_capacity is None and not self.enable_cache_trimming):
                prefill_cost = self._prefill_cost_function(module)
                batched_cost = prefill_cost(batch_size, full_width, include_kv=False)
                decode_cost = self.continuous_decode_cost_ms * (sampling.max_new_tokens - 1)
                # Every partition pays at least one Forward plus each request's unpadded work.
                # If ordinary generation already beats that bound, there is no reason to run DP.
                if min(prompt_lengths.values()) == prompt_len:
                    lower_bound = prefill_cost(batch_size, prompt_len)
                else:
                    lower_bound = self.prefill_fixed_cost_ms + sum(
                        prefill_cost(1, length) - self.prefill_fixed_cost_ms for length in prompt_lengths.values())
                use_batched = batched_cost <= lower_bound + decode_cost
                if not use_batched:
                    _, bucket_cost = plan_prefill_buckets(list(prompt_lengths.values()), prefill_cost,
                                                          self.prefill_max_tokens)
                    use_batched = batched_cost <= bucket_cost + decode_cost
        if use_batched:
            # A bounded rollout with no pending rows can use the existing batch path.
            output = self._generate_batched(request,
                                            replace(sampling, continuous_batch_size=None),
                                            generation_config=prepared_generation_config,
                                            mask_padding=False)
            self._last_continuous_stats = None
            if self.enable_profiling:
                total_ms = (self._profile_start(profile) - profile["start"]) * 1000
                profile = self._last_profile
                routing_ms = total_ms - profile["total_ms"]
                profile["generation_ms"] += routing_ms
                profile["generation_overhead_ms"] += routing_ms
                profile["total_ms"] = total_ms
                profile["generation_strategy"] = "batched"
                generated_tokens = int(output.attention_mask[:, request.prompt_ids.shape[1]:].sum().item())
                profile["num_generated_tokens"] = generated_tokens
                total_ms = profile["total_ms"]
                profile["tokens_per_second"] = generated_tokens * 1000 / total_ms if total_ms > 0 else 0.0
                profile["active_batch_size"] = batch_size
                profile["continuous_batch_size"] = max_batch_size
            return output

        if self.enable_profiling and profile is None:
            profile = self._start_continuous_profile()
        profile_accelerator = profile["accelerator"] if profile is not None else None
        profile_start = profile["start"] if profile is not None else None

        from transformers import StaticCache
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        requests = tuple(
            RolloutRequest(request.prompt_ids[index:index + 1], request.prompt_attention_mask[index:index + 1])
            for index in range(batch_size))
        device = requests[0].prompt_ids.device
        model_dtype = next(module.parameters()).dtype

        scheduler_start = self._profile_start(profile)
        scheduler = ContinuousBatchScheduler(max_batch_size, sampling.max_new_tokens)
        request_by_id = {}
        responses = {}
        request_order = (sorted(range(len(requests)), key=lambda request_id: -prompt_lengths[request_id])
                         if align_decode_fronts else range(len(requests)))
        for request_id in request_order:
            request = requests[request_id]
            scheduler.submit(ContinuousBatchRequest(request_id))
            request_by_id[request_id] = request
            responses[request_id] = []
        self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)

        cache_start = self._profile_start(profile)
        # Layers are sized by the first prefill because AutoTP shards KV heads per rank.
        cache = DeepSpeedStaticCache(module.config, max_cache_len=max_cache_len)
        write_positions = torch.full((max_batch_size, ), -1, dtype=torch.long, device=device)
        cache.set_write_position(write_positions)
        attention_mask = torch.zeros((max_batch_size, max_cache_len), dtype=torch.long, device=device)
        stats = {
            "cache_capacity": max_cache_len,
            "peak_cache_length": prompt_len,
            "trim_count": 0,
            "trimmed_columns": 0,
            "trim_latency_ms": 0.0,
            "decode_steps": 0,
        }
        next_tokens = {}
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
                    if align_decode_fronts:
                        span_starts[:survivor_count] = [span_starts[index] for index in update.keep_slots]
                        span_starts[survivor_count:] = [0] * (max_batch_size - survivor_count)

                dead_prefix = 0
                if self.enable_cache_trimming:
                    if align_decode_fronts:
                        dead_prefix = min(span_starts[:survivor_count])
                    else:
                        trim_threshold = max(1, prompt_len)
                        dead_prefix = self._continuous_dead_prefix(attention_mask, survivor_count)
                        if dead_prefix < trim_threshold and cache_position < max_cache_len - trim_threshold:
                            dead_prefix = 0
                elif cache_position >= max_cache_len:
                    if align_decode_fronts:
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
                    if align_decode_fronts:
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
            admitted_tokens = self._continuous_prefill(
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
                align_decode_fronts,
                profile,
                generation_config=prepared_generation_config,
            )
            if align_decode_fronts:
                for admitted, target_row in zip(update.admitted, update.admitted_slots):
                    span_starts[target_row] = cache_position - prompt_lengths[admitted.request_id]

            decoded_tokens = {}
            if survivor_count:
                survivor_ids = update.active_ids[:survivor_count]
                decode_input = torch.cat([next_tokens[request_id] for request_id in survivor_ids], dim=0)
                if cache_position >= max_cache_len:
                    raise self._continuous_cache_exhaustion_error(max_cache_len)
                write_positions[:survivor_count].fill_(cache_position)
                if align_decode_fronts:
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
                decoded = self._continuous_next_tokens(
                    output.logits[:, -1, :],
                    survivor_ids,
                    request_by_id,
                    responses,
                    module,
                    select_next_tokens,
                    generation_config=prepared_generation_config,
                )
                stats["decode_steps"] += 1
                decoded_tokens = dict(zip(survivor_ids, decoded.split(1, dim=0)))
                for request_id in survivor_ids:
                    logical_positions[request_id] += 1

            next_tokens = decoded_tokens | admitted_tokens
            finished_ids = []
            for request_id in update.active_ids:
                token = next_tokens[request_id]
                responses[request_id].append(token)
                if self._is_eos(token):
                    finished_ids.append(request_id)

            if survivor_count:
                cache_position += 1
            stats["peak_cache_length"] = max(stats["peak_cache_length"], cache_position)
            scheduler_start = self._profile_start(profile)
            update = scheduler.advance(finished_ids)
            self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)

        stats["cache_memory_bytes"] = self._continuous_cache_memory_bytes(cache, attention_mask, write_positions)
        generation_end = self._profile_start(profile)
        post_processing_start = generation_end
        output = self._build_continuous_batch(original_request, responses)
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
            self._finish_continuous_profile(profile, original_request, responses, max_batch_size, prompt_len,
                                            generation_end, post_processing_end)
            self._last_profile["generation_strategy"] = ("bucketed" if self.adaptive_prefill else
                                                         "aligned" if align_decode_fronts else "equal_width")
        return output

    @torch.no_grad()
    def _generate_continuous_graph(self, original_request, requests, sampling, max_batch_size, *, prompt_lengths,
                                   generation_config, profile):
        """Run continuous decoding with a fixed-capacity graph.

        Scheduler slots stay attached to physical cache rows for their whole
        lifetime. Retired rows are masked until a new prompt is prefetched into
        them, so the captured decode always receives the same batch shape and
        static tensor addresses.
        """
        module = self.engine.module
        select_next_tokens = self._token_selector(sampling)
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

        if self.enable_profiling and profile is None:
            profile = self._start_continuous_profile()
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
        request_by_id = {}
        responses = {}
        for request_id, request in enumerate(requests):
            scheduler.submit(ContinuousBatchRequest(request_id))
            request_by_id[request_id] = request
            responses[request_id] = []

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
            "cache_capacity": max_cache_len,
            "peak_cache_length": prompt_len,
            "cache_memory_bytes": self._continuous_cache_memory_bytes(cache, static_attention, static_write_positions),
            "trim_count": 0,
            "trimmed_columns": 0,
            "trim_latency_ms": 0.0,
            "trim_bytes_moved": 0,
            "decode_steps": 0,
        }

        slot_by_request = {}
        write_positions = {}
        next_tokens = {}
        update = scheduler.schedule()
        while update.active:
            if profile is not None:
                profile["active_batch_sizes"].append(len(update.active))

            for request_id in update.retired:
                static_attention[slot_by_request[request_id]].zero_()
                slot_by_request.pop(request_id, None)
                write_positions.pop(request_id, None)
                next_tokens.pop(request_id, None)

            free_slots = [slot for slot in range(max_batch_size) if slot not in slot_by_request.values()]
            admitted_ids = tuple(request.request_id for request in update.admitted)
            admitted_slots = []
            for request_id in admitted_ids:
                slot = free_slots.pop(0)
                slot_by_request[request_id] = slot
                write_positions[request_id] = prompt_len
                admitted_slots.append(slot)

            if admitted_ids:
                # A full-width admission that fits the token limit is already the optimal single bucket.
                bucket_prefill = self.adaptive_prefill and (
                    any(prompt_lengths[i] != prompt_len for i in admitted_ids) or
                    (self.prefill_max_tokens is not None and len(admitted_ids) * prompt_len > self.prefill_max_tokens))
                admitted_tokens = self._continuous_prefill(
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
                    align_decode_fronts=bucket_prefill,
                    target_slots=tuple(admitted_slots),
                    generation_config=generation_config,
                )
                next_tokens.update(admitted_tokens)

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
                decoded = self._continuous_next_tokens(
                    decode_logits,
                    decode_ids,
                    request_by_id,
                    responses,
                    module,
                    select_next_tokens,
                    generation_config=generation_config,
                )
                for request_id, token in zip(decode_ids, decoded.split(1, dim=0)):
                    decoded_tokens[request_id] = token
                    write_positions[request_id] += 1
                stats["decode_steps"] += 1
                next_tokens.update(decoded_tokens)

            finished_ids = []
            for request_id in update.active_ids:
                token = next_tokens[request_id]
                responses[request_id].append(token)
                if self._is_eos(token):
                    finished_ids.append(request_id)

            scheduler_start = self._profile_start(profile)
            update = scheduler.advance(finished_ids)
            self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)
            if write_positions:
                stats["peak_cache_length"] = max(stats["peak_cache_length"], max(write_positions.values()))

        generation_end = self._profile_start(profile)
        output = self._build_continuous_batch(original_request, responses)
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
            self._finish_continuous_profile(profile, original_request, responses, max_batch_size, prompt_len,
                                            generation_end, post_processing_end)
            if self.adaptive_prefill:
                self._last_profile["generation_strategy"] = "bucketed_graph"
        return output

    def _validate_continuous_graph_options(self, prompt_len, sampling):
        if self.align_decode_fronts and not self.adaptive_prefill:
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
            # Capture sizes the layers from the model's local KV tensors.
            cache = cache_type(module.config, max_cache_len=max_cache_len)
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
            if not all(layer.is_initialized for layer in cache.layers):
                # Allocate uninitialized KV layers on the current stream, not the warmup stream.
                module(
                    static_input,
                    attention_mask=static_attention,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=static_cache_position,
                    position_ids=static_position_ids,
                )
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

    def _validate_continuous_inputs(self, request, sampling, max_batch_size):
        if request.prompt_ids.shape[0] == 0:
            raise ValueError("continuous batching requires at least one request")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.use_shared_prefill:
            raise ValueError("continuous batching does not support shared prompt prefill")

        if sampling.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.adaptive_prefill and sampling.temperature > 0:
            raise ValueError("adaptive prefill currently supports greedy decoding only")
        if sampling.n_samples_per_prompt != 1:
            raise ValueError("continuous batching currently supports one sample per prompt")
        # Rollout masks contain 0/1; validate and read all lengths in one device-to-host transfer.
        # Count like the boolean token selection; a low-precision sum would round long prompts.
        lengths = request.prompt_attention_mask.count_nonzero(dim=1).tolist()
        if any(length <= 0 for length in lengths):
            raise ValueError("continuous batching requires at least one prompt token per request")
        return lengths

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
                            target_slots=None,
                            generation_config=None):
        if not update.admitted:
            return {}

        if target_slots is None:
            target_slots = update.admitted_slots
        if len(target_slots) != len(update.admitted):
            raise ValueError("continuous prefill needs one target slot per admitted request")
        if not align_decode_fronts:
            admitted_ids = tuple(request.request_id for request in update.admitted)
            prompt_ids = torch.cat([request_by_id[request_id].prompt_ids for request_id in admitted_ids], dim=0)
            prompt_attention = torch.cat(
                [request_by_id[request_id].prompt_attention_mask for request_id in admitted_ids], dim=0)
            prompt_len = prompt_ids.shape[1]
            prefill_start = self._profile_start(profile)
            prefill_tokens, prefill_cache = self._continuous_prefill_cache(
                module,
                static_cache_type,
                prompt_ids,
                prompt_attention,
                model_dtype,
                device,
                sampling,
                generation_config=generation_config,
            )
            self._profile_end(profile, "prefill_forward_ms", prefill_start, count="num_prefill_forwards")
            cache_copy_start = self._profile_start(profile)
            cache_start = cache_position - prompt_len
            if cache_start < 0:
                raise ValueError("continuous batching prompt does not fit behind the decode front")
            for layer_idx in range(self._cache_layer_count(prefill_cache)):
                prefill_keys, prefill_values = self._cache_layer_tensors(prefill_cache, layer_idx)
                target_layer = self._continuous_cache_layer(cache, layer_idx, prefill_keys, prefill_values,
                                                            attention_mask.shape[0])
                for source_row, target_row in enumerate(target_slots):
                    target_layer.keys[target_row, :, cache_start:cache_position].copy_(prefill_keys[source_row])
                    target_layer.values[target_row, :, cache_start:cache_position].copy_(prefill_values[source_row])
            for source_row, target_row in enumerate(target_slots):
                attention_mask[target_row, cache_start:cache_position].copy_(prompt_attention[source_row])
                write_positions[target_row] = cache_position
            self._profile_end(profile, "cache_management_overhead_ms", cache_copy_start)
            return dict(zip(admitted_ids, prefill_tokens.split(1, dim=0)))

        admitted_tokens = {}
        admitted = list(zip(update.admitted, target_slots))
        if self.adaptive_prefill:
            lengths = [prompt_lengths[item.request_id] for item, _ in admitted]
            buckets, _ = plan_prefill_buckets(lengths, self._prefill_cost_function(module), self.prefill_max_tokens)
        else:
            buckets = tuple((row, ) for row in range(len(admitted)))
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        for bucket in buckets:
            width = max(prompt_lengths[admitted[row][0].request_id] for row in bucket)
            prompt_ids = torch.full((len(bucket), width), pad_token_id, dtype=torch.long, device=device)
            prompt_attention = torch.zeros_like(prompt_ids)
            prompt_requests = {}
            for source_row, row in enumerate(bucket):
                item, _ = admitted[row]
                request = request_by_id[item.request_id]
                length = prompt_lengths[item.request_id]
                prompt_ids[source_row, -length:] = request.prompt_ids[0, request.prompt_attention_mask[0].bool()]
                prompt_attention[source_row, -length:] = 1
                prompt_requests[source_row] = request
            prefill_start = self._profile_start(profile)
            prefill_tokens, prefill_cache = self._continuous_prefill_cache(module,
                                                                           static_cache_type,
                                                                           prompt_ids,
                                                                           prompt_attention,
                                                                           model_dtype,
                                                                           device,
                                                                           sampling,
                                                                           prompt_requests=prompt_requests,
                                                                           generation_config=generation_config)
            self._profile_end(profile, "prefill_forward_ms", prefill_start, count="num_prefill_forwards")
            cache_copy_start = self._profile_start(profile)
            for source_row, row in enumerate(bucket):
                item, target_row = admitted[row]
                length = prompt_lengths[item.request_id]
                cache_start = cache_position - length
                if cache_start < 0:
                    raise ValueError("continuous batching prompt does not fit behind the decode front")
                admitted_tokens[item.request_id] = prefill_tokens[source_row:source_row + 1]
                for layer_idx in range(self._cache_layer_count(prefill_cache)):
                    keys, values = self._cache_layer_tensors(prefill_cache, layer_idx)
                    target_layer = self._continuous_cache_layer(cache, layer_idx, keys, values,
                                                                attention_mask.shape[0])
                    target_layer.keys[target_row, :, cache_start:cache_position].copy_(keys[source_row, :, -length:])
                    target_layer.values[target_row, :, cache_start:cache_position].copy_(values[source_row, :,
                                                                                                -length:])
                attention_mask[target_row, cache_start:cache_position].fill_(1)
                write_positions[target_row] = cache_position
            self._profile_end(profile, "cache_management_overhead_ms", cache_copy_start)
        return admitted_tokens

    @staticmethod
    def _continuous_cache_memory_bytes(cache, attention_mask, write_positions):
        # Eager and graph decoding report the same quantity: the KV cache plus the per-row metadata that
        # tracks it. Graph-only staging buffers are not counted.
        return cache.memory_bytes() + attention_mask.nbytes + write_positions.nbytes

    @staticmethod
    def _continuous_cache_layer(cache, layer_idx, prefill_keys, prefill_values, max_batch_size):
        layer = cache.layers[layer_idx]
        if not layer.is_initialized:
            layer.lazy_initialization(prefill_keys, prefill_values, max_batch_size=max_batch_size)
        return layer

    def _prefill_cost_function(self, module):
        """Read model geometry once; DP candidates use only scalar arithmetic."""
        config = getattr(module.config, "text_config", module.config)
        layers = getattr(config, "num_hidden_layers", 1)
        heads = getattr(config, "num_attention_heads", 1)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = getattr(config, "hidden_size", heads) // heads
        kv_heads = getattr(config, "num_key_value_heads", None) or heads
        attention_cost = self.prefill_attention_cost_ms * 4 * layers * heads * head_dim / 1e9
        element_size = next(module.parameters()).element_size()
        kv_cost = self.prefill_kv_cost_ms * 2 * layers * kv_heads * head_dim * element_size / (1024 * 1024)

        def cost(count, width, include_kv=True):
            tokens = count * width
            return self.prefill_fixed_cost_ms + tokens * (self.prefill_token_cost_ms + attention_cost * width +
                                                          (kv_cost if include_kv else 0))

        return cost

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

    def _finish_continuous_profile(self, profile, request, responses, max_batch_size, prompt_len, generation_end,
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
            "batch_size": request.prompt_ids.shape[0],
            "num_samples_per_prompt": 1,
            "prompt_length": prompt_len,
            "response_length": max(response_lengths, default=0),
            "active_batch_size": max(profile["active_batch_sizes"], default=0),
            "continuous_batch_size": max_batch_size,
        }

    def _continuous_prefill_cache(self,
                                  module,
                                  static_cache_type,
                                  prompt_ids,
                                  prompt_attention,
                                  model_dtype,
                                  device,
                                  sampling,
                                  prompt_requests=None,
                                  generation_config=None):
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
                **self._hf_sampling_kwargs(sampling),
                **({
                    "output_logits": True
                } if self.adaptive_prefill else {}),
            )
            prefill_cache = prefill_output.past_key_values
            if not self.adaptive_prefill:
                return prefill_output.sequences[:, -1:], prefill_cache
            logits = prefill_output.logits[-1]
        else:
            prompt_len = prompt_ids.shape[1]
            prefill_cache = self._create_static_cache(static_cache_type, module.config, prompt_ids.shape[0],
                                                      prompt_len, device, model_dtype)
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
            logits = prefill_output.logits[:, -1, :]
        request_ids = range(prompt_ids.shape[0])
        if prompt_requests is None:
            prompt_requests = {
                row: RolloutRequest(prompt_ids[row:row + 1], prompt_attention[row:row + 1])
                for row in request_ids
            }
        # Trimming model input must not trim HF's repetition-penalty history.
        next_tokens = self._continuous_next_tokens(logits,
                                                   request_ids,
                                                   prompt_requests,
                                                   dict.fromkeys(request_ids, ()),
                                                   module,
                                                   self._token_selector(sampling),
                                                   generation_config=generation_config)
        return next_tokens, prefill_cache

    @staticmethod
    def _repetition_penalty(module, generation_config=None):
        if generation_config is None:
            generation_config = getattr(module, "generation_config", None)
        repetition_penalty = getattr(generation_config, "repetition_penalty", 1.0)
        return 1.0 if repetition_penalty is None else repetition_penalty

    @staticmethod
    def _apply_repetition_penalty(logits, input_ids, repetition_penalty):
        """Penalize the tokens of ``input_ids`` in last-position ``logits``, without changing ``logits``.

        generate() applies the penalty in float32, before the sampling warpers, so it shapes the sampled distribution.
        """
        if repetition_penalty == 1.0:
            return logits
        logits = logits.float()
        scores = logits.gather(1, input_ids)
        scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
        return logits.scatter(1, input_ids, scores)

    @classmethod
    def _continuous_next_tokens(cls,
                                logits,
                                request_ids,
                                request_by_id,
                                responses,
                                module,
                                select_next_tokens,
                                generation_config=None):
        repetition_penalty = cls._repetition_penalty(module, generation_config)
        if repetition_penalty != 1.0:
            logits = torch.cat([
                cls._apply_repetition_penalty(
                    logits[row:row + 1],
                    torch.cat((request_by_id[request_id].prompt_ids, *responses[request_id]), dim=1),
                    repetition_penalty,
                ) for row, request_id in enumerate(request_ids)
            ],
                               dim=0)
        return select_next_tokens(logits)

    @staticmethod
    def _hf_sampling_kwargs(sampling):
        """Hugging Face ``generate()`` arguments for ``sampling``; ``temperature <= 0`` is greedy."""
        if sampling.temperature <= 0.0:
            return {"do_sample": False, "temperature": 1.0, "top_p": 1.0}
        return {
            "do_sample": True,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": max(sampling.top_k, 0),
        }

    @staticmethod
    def _token_selector(sampling):
        """Return a function that picks the next tokens ``[B, 1]`` from last-position logits ``[B, V]``.

        Tokens are picked after each forward or graph replay, outside any
        captured graph. Sampling applies Hugging Face's temperature, top-k and
        top-p warpers in the order ``generate()`` applies them, so every path
        draws from the same distribution as ``generate()``.
        """
        if sampling.temperature <= 0.0:
            return lambda logits: logits.argmax(dim=-1, keepdim=True)

        from transformers.generation.logits_process import (TemperatureLogitsWarper, TopKLogitsWarper,
                                                            TopPLogitsWarper)

        warpers = []
        if sampling.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(sampling.temperature))
        if sampling.top_k > 0:
            warpers.append(TopKLogitsWarper(top_k=sampling.top_k))
        if sampling.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=sampling.top_p))

        def sample(logits):
            # generate() also warps and samples in float32.
            scores = logits.float()
            for warper in warpers:
                # The warpers read only the scores, not the generated ids.
                scores = warper(None, scores)
            return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)

        return sample

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

    def _build_continuous_batch(self, request, responses):
        response_ids = [torch.cat(responses[index], dim=1) for index in range(request.prompt_ids.shape[0])]
        max_response_len = max(response.shape[1] for response in response_ids)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("The tokenizer must define pad_token_id or eos_token_id")
        input_rows = []
        attention_rows = []
        for index, response in enumerate(response_ids):
            padding = max_response_len - response.shape[1]
            if padding:
                response_padding = torch.full((1, padding), pad_token_id, dtype=response.dtype, device=response.device)
                attention_padding = torch.zeros((1, padding),
                                                dtype=request.prompt_attention_mask.dtype,
                                                device=response.device)
                response = torch.cat((response, response_padding), dim=1)
            else:
                attention_padding = torch.empty((1, 0),
                                                dtype=request.prompt_attention_mask.dtype,
                                                device=response.device)
            input_rows.append(torch.cat((request.prompt_ids[index:index + 1], response), dim=1))
            response_attention = torch.ones((1, response_ids[index].shape[1]),
                                            dtype=request.prompt_attention_mask.dtype,
                                            device=response.device)
            attention_rows.append(
                torch.cat((request.prompt_attention_mask[index:index + 1], response_attention, attention_padding),
                          dim=1))

        input_ids = torch.cat(input_rows, dim=0)
        attention_mask = torch.cat(attention_rows, dim=0)
        response_start = request.prompt_ids.shape[1]
        return RolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_start_idx=torch.full((request.prompt_ids.shape[0], ),
                                          response_start,
                                          dtype=torch.long,
                                          device=input_ids.device),
        )

    def _zero3_work_remaining(self):
        """Return a rendezvous over the group used for ZeRO-3 parameter gathers."""
        is_zero3 = getattr(self.engine, "zero_optimization_partition_weights", None)
        if not callable(is_zero3) or not is_zero3():
            return None

        process_group = None
        if dist.is_initialized():
            optimizer = getattr(self.engine, "optimizer", None)
            process_group = getattr(optimizer, "dp_process_group", None)
            if process_group is None:
                raise RuntimeError("ZeRO-3 synchronized rollout requires optimizer.dp_process_group")
        return GlobalWorkRemaining(process_group)

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
    def _pad_after_eos(output_ids, response_start, eos_token_id, pad_token_id, mask_padding=True):
        """Retain the first response EOS and pad every subsequent position."""
        response_ids = output_ids[:, response_start:]
        response_attn = (response_ids != pad_token_id) if mask_padding else torch.ones_like(response_ids,
                                                                                            dtype=torch.bool)

        if eos_token_id is None or response_ids.shape[1] == 0:
            return output_ids, response_attn.long()

        eos_ids = torch.as_tensor(eos_token_id, device=response_ids.device, dtype=response_ids.dtype).flatten()
        is_eos = (response_ids.unsqueeze(-1) == eos_ids).any(dim=-1)
        has_eos = is_eos.any(dim=-1)
        first_eos_idx = is_eos.long().argmax(dim=-1)
        positions = torch.arange(response_ids.shape[1], device=response_ids.device).unsqueeze(0)
        after_first_eos = has_eos.unsqueeze(1) & (positions > first_eos_idx.unsqueeze(1))
        first_eos = has_eos.unsqueeze(1) & (positions == first_eos_idx.unsqueeze(1))

        output_ids = output_ids.clone()
        output_ids[:, response_start:].masked_fill_(after_first_eos, pad_token_id)
        # EOS is a valid generated token even when pad_token_id == eos_token_id.
        response_attn = (response_attn | first_eos) & ~after_first_eos
        return output_ids, response_attn.long()

    # ------------------------------------------------------------------
    # Graph capture decode loop
    # ------------------------------------------------------------------

    def _generate_graph(self, prompt_ids, prompt_attn, max_new_tokens, pad_token_id, module, device, sampling):
        """Decode with DeepSpeedStaticCache + graph capture, greedy or sampled per ``sampling``."""
        from transformers import StaticCache
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        if getattr(module.config, "_attn_implementation", None) == "flash_attention_2":
            raise ValueError("graph capture does not support flash_attention_2")

        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        max_len = prompt_len + max_new_tokens
        eos_token_id = self.tokenizer.eos_token_id
        model_dtype = next(module.parameters()).dtype
        select_next_tokens = self._token_selector(sampling)
        repetition_penalty = self._repetition_penalty(module)
        output_ids = [prompt_ids]

        def next_tokens(logits):
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, torch.cat(output_ids, dim=1), repetition_penalty)
            return select_next_tokens(logits)

        # --- Prefill with HF StaticCache (correct attention semantics) ---
        prefill_cache = self._create_static_cache(StaticCache, module.config, batch_size, max_len, device, model_dtype)
        prefill_attn = torch.ones(batch_size, prompt_len, dtype=torch.long, device=device)
        prefill_attn[:, :prompt_len] = prompt_attn
        prefill_out = module(
            prompt_ids,
            attention_mask=prefill_attn,
            past_key_values=prefill_cache,
            use_cache=True,
            cache_position=torch.arange(prompt_len, device=device),
            **self._last_logits_kwargs(module),
        )
        next_token = next_tokens(prefill_out.logits[:, -1, :])
        output_ids.append(next_token)

        # --- Copy prefill KV into DeepSpeedStaticCache ---
        write_pos = torch.tensor(prompt_len - 1, dtype=torch.long, device=device)
        ds_cache = DeepSpeedStaticCache(module.config, max_cache_len=max_len)
        ds_cache.set_write_position(write_pos)
        # Infer each layer's local KV shape from prefill before graph capture.
        for layer_idx in range(self._cache_layer_count(ds_cache)):
            ds_layer = ds_cache.layers[layer_idx]
            hf_keys, hf_values = self._cache_layer_tensors(prefill_cache, layer_idx)
            if not ds_layer.is_initialized:
                ds_layer.lazy_initialization(hf_keys, hf_values)
            ds_layer.keys[:, :, :prompt_len, :].copy_(hf_keys[:, :, :prompt_len, :])
            ds_layer.values[:, :, :prompt_len, :].copy_(hf_values[:, :, :prompt_len, :])

        # --- Static buffers for graph capture ---
        static_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        static_attn = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        static_attn[:, :prompt_len] = prompt_attn
        static_attn[:, prompt_len] = 1  # first decode position
        # Transformers builds its SDPA mask from a 2-D attention_mask with host-side checks
        # (``padding_mask.all()`` in ``_ignore_causal_mask_sdpa``). Those synchronize the stream and
        # invalidate CUDA graph capture, so the captured forward receives a pre-built 4-D additive
        # mask instead, as the continuous graph path does; each decode step unmasks one more column.
        static_causal_attn = torch.zeros((batch_size, 1, 1, max_len), dtype=model_dtype, device=device)
        self._update_static_causal_attention(static_attn, static_causal_attn)
        static_pos = torch.tensor(prompt_len, dtype=torch.long, device=device)
        static_cache_pos = static_pos.unsqueeze(0)  # [1] for cache_position
        static_pos_ids = static_pos.reshape(1, 1).expand(batch_size, 1)  # [batch, 1]

        write_pos.fill_(prompt_len)

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
        finally:
            module._forward_pre_hooks.update(saved_pre)
            module._forward_hooks.update(saved_post)

        # --- Decode loop ---
        eos_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for step in range(max_new_tokens - 1):
            if eos_mask.all():
                output_ids.append(torch.full((batch_size, 1), pad_token_id, dtype=torch.long, device=device))
                continue

            # Update static inputs
            static_token.copy_(next_token)
            pos = prompt_len + step
            write_pos.fill_(pos)
            static_cache_pos.fill_(pos)
            static_pos_ids.fill_(pos)
            static_attn[:, pos] = 1
            self._update_static_causal_attention(static_attn, static_causal_attn)

            # Replay
            get_accelerator().replay_graph(graph)
            next_token = next_tokens(static_logits[:, -1, :])
            output_ids.append(next_token)
            eos_mask |= (next_token.squeeze(1) == eos_token_id)

        return torch.cat(output_ids, dim=1)

    def sync_weights(self, step: int) -> None:  # noqa: ARG002
        """No-op: hybrid engine reads model weights live."""
        return None

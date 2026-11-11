# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engine backed by DeepSpeed's hybrid engine.

Every call runs one driver, and single-turn generation is a rollout of one turn without tool calls. Turns run
in static batches by default, or share a fixed number of decode rows with ``continuous_batch_size``, with or
without graph capture.
"""

import asyncio
import time
from copy import copy
from dataclasses import dataclass, field
from inspect import signature
from typing import Optional

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger
from deepspeed.runtime.rollout.base import (RolloutBatch, RolloutConfig, RolloutEngine, RolloutRecord, RolloutRequest,
                                            SamplingConfig)
from deepspeed.runtime.rollout.continuous_batching import _ContinuousRunGraph, _ContinuousRun
from deepspeed.runtime.rollout.tool import TOOL_CALL_PARSERS, ToolSet
from deepspeed.runtime.rollout.utils import (
    expand_samples,
    get_eos_token_ids,
    get_pad_token_id,
    get_repetition_penalty,
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


class _TokenSelector:
    """Picks next tokens from logits the way ``generate()`` does: repetition penalty, then temperature and top-k or
    top-p, returning each token with its log-prob, or None when log-probs are not wanted.

    With a repetition penalty it keeps a mask of the tokens each row has seen, so a caller starts each row's history
    from its prompt and the selector adds every token it picks. Rows are cache rows; ``rows=None`` means the leading
    ``logits.shape[0]`` rows.
    """

    def __init__(self, sampling, repetition_penalty=1.0, num_rows=None, return_log_probs=True):
        self.temperature = sampling.temperature
        self.return_log_probs = return_log_probs
        self.repetition_penalty = repetition_penalty
        self.num_rows = num_rows
        self.seen = None
        self.warpers = []
        if self.temperature > 0.0 and (sampling.top_k > 0 or sampling.top_p < 1.0):
            from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper
            if sampling.top_k > 0:
                self.warpers.append(TopKLogitsWarper(top_k=sampling.top_k))
            if sampling.top_p < 1.0:
                self.warpers.append(TopPLogitsWarper(top_p=sampling.top_p))

    def start(self, prompt_ids, prompt_mask=None, rows=None, vocab_size=None):
        """Reset the history of ``rows`` to ``prompt_ids``, leaving out positions where ``prompt_mask`` is 0."""
        if self.repetition_penalty == 1.0:
            return
        if self.seen is None:
            self.seen = torch.zeros(self.num_rows or prompt_ids.shape[0],
                                    vocab_size,
                                    dtype=torch.bool,
                                    device=prompt_ids.device)
        vocab_size = self.seen.shape[1]
        # Masked positions write to a spare column that is then dropped.
        if prompt_mask is not None:
            prompt_ids = prompt_ids.masked_fill(prompt_mask == 0, vocab_size)
        seen = torch.zeros(prompt_ids.shape[0], vocab_size + 1, dtype=torch.bool, device=prompt_ids.device)
        seen = seen.scatter_(1, prompt_ids, True)[:, :vocab_size]
        if rows is None:
            self.seen[:prompt_ids.shape[0]] = seen
        else:
            self.seen[rows] = seen

    def first(self, logits, prompt_ids, prompt_mask=None, rows=None):
        """Start the history of ``rows`` from their prompts, then pick their first tokens."""
        self.start(prompt_ids, prompt_mask, rows, vocab_size=logits.shape[-1])
        return self(logits, rows)

    def record(self, tokens, rows=None):
        """Add tokens picked elsewhere to the history of ``rows``."""
        if self.seen is None:
            return
        if rows is None:
            self.seen[:tokens.shape[0]].scatter_(1, tokens, True)
        else:
            self.seen[rows] = self.seen[rows].scatter(1, tokens, True)

    def keep(self, keep_rows):
        """Move the histories of ``keep_rows`` to the leading rows, as a batch compaction does."""
        if self.seen is not None and keep_rows.numel():
            self.seen[:keep_rows.numel()] = self.seen[keep_rows]

    def __call__(self, logits, rows=None):
        if self.seen is not None:
            seen = self.seen[:logits.shape[0]] if rows is None else self.seen[rows]
            # generate() penalizes in float32, before any warper.
            logits = logits.float()
            penalty = self.repetition_penalty
            logits = torch.where(seen, torch.where(logits < 0, logits * penalty, logits / penalty), logits)
        if self.temperature <= 0.0:
            tokens = logits.argmax(dim=-1, keepdim=True)
            log_probs = _chosen_log_probs(logits, tokens) if self.return_log_probs else None
        else:
            scores = logits.float() / self.temperature
            for warper in self.warpers:
                # The warpers read only the scores, not the generated ids.
                scores = warper(None, scores)
            tokens = torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)
            log_probs = torch.log_softmax(scores, dim=-1).gather(1, tokens) if self.return_log_probs else None
        self.record(tokens, rows)
        return tokens, log_probs


@dataclass
class HybridEngineRolloutConfig(RolloutConfig):
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
    chat_template_kwargs: dict = field(default_factory=dict)
    # The model's tool-call format, a key of TOOL_CALL_PARSERS such as "hermes" (Qwen2.5, Qwen3) or
    # "qwen3_coder" (Qwen3-Coder, Qwen3.5).
    tool_call_parser: str = "hermes"
    continuous_batch_size: Optional[int] = None
    return_log_probs: bool = False

    def __post_init__(self) -> None:
        if self.multi_turn and (not isinstance(self.max_turns, int) or self.max_turns < 1):
            raise ValueError("max_turns must be a positive integer when multi_turn is enabled")
        if self.continuous_batch_size is not None and self.continuous_batch_size < 1:
            raise ValueError("continuous_batch_size must be positive")
        if self.tool_call_parser not in TOOL_CALL_PARSERS:
            raise ValueError(f"unknown tool_call_parser {self.tool_call_parser!r}; "
                             f"choose from {', '.join(TOOL_CALL_PARSERS)}")


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
        self.partitions_weights = engine.zero_optimization_partition_weights()
        self.autotp_size = engine.autotp_size()
        self.inference_tp_size = engine._config.hybrid_engine.inference_tp_size
        # The rollout runs every trajectory to completion, so it needs an explicit cap on total length rather than
        # the hybrid engine's default.
        if "max_out_tokens" not in engine._config.hybrid_engine.model_fields_set:
            raise ValueError("HybridEngineRollout caps each trajectory's prompt, responses and tool output at "
                             "hybrid_engine.max_out_tokens; set it in the DeepSpeed config")
        self.max_out_tokens = engine._config.hybrid_engine.max_out_tokens
        self.tools = tools if tools is not None else ToolSet([])
        if cfg is None:
            cfg = HybridEngineRolloutConfig()
        # Any other config would leave every setting at its default without a word.
        if not isinstance(cfg, HybridEngineRolloutConfig):
            raise TypeError(f"HybridEngineRollout needs a HybridEngineRolloutConfig, got {type(cfg).__name__}")
        self.use_graph_capture = cfg.use_graph_capture
        self.enable_profiling = cfg.enable_profiling
        self.use_shared_prefill = cfg.use_shared_prefill
        self.align_decode_fronts = cfg.align_decode_fronts
        self.enable_cache_trimming = cfg.enable_cache_trimming
        self.continuous_cache_capacity = cfg.continuous_cache_capacity
        self.continuous_batch_size = cfg.continuous_batch_size
        self.return_log_probs = cfg.return_log_probs
        self.multi_turn = cfg.multi_turn
        self.max_turns = cfg.max_turns
        self.chat_template_kwargs = cfg.chat_template_kwargs
        self.parse_tool_calls = TOOL_CALL_PARSERS[cfg.tool_call_parser]
        self._last_profile = None
        self._last_continuous_stats = None
        if self.use_shared_prefill:
            if self.continuous_batch_size is not None:
                raise ValueError("continuous batching does not support shared prompt prefill")
            if self.use_graph_capture:
                raise ValueError("shared prefill does not support graph capture")
            if not getattr(engine, "_inference_containers", None):
                raise ValueError("shared prefill needs a model with a DeepSpeed inference policy")

    @torch.no_grad()
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        samples = request.samples
        if not samples:
            return RolloutBatch(samples=[])
        self._check_sampling(sampling)
        self._check_parallelism()
        self._check_attention()
        # Once per call is enough: the ranks then run the same forwards and samples, and take tool output from one
        # rank, so their random states stay equal across turns.
        self._sync_rng_across_tp()
        static_batching = self.continuous_batch_size is None
        call_start = None
        if static_batching and self.enable_profiling:
            get_accelerator().synchronize()
            call_start = time.perf_counter()
        samples = render_prompts(samples, self.tokenizer, self.tools.specs, self.chat_template_kwargs)
        trajectories = expand_samples(samples, sampling.n_samples_per_prompt)
        if static_batching:
            self._generate(trajectories, sampling, len(samples), call_start)
        elif self._graph_capture_enabled():
            self._generate_continuous_graph(trajectories, sampling, len(samples))
        else:
            self._generate_continuous(trajectories, sampling, len(samples))
        return RolloutBatch(samples=trajectories)

    def _check_parallelism(self):
        if self.partitions_weights:
            if self.multi_turn:
                raise ValueError("multi-turn rollout does not support ZeRO-3")
            if self.use_graph_capture:
                raise ValueError("graph capture does not support ZeRO-3")
        # The rollout calls the module directly, so it skips the input gather of the hybrid engine's generate().
        if self.inference_tp_size > 1:
            raise ValueError("HybridEngineRollout does not support hybrid engine inference tensor parallelism")
        # Continuous batching admits a turn when its tools return, which happens at different steps on each rank, so
        # tensor-parallel ranks would run different forwards.
        if self.autotp_size > 1 and self.continuous_batch_size is not None:
            raise ValueError("continuous batching does not support AutoTP")

    def _check_attention(self):
        # Graph decode and continuous batching pass their own 4-D attention masks, which flash_attention_2 cannot take.
        if getattr(getattr(self.engine.module, "config", None), "_attn_implementation", None) != "flash_attention_2":
            return
        if self.continuous_batch_size is not None:
            raise ValueError("continuous batching does not support flash_attention_2")
        if self._graph_capture_enabled():
            raise ValueError("graph capture does not support flash_attention_2")

    def _sync_rng_across_tp(self):
        """Share the random state across each AutoTP group, so its ranks sample the same tokens."""
        if self.autotp_size <= 1:
            return
        import deepspeed.comm as dist
        from deepspeed.utils import groups

        accelerator = get_accelerator()
        state = accelerator.get_rng_state().to(accelerator.current_device_name())
        dist.broadcast(state,
                       groups.get_tensor_model_parallel_src_rank(),
                       group=groups.get_tensor_model_parallel_group())
        accelerator.set_rng_state(state.cpu())

    def _run_tools(self, calls_by_row):
        if self.autotp_size <= 1:
            return asyncio.run(self._run_tool_calls(calls_by_row))

        # Run a turn's tool calls once per AutoTP group, so every rank gets the same output.
        import deepspeed.comm as dist
        from deepspeed.utils import groups

        src = groups.get_tensor_model_parallel_src_rank()
        outputs = [asyncio.run(self._run_tool_calls(calls_by_row)) if dist.get_rank() == src else None]
        dist.broadcast_object_list(outputs, src, group=groups.get_tensor_model_parallel_group())
        return outputs[0]

    def _max_across_ranks(self, value):
        """The largest ``value`` over the ranks that gather ZeRO-3 parameters together."""
        import deepspeed.comm as dist

        tensor = torch.tensor([value], device=get_accelerator().current_device_name())
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self.engine.seq_data_parallel_group)
        return int(tensor.item())

    def _turn_budget(self, sample, generated_length, sampling):
        budget = sampling.max_new_tokens - generated_length
        if sampling.max_new_tokens_per_turn is not None:
            budget = min(budget, sampling.max_new_tokens_per_turn)
        return min(budget, self.max_out_tokens - sum(sample.attention_mask))

    def _end_turn(self, sample, response, turn, generated, max_new_tokens):
        if not response or response[-1] not in get_eos_token_ids(self.tokenizer, self.engine.module):
            sample.status = RolloutRecord.Status.TRUNCATED
            return None
        if not self.multi_turn:
            sample.status = RolloutRecord.Status.COMPLETED
            return None
        calls = self.parse_tool_calls(self.tokenizer.decode(response), self.tools)
        if not calls:
            sample.status = RolloutRecord.Status.COMPLETED
            return None
        if generated >= max_new_tokens or turn == self.max_turns - 1:
            sample.status = RolloutRecord.Status.TRUNCATED
            return None
        return calls

    def _tool_output_fits(self, sample, response, tool_response):
        """Whether a turn's response and tool output stay within ``max_out_tokens``."""
        return sum(sample.attention_mask) + len(response) + len(tool_response) <= self.max_out_tokens

    def _tool_response(self, calls, outputs):
        tool_messages = [{
            "role": "tool",
            "name": call.name or "unknown",
            "content": output
        } for call, output in zip(calls, outputs)]
        return tokenize_tool_responses(tool_messages, self.tokenizer,
                                       get_eos_token_ids(self.tokenizer, self.engine.module),
                                       self.chat_template_kwargs)

    @torch.no_grad()
    def _generate(self, trajectories, sampling, num_prompts, call_start=None):
        """Run turns in static batches."""
        device = get_accelerator().current_device_name()
        n = sampling.n_samples_per_prompt
        max_new_tokens = sampling.max_new_tokens
        return_log_probs = self.return_log_probs
        pad_token_id = get_pad_token_id(self.tokenizer)
        module = self.engine.module
        max_prompt_length = max(len(sample.tokens) for sample in trajectories)

        if self.enable_profiling:
            accelerator = get_accelerator()
            _curr_time = call_start
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

        generated_length = [0] * len(trajectories)
        active = list(range(len(trajectories)))

        for turn in range(self.max_turns if self.multi_turn else 1):
            budgets = {}
            for i in active:
                budgets[i] = self._turn_budget(trajectories[i], generated_length[i], sampling)
                if budgets[i] <= 0:
                    trajectories[i].status = RolloutRecord.Status.TRUNCATED
            active = [i for i in active if budgets[i] > 0]
            turn_max_new_tokens = max((budgets[i] for i in active), default=0)
            if self.partitions_weights:
                # ZeRO-3 gathers parameters in every forward, so all ranks run as many as the longest budget needs.
                turn_max_new_tokens = self._max_across_ranks(turn_max_new_tokens)
                if not active and turn_max_new_tokens > 0:
                    dummy = torch.full((1, 1), pad_token_id, dtype=torch.long, device=device)
                    self._generate_tokens(dummy, torch.ones_like(dummy), turn_max_new_tokens, sampling, pad_token_id)
            if not active:
                break
            turn_samples = [trajectories[i] for i in active]
            row_budgets = [budgets[i] for i in active]

            if self.enable_profiling:
                turn_start = _curr_time

            prompt_ids, prompt_attn_mask = left_pad_samples(turn_samples, pad_token_id)
            prompt_len = prompt_ids.shape[1]
            prompt_ids = prompt_ids.to(device)
            prompt_attn_mask = prompt_attn_mask.to(device)

            if self.enable_profiling:
                accelerator.synchronize()
                expansion_end = time.perf_counter()

            forward_profiler = None
            handles = []
            if self.enable_profiling and not self._graph_capture_enabled():
                forward_profiler = _ForwardProfiler(accelerator)
                handles.extend(forward_profiler.register(module))

            try:
                if self.use_shared_prefill and n > 1 and turn == 0:
                    num_groups = len(active) // n
                    self.engine.prepare_shared_prefill(num_groups, n, prompt_len)
                    handles.extend(self._register_shared_prefill_hooks(module, num_groups, n))
                output_ids, log_probs = self._generate_tokens(prompt_ids, prompt_attn_mask, turn_max_new_tokens,
                                                              sampling, pad_token_id)
            finally:
                for handle in handles:
                    handle.remove()

            if self.enable_profiling:
                accelerator.synchronize()
                generation_end = time.perf_counter()

            # Under ZeRO-3 generation runs past EOS, so ranks run the same number of parameter gathers; otherwise
            # rows that end early keep decoding until every row has. Either way, each response keeps its first EOS
            # and every later token becomes padding.
            output_ids, response_attn = self._pad_after_eos(
                output_ids,
                response_start=prompt_len,
                eos_token_id=get_eos_token_ids(self.tokenizer, self.engine.module),
                pad_token_id=pad_token_id,
            )
            responses = masked_lists(output_ids[:, prompt_len:], response_attn)
            response_log_probs = masked_lists(log_probs, response_attn) if return_log_probs else None

            turn_responses = []
            turn_log_probs = []
            turn_calls = {}
            for row, i in enumerate(active):
                response = responses[row][:row_budgets[row]]
                turn_responses.append(response)
                turn_log_probs.append(response_log_probs[row][:row_budgets[row]] if return_log_probs else None)
                generated_length[i] += len(response)
                calls = self._end_turn(trajectories[i], response, turn, generated_length[i], max_new_tokens)
                if calls is not None:
                    turn_calls[row] = calls

            tool_outputs = {}
            tool_ms = 0.0
            if turn_calls:
                tool_start = time.perf_counter()
                tool_outputs = self._run_tools(turn_calls)
                tool_ms = (time.perf_counter() - tool_start) * 1000.0

            tool_active = []
            for row, i in enumerate(active):
                tool_response = []
                if row in turn_calls:
                    tool_response = self._tool_response(turn_calls[row], tool_outputs[row])
                    if self._tool_output_fits(trajectories[i], turn_responses[row], tool_response):
                        tool_active.append(i)
                    else:
                        trajectories[i].status = RolloutRecord.Status.TRUNCATED
                        tool_response = []
                update_sample_from_response(trajectories[i], turn_responses[row], turn_log_probs[row], tool_response,
                                            self.tokenizer)
            active = tool_active

            if self.enable_profiling:
                accelerator.synchronize()
                turn_end = time.perf_counter()
                _curr_time = turn_end
                profile["prompt_expansion_ms"] += (expansion_end - turn_start) * 1000.0
                profile["generation_ms"] += (generation_end - expansion_end) * 1000.0
                profile["tool_ms"] += tool_ms
                profile["post_processing_ms"] += (turn_end - generation_end) * 1000.0 - tool_ms
                profile["num_generated_tokens"] += int(output_ids.shape[0] * (output_ids.shape[1] - prompt_len))
                profile["num_turns"] += 1
                prefill_ms, decode_ms, num_decodes = (None, None, 0)
                if forward_profiler is not None:
                    prefill_ms, decode_ms, num_decodes = forward_profiler.get_stage_times()
                if prefill_ms is None or profile["prefill_forward_ms"] is None:
                    profile["prefill_forward_ms"] = None
                    profile["decode_forward_ms"] = None
                else:
                    profile["prefill_forward_ms"] += prefill_ms
                    profile["decode_forward_ms"] += decode_ms
                profile["num_decode_forwards"] += num_decodes

        if self.enable_profiling:
            total_ms = (_curr_time - call_start) * 1000.0
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
                "batch_size": num_prompts,
                "num_samples_per_prompt": n,
                "max_prompt_length": max_prompt_length,
                "response_length": max(sample.response_length for sample in trajectories),
            })
            self._last_profile = profile

    @torch.no_grad()
    def _generate_continuous_graph(self, trajectories, sampling, num_prompts):
        """Continuous batching on a captured graph."""
        module = self.engine.module
        self._validate_continuous(module)
        if self.align_decode_fronts:
            raise ValueError("continuous graph capture does not support aligned decode fronts")
        if self.enable_cache_trimming:
            raise ValueError("continuous graph capture does not support cache trimming")
        turn_cap = sampling.max_new_tokens
        if sampling.max_new_tokens_per_turn is not None:
            turn_cap = min(turn_cap, sampling.max_new_tokens_per_turn)
        longest_prompt = max(sum(sample.attention_mask) for sample in trajectories)
        # A single turn needs no more than its prompt and budget; a multi-turn row may grow to the cap.
        needed = self.max_out_tokens if self.multi_turn else min(self.max_out_tokens, longest_prompt + turn_cap)
        capacity = self.continuous_cache_capacity
        if capacity is None:
            capacity = needed
        elif not self.multi_turn and capacity < needed:
            raise ValueError("continuous_cache_capacity must fit the prompt and all generated tokens")
        max_positions = getattr(getattr(module, "config", None), "max_position_embeddings", None)
        if isinstance(max_positions, int) and capacity > max_positions:
            raise ValueError("continuous batching cache exceeds the model maximum position embeddings")

        selector = _TokenSelector(sampling, get_repetition_penalty(sampling, module), self.continuous_batch_size,
                                  self.return_log_probs)
        _ContinuousRunGraph(self, trajectories, sampling, num_prompts, capacity, selector).run()

    @staticmethod
    def _sync_time(accelerator):
        accelerator.synchronize()
        return time.perf_counter()

    def _validate_continuous(self, module):
        if self.partitions_weights:
            raise ValueError("continuous batching does not support ZeRO-3")
        if getattr(module, "_supports_cache_class", None) is False:
            raise ValueError("continuous batching requires a model with cache-class support; use the default "
                             "generate() path or upgrade transformers")

    def _check_sampling(self, sampling: SamplingConfig) -> None:
        return_log_probs = self.return_log_probs
        if return_log_probs and sampling.temperature > 0.0 and (sampling.top_p < 1.0 or sampling.top_k > 0):
            raise ValueError("return_log_probs requires top_p=1 and top_k<=0 when sampling")
        repetition_penalty = get_repetition_penalty(sampling, self.engine.module)
        if return_log_probs and repetition_penalty != 1.0:
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

    def _generate_tokens(self, prompt_ids, prompt_attn_mask, max_new_tokens, sampling, pad_token_id):
        module = self.engine.module
        if self._graph_capture_enabled():
            return self._generate_graph(prompt_ids, prompt_attn_mask, max_new_tokens, sampling, module,
                                        prompt_ids.device)

        # ZeRO-3 gathers parameters during each decode forward, so every
        # data-parallel rank must run the same number of decode steps and
        # generation cannot stop at EOS. Other stages stop once every row ends.
        stop_at_eos = not self.engine.zero_optimization_partition_weights()
        generate_kwargs = {
            "attention_mask": prompt_attn_mask,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": (get_eos_token_ids(self.tokenizer, self.engine.module) or None) if stop_at_eos else None,
            "pad_token_id": pad_token_id,
            **self._hf_sampling_kwargs(sampling),
        }
        finish_log_probs = None
        if self.return_log_probs:
            # output_scores / output_logits would keep a [batch, vocab] tensor per step until generate() returns;
            # the recorder keeps only each sampled token's log-prob.
            from transformers import LogitsProcessorList
            processor, finish_log_probs = _sampled_log_prob_recorder(generate_kwargs["temperature"])
            generate_kwargs["logits_processor"] = LogitsProcessorList([processor])
        output_ids = module.generate(prompt_ids, **generate_kwargs)
        log_probs = finish_log_probs(output_ids[:, -1:]) if finish_log_probs is not None else None
        return output_ids, log_probs

    async def _run_tool_calls(self, calls_by_row):
        rows = list(calls_by_row)
        outputs = await asyncio.gather(*(asyncio.gather(*(self.tools.run(call) for call in calls_by_row[row]))
                                         for row in rows))
        return dict(zip(rows, outputs))

    @torch.no_grad()
    def _generate_continuous(self, trajectories, sampling, num_prompts):
        """Continuous batching without graph capture, on one write column shared by all rows."""
        module = self.engine.module
        max_new_tokens = sampling.max_new_tokens
        turn_cap = max_new_tokens
        if sampling.max_new_tokens_per_turn is not None:
            turn_cap = min(turn_cap, sampling.max_new_tokens_per_turn)
        if self.align_decode_fronts:
            prompt_len = max(sum(sample.attention_mask) for sample in trajectories)
        else:
            prompt_len = max(len(sample.tokens) for sample in trajectories)
        max_cache_len = self._shared_column_width(module, trajectories, prompt_len, turn_cap)
        self._validate_continuous(module)
        selector = _TokenSelector(sampling, get_repetition_penalty(sampling, module), self.continuous_batch_size,
                                  self.return_log_probs)
        _ContinuousRun(self, trajectories, sampling, num_prompts, prompt_len, max_cache_len, selector).run()

    def _shared_column_width(self, module, trajectories, prompt_len, turn_cap):
        """The number of cache columns for the shared write column.

        Without ``continuous_cache_capacity``, it is the longest context a turn can be admitted with plus one turn's
        budget, which trimming at full always leaves room for.
        """
        max_positions = getattr(getattr(module, "config", None), "max_position_embeddings", None)
        if not isinstance(max_positions, int):
            max_positions = None
        if self.multi_turn:
            if max_positions is not None and self.max_out_tokens > max_positions:
                raise ValueError("hybrid_engine.max_out_tokens exceeds the model maximum position embeddings")
            longest_context = self.max_out_tokens
        else:
            if max_positions is not None:
                if any(sum(sample.attention_mask) + turn_cap > max_positions for sample in trajectories):
                    raise ValueError("continuous batching request exceeds the model maximum position embeddings")
            longest_context = prompt_len
        if self.continuous_cache_capacity is not None:
            if self.continuous_cache_capacity < prompt_len:
                raise ValueError("continuous_cache_capacity must be at least the padded prompt length")
            return self.continuous_cache_capacity
        return longest_context + turn_cap

    @staticmethod
    def _update_static_causal_attention(static_atten_mask, static_causal_atten_mask):
        static_causal_atten_mask.copy_(static_atten_mask[:, None, None, :])
        static_causal_atten_mask.neg_().add_(1).mul_(torch.finfo(static_causal_atten_mask.dtype).min)

    @staticmethod
    def _continuous_cache_exhaustion_error(max_cache_len):
        return ValueError(
            f"continuous batching cache capacity ({max_cache_len}) exhausted before active requests retired; "
            "increase HybridEngineRolloutConfig.continuous_cache_capacity or enable cache trimming with "
            "HybridEngineRolloutConfig(enable_cache_trimming=True)")

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
                            selector,
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
            rows = [request_by_id[request_id] for request_id in admitted_ids]
            # Returning turns have longer contexts than new ones; left-pad the admission to its widest.
            prompt_len = max(row.prompt_ids.shape[1] for row in rows)
            prompt_ids = torch.cat([
                torch.nn.functional.pad(row.prompt_ids, (prompt_len - row.prompt_ids.shape[1], 0),
                                        value=get_pad_token_id(self.tokenizer)) for row in rows
            ],
                                   dim=0)
            prompt_attention = torch.cat([
                torch.nn.functional.pad(row.prompt_attention_mask, (prompt_len - row.prompt_ids.shape[1], 0))
                for row in rows
            ],
                                         dim=0)
            prefill_start = self._profile_start(profile)
            prefill_tokens, prefill_log_probs, prefill_cache = self._continuous_prefill_cache(
                module,
                static_cache_type,
                prompt_ids,
                prompt_attention,
                model_dtype,
                device,
                sampling,
                selector,
                torch.tensor(target_slots, dtype=torch.long, device=device),
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
            admitted_log_probs = {}
            if prefill_log_probs is not None:
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
                selector,
                torch.tensor([target_row], dtype=torch.long, device=device),
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

    def _finish_continuous_profile(self, profile, batch_size, num_samples_per_prompt, generated, response_lengths,
                                   num_turns, tool_ms, max_batch_size, prompt_len, generation_end,
                                   post_processing_end):
        total_ms = (post_processing_end - profile["start"]) * 1000.0
        generation_ms = (generation_end - profile["start"]) * 1000.0
        prefill_ms = profile["prefill_forward_ms"]
        decode_ms = profile["decode_forward_ms"]
        num_generated_tokens = sum(generated)
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
            "tool_ms": tool_ms,
            "num_turns": num_turns,
            "batch_size": batch_size,
            "num_samples_per_prompt": num_samples_per_prompt,
            "max_prompt_length": prompt_len,
            "response_length": max(response_lengths, default=0),
            "active_batch_size": max(profile["active_batch_sizes"], default=0),
            "continuous_batch_size": max_batch_size,
        }

    def _continuous_prefill_cache(self, module, static_cache_type, prompt_ids, prompt_attention, model_dtype, device,
                                  sampling, selector, rows):
        position_ids = self._prefill_position_ids(prompt_attention)
        if getattr(module, "_supports_cache_class", None) is None:
            pad_token_id = get_pad_token_id(self.tokenizer)
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
            log_probs = _chosen_log_probs(prefill_output.logits[-1] / scale, tokens) if self.return_log_probs else None
            # generate() picked these tokens, so the selector only records them for the decode steps.
            selector.start(prompt_ids, prompt_attention, rows, vocab_size=prefill_output.logits[-1].shape[-1])
            selector.record(tokens, rows)
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
        tokens, log_probs = selector.first(prefill_output.logits[:, -1, :], prompt_ids, prompt_attention, rows)
        return tokens, log_probs, prefill_cache

    @staticmethod
    def _hf_sampling_kwargs(sampling):
        """Hugging Face ``generate()`` arguments for ``sampling``."""
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

    def _eos_tensor(self, device):
        return torch.tensor(get_eos_token_ids(self.tokenizer, self.engine.module) or [-1],
                            dtype=torch.long,
                            device=device)

    @staticmethod
    def _matches_eos(tokens, eos_ids):
        return (tokens.unsqueeze(-1) == eos_ids).any(dim=-1)

    def _is_eos(self, token):
        eos_ids = get_eos_token_ids(self.tokenizer, self.engine.module)
        if not eos_ids:
            return False
        eos_ids = torch.as_tensor(eos_ids, dtype=token.dtype, device=token.device)
        return bool((token == eos_ids).any().item())

    @staticmethod
    def _last_logits_kwargs(module):
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

        Generation pads a row only after it ends, so a token equal to
        ``pad_token_id`` before the first EOS was generated and stays part of the
        response.
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

    def _prefill_rows(self, module, prompt_ids, prompt_attention, ds_cache, logits_kwargs, model_dtype, device,
                      target_rows, cache_rows):
        """Prefill left-padded prompts in one forward, store each row's tokens in its cache row without padding,
        and return the last-position logits."""
        from transformers import StaticCache

        batch_size, width = prompt_ids.shape
        prefill_cache = self._create_static_cache(StaticCache, module.config, batch_size, width, device, model_dtype)
        prefill_out = module(
            prompt_ids,
            attention_mask=prompt_attention,
            past_key_values=prefill_cache,
            use_cache=True,
            cache_position=torch.arange(width, device=device),
            position_ids=self._prefill_position_ids(prompt_attention),
            **logits_kwargs,
        )
        last_logits = prefill_out.logits[:, -1, :]
        lengths = prompt_attention.sum(dim=1).tolist()
        targets = target_rows.tolist()
        for layer_idx, ds_layer in enumerate(ds_cache.layers):
            hf_keys, hf_values = self._cache_layer_tensors(prefill_cache, layer_idx)
            if not ds_layer.is_initialized:
                ds_layer.lazy_initialization(hf_keys.new_empty(cache_rows, *hf_keys.shape[1:]),
                                             hf_values.new_empty(cache_rows, *hf_values.shape[1:]))
            # Left padding puts each row's real tokens at the end of the prefill.
            for row, (target, length) in enumerate(zip(targets, lengths)):
                ds_layer.keys[target, :, :length] = hf_keys[row, :, width - length:]
                ds_layer.values[target, :, :length] = hf_values[row, :, width - length:]
        return last_logits

    def _generate_graph(self, prompt_ids, prompt_attn_mask, max_new_tokens, sampling, module, device):
        """Decode a static batch on a captured graph, greedy or sampled."""
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        max_len = prompt_len + max_new_tokens
        eos_ids = self._eos_tensor(device)
        model_dtype = next(module.parameters()).dtype
        row_indices = torch.arange(batch_size, device=device)

        cache_config = module.config
        tp_size = self.autotp_size
        if tp_size > 1:
            cache_config = copy(module.config)
            cache_config.num_key_value_heads = max(1, module.config.num_key_value_heads // tp_size)
            cache_config.num_attention_heads = max(1, module.config.num_attention_heads // tp_size)
        ds_cache = DeepSpeedStaticCache(
            cache_config,
            batch_size=batch_size,
            max_cache_len=max_len,
            device=device,
            dtype=model_dtype,
        )

        ds_cache.set_write_position(torch.tensor(0, dtype=torch.long, device=device))
        prefill_out = module(
            prompt_ids,
            attention_mask=prompt_attn_mask,
            past_key_values=ds_cache,
            use_cache=True,
            cache_position=torch.arange(prompt_len, device=device),
            position_ids=torch.arange(prompt_len, device=device).expand(batch_size, -1),
            **self._last_logits_kwargs(module),
        )
        prefill_logits = prefill_out.logits[:, -1, :]
        decode_start = torch.full((batch_size, ), prompt_len, dtype=torch.long, device=device)
        prompt_slots = prompt_attn_mask.bool()
        selector = _TokenSelector(sampling, get_repetition_penalty(sampling, module), batch_size,
                                  self.return_log_probs)
        next_token, first_log_probs = selector.first(prefill_logits, prompt_ids)
        log_probs = [first_log_probs]

        output_ids = [prompt_ids, next_token]

        write_pos = decode_start.clone()
        ds_cache.set_write_position(write_pos)

        # --- Static buffers for graph capture ---
        static_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        static_attn_mask = torch.zeros(batch_size, 1, 1, max_len, dtype=torch.bool, device=device)
        static_attn_mask[:, 0, 0, :prompt_len] = prompt_slots
        static_attn_mask[row_indices, 0, 0, decode_start] = True
        static_cache_pos = write_pos[:1]
        static_pos_ids = write_pos.unsqueeze(1)

        # Transformers checks a 2-D mask on the host during capture, and eager attention adds its mask to the
        # attention weights, so the mask is pre-built, 4-D and additive.
        static_causal_attn_mask = torch.zeros((batch_size, 1, 1, max_len), dtype=model_dtype, device=device)
        self._update_static_causal_attention(static_attn_mask[:, 0, 0], static_causal_attn_mask)

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
                        attention_mask=static_causal_attn_mask,
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
                    attention_mask=static_causal_attn_mask,
                    past_key_values=ds_cache,
                    use_cache=True,
                    cache_position=static_cache_pos,
                    position_ids=static_pos_ids,
                )
                static_logits = out.logits
            write_pos.copy_(decode_start)
            static_token.copy_(next_token)
        finally:
            module._forward_pre_hooks.update(saved_pre)
            module._forward_hooks.update(saved_post)

        # --- Decode loop ---
        eos_mask = self._matches_eos(next_token.squeeze(1), eos_ids)
        for step in range(max_new_tokens - 1):
            if bool(eos_mask.all().item()):
                break

            # Update static inputs
            static_token.copy_(next_token)
            torch.add(decode_start, step, out=write_pos)
            static_attn_mask[row_indices, 0, 0, write_pos] = True
            self._update_static_causal_attention(static_attn_mask[:, 0, 0], static_causal_attn_mask)

            # Replay
            get_accelerator().replay_graph(graph)
            next_token, step_log_probs = selector(static_logits[:, -1, :])
            output_ids.append(next_token)
            log_probs.append(step_log_probs)
            eos_mask |= self._matches_eos(next_token.squeeze(1), eos_ids)

        return torch.cat(output_ids, dim=1), torch.cat(log_probs, dim=1) if self.return_log_probs else None

    def sync_weights(self, step: int) -> None:  # noqa: ARG002
        """No-op: hybrid engine reads model weights live."""
        return None

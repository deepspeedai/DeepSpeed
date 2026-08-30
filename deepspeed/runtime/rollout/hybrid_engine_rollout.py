# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engine backed by DeepSpeed's hybrid engine.

Two generation paths:
  1. **model.generate()** (default): delegates to HuggingFace generate.
     Supports sampling (temperature, top_p) and greedy.
  2. **graph capture + DeepSpeedStaticCache**: only for greedy (temperature=0).
     Pre-allocates a StaticCache, captures the decode forward pass with a
     CUDA graph, and replays it for each decode step.  Eliminates kernel
     launch overhead.
  3. **generate_continuous()**: a bounded greedy prototype that refills
     retired cache rows with pending prompts.
"""

import time
from copy import copy
from dataclasses import dataclass
from inspect import signature

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutEngine, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.continuous_batching import ContinuousBatchRequest, ContinuousBatchScheduler


@dataclass
class HybridEngineRolloutConfig:
    """Configuration for HybridEngineRollout."""
    use_graph_capture: bool = False
    enable_profiling: bool = False
    use_shared_prefill: bool = False


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
        self._last_profile = None

    @torch.no_grad()
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
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

        is_greedy = sampling.temperature <= 0.0

        shared_prefill_handles = []
        if self.use_shared_prefill and n > 1:
            if self.use_graph_capture:
                raise RuntimeError("Shared prefill does not support CUDA graph capture")
            self.engine.prepare_shared_prefill(B, n, prompt_len)
            shared_prefill_handles = self._register_shared_prefill_hooks(module, B, n)
        try:
            if self.use_graph_capture and is_greedy:
                output_ids = self._generate_graph(prompt_ids, prompt_attn, max_new_tokens, pad_token_id, module,
                                                  device)
            else:
                temperature = max(sampling.temperature, 1e-8)
                do_sample = not is_greedy
                output_ids = module.generate(
                    prompt_ids,
                    attention_mask=prompt_attn,
                    max_new_tokens=max_new_tokens,
                    # ZeRO-3 gathers parameters during each decode forward, so every
                    # data-parallel rank must execute the same number of iterations.
                    eos_token_id=None,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                    top_p=sampling.top_p if do_sample else 1.0,
                    pad_token_id=pad_token_id,
                )
        finally:
            for handle in shared_prefill_handles:
                handle.remove()

        if self.enable_profiling:
            accelerator.synchronize()
            generation_end = time.perf_counter()

        # Generation deliberately ignores EOS above so ZeRO-3 ranks execute
        # the same number of parameter-gather collectives. Restore the usual
        # generation semantics before returning: retain the first EOS in each
        # response and replace every later token with padding.
        output_ids, response_attn = self._pad_after_eos(
            output_ids,
            response_start=prompt_len,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
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
            response_length = int(output_ids.shape[1] - prompt_len)
            num_generated_tokens = int(output_ids.shape[0] * response_length)
            tokens_per_second = 0.0
            if total_ms > 0.0:
                tokens_per_second = num_generated_tokens / (total_ms / 1000.0)
            self._last_profile = {
                "prompt_expansion_ms": prompt_expansion_ms,
                "generation_ms": generation_ms,
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
    def generate_continuous(self, requests, sampling_configs, max_batch_size):
        """Generate independent greedy requests in a continuously refilled batch.

        This first integration targets the OPSD prototype: every request has a
        single prompt row and one greedy response. Requests may use different
        response budgets. Completed rows retire immediately and pending prompts
        prefill into the released rows before the next decode step.
        """
        requests = tuple(requests)
        sampling_configs = tuple(sampling_configs)
        self._validate_continuous_inputs(requests, sampling_configs, max_batch_size)

        from transformers import StaticCache
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        module = self.engine.module
        device = requests[0].prompt_ids.device
        prompt_len = requests[0].prompt_ids.shape[1]
        model_dtype = next(module.parameters()).dtype
        max_cache_len = prompt_len + sum(config.max_new_tokens for config in sampling_configs)
        max_positions = getattr(module.config, "max_position_embeddings", max_cache_len)
        if max_cache_len > max_positions:
            raise ValueError("continuous batching cache exceeds the model maximum position embeddings")

        scheduler = ContinuousBatchScheduler(max_batch_size)
        request_by_id = {}
        responses = {}
        for request_id, (request, config) in enumerate(zip(requests, sampling_configs)):
            scheduler.submit(ContinuousBatchRequest(request_id, config.max_new_tokens))
            request_by_id[request_id] = request
            responses[request_id] = []

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
        next_tokens = {}
        cache_position = prompt_len
        update = scheduler.schedule()

        while update.active:
            keep_slots = torch.tensor(update.keep_slots, dtype=torch.long, device=device)
            survivor_count = keep_slots.numel()
            if survivor_count:
                cache.compact(keep_slots)
                survivor_attention = attention_mask.index_select(0, keep_slots).clone()
                attention_mask.zero_()
                attention_mask[:survivor_count].copy_(survivor_attention)
            else:
                cache.reset()
                write_positions.fill_(-1)
                attention_mask.zero_()
                cache_position = prompt_len

            admitted_tokens = self._continuous_prefill(
                module,
                StaticCache,
                cache,
                update,
                request_by_id,
                attention_mask,
                cache_position,
                prompt_len,
                model_dtype,
                device,
            )

            decoded_tokens = {}
            if survivor_count:
                survivor_ids = update.active_ids[:survivor_count]
                decode_input = torch.cat([next_tokens[request_id] for request_id in survivor_ids], dim=0)
                write_positions[:survivor_count].fill_(cache_position)
                position_ids = attention_mask[:survivor_count, :cache_position].sum(dim=1, keepdim=True)
                attention_mask[:survivor_count, cache_position] = 1
                output = module(
                    decode_input,
                    attention_mask=attention_mask[:survivor_count],
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=torch.tensor([cache_position], dtype=torch.long, device=device),
                    position_ids=position_ids,
                )
                decoded = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                decoded_tokens = dict(zip(survivor_ids, decoded.split(1, dim=0)))

            next_tokens = decoded_tokens | admitted_tokens
            finished_ids = []
            for request_id in update.active_ids:
                token = next_tokens[request_id]
                responses[request_id].append(token)
                if self._is_eos(token):
                    finished_ids.append(request_id)

            update = scheduler.advance(finished_ids)
            if survivor_count:
                cache_position += 1

        return [
            self._build_continuous_output(request, responses[request_id])
            for request_id, request in enumerate(requests)
        ]

    def _validate_continuous_inputs(self, requests, sampling_configs, max_batch_size):
        if not requests:
            raise ValueError("continuous batching requires at least one request")
        if len(requests) != len(sampling_configs):
            raise ValueError("requests and sampling_configs must have the same length")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.use_graph_capture:
            raise ValueError("continuous batching does not yet support CUDA graph capture")

        prompt_len = requests[0].prompt_ids.shape[1]
        device = requests[0].prompt_ids.device
        for request, config in zip(requests, sampling_configs):
            if request.prompt_ids.shape[0] != 1:
                raise ValueError("continuous batching requires one prompt row per request")
            if request.prompt_ids.shape[1] != prompt_len:
                raise ValueError("continuous batching currently requires equal prompt widths")
            if request.prompt_ids.device != device:
                raise ValueError("continuous batching requests must use the same device")
            if config.temperature > 0:
                raise ValueError("continuous batching currently supports greedy decoding only")
            if config.n_samples_per_prompt != 1:
                raise ValueError("continuous batching currently supports one sample per prompt")

    def _continuous_prefill(self, module, static_cache_type, cache, update, request_by_id, attention_mask,
                            cache_position, prompt_len, model_dtype, device):
        if not update.admitted:
            return {}

        admitted_ids = tuple(request.request_id for request in update.admitted)
        prompt_ids = torch.cat([request_by_id[request_id].prompt_ids for request_id in admitted_ids], dim=0)
        prompt_attention = torch.cat([request_by_id[request_id].prompt_attention_mask for request_id in admitted_ids],
                                     dim=0)
        prefill_cache = self._create_static_cache(static_cache_type, module.config, len(admitted_ids), prompt_len,
                                                  device, model_dtype)
        prefill_output = module(
            prompt_ids,
            attention_mask=prompt_attention,
            past_key_values=prefill_cache,
            use_cache=True,
            cache_position=torch.arange(prompt_len, device=device),
        )
        prefill_tokens = prefill_output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cache_start = cache_position - prompt_len
        for layer_idx, prefill_layer in enumerate(prefill_cache.layers):
            target_layer = cache.layers[layer_idx]
            for source_row, target_row in enumerate(update.admitted_slots):
                target_layer.keys[target_row, :, cache_start:cache_position].copy_(prefill_layer.keys[source_row])
                target_layer.values[target_row, :, cache_start:cache_position].copy_(prefill_layer.values[source_row])
        for source_row, target_row in enumerate(update.admitted_slots):
            attention_mask[target_row, cache_start:cache_position].copy_(prompt_attention[source_row])
            write_positions = cache._write_position
            write_positions[target_row] = cache_position
        return dict(zip(admitted_ids, prefill_tokens.split(1, dim=0)))

    def _is_eos(self, token):
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            return False
        eos_ids = torch.as_tensor(eos_token_id, dtype=token.dtype, device=token.device).flatten()
        return bool((token == eos_ids).any().item())

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
    def _build_continuous_output(request, response_tokens):
        response_ids = torch.cat(response_tokens, dim=1)
        input_ids = torch.cat((request.prompt_ids, response_ids), dim=1)
        response_attention = torch.ones_like(response_ids)
        attention_mask = torch.cat((request.prompt_attention_mask, response_attention), dim=1)
        response_start = request.prompt_ids.shape[1]
        return RolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_start_idx=torch.tensor([response_start], dtype=torch.long, device=input_ids.device),
        )

    def get_last_profile(self):
        """Return the most recent profiling snapshot for this rollout instance."""
        return self._last_profile

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
        """Retain the first response EOS and pad every subsequent position."""
        response_ids = output_ids[:, response_start:]
        response_attn = (response_ids != pad_token_id)

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
        response_attn = ((response_ids != pad_token_id) | first_eos) & ~after_first_eos
        return output_ids, response_attn.long()

    # ------------------------------------------------------------------
    # Graph capture decode loop (greedy only)
    # ------------------------------------------------------------------

    def _generate_graph(self, prompt_ids, prompt_attn, max_new_tokens, pad_token_id, module, device):
        """Greedy decode with DeepSpeedStaticCache + CUDA graph capture."""
        from transformers import StaticCache
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        max_len = prompt_len + max_new_tokens
        eos_token_id = self.tokenizer.eos_token_id
        model_dtype = next(module.parameters()).dtype

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
        )
        next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # --- Copy prefill KV into DeepSpeedStaticCache ---
        write_pos = torch.tensor(prompt_len - 1, dtype=torch.long, device=device)
        ds_cache = DeepSpeedStaticCache(
            module.config,
            batch_size=batch_size,
            max_cache_len=max_len,
            device=device,
            dtype=model_dtype,
        )
        ds_cache.set_write_position(write_pos)
        # Trigger lazy init then copy real data
        for layer_idx in range(len(ds_cache.layers)):
            ds_layer = ds_cache.layers[layer_idx]
            hf_layer = prefill_cache.layers[layer_idx]
            if not ds_layer.is_initialized:
                ds_layer.lazy_initialization(hf_layer.keys, hf_layer.values)
            ds_layer.keys[:, :, :prompt_len, :].copy_(hf_layer.keys[:, :, :prompt_len, :])
            ds_layer.values[:, :, :prompt_len, :].copy_(hf_layer.values[:, :, :prompt_len, :])

        output_ids = [prompt_ids, next_token]

        # --- Static buffers for graph capture ---
        static_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        static_attn = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        static_attn[:, :prompt_len] = prompt_attn
        static_attn[:, prompt_len] = 1  # first decode position
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
                        attention_mask=static_attn,
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
                    attention_mask=static_attn,
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

            # Replay
            get_accelerator().replay_graph(graph)
            next_token = static_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            output_ids.append(next_token)
            eos_mask |= (next_token.squeeze(1) == eos_token_id)

        return torch.cat(output_ids, dim=1)

    @staticmethod
    def _sample_top_p(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
        """Sample from logits with temperature and nucleus (top-p) filtering."""
        logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = (cumulative_probs - torch.softmax(sorted_logits, dim=-1)) >= top_p
            sorted_logits[mask] = -float('inf')
            probs = torch.softmax(sorted_logits, dim=-1)
            sampled = torch.multinomial(probs, 1)
            tokens = sorted_indices.gather(1, sampled)
        else:
            probs = torch.softmax(logits, dim=-1)
            tokens = torch.multinomial(probs, 1)
        return tokens

    def sync_weights(self, step: int) -> None:  # noqa: ARG002
        """No-op: hybrid engine reads model weights live."""
        return None

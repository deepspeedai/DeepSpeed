# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engine backed by DeepSpeed's hybrid engine, with a ZeRO-3 fallback.

Supports two generation strategies:
  1. **naive** (default): HF generate with ``num_return_sequences``.
  2. **continuous_batching**: Custom decode loop with shared-prefix prefill,
     continuous batching (slot replacement on EOS), KV cache left-trim, and
     early-exit batch compaction.

The continuous batching path is activated when ``continuous_batching_size > 0``
is passed to the constructor.
"""

import copy
from dataclasses import dataclass
from typing import Optional

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutEngine, RolloutRequest, SamplingConfig


@dataclass
class HybridEngineRolloutConfig:
    """Configuration for HybridEngineRollout.

    Attributes:
        continuous_batching_size: Number of decode slots for the custom CB loop.
            When > 0, uses shared-prefix prefill + CB + KV trim + early exit.
            When 0 (default), falls back to HF generate with num_return_sequences.
        kv_trim_threshold: Minimum common leading padding before KV cache trim
            is applied. 0 = never trim, 1 = trim every step, 16 = trim only
            when >=16 tokens of common padding have accumulated (default).
    """
    continuous_batching_size: int = 0
    kv_trim_threshold: int = 16
    use_graph_capture: bool = False


def _hybrid_engine_has_accel(engine) -> bool:
    """Check if the DeepSpeed engine has accelerated inference containers."""
    return getattr(engine, "_generate", None) is not None


class HybridEngineRollout(RolloutEngine):
    """Rollout engine using DeepSpeed hybrid engine or ZeRO-3 fallback.

    Args:
        engine: DeepSpeed engine wrapping the model.
        tokenizer: HuggingFace tokenizer (must have pad_token_id or eos_token_id).
        continuous_batching_size: Number of CB decode slots (0 = use HF generate).
        kv_trim_threshold: Min common padding before KV trim fires (0 = disabled).
    """

    name = "hybrid_engine"

    def __init__(self, engine, tokenizer, continuous_batching_size: int = 0,
                 kv_trim_threshold: int = 16, use_graph_capture: bool = False):
        self.engine = engine
        self.tokenizer = tokenizer
        self.continuous_batching_size = continuous_batching_size
        self.kv_trim_threshold = kv_trim_threshold
        self.use_graph_capture = use_graph_capture
        self._has_accel = _hybrid_engine_has_accel(engine)
        # Graph capture state (lazily initialized)
        self._graph = None
        self._graph_batch_size: int = 0
        self._graph_max_cache_len: int = 0
        self._graph_input_ids: Optional[torch.Tensor] = None
        self._graph_cache_pos: Optional[torch.Tensor] = None
        self._graph_static_cache = None
        self._graph_output_logits: Optional[torch.Tensor] = None
        self._graph_attn_mask: Optional[torch.Tensor] = None
        self._graph_position_ids: Optional[torch.Tensor] = None

    @torch.no_grad()
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        if self.continuous_batching_size > 0:
            if self.use_graph_capture:
                return self._generate_graph_capture_cb(request, sampling, self.continuous_batching_size)
            return self._generate_continuous_batching(request, sampling, self.continuous_batching_size)
        return self._generate_naive(request, sampling)

    # ------------------------------------------------------------------
    # Naive path: delegate to HF generate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_naive(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        gen_kwargs = dict(
            input_ids=request.prompt_ids,
            attention_mask=request.prompt_attention_mask,
            max_new_tokens=sampling.max_new_tokens,
            do_sample=sampling.temperature > 0.0,
            temperature=max(sampling.temperature, 1e-8),
            top_p=sampling.top_p,
            top_k=sampling.top_k if sampling.top_k > 0 else 0,
            num_return_sequences=sampling.n_samples_per_prompt,
            pad_token_id=pad_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        self.engine.eval()
        try:
            if self._has_accel:
                seqs = self.engine.generate(**gen_kwargs)
            else:
                seqs = self._fallback_generate(**gen_kwargs)
        finally:
            self.engine.train()

        return self._build_rollout_batch(request, seqs, sampling.n_samples_per_prompt, pad_id)

    # ------------------------------------------------------------------
    # Continuous batching path: shared prefix + CB + KV trim + early exit
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_continuous_batching(self, request: RolloutRequest, sampling: SamplingConfig,
                                      cb_size: int) -> RolloutBatch:
        """Custom decode loop with:
        - Shared prefix: prefill prompt once, expand KV for all samples
        - Continuous batching: fixed slot count, replace finished slots
        - KV cache left-trim: trim common leading padding (threshold=16)
        - Early exit: batch compaction when no pending rollouts remain
        """
        device = request.prompt_ids.device
        B = request.prompt_ids.shape[0]
        n = sampling.n_samples_per_prompt
        total = B * n
        prompt_len = request.prompt_ids.shape[1]
        max_new_tokens = sampling.max_new_tokens
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id or eos_token_id
        batch_size = min(total, cb_size)

        temperature = max(sampling.temperature, 1e-8)
        top_p = sampling.top_p

        # === Phase 1: Shared prefix prefill ===
        self.engine.eval()
        gather_ctx = None
        try:
            module, gather_ctx = self._get_module_ctx()
            if gather_ctx is not None:
                gather_ctx.__enter__()
            out = module(
                request.prompt_ids,
                attention_mask=request.prompt_attention_mask,
                use_cache=True,
            )
            prompt_past = out.past_key_values
            first_logits = out.logits[:, -1, :]  # [B, vocab]

            # Storage for all generated tokens
            all_tokens = torch.full(
                (total, max_new_tokens), pad_token_id,
                dtype=torch.long, device=device)
            gen_lens = torch.zeros(total, dtype=torch.long, device=device)
            completed_count = 0
            next_rollout_idx = 0

            # === Phase 2: Initialize slots ===
            init_count = min(batch_size, total)

            # Expand prompt KV for initial slots (shared prefix reuse)
            past = copy.deepcopy(prompt_past)
            if B == 1:
                past.batch_repeat_interleave(init_count)
                init_logits = first_logits.repeat(init_count, 1)
            else:
                past.batch_repeat_interleave(n)
                if init_count < B * n:
                    past.reorder_cache(torch.arange(init_count, device=device))
                prompt_indices = torch.arange(init_count, device=device) // n
                init_logits = first_logits[prompt_indices]

            # Per-slot state
            slot_rollout = list(range(init_count))
            slot_decode_step = [0] * init_count
            slot_position = [prompt_len] * init_count
            slot_active = [True] * init_count
            slot_pad_start = [0] * init_count
            next_rollout_idx = init_count

            # Attention mask
            if B == 1:
                attn_mask = request.prompt_attention_mask.repeat(init_count, 1)
            else:
                attn_mask = request.prompt_attention_mask.repeat_interleave(n, dim=0)[:init_count]

            # Sample first token for each slot
            next_tokens = self._sample_top_p(init_logits, temperature, top_p)
            for i in range(init_count):
                all_tokens[slot_rollout[i], 0] = next_tokens[i, 0]
                gen_lens[slot_rollout[i]] = 1
                slot_decode_step[i] = 1

            # Extend attn_mask for first generated token
            attn_mask = torch.cat([
                attn_mask,
                torch.ones(init_count, 1, dtype=attn_mask.dtype, device=device)
            ], dim=1)

            # Check immediate EOS
            eos_now = (next_tokens.squeeze(1) == eos_token_id).cpu().tolist()
            for i in range(init_count):
                if eos_now[i]:
                    completed_count += 1
                    if next_rollout_idx < total:
                        self._cb_replace_slot(
                            i, past, prompt_past, attn_mask, first_logits,
                            next_tokens, all_tokens, gen_lens, slot_rollout,
                            slot_decode_step, slot_position, slot_pad_start,
                            next_rollout_idx, prompt_len, B, n, device,
                            temperature, top_p)
                        next_rollout_idx += 1
                    else:
                        slot_active[i] = False

            # === Phase 3: Main decode loop ===
            while completed_count < total:
                if not any(slot_active):
                    break

                num_slots = len(slot_rollout)

                # Build position_ids
                pos_ids = torch.tensor(
                    [[slot_position[i]] for i in range(num_slots)],
                    device=device)

                # Forward pass (single token per slot)
                out = module(
                    next_tokens, attention_mask=attn_mask,
                    position_ids=pos_ids,
                    past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_tokens = self._sample_top_p(out.logits[:, -1, :], temperature, top_p)

                # Extend attention mask
                attn_mask = torch.cat([
                    attn_mask,
                    torch.ones(num_slots, 1, dtype=attn_mask.dtype, device=device)
                ], dim=1)

                # Update positions and store tokens
                for i in range(num_slots):
                    if not slot_active[i]:
                        continue
                    slot_position[i] += 1
                    ds = slot_decode_step[i]
                    if ds < max_new_tokens:
                        all_tokens[slot_rollout[i], ds] = next_tokens[i, 0]
                        gen_lens[slot_rollout[i]] = ds + 1
                    slot_decode_step[i] += 1

                # Check for finished slots (EOS or max_len)
                eos_mask = (next_tokens.squeeze(1) == eos_token_id).cpu().tolist()
                slots_finished = []
                for i in range(num_slots):
                    if not slot_active[i]:
                        continue
                    if eos_mask[i] or slot_decode_step[i] >= max_new_tokens:
                        completed_count += 1
                        if next_rollout_idx < total:
                            self._cb_replace_slot(
                                i, past, prompt_past, attn_mask, first_logits,
                                next_tokens, all_tokens, gen_lens, slot_rollout,
                                slot_decode_step, slot_position, slot_pad_start,
                                next_rollout_idx, prompt_len, B, n, device,
                                temperature, top_p)
                            next_rollout_idx += 1
                        else:
                            slots_finished.append(i)

                # Batch compaction (early exit): remove finished slots
                if slots_finished:
                    keep = [i for i in range(num_slots) if i not in slots_finished]
                    if not keep:
                        break
                    keep_t = torch.tensor(keep, device=device)
                    next_tokens = next_tokens[keep_t]
                    past.reorder_cache(keep_t)
                    attn_mask = attn_mask[keep_t]
                    slot_rollout = [slot_rollout[i] for i in keep]
                    slot_decode_step = [slot_decode_step[i] for i in keep]
                    slot_position = [slot_position[i] for i in keep]
                    slot_active = [slot_active[i] for i in keep]
                    slot_pad_start = [slot_pad_start[i] for i in keep]

                # KV cache left-trim: remove common leading padding
                if self.kv_trim_threshold > 0:
                    active_pads = [slot_pad_start[i] for i in range(len(slot_pad_start))
                                   if slot_active[i]]
                    if active_pads:
                        min_pad = min(active_pads)
                        if min_pad >= self.kv_trim_threshold:
                            for layer_idx in range(len(past)):
                                past.layers[layer_idx].keys = past.layers[layer_idx].keys[:, :, min_pad:]
                                past.layers[layer_idx].values = past.layers[layer_idx].values[:, :, min_pad:]
                            attn_mask = attn_mask[:, min_pad:]
                            for i in range(len(slot_pad_start)):
                                slot_pad_start[i] -= min_pad

        finally:
            if gather_ctx is not None:
                gather_ctx.__exit__(None, None, None)
            self.engine.train()

        # === Build output ===
        max_gen = gen_lens.max().item() if gen_lens.max().item() > 0 else 1
        all_tokens = all_tokens[:, :max_gen]

        # Construct [prompt | response] sequences
        if B == 1:
            expanded_prompts = request.prompt_ids.repeat(total, 1)
            expanded_prompt_mask = request.prompt_attention_mask.repeat(total, 1)
        else:
            expanded_prompts = request.prompt_ids.repeat_interleave(n, dim=0)
            expanded_prompt_mask = request.prompt_attention_mask.repeat_interleave(n, dim=0)

        seqs = torch.cat([expanded_prompts, all_tokens], dim=1)
        response_mask = (all_tokens != pad_token_id).to(expanded_prompt_mask.dtype)
        attention_mask = torch.cat([expanded_prompt_mask, response_mask], dim=1)
        response_start_idx = torch.full((total,), prompt_len, dtype=torch.long, device=device)

        return RolloutBatch(
            input_ids=seqs,
            attention_mask=attention_mask,
            response_start_idx=response_start_idx,
        )

    # ------------------------------------------------------------------
    # Graph capture + continuous batching path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_graph_capture_cb(self, request: RolloutRequest, sampling: SamplingConfig,
                                cb_size: int) -> RolloutBatch:
        """Decode loop using graph capture for the forward pass.

        Uses StaticCache (pre-allocated, fixed-shape KV) so the decode forward
        can be captured as a graph and replayed each step.  Batch size
        stays fixed (no compaction); finished slots are masked via attention_mask.
        """
        from transformers import StaticCache

        device = request.prompt_ids.device
        B = request.prompt_ids.shape[0]
        n = sampling.n_samples_per_prompt
        total = B * n
        prompt_len = request.prompt_ids.shape[1]
        max_new_tokens = sampling.max_new_tokens
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id or eos_token_id
        batch_size = min(total, cb_size)
        # With CB, decode_pos advances globally: ceil(total/batch_size) rounds of max_new_tokens
        import math
        max_cache_len = prompt_len + max_new_tokens

        temperature = max(sampling.temperature, 1e-8)
        top_p = sampling.top_p

        self.engine.eval()
        gather_ctx = None
        try:
            module, gather_ctx = self._get_module_ctx()
            if gather_ctx is not None:
                gather_ctx.__enter__()
            model_dtype = next(module.parameters()).dtype

            # === Phase 1: Prefill into a reference cache ===
            prefill_bs = 1 if B == 1 else B
            prefill_cache = StaticCache(
                module.config, batch_size=prefill_bs,
                max_cache_len=max_cache_len, device=device, dtype=model_dtype)
            cache_pos_prefill = torch.arange(prompt_len, device=device)
            pfkw = dict(input_ids=request.prompt_ids[:prefill_bs],
                        past_key_values=prefill_cache, use_cache=True,
                        cache_position=cache_pos_prefill)
            if B > 1:
                pfkw['attention_mask'] = request.prompt_attention_mask
            out = module(**pfkw)
            first_logits = out.logits[:, -1, :]  # [prefill_bs, vocab]

            # === Phase 2: Build batch static cache + static buffers ===
            static_cache = StaticCache(
                module.config, batch_size=batch_size,
                max_cache_len=max_cache_len, device=device, dtype=model_dtype)

            static_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            static_cache_position = torch.zeros(1, dtype=torch.long, device=device)
            static_attn_mask = torch.zeros(
                batch_size, max_cache_len, dtype=torch.long, device=device)
            static_position_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

            # Temporarily remove DeepSpeed forward hooks (they call synchronize()
            # which is illegal during CUDA graph capture).
            _saved_pre_hooks = dict(module._forward_pre_hooks)
            _saved_post_hooks = dict(module._forward_hooks)
            module._forward_pre_hooks.clear()
            module._forward_hooks.clear()

            # Initialize static_cache with a dummy forward, then restore with prefill KV
            dummy_cp = torch.zeros(1, dtype=torch.long, device=device)
            module(torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                   past_key_values=static_cache, use_cache=True, cache_position=dummy_cp)

            # === Phase 3: Warmup + capture (or reuse cached graph) ===
            need_new_graph = (self._graph is None or
                              self._graph_batch_size != batch_size or
                              self._graph_max_cache_len != max_cache_len)

            if need_new_graph:
                # Warmup
                static_cache_position[0] = prompt_len
                static_input_ids[:, 0] = 0
                static_position_ids[:, 0] = prompt_len
                module(static_input_ids, past_key_values=static_cache, use_cache=True,
                       cache_position=static_cache_position, attention_mask=static_attn_mask,
                       position_ids=static_position_ids)
                # Capture — must be on correct device
                graph = get_accelerator().create_graph()
                with torch.cuda.device(device):
                    with get_accelerator().capture_to_graph(graph):
                        graph_out = module(static_input_ids, past_key_values=static_cache,
                                          use_cache=True, cache_position=static_cache_position,
                                          attention_mask=static_attn_mask,
                                          position_ids=static_position_ids)

                # Cache graph state
                self._graph = graph
                self._graph_batch_size = batch_size
                self._graph_max_cache_len = max_cache_len
                self._graph_input_ids = static_input_ids
                self._graph_cache_pos = static_cache_position
                self._graph_static_cache = static_cache
                self._graph_output_logits = graph_out.logits
                self._graph_attn_mask = static_attn_mask
                self._graph_position_ids = static_position_ids
                graph_logits = graph_out.logits
            else:
                # Reuse cached graph — reassign static buffers to the cached ones
                # The graph was captured with specific tensor addresses; we must
                # use those same tensors.
                graph = self._graph
                static_input_ids = self._graph_input_ids
                static_cache_position = self._graph_cache_pos
                static_cache = self._graph_static_cache
                static_attn_mask = self._graph_attn_mask
                static_position_ids = self._graph_position_ids
                graph_logits = self._graph_output_logits

                # Just restore prefill KV + reset attn mask (no warmup/capture needed)

            # === Phase 4: Restore prefill KV into all initial slots ===
            init_count = min(batch_size, total)
            for layer_idx in range(len(static_cache.layers)):
                src_k = prefill_cache.layers[layer_idx].keys
                src_v = prefill_cache.layers[layer_idx].values
                for s in range(init_count):
                    pi = s // n if B > 1 else 0
                    static_cache.layers[layer_idx].keys[s] = src_k[pi]
                    static_cache.layers[layer_idx].values[s] = src_v[pi]

            static_attn_mask[:] = 0
            static_attn_mask[:init_count, :prompt_len] = 1

            if B == 1:
                init_logits = first_logits.expand(init_count, -1)
            else:
                prompt_indices = torch.arange(init_count, device=device) // n
                init_logits = first_logits[prompt_indices]

            # === Phase 5: Slot state + sample first token ===
            all_tokens = torch.full(
                (total, max_new_tokens), pad_token_id, dtype=torch.long, device=device)
            gen_lens = torch.zeros(total, dtype=torch.long, device=device)
            completed_count = 0
            next_rollout_idx = init_count

            slot_rollout = list(range(init_count)) + [-1] * (batch_size - init_count)
            slot_decode_step = [0] * batch_size
            slot_active = [i < init_count for i in range(batch_size)]
            slot_position = [prompt_len] * batch_size

            first_tokens = self._sample_top_p(init_logits, temperature, top_p)
            for i in range(init_count):
                all_tokens[i, 0] = first_tokens[i, 0]
                gen_lens[i] = 1
                slot_decode_step[i] = 1

            # Check immediate EOS
            eos_now = (first_tokens.squeeze(1) == eos_token_id).cpu().tolist()
            for i in range(init_count):
                if eos_now[i]:
                    completed_count += 1
                    if next_rollout_idx < total:
                        self._graph_capture_replace_slot_kv(
                            i, static_cache, prefill_cache, static_attn_mask,
                            first_logits, first_tokens, all_tokens, gen_lens,
                            slot_rollout, slot_decode_step, slot_position,
                            next_rollout_idx, prompt_len, B, n, device,
                            temperature, top_p)
                        next_rollout_idx += 1
                    else:
                        slot_active[i] = False

            current_tokens = first_tokens
            decode_pos = prompt_len

            # Vectorized slot state tensors for the hot loop
            slot_rollout_t = torch.tensor(slot_rollout, dtype=torch.long, device=device)
            slot_decode_step_t = torch.tensor(slot_decode_step, dtype=torch.long, device=device)
            slot_active_t = torch.tensor(slot_active, dtype=torch.bool, device=device)
            slot_position_t = torch.tensor(slot_position, dtype=torch.long, device=device)

            # === Phase 6: Main decode loop with graph replay ===
            while completed_count < total:
                if not slot_active_t.any():
                    # Round complete: all batch slots finished.
                    # Reset decode_pos and start a new round with fresh slots.
                    if next_rollout_idx >= total:
                        break
                    decode_pos = prompt_len
                    static_attn_mask[:] = 0
                    refill_count = min(batch_size, total - next_rollout_idx)
                    for s in range(refill_count):
                        src_prompt_idx = next_rollout_idx // n if B > 1 else 0
                        for layer_idx in range(len(static_cache.layers)):
                            static_cache.layers[layer_idx].keys[s].copy_(
                                prefill_cache.layers[layer_idx].keys[src_prompt_idx])
                            static_cache.layers[layer_idx].values[s].copy_(
                                prefill_cache.layers[layer_idx].values[src_prompt_idx])
                        static_attn_mask[s, :prompt_len] = 1
                        new_first_logits = first_logits[src_prompt_idx:src_prompt_idx + 1]
                        new_token = self._sample_top_p(new_first_logits, temperature, top_p)
                        current_tokens[s] = new_token[0]
                        all_tokens[next_rollout_idx, 0] = new_token[0, 0]
                        gen_lens[next_rollout_idx] = 1
                        slot_rollout_t[s] = next_rollout_idx
                        slot_decode_step_t[s] = 1
                        slot_position_t[s] = prompt_len
                        slot_active_t[s] = True
                        next_rollout_idx += 1
                    for s in range(refill_count, batch_size):
                        slot_active_t[s] = False
                    continue

                # Update static buffers in-place (tensor ops, no Python loop)
                static_input_ids[:, 0] = current_tokens[:, 0]
                static_cache_position[0] = decode_pos
                static_attn_mask[:, decode_pos] = slot_active_t.long()
                static_position_ids[:, 0] = slot_position_t

                get_accelerator().replay_graph(graph)

                current_tokens = self._sample_top_p(
                    graph_logits[:, -1, :], temperature, top_p)

                # Store tokens for active slots (vectorized)
                active_mask = slot_active_t & (slot_decode_step_t < max_new_tokens)
                if active_mask.any():
                    active_indices = active_mask.nonzero(as_tuple=True)[0]
                    for idx in active_indices.tolist():
                        ds = slot_decode_step_t[idx].item()
                        ri = slot_rollout_t[idx].item()
                        all_tokens[ri, ds] = current_tokens[idx, 0]
                        gen_lens[ri] = ds + 1
                slot_decode_step_t[slot_active_t] += 1
                slot_position_t[slot_active_t] += 1

                decode_pos += 1

                # Check finished (EOS or max_len)
                eos_hit = (current_tokens.squeeze(1) == eos_token_id) & slot_active_t
                maxlen_hit = (slot_decode_step_t >= max_new_tokens) & slot_active_t
                finished = eos_hit | maxlen_hit

                if finished.any():
                    for idx in finished.nonzero(as_tuple=True)[0].tolist():
                        completed_count += 1
                        slot_active_t[idx] = False

        finally:
            # Restore DeepSpeed forward hooks
            module._forward_pre_hooks.update(_saved_pre_hooks)
            module._forward_hooks.update(_saved_post_hooks)
            if gather_ctx is not None:
                gather_ctx.__exit__(None, None, None)
            self.engine.train()
            del prefill_cache

        # === Build output ===
        max_gen = gen_lens.max().item() if gen_lens.max().item() > 0 else 1
        all_tokens = all_tokens[:, :max_gen]

        if B == 1:
            expanded_prompts = request.prompt_ids.repeat(total, 1)
            expanded_prompt_mask = request.prompt_attention_mask.repeat(total, 1)
        else:
            expanded_prompts = request.prompt_ids.repeat_interleave(n, dim=0)
            expanded_prompt_mask = request.prompt_attention_mask.repeat_interleave(n, dim=0)

        seqs = torch.cat([expanded_prompts, all_tokens], dim=1)
        response_mask = (all_tokens != pad_token_id).to(expanded_prompt_mask.dtype)
        attention_mask = torch.cat([expanded_prompt_mask, response_mask], dim=1)
        response_start_idx = torch.full((total,), prompt_len, dtype=torch.long, device=device)

        return RolloutBatch(
            input_ids=seqs,
            attention_mask=attention_mask,
            response_start_idx=response_start_idx,
        )

    def _graph_capture_replace_slot_kv(self, slot_idx, static_cache, prefill_cache,
                                     static_attn_mask, first_logits, next_tokens,
                                     all_tokens, gen_lens, slot_rollout,
                                     slot_decode_step, slot_position,
                                     new_rollout_idx,
                                     prompt_len, B, n, device, temperature, top_p):
        """Replace a finished slot by copying prompt KV from prefill_cache."""
        src_prompt_idx = new_rollout_idx // n if B > 1 else 0

        # Copy prompt KV from prefill cache
        for layer_idx in range(len(static_cache.layers)):
            static_cache.layers[layer_idx].keys[slot_idx] = \
                prefill_cache.layers[layer_idx].keys[src_prompt_idx]
            static_cache.layers[layer_idx].values[slot_idx] = \
                prefill_cache.layers[layer_idx].values[src_prompt_idx]

        # Reset attention mask: only prompt positions are valid
        static_attn_mask[slot_idx] = 0
        static_attn_mask[slot_idx, :prompt_len] = 1

        # Sample first token for new slot
        new_first_logits = first_logits[src_prompt_idx:src_prompt_idx + 1]
        new_token = self._sample_top_p(new_first_logits, temperature, top_p)
        next_tokens[slot_idx] = new_token[0]

        slot_rollout[slot_idx] = new_rollout_idx
        all_tokens[new_rollout_idx, 0] = new_token[0, 0]
        gen_lens[new_rollout_idx] = 1
        slot_decode_step[slot_idx] = 1
        slot_position[slot_idx] = prompt_len

    # ------------------------------------------------------------------
    # CB helper: replace a finished slot with a new rollout
    # ------------------------------------------------------------------

    def _cb_replace_slot(self, slot_idx, past, prompt_past, attn_mask,
                         first_logits, next_tokens, all_tokens, gen_lens,
                         slot_rollout, slot_decode_step, slot_position,
                         slot_pad_start, new_rollout_idx, prompt_len, B, n,
                         device, temperature, top_p):
        """Replace a finished slot with a new rollout by left-padding prompt KV."""
        current_kv_len = past.layers[0].keys.shape[2]
        pad_len = current_kv_len - prompt_len

        src_prompt_idx = new_rollout_idx // n if B > 1 else 0

        for layer_idx in range(len(past)):
            key, value = past[layer_idx]
            src_key, src_value = prompt_past[layer_idx]
            heads = key.shape[1]
            head_dim = key.shape[3]

            pad_kv = torch.zeros(1, heads, pad_len, head_dim,
                                 dtype=key.dtype, device=device)
            new_key = torch.cat([pad_kv, src_key[src_prompt_idx:src_prompt_idx + 1]], dim=2)
            new_value = torch.cat([pad_kv, src_value[src_prompt_idx:src_prompt_idx + 1]], dim=2)
            key[slot_idx] = new_key[0]
            value[slot_idx] = new_value[0]

        mask_len = attn_mask.shape[1]
        new_mask = torch.zeros(mask_len, dtype=attn_mask.dtype, device=device)
        new_mask[pad_len:pad_len + prompt_len] = 1
        attn_mask[slot_idx] = new_mask

        new_first_logits = first_logits[src_prompt_idx:src_prompt_idx + 1]
        new_token = self._sample_top_p(new_first_logits, temperature, top_p)
        next_tokens[slot_idx] = new_token[0]

        slot_rollout[slot_idx] = new_rollout_idx
        all_tokens[new_rollout_idx, 0] = new_token[0, 0]
        gen_lens[new_rollout_idx] = 1
        slot_decode_step[slot_idx] = 1
        slot_position[slot_idx] = prompt_len
        slot_pad_start[slot_idx] = pad_len

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_top_p(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
        """Sample from logits with temperature and nucleus (top-p) filtering."""
        logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = (cumulative_probs - torch.softmax(sorted_logits, dim=-1)) >= top_p
            sorted_logits[mask] = -float('inf')
            probs = torch.softmax(sorted_logits, dim=-1)
            sampled = torch.multinomial(probs, 1)
            tokens = sorted_indices.gather(1, sampled)
        else:
            probs = torch.softmax(logits, dim=-1)
            tokens = torch.multinomial(probs, 1)
        return tokens

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_module_ctx(self):
        """Get the underlying HF model and optional GatheredParameters context.

        Returns (module, context_manager_or_None). For ZeRO-3 the caller must
        enter/exit the context to keep params gathered for the decode loop.
        """
        from deepspeed.runtime.zero import GatheredParameters

        module = self.engine.module
        params = list(module.parameters())
        if params and hasattr(params[0], 'ds_id'):
            ctx = GatheredParameters(params)
            return module, ctx
        return module, None

    def _build_rollout_batch(self, request, seqs, n, pad_id):
        """Build RolloutBatch from HF generate output."""
        B = request.prompt_ids.shape[0]
        T_p = request.prompt_ids.shape[1]
        if seqs.shape[0] != B * n:
            raise RuntimeError(f"generate returned batch {seqs.shape[0]}, expected {B * n}")

        response_start_idx = torch.full((B * n,), T_p, dtype=torch.long, device=seqs.device)
        attention_mask = (seqs != pad_id).to(request.prompt_attention_mask.dtype)
        prompt_mask_expanded = request.prompt_attention_mask.repeat_interleave(n, dim=0)
        attention_mask[:, :T_p] = prompt_mask_expanded

        return RolloutBatch(input_ids=seqs, attention_mask=attention_mask,
                           response_start_idx=response_start_idx)

    def sync_weights(self, step: int) -> None:  # noqa: ARG002
        """No-op: hybrid engine reads model weights live."""
        return None

    @torch.no_grad()
    def _fallback_generate(self, **gen_kwargs) -> torch.Tensor:
        """Manual ZeRO-3 generate: gather all params, call HF generate."""
        from deepspeed.runtime.zero import GatheredParameters

        module = self.engine.module
        all_params = list(module.parameters())
        with GatheredParameters(all_params):
            return module.generate(**gen_kwargs)

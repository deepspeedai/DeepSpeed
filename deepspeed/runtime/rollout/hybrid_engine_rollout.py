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
"""

import time
from dataclasses import dataclass

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutEngine, RolloutRequest, SamplingConfig


@dataclass
class HybridEngineRolloutConfig:
    """Configuration for HybridEngineRollout."""
    use_graph_capture: bool = False
    enable_profiling: bool = False


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

        if self.use_graph_capture and is_greedy:
            output_ids = self._generate_graph(prompt_ids, prompt_attn, max_new_tokens, pad_token_id, module, device)
        else:
            temperature = max(sampling.temperature, 1e-8)
            do_sample = not is_greedy
            output_ids = module.generate(
                prompt_ids,
                attention_mask=prompt_attn,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=sampling.top_p if do_sample else 1.0,
                pad_token_id=pad_token_id,
            )

        if self.enable_profiling:
            accelerator.synchronize()
            generation_end = time.perf_counter()

        # Build attention mask: pad positions (both left padding from prompt
        # and right padding from EOS / shorter sequences) are 0.
        response_start = prompt_len
        attention_mask = (output_ids != pad_token_id).long()
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

    def get_last_profile(self):
        """Return the most recent profiling snapshot for this rollout instance."""
        return self._last_profile

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
        prefill_cache = StaticCache(
            config=module.config,
            batch_size=batch_size,
            max_cache_len=max_len,
            device=device,
            dtype=model_dtype,
        )
        # Mirror generate's hybrid-model kwargs exactly: a per-type mask dict
        # (GDN must see None) and no explicit cache_position. A 2D mask here
        # corrupts the GDN conv-state initialization and collapses decode.
        prefill_out = module(
            prompt_ids,
            attention_mask={
                "full_attention": None,
                "linear_attention": None
            },
            past_key_values=prefill_cache,
            use_cache=True,
        )
        next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # --- Copy prefill KV into DeepSpeedStaticCache ---
        write_pos = torch.tensor(prompt_len, dtype=torch.long, device=device)
        ds_cache = DeepSpeedStaticCache(
            module.config,
            batch_size=batch_size,
            max_cache_len=max_len,
            device=device,
            dtype=model_dtype,
        )
        ds_cache.set_write_position(write_pos)
        # Trigger lazy init then copy real data. Hybrid models carry
        # linear-attention (GDN) slots alongside KV slots; the GDN slots are
        # bound by reference to the prefill cache's HF slot objects, whose
        # state management is already cudagraph-safe by construction.
        from deepspeed.utils.static_cache import DSStaticGDNSlot
        for layer_idx in range(len(ds_cache.layers)):
            ds_layer = ds_cache.layers[layer_idx]
            hf_layer = prefill_cache.layers[layer_idx]
            if isinstance(ds_layer, DSStaticGDNSlot):
                ds_layer.bind(hf_layer)
                continue
            if not ds_layer.is_initialized:
                ds_layer.lazy_initialization(hf_layer.keys, hf_layer.values)
            ds_layer.keys[:, :, :prompt_len, :].copy_(hf_layer.keys[:, :, :prompt_len, :])
            ds_layer.values[:, :, :prompt_len, :].copy_(hf_layer.values[:, :, :prompt_len, :])

        output_ids = [prompt_ids, next_token]

        # --- Static buffers for graph capture ---
        static_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        # Full-width static causal mask for the full-attention layers; revealed
        # one position per decode step by in-place writes (graph-replay safe).
        # GDN layers must receive None: their recurrence ignores attention masks
        # and a non-None value corrupts the conv-state updates.
        static_attn = torch.zeros(batch_size, 1, 1, max_len, dtype=torch.bool, device=device)
        static_attn[:, :, :, :prompt_len] = prompt_attn.unsqueeze(1).unsqueeze(1).bool()
        static_attn[:, :, :, prompt_len] = True

        full_token_buf = torch.zeros(max_len, dtype=torch.long, device=device)
        full_token_buf[:prompt_len] = prompt_ids.view(-1)

        write_pos.fill_(prompt_len)

        # Snapshot the GDN states right after prefill: the warmup forwards
        # advance conv/recurrent states by extra steps, so they must be
        # restored before capture or every replay starts from corrupted state.
        gdn_snapshot = []
        for ds_layer in ds_cache.layers:
            if type(ds_layer).__name__ == "DSStaticGDNSlot":
                cs, rs = ds_layer.conv_states[0], ds_layer.recurrent_states[0]
                gdn_snapshot.append((cs, rs, cs.clone(), rs.clone()))

        def restore_gdn_states():
            for cs, rs, cs0, rs0 in gdn_snapshot:
                cs.copy_(cs0)
                rs.copy_(rs0)

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
                        attention_mask={
                            "full_attention": static_attn,
                            "linear_attention": None
                        },
                        past_key_values=ds_cache,
                        use_cache=True,
                    )
            get_accelerator().current_stream().wait_stream(s)
            restore_gdn_states()

            # Capture: full-step graph (forward + argmax + buffer updates).
            # The step-update kernel indexes token_buf via write_pos (a GPU
            # tensor the kernel itself advances), so no host-side step counter
            # is needed — the graph is fully self-contained for replay.
            graph_op = None
            try:
                from deepspeed.ops.module_inject import get_fused_glu_op
                candidate = get_fused_glu_op()
                if hasattr(candidate, "decode_step_graph"):
                    graph_op = candidate
            except Exception:
                graph_op = None

            graph = get_accelerator().create_graph()
            with get_accelerator().capture_to_graph(graph):
                out = module(
                    static_token,
                    attention_mask={
                        "full_attention": static_attn,
                        "linear_attention": None
                    },
                    past_key_values=ds_cache,
                    use_cache=True,
                )
                static_logits = out.logits
                if graph_op is not None:
                    graph_op.decode_step_graph(static_logits[:, -1, :].contiguous(), static_token.view(batch_size, 1),
                                               write_pos, static_attn, full_token_buf)

            if graph_op is not None:
                # The capture run advanced write_pos and mutated buffers;
                # restore to the pre-decode state before the replay loop.
                restore_gdn_states()
                write_pos.fill_(prompt_len)
                static_token.copy_(next_token)
        finally:
            module._forward_pre_hooks.update(saved_pre)
            module._forward_hooks.update(saved_post)

        # --- Decode loop: full-step graph > C++ loop > fused step > Python ---
        if graph_op is not None:
            # Full-step graph: one replay = forward + argmax + buffer updates.
            # Python only does replay + periodic EOS check (every 16 steps).
            eos_check_every = 16
            for step in range(max_new_tokens - 1):
                get_accelerator().replay_graph(graph)
                if step % eos_check_every == eos_check_every - 1:
                    tok_val = static_token.view(-1)[0].item()
                    if tok_val == eos_token_id:
                        full_token_buf[step + 2:] = pad_token_id
                        break
            gen_ids = full_token_buf[prompt_len:prompt_len + max_new_tokens].unsqueeze(0)
            return torch.cat([prompt_ids, gen_ids], dim=1)

        loop_op = None
        ds_op = None
        try:
            from deepspeed.ops.module_inject.decode_loop import get_decode_loop_op
            loop_op = get_decode_loop_op()
        except Exception:
            pass
        if loop_op is None:
            try:
                from deepspeed.ops.module_inject import get_fused_glu_op
                ds_op = get_fused_glu_op()
            except Exception:
                ds_op = None

        static_token.copy_(next_token)
        token_buf = torch.zeros(max_new_tokens, dtype=torch.long, device=device)
        token_buf[0] = next_token.squeeze(0)[0] if batch_size == 1 else next_token[0, 0]

        if loop_op is not None and hasattr(loop_op, "decode_loop"):
            loop_op.decode_loop(graph, static_logits[:,
                                                     -1, :].contiguous(), static_token.view(1,
                                                                                            1), write_pos, static_attn,
                                token_buf, max_new_tokens, eos_token_id if eos_token_id is not None else -1,
                                pad_token_id if pad_token_id is not None else 0, 16)
            gen_ids = token_buf.unsqueeze(0)
            return torch.cat([prompt_ids, gen_ids], dim=1)

        if ds_op is not None and hasattr(ds_op, "decode_step"):
            eos_check_every = 16
            for step in range(max_new_tokens - 1):
                get_accelerator().replay_graph(graph)
                ds_op.decode_step(static_logits[:, -1, :], static_token.view(batch_size, 1), write_pos, static_attn,
                                  token_buf, step + 1)
                if step % eos_check_every == eos_check_every - 1:
                    tok_val = static_token.view(-1)[0].item()
                    if tok_val == eos_token_id:
                        token_buf[step + 2:] = pad_token_id
                        break
            gen_ids = token_buf.unsqueeze(0)
            return torch.cat([prompt_ids, gen_ids], dim=1)

        eos_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for step in range(max_new_tokens - 1):
            if eos_mask.all():
                output_ids.append(torch.full((batch_size, 1), pad_token_id, dtype=torch.long, device=device))
                continue
            static_token.copy_(next_token)
            pos = prompt_len + step
            write_pos.fill_(pos)
            static_attn[:, :, :, pos + 1] = True
            get_accelerator().replay_graph(graph)
            next_token = static_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            output_ids.append(next_token)
            eos_mask |= (next_token.view(1) == eos_token_id)
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

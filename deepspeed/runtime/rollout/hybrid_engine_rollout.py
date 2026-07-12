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

from dataclasses import dataclass

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutEngine, RolloutRequest, SamplingConfig


@dataclass
class HybridEngineRolloutConfig:
    """Configuration for HybridEngineRollout."""
    use_graph_capture: bool = False


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

    @torch.no_grad()
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        device = request.prompt_ids.device
        B = request.prompt_ids.shape[0]
        n = sampling.n_samples_per_prompt
        total = B * n
        prompt_len = request.prompt_ids.shape[1]
        max_new_tokens = sampling.max_new_tokens
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        module = self.engine.module

        # Expand prompts for n samples per prompt
        if n > 1:
            prompt_ids = request.prompt_ids.repeat_interleave(n, dim=0)
            prompt_attn = request.prompt_attention_mask.repeat_interleave(n, dim=0)
        else:
            prompt_ids = request.prompt_ids
            prompt_attn = request.prompt_attention_mask

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

        # Build attention mask: pad positions (both left padding from prompt
        # and right padding from EOS / shorter sequences) are 0.
        full_len = output_ids.shape[1]
        response_start = prompt_len
        attention_mask = (output_ids != pad_token_id).long()
        for i in range(total):
            prompt_valid = request.prompt_attention_mask[i // n if B > 1 else 0]
            attention_mask[i, :prompt_len] = prompt_valid

        return RolloutBatch(
            input_ids=output_ids,
            attention_mask=attention_mask,
            response_start_idx=torch.full((total, ), response_start, dtype=torch.long, device=device),
        )

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

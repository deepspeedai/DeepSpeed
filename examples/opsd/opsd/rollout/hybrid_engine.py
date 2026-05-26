# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout backed by DeepSpeed's hybrid engine, with a ZeRO-3 fallback.

For architectures in DeepSpeed's inference-container policy list
(GPT2 / GPT-NeoX / OPT / BLOOM / LLAMA / LLAMA2 / InternLM as of 0.15) the
hybrid engine gives accelerated decode by swapping in optimized inference
kernels for the duration of the rollout. For everything else (Qwen2 / Qwen3
/ any model without a policy), no inference container is created and
``DeepSpeedHybridEngine.generate`` would AttributeError on its unbound
``_generate`` slot — so we detect that case at construction time and fall
back to a manual path that just gathers ZeRO-3 partitions and calls the
HuggingFace model's ``generate`` directly. Correct, just slower than the
accelerated path.
"""

import torch

from opsd.config import RolloutConfig
from opsd.rollout.base import RolloutBatch, RolloutEngine, RolloutRequest, SamplingConfig


def _hybrid_engine_has_accel(engine) -> bool:
    # The accelerated path is only wired up when at least one inference
    # container was populated for the model's layers. ``_inference_containers``
    # and ``_generate`` are both internal but they are the only two reliable
    # signals across DeepSpeed 0.14–0.19; ``_generate`` is bound exactly when
    # the container list is non-empty.
    return getattr(engine, "_generate", None) is not None


class HybridEngineRollout(RolloutEngine):
    name = "hybrid_engine"

    def __init__(self, student_engine, tokenizer, cfg: RolloutConfig):
        if cfg.engine != "hybrid_engine":
            raise ValueError(f"RolloutConfig.engine must be 'hybrid_engine'; got {cfg.engine!r}")
        self.engine = student_engine
        self.tokenizer = tokenizer
        self.cfg = cfg
        self._has_accel = _hybrid_engine_has_accel(student_engine)

    @torch.no_grad()
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # Many decoder-only tokenizers (Llama, Qwen) ship without a pad
            # token. Fall back to eos so that generate doesn't crash on the
            # left-padded prompts.
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

        # Hybrid engine expects training mode toggled off so the inference
        # containers take over. eval() is cheap (boolean flip + module walk).
        self.engine.eval()
        try:
            if self._has_accel:
                seqs = self.engine.generate(**gen_kwargs)
            else:
                seqs = self._fallback_generate(**gen_kwargs)
        finally:
            self.engine.train()

        # ``seqs`` is [B * n, T_p + T_r_actual], left-padded prompt + response.
        # With left-padded prompts every sample's response starts at column T_p.
        B = request.prompt_ids.shape[0]
        n = sampling.n_samples_per_prompt
        T_p = request.prompt_ids.shape[1]
        if seqs.shape[0] != B * n:
            raise RuntimeError(f"generate returned batch {seqs.shape[0]}, expected {B * n}")

        response_start_idx = torch.full((B * n, ), T_p, dtype=torch.long, device=seqs.device)

        # Response positions are anything past the prompt that is also not pad.
        attention_mask = (seqs != pad_id).to(request.prompt_attention_mask.dtype)
        # Keep the prompt portion of the mask aligned with what the caller
        # passed in (a prompt token equal to pad_id should still be attended);
        # for typical left-padded prompts the overlap is identical.
        prompt_mask_expanded = request.prompt_attention_mask.repeat_interleave(n, dim=0)
        attention_mask[:, :T_p] = prompt_mask_expanded

        return RolloutBatch(input_ids=seqs, attention_mask=attention_mask, response_start_idx=response_start_idx)

    def sync_weights_from_student(self, step: int) -> None:  # noqa: ARG002
        # The hybrid engine reads the student's live weights every generate
        # call, so there is nothing to sync.
        return None

    @torch.no_grad()
    def _fallback_generate(self, **gen_kwargs) -> torch.Tensor:
        """Manual ZeRO-3 generate for architectures the hybrid engine doesn't
        have an inference policy for.

        Walks every parameter into a ``GatheredParameters`` context so the full
        weight is materialized on each rank for the duration of generation,
        then calls the underlying HF model's ``generate``. Re-partitions on
        exit. This is correct but does not get the hybrid engine's optimized
        kernels — expect ~3-5x slower decode than the accelerated path.
        """
        from deepspeed.runtime.zero import GatheredParameters

        module = self.engine.module
        all_params = list(module.parameters())
        with GatheredParameters(all_params):
            return module.generate(**gen_kwargs)

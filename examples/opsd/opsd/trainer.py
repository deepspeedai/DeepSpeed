# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""On-policy distillation training loop.

Each step is three phases:

  0. **Rollout.** The student generates responses for the batch's prompts
     (via the configured :class:`~opsd.rollout.RolloutEngine` — hybrid engine
     or vLLM).
  1. **Teacher.** The frozen teacher runs a forward over prompt+response. The
     full logit tensor is parked on the host via
     :class:`~opsd.teacher.TeacherLogitCache` so teacher GPU buffers can be
     released before the student backward.
  2. **Student.** The student runs forward+backward on prompt+response. The
     loss is the per-token divergence to the teacher, streamed from the
     host-resident cache one sequence chunk at a time
     (:func:`~opsd.losses.streamed_distillation_loss`), so the full
     ``[B, T, V]`` teacher tensor never co-resides with the student logits on
     the training device.

The trainer itself contains no DeepSpeed-specific control flow beyond the
``backward`` / ``step`` calls on the student engine; backend choice (ZeRO
stage, offload, hybrid engine) is owned entirely by the DeepSpeed JSON config.
"""

import os
import time
from typing import Any

import torch
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator

from opsd.config import OPSDConfig
from opsd.losses import streamed_distillation_loss
from opsd.rollout import RolloutEngine, RolloutRequest, SamplingConfig
from opsd.utils import build_response_mask


def _is_rank_zero() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


class OPSDTrainer:

    def __init__(
        self,
        cfg: OPSDConfig,
        student_engine: Any,
        teacher: Any,
        tokenizer: Any,
        rollout: RolloutEngine,
        dataloader: Any,
    ):
        self.cfg = cfg
        self.student_engine = student_engine
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.rollout = rollout
        self.dataloader = dataloader

        self.device = get_accelerator().current_device_name()
        self.step = 0

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def train(self) -> None:
        max_steps = self.cfg.training.max_steps
        for epoch in range(self.cfg.training.num_train_epochs):
            for batch in self.dataloader:
                if max_steps > 0 and self.step >= max_steps:
                    return
                metrics = self._train_step(batch)
                self._maybe_log(metrics)
                self._maybe_save()
                self.step += 1
            if max_steps > 0 and self.step >= max_steps:
                return

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------

    def _train_step(self, batch) -> dict:
        t_start = time.time()

        prompt_ids = batch["prompt_ids"].to(self.device, non_blocking=True)
        prompt_attn = batch["prompt_attention_mask"].to(self.device, non_blocking=True)

        # Push student weights into the rollout backend if it's time to.
        # No-op for the hybrid engine; meaningful for vLLM.
        if self.step % self.cfg.rollout.weight_sync_interval == 0:
            self.rollout.sync_weights(self.step)

        # --- Phase 0: rollout (student generates responses) ---------------
        # Switch hybrid engine to inference mode (gathers ZeRO-3 params).
        self.student_engine.eval()
        sampling = SamplingConfig(
            max_new_tokens=self.cfg.rollout.max_response_length,
            temperature=self.cfg.rollout.temperature,
            top_p=self.cfg.rollout.top_p,
            top_k=self.cfg.rollout.top_k,
            n_samples_per_prompt=self.cfg.rollout.n_samples_per_prompt,
        )
        roll = self.rollout.generate(
            RolloutRequest(prompt_ids=prompt_ids, prompt_attention_mask=prompt_attn),
            sampling,
        )
        self.student_engine.train()
        input_ids = roll.input_ids.to(self.device, non_blocking=True)
        attention_mask = roll.attention_mask.to(self.device, non_blocking=True)
        response_start_idx = roll.response_start_idx.to(self.device, non_blocking=True)
        response_mask = build_response_mask(response_start_idx, attention_mask)
        t_rollout = time.time() - t_start

        # --- Phase 1: teacher forward → host-cached logits ----------------
        t1 = time.time()
        teacher_cache = self.teacher.forward_to_cache(input_ids, attention_mask)
        t_teacher = time.time() - t1

        # --- Phase 2: student forward + streamed KL + backward ------------
        t2 = time.time()
        self.student_engine.train()
        outputs = self.student_engine(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = outputs.logits  # [B, T, V]

        # Shift for next-token prediction: logits at position t predict token
        # at t+1, so the loss aligns student_logits[:, :-1] with the position
        # t+1 entries of the response mask.
        student_logits_shifted = student_logits[:, :-1, :]
        mask_shifted = response_mask[:, 1:].contiguous()

        def _fetch(start: int, end: int) -> torch.Tensor:
            # The cache holds *unshifted* teacher logits; for the next-token
            # objective we ask the cache for positions [start, end) of the
            # shifted teacher, which is positions [start, end) of the original
            # since we already lopped off the final column in the student.
            return teacher_cache.chunk_to_device(start,
                                                 end,
                                                 device=student_logits_shifted.device,
                                                 dtype=student_logits_shifted.dtype)

        loss = streamed_distillation_loss(
            student_logits=student_logits_shifted,
            teacher_chunk_fetcher=_fetch,
            response_mask=mask_shifted,
            loss_type=self.cfg.distillation.loss_type,
            temperature=self.cfg.distillation.temperature,
            chunk_size=self.cfg.distillation.chunk_size,
        )

        self.student_engine.backward(loss)
        self.student_engine.step()

        teacher_cache.free()
        t_student = time.time() - t2

        # Reduce loss across ranks for a clean log line.
        loss_for_log = loss.detach().clone()
        if dist.is_initialized():
            dist.all_reduce(loss_for_log)
            loss_for_log /= dist.get_world_size()

        return {
            "loss": float(loss_for_log.item()),
            "rollout_s": t_rollout,
            "teacher_s": t_teacher,
            "student_s": t_student,
            "step_s": time.time() - t_start,
            "response_tokens": int(mask_shifted.sum().item()),
        }

    # ------------------------------------------------------------------
    # Logging / checkpointing
    # ------------------------------------------------------------------

    def _maybe_log(self, metrics: dict) -> None:
        if self.step % self.cfg.training.logging_steps != 0:
            return
        if not _is_rank_zero():
            return
        print(f"[opsd][step {self.step}] loss={metrics['loss']:.4f} "
              f"rollout={metrics['rollout_s']:.2f}s teacher={metrics['teacher_s']:.2f}s "
              f"student={metrics['student_s']:.2f}s step={metrics['step_s']:.2f}s "
              f"resp_tok={metrics['response_tokens']}")

    def _maybe_save(self) -> None:
        if self.step == 0:
            return
        if self.step % self.cfg.training.save_steps != 0:
            return
        tag = f"step_{self.step}"
        os.makedirs(self.cfg.training.save_dir, exist_ok=True)
        self.student_engine.save_checkpoint(self.cfg.training.save_dir, tag=tag)
        if _is_rank_zero():
            print(f"[opsd] saved checkpoint to {self.cfg.training.save_dir}/{tag}")

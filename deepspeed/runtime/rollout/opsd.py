# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Response-token batching and distillation losses for OPSD-style training."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from deepspeed.runtime.rollout.base import RolloutBatch
from deepspeed.runtime.rollout.utils import left_pad_samples


@dataclass(frozen=True)
class ResponseTokenBatch:
    """Causal training targets derived from an on-policy rollout.

    ``target_ids`` and ``response_mask`` are shifted by one position relative
    to ``input_ids``: logits at position ``i`` predict ``target_ids[:, i]``.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_ids: torch.Tensor
    response_mask: torch.Tensor

    @classmethod
    def from_rollout(cls,
                     rollout: RolloutBatch,
                     pad_token_id: int = 0,
                     device: Optional[torch.device] = None) -> "ResponseTokenBatch":
        """Derive response-only causal targets from ``rollout``.

        Samples are left-padded to one width with ``pad_token_id``. A sample's
        response is its last ``response_length`` tokens; within it, tokens with
        ``loss_mask`` 0 (e.g. multi-turn tool output) are not targets.
        """
        input_ids, attention_mask = left_pad_samples(rollout.samples, pad_token_id)
        width = input_ids.shape[1]
        # 1 where the token is a training target; rows are left-padded, so each response ends at the last column.
        target_weights = torch.zeros_like(attention_mask)
        for row, sample in enumerate(rollout.samples):
            prompt_length = len(sample.tokens) - sample.response_length
            # The first token has no preceding logit, so it can never be a target.
            if sample.response_length < 0 or prompt_length < 1:
                raise ValueError(f"samples[{row}] response_length must leave at least one prompt token")
            loss_mask = sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length
            if len(loss_mask) != sample.response_length:
                raise ValueError(f"samples[{row}] loss_mask length {len(loss_mask)} does not match "
                                 f"response_length {sample.response_length}")
            response_start = width - sample.response_length
            target_weights[row, response_start:] = torch.tensor(loss_mask, dtype=torch.long)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        response_mask = attention_mask[:, 1:].bool() & target_weights[:, 1:].to(device).bool()
        return cls(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_ids=input_ids[:, 1:],
            response_mask=response_mask,
        )

    def select_causal_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Align model logits with this batch's shifted response targets."""
        if logits.dim() != 3:
            raise ValueError("logits must be 3-D [batch, sequence_length, vocabulary_size]")
        if logits.shape[:2] != self.input_ids.shape:
            raise ValueError("logits batch and sequence dimensions must match input_ids")
        return logits[:, :-1, :]


@dataclass(frozen=True)
class JSDLossOutput:
    """JSD loss and its unreduced statistics for distributed aggregation."""

    loss: torch.Tensor
    loss_sum: torch.Tensor
    valid_token_count: torch.Tensor


def generalized_jsd_loss(student_logits: torch.Tensor,
                         teacher_logits: torch.Tensor,
                         response_mask: torch.Tensor,
                         beta: float = 0.5,
                         temperature: float = 1.0,
                         teacher_top_k: Optional[int] = None,
                         pointwise_clip: Optional[float] = None) -> JSDLossOutput:
    """Compute generalized JSD on valid response tokens.

    ``beta=0`` yields forward KL from teacher to student; ``beta=1`` yields
    reverse KL from student to teacher. A positive ``teacher_top_k`` restricts
    both distributions to the teacher's most likely vocabulary entries before
    they are normalized. ``pointwise_clip`` caps individual vocabulary-level
    divergence contributions before their per-token reduction.
    """
    if student_logits.shape != teacher_logits.shape or student_logits.dim() != 3:
        raise ValueError("student_logits and teacher_logits must have matching 3-D shapes")
    if response_mask.shape != student_logits.shape[:2]:
        raise ValueError("response_mask shape must match logits batch and sequence dimensions")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if teacher_top_k is not None and teacher_top_k <= 0:
        raise ValueError("teacher_top_k must be positive when specified")
    if teacher_top_k is not None and teacher_top_k > student_logits.shape[-1]:
        raise ValueError("teacher_top_k must not exceed the vocabulary size")
    if pointwise_clip is not None and pointwise_clip < 0.0:
        raise ValueError("pointwise_clip must be non-negative when specified")

    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    if teacher_top_k is not None:
        teacher_top_indices = torch.topk(teacher_logits, k=teacher_top_k, dim=-1).indices
        student_logits = torch.gather(student_logits, dim=-1, index=teacher_top_indices)
        teacher_logits = torch.gather(teacher_logits, dim=-1, index=teacher_top_indices)

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()

    if beta == 0.0:
        pointwise_jsd = teacher_probs * (teacher_log_probs - student_log_probs)
    elif beta == 1.0:
        pointwise_jsd = student_probs * (student_log_probs - teacher_log_probs)
    else:
        mixture_log_probs = torch.logaddexp(
            student_log_probs +
            torch.log1p(torch.tensor(-beta, dtype=student_logits.dtype, device=student_logits.device)),
            teacher_log_probs +
            torch.log(torch.tensor(beta, dtype=student_logits.dtype, device=student_logits.device)))
        pointwise_jsd = beta * teacher_probs * (teacher_log_probs - mixture_log_probs)
        pointwise_jsd += (1.0 - beta) * student_probs * (student_log_probs - mixture_log_probs)

    if pointwise_clip is not None:
        pointwise_jsd = pointwise_jsd.clamp(max=pointwise_clip)

    response_mask = response_mask.to(dtype=pointwise_jsd.dtype)
    loss_sum = (pointwise_jsd.sum(dim=-1) * response_mask).sum()
    valid_token_count = response_mask.sum().to(dtype=torch.long)
    loss = loss_sum / valid_token_count.clamp_min(1)
    return JSDLossOutput(loss=loss, loss_sum=loss_sum, valid_token_count=valid_token_count)

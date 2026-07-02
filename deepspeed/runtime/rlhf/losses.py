# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Per-token distillation divergences with sequence-axis chunking.

The full ``[B, T, V]`` tensor produced by a forward pass on a modern LLM can
easily exceed several GB in fp32 (e.g. 8 * 1024 * 150k * 4 B ~ 4.9 GB). Holding
both student *and* teacher logits at once would double that. We chunk along the
sequence axis so the per-chunk softmax + difference only ever needs
``[B, chunk, V]`` of working memory, regardless of T.

Math conventions:
    * ``forward_kl``  = D_KL(teacher || student) — mode-covering for student
    * ``reverse_kl``  = D_KL(student || teacher) — mode-seeking for student
    * ``jsd``         = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M), M = (P+Q)/2

All three follow the standard knowledge-distillation temperature convention:
divide logits by T before softmax, then multiply the result by T**2 so that
gradient magnitudes are comparable across temperatures.
"""

from typing import Callable

import torch
import torch.nn.functional as F


def _forward_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    t_probs = t_log_probs.exp()
    kl = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
    return kl * (temperature**2)


def _reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_probs = s_log_probs.exp()
    kl = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)
    return kl * (temperature**2)


def _jsd(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_probs = s_log_probs.exp()
    t_probs = t_log_probs.exp()
    m_probs = 0.5 * (s_probs + t_probs)
    # Clamp guards against log(0) when both distributions have ~0 mass on the
    # same vocab id (rare in practice but possible after temperature scaling).
    m_log_probs = m_probs.clamp_min(1e-12).log()
    kl_s = (s_probs * (s_log_probs - m_log_probs)).sum(dim=-1)
    kl_t = (t_probs * (t_log_probs - m_log_probs)).sum(dim=-1)
    return 0.5 * (kl_s + kl_t) * (temperature**2)


_LOSS_FNS: "dict[str, Callable[..., torch.Tensor]]" = {
    "forward_kl": _forward_kl,
    "reverse_kl": _reverse_kl,
    "jsd": _jsd,
}


def chunked_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    loss_type: str = "reverse_kl",
    temperature: float = 1.0,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Mean per-token divergence over response positions, chunked over the
    sequence axis to bound peak memory.

    Args:
        student_logits: ``[B, T, V]`` — gradient flows here.
        teacher_logits: ``[B, T, V]`` — caller is responsible for ``detach()``
            (we do not detach here so the function stays cheap).
        response_mask:  ``[B, T]`` — 1 where the position should contribute to
            the loss (i.e. response tokens, not prompt or padding), 0 elsewhere.
        loss_type:      ``"forward_kl"`` | ``"reverse_kl"`` | ``"jsd"``.
        temperature:    KD temperature; >1 softens both distributions.
        chunk_size:     Sequence-axis chunk size.

    Returns:
        Scalar loss = sum-over-positions(per_tok * mask) / sum(mask), promoted
        to fp32 internally for numerical stability.
    """
    if loss_type not in _LOSS_FNS:
        raise ValueError(f"Unknown loss_type {loss_type!r}; choose from {sorted(_LOSS_FNS)}")
    fn = _LOSS_FNS[loss_type]

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(f"shape mismatch: student {tuple(student_logits.shape)} vs teacher "
                         f"{tuple(teacher_logits.shape)}")
    B, T, _ = student_logits.shape
    if response_mask.shape != (B, T):
        raise ValueError(f"response_mask {tuple(response_mask.shape)} does not match logits "
                         f"prefix ({B}, {T})")

    mask_f = response_mask.to(torch.float32)
    total_tokens = mask_f.sum().clamp_min(1.0)
    total_loss = student_logits.new_zeros((), dtype=torch.float32)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_mask = mask_f[:, start:end]
        # Skipping empty chunks avoids a redundant forward through the softmax
        # path on chunks that wouldn't contribute anything to the sum.
        if chunk_mask.sum().item() == 0:
            continue
        per_tok = fn(
            student_logits[:, start:end].float(),
            teacher_logits[:, start:end].float(),
            temperature,
        )
        total_loss = total_loss + (per_tok * chunk_mask).sum()

    return total_loss / total_tokens


def streamed_distillation_loss(
    student_logits: torch.Tensor,
    teacher_chunk_fetcher: Callable[[int, int], torch.Tensor],
    response_mask: torch.Tensor,
    loss_type: str = "reverse_kl",
    temperature: float = 1.0,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Same math as :func:`chunked_distillation_loss`, but teacher logits are
    pulled chunk-by-chunk via a fetcher so the full ``[B, T, V]`` teacher
    tensor never needs to live on the same device as the student.

    Args:
        student_logits: ``[B, T, V]`` on the training device.
        teacher_chunk_fetcher: ``fn(start, end) -> [B, end - start, V]``, already
            on the same device and broadcastable dtype as ``student_logits``.
            Typically wraps ``TeacherLogitCache.chunk_to_device``.
        response_mask:  ``[B, T]`` — 1 where the position should contribute.
        loss_type:      one of ``"forward_kl" | "reverse_kl" | "jsd"``.
        temperature:    KD temperature.
        chunk_size:     Sequence-axis chunk size.
    """
    if loss_type not in _LOSS_FNS:
        raise ValueError(f"Unknown loss_type {loss_type!r}; choose from {sorted(_LOSS_FNS)}")
    fn = _LOSS_FNS[loss_type]

    B, T, _ = student_logits.shape
    if response_mask.shape != (B, T):
        raise ValueError(f"response_mask {tuple(response_mask.shape)} does not match logits "
                         f"prefix ({B}, {T})")

    mask_f = response_mask.to(torch.float32)
    total_tokens = mask_f.sum().clamp_min(1.0)
    total_loss = student_logits.new_zeros((), dtype=torch.float32)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_mask = mask_f[:, start:end]
        if chunk_mask.sum().item() == 0:
            continue
        teacher_chunk = teacher_chunk_fetcher(start, end)
        if teacher_chunk.shape[1] != (end - start):
            raise RuntimeError(f"fetcher returned chunk of length {teacher_chunk.shape[1]}, "
                               f"expected {end - start}")
        per_tok = fn(
            student_logits[:, start:end].float(),
            teacher_chunk.float(),
            temperature,
        )
        total_loss = total_loss + (per_tok * chunk_mask).sum()

    return total_loss / total_tokens


def per_token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Gather log p(label_t | context_<t) for each position.

    Convenience helper used by the trainer for diagnostic logging (e.g. mean
    student log-prob of the rollout, mean teacher log-prob of the rollout).

    Args:
        logits: ``[B, T, V]``.
        labels: ``[B, T]`` token ids.

    Returns:
        ``[B, T]`` of log-probabilities.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

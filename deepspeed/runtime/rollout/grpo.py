# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Group-relative advantages and the PPO-clip policy loss for GRPO training."""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from deepspeed.runtime.rollout.base import RolloutBatch, RolloutRecord

_AGGREGATIONS = ("token_mean", "sample_mean")


def group_advantages(records: Sequence[RolloutRecord],
                     rewards: Sequence[float],
                     scale_by_std: bool = True,
                     eps: float = 1e-6) -> torch.Tensor:
    """GRPO advantages ``[len(records)]``: each reward minus its group's mean reward.

    A group is the records that share a ``group_index``, i.e. the responses to
    one prompt. With ``scale_by_std`` the difference is also divided by the
    group's standard deviation (plus ``eps``). A group of one record, or of
    equal rewards, gets advantage 0.
    """
    if len(records) != len(rewards):
        raise ValueError(f"got {len(rewards)} rewards for {len(records)} records")
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    group_ids = [record.group_index for record in records]
    if any(group_id is None for group_id in group_ids):
        raise ValueError("every record needs a group_index; rollouts set it to the record's prompt")

    advantages = torch.zeros_like(rewards)
    for group_id in set(group_ids):
        rows = torch.tensor([row for row, other in enumerate(group_ids) if other == group_id])
        group_rewards = rewards[rows]
        centered = group_rewards - group_rewards.mean()
        # Equal rewards leave every difference 0, which no scale changes.
        if scale_by_std and len(rows) > 1 and centered.any():
            centered = centered / (group_rewards.std() + eps)
        advantages[rows] = centered
    return advantages


def old_log_probs_from_rollout(rollout: RolloutBatch, device: Optional[torch.device] = None) -> torch.Tensor:
    """The records' ``rollout_log_probs`` laid out like ``ResponseTokenBatch.from_rollout``'s targets.

    Column ``i`` holds the log-prob of target ``i`` (token ``i + 1``). Rows are
    left-padded, so each response ends at the last column. Positions outside a
    response are 0; use them only under ``response_mask``.
    """
    width = max(len(record.tokens) for record in rollout.samples)
    old_log_probs = torch.zeros((len(rollout.samples), width - 1), dtype=torch.float32)
    for row, record in enumerate(rollout.samples):
        if record.response_length == 0:
            continue
        if record.rollout_log_probs is None:
            raise ValueError(f"samples[{row}] has no rollout_log_probs; generate with return_log_probs=True")
        if len(record.rollout_log_probs) != record.response_length:
            raise ValueError(f"samples[{row}] rollout_log_probs length {len(record.rollout_log_probs)} does not "
                             f"match response_length {record.response_length}")
        old_log_probs[row, width - 1 - record.response_length:] = torch.tensor(record.rollout_log_probs)
    return old_log_probs.to(device)


@dataclass(frozen=True)
class GRPOLossOutput:
    """GRPO policy loss, its unreduced statistics for distributed aggregation, and logging metrics.

    ``loss`` is ``loss_sum / loss_normalizer``. For ``token_mean`` the
    normalizer counts response tokens; for ``sample_mean`` it counts samples
    with at least one response token. Summing ``loss_sum`` and
    ``loss_normalizer`` across ranks gives the loss of the global batch.
    """

    loss: torch.Tensor
    loss_sum: torch.Tensor
    loss_normalizer: torch.Tensor
    valid_token_count: torch.Tensor
    # Fraction of response tokens whose ratio the clip bounded.
    clip_fraction: torch.Tensor
    # Mean per-token KL to the reference policy; None without ``ref_log_probs``.
    kl: Optional[torch.Tensor]


def grpo_policy_loss(log_probs: torch.Tensor,
                     old_log_probs: torch.Tensor,
                     advantages: torch.Tensor,
                     response_mask: torch.Tensor,
                     clip_low: float = 0.2,
                     clip_high: Optional[float] = None,
                     ref_log_probs: Optional[torch.Tensor] = None,
                     kl_coef: float = 0.0,
                     aggregation: str = "token_mean") -> GRPOLossOutput:
    """PPO-clip loss on per-token log-probs, as GRPO uses it.

    ``log_probs``, ``old_log_probs`` and the optional ``ref_log_probs`` are
    ``[B, T]`` log-probs of the target tokens under the current, rollout and
    reference policies; ``response_mask`` ``[B, T]`` selects the tokens to
    train on (``ResponseTokenBatch.response_mask`` excludes prompts, padding
    and tool output). ``advantages`` is ``[B]``.

    The ratio is clipped to ``[1 - clip_low, 1 + clip_high]``; ``clip_high``
    defaults to ``clip_low`` (set it higher for DAPO's clip-higher). With
    ``ref_log_probs``, the k3 estimator of KL(current || reference), weighted
    by ``kl_coef``, is added to each token's loss.
    """
    if log_probs.dim() != 2:
        raise ValueError("log_probs must be [batch, tokens]")
    for name, tensor in (("old_log_probs", old_log_probs), ("response_mask", response_mask)):
        if tensor.shape != log_probs.shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} must match log_probs {tuple(log_probs.shape)}")
    if ref_log_probs is not None and ref_log_probs.shape != log_probs.shape:
        raise ValueError("ref_log_probs shape must match log_probs")
    if advantages.shape != log_probs.shape[:1]:
        raise ValueError("advantages must be [batch]")
    if clip_low < 0.0 or (clip_high is not None and clip_high < 0.0):
        raise ValueError("clip ranges must be non-negative")
    if kl_coef != 0.0 and ref_log_probs is None:
        raise ValueError("kl_coef needs ref_log_probs")
    if aggregation not in _AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {', '.join(_AGGREGATIONS)}")
    if clip_high is None:
        clip_high = clip_low

    mask = response_mask.to(dtype=log_probs.dtype)
    advantages = advantages.to(dtype=log_probs.dtype).unsqueeze(1)
    ratio = torch.exp(log_probs - old_log_probs)
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
    token_loss = torch.maximum(unclipped, clipped)

    kl = None
    if ref_log_probs is not None:
        # The k3 estimator (Schulman, "Approximating KL Divergence"): non-negative and low-variance.
        log_ratio = ref_log_probs - log_probs
        token_kl = torch.exp(log_ratio) - log_ratio - 1.0
        token_loss = token_loss + kl_coef * token_kl

    valid_token_count = mask.sum().to(dtype=torch.long)
    token_normalizer = valid_token_count.clamp_min(1).to(dtype=log_probs.dtype)
    if ref_log_probs is not None:
        kl = (token_kl.detach() * mask).sum() / token_normalizer
    clip_fraction = ((clipped > unclipped).to(dtype=log_probs.dtype) * mask).sum().detach() / token_normalizer

    if aggregation == "token_mean":
        loss_sum = (token_loss * mask).sum()
        loss_normalizer = valid_token_count
    else:
        sample_tokens = mask.sum(dim=1)
        has_tokens = sample_tokens > 0
        # A sample with no response tokens (cut before its first turn) adds nothing.
        sample_loss = (token_loss * mask).sum(dim=1) / sample_tokens.clamp_min(1)
        loss_sum = sample_loss[has_tokens].sum()
        loss_normalizer = has_tokens.sum().to(dtype=torch.long)
    loss = loss_sum / loss_normalizer.clamp_min(1)
    return GRPOLossOutput(loss=loss,
                          loss_sum=loss_sum,
                          loss_normalizer=loss_normalizer,
                          valid_token_count=valid_token_count,
                          clip_fraction=clip_fraction,
                          kl=kl)

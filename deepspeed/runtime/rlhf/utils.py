# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Small tensor/masking helpers shared by trainer, losses, and tests.

These intentionally stay free of DeepSpeed / distributed imports so the
non-distributed unit tests can exercise them on CPU without a torchrun
launcher.
"""

import torch


def build_response_mask(response_start_idx: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mark positions belonging to the response (not prompt, not padding).

    Args:
        response_start_idx: ``[B]`` int tensor — the first column index that is
            part of the response, per sample. For *right-padded* prompts this
            equals the prompt's token count; for the more common *left-padded*
            convention used by causal generation it equals the prompt section
            length (i.e. the column where prompt ends and response begins).
        attention_mask: ``[B, T]`` — 1 on real tokens (prompt + response), 0 on
            padding.

    Returns:
        ``[B, T]`` 0/1 mask with the same dtype as ``attention_mask``. 1 only
        at positions ``t >= response_start_idx[b]`` that are also attended.
    """
    if response_start_idx.dim() != 1:
        raise ValueError(f"response_start_idx must be 1-D, got shape {tuple(response_start_idx.shape)}")
    if attention_mask.dim() != 2:
        raise ValueError(f"attention_mask must be 2-D, got shape {tuple(attention_mask.shape)}")
    B, T = attention_mask.shape
    if response_start_idx.shape[0] != B:
        raise ValueError(f"response_start_idx batch ({response_start_idx.shape[0]}) != "
                         f"attention_mask batch ({B})")

    pos = torch.arange(T, device=attention_mask.device).unsqueeze(0).expand(B, T)
    is_response = pos >= response_start_idx.to(pos.dtype).unsqueeze(1)
    return is_response.to(attention_mask.dtype) * attention_mask


def shift_for_next_token_prediction(logits: torch.Tensor, labels: torch.Tensor):
    """Align logits at position t with the label at position t+1.

    Returns:
        Tuple ``(shifted_logits[:, :-1, :], shifted_labels[:, 1:])`` — both
        contiguous, so they can be safely indexed for the divergence loss.
    """
    return logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Helpers for expert token counting in AutoEP routing paths."""

import torch

from deepspeed.accelerator import get_accelerator


def count_tokens_per_expert(
    selected_experts_indices: torch.Tensor,
    num_experts: int,
    *,
    out_dtype: torch.dtype = torch.float32,
    deterministic_safe: bool = False,
) -> torch.Tensor:
    """Count routed tokens per expert.

    Fast path uses ``torch.bincount`` on the current device.
    If ``deterministic_safe=True`` and deterministic algorithms are enabled
    on CUDA, this falls back to CPU bincount to avoid non-deterministic kernel
    restrictions.
    """
    flat_indices = selected_experts_indices.reshape(-1).to(torch.int64)

    if deterministic_safe and torch.are_deterministic_algorithms_enabled() and get_accelerator().on_accelerator(
            flat_indices):
        counts = torch.bincount(flat_indices.detach().cpu(), minlength=num_experts)
        counts = counts.to(selected_experts_indices.device)
    else:
        counts = torch.bincount(flat_indices, minlength=num_experts)

    if counts.numel() < num_experts:
        pad = torch.zeros(num_experts - counts.numel(), device=counts.device, dtype=counts.dtype)
        counts = torch.cat([counts, pad], dim=0)
    elif counts.numel() > num_experts:
        counts = counts[:num_experts]

    return counts.to(out_dtype)

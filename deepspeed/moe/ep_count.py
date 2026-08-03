# Copyright (c) DeepSpeed Team.
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

    The fast path scatter-adds into a fixed ``[num_experts]`` buffer. Because the
    output shape is known up front, it avoids the device-to-host synchronization
    that ``torch.bincount`` incurs to size its output from ``max_index + 1``.

    Counting integers is order-independent, so the result is deterministic even
    though ``scatter_add_`` accumulates with atomics: integer addition is
    associative, unlike float addition. The ``deterministic_safe`` fallback to
    CPU bincount exists only because PyTorch's categorical determinism guard may
    reject atomic scatter on some builds, not because the counts would differ.
    """
    flat_indices = selected_experts_indices.reshape(-1).to(torch.int64)

    if deterministic_safe and torch.are_deterministic_algorithms_enabled() and get_accelerator().on_accelerator(
            flat_indices):
        counts = torch.bincount(flat_indices.detach().cpu(), minlength=num_experts)
        counts = counts.to(selected_experts_indices.device)
    else:
        counts = torch.zeros(num_experts, dtype=torch.int64, device=flat_indices.device)
        counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices))

    if counts.numel() < num_experts:
        pad = torch.zeros(num_experts - counts.numel(), device=counts.device, dtype=counts.dtype)
        counts = torch.cat([counts, pad], dim=0)
    elif counts.numel() > num_experts:
        counts = counts[:num_experts]

    return counts.to(out_dtype)

# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Helpers for expert token counting in AutoEP routing paths."""

import torch


def count_tokens_per_expert(
    selected_experts_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Count routed tokens per expert.

    Because the output shape is known up front, it avoids the device-to-host synchronization
    that ``torch.bincount`` incurs to size its output from ``max_index + 1``.
    """
    flat_indices = selected_experts_indices.reshape(-1)

    counts = torch.zeros(num_experts, dtype=torch.int32, device=flat_indices.device)
    counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=counts.dtype))

    return counts

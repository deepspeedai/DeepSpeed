# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Token count utilities for expert parallelism."""

import torch


def count_tokens_per_expert(
    selected_experts: torch.Tensor,
    num_experts: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Count the number of tokens routed to each expert.

    Args:
        selected_experts: Expert indices per token, shape ``(T, top_k)`` or ``(N,)``.
        num_experts: Total number of experts (global, before EP slicing).
        out_dtype: Output dtype.  Defaults to float32 because ``torch.histc``
            requires float input on CPU.

    Returns:
        Token-count histogram, shape ``(num_experts,)``.
    """
    return torch.histc(
        selected_experts.view(-1).float(),
        bins=num_experts,
        min=0,
        max=num_experts,
    ).to(out_dtype)

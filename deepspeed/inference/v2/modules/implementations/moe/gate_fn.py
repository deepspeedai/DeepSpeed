# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn.functional as F
from torch import Tensor

from typing import Tuple

#TODO(cmikeh2): DELETE


@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[-1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]


def top1gating(logits: Tensor,
               capacity_factor: float,
               min_capacity: int,
               drop_tokens: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    assert logits.shape[
        0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."

    top_idx = _top_idx(mask1, capacity)

    mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)

    indices_mask = mask1.sum(dim=1) * num_experts - 1
    indices1_s = torch.min(indices1_s, indices_mask)

    gates1_s = (gates * mask1).sum(dim=1)

    return indices1_s, gates1_s

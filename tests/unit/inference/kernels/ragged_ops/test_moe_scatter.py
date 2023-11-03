# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.inference_utils import DtypeEnum
from deepspeed.inference.v2.kernels.ragged_ops import MoEScatter, RaggedTop1Gating
from .ragged_testing_utils import build_simple_batch
"""
For simplicity's sake, these tests do rely on ``RaggedTop1Gating`` to produce correct
inputs. If ``RaggedTop1Gating`` is broken, these tests will fail, so double check
the unit test results there before debugging here.
"""


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("n_tokens, n_experts", [(13, 64), (278, 64), (1977, 64)])
@pytest.mark.parametrize("do_padding", [True, False])
def test_moe_scatter(n_tokens, n_experts, do_padding):

    # Sequence composition shouldn't matter here
    batch = build_simple_batch([n_tokens], padding=do_padding)

    logits = torch.randn((batch.tensor_toks, n_experts),
                         dtype=torch.float16,
                         device=get_accelerator().current_device())

    # This will make each token's value equal to its index. NOTE: This will break for
    # tokens with index > 2048.
    hidden_states = torch.arange(batch.tensor_toks, dtype=torch.float16,
                                 device=get_accelerator().current_device()).repeat_interleave(4096, dim=0).reshape(
                                     batch.tensor_toks, 4096).contiguous()

    gate = RaggedTop1Gating(DtypeEnum.fp16)

    # Gating outputs
    expert_counts = torch.zeros((n_experts, ), dtype=torch.int32, device=get_accelerator().current_device())
    scores = torch.empty((batch.tensor_toks, ), dtype=torch.float32, device=get_accelerator().current_device())
    expert_assignment = torch.empty((batch.tensor_toks, ),
                                    dtype=torch.int32,
                                    device=get_accelerator().current_device())
    expert_offset = torch.empty((batch.tensor_toks, ), dtype=torch.int32, device=get_accelerator().current_device())

    gate(expert_counts, scores, expert_assignment, expert_offset, logits, batch)

    # Scatter outputs
    moe_input = torch.empty((batch.tensor_toks, 4096), dtype=torch.float16, device=get_accelerator().current_device())
    expert_cumsum = torch.empty((n_experts, ), dtype=torch.int64, device=get_accelerator().current_device())
    mapped_slots = torch.empty((batch.tensor_toks, ), dtype=torch.int32, device=get_accelerator().current_device())

    scatter = MoEScatter(DtypeEnum.fp16, 4096)
    scatter(moe_input, expert_cumsum, mapped_slots, hidden_states, expert_counts, expert_assignment, expert_offset)
    assert torch.equal(expert_cumsum, torch.cumsum(expert_counts, dim=0).to(torch.int64))

    for token_idx in range(batch.tensor_toks):
        if token_idx < n_tokens:
            expert_idx = expert_assignment[token_idx].item()
            if expert_idx == 0:
                expert_cumsum_val = 0
            else:
                expert_cumsum_val = expert_cumsum[expert_idx - 1]
            offset = expert_offset[token_idx]
            total_offset = offset + expert_cumsum_val

            assert total_offset == mapped_slots[token_idx].item()
            assert torch.equal(moe_input[total_offset], hidden_states[token_idx])
        else:
            assert mapped_slots[token_idx].item() == -1

    assert expert_cumsum[-1] == n_tokens

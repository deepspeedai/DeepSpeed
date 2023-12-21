# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.inference_utils import DtypeEnum
from deepspeed.inference.v2.kernels.ragged_ops import MoEScatter, RaggedTopKGating
from .ragged_testing_utils import build_simple_batch
"""
For simplicity's sake, these tests do rely on ``RaggedTopKGating`` to produce correct
inputs. If ``RaggedTopKGating`` is broken, these tests will fail, so double check
the unit test results there before debugging here.
"""

TEST_CONFIGS = [
    (13, 64, 1),
    (278, 64, 1),
    (1977, 64, 1),
    (13, 8, 2),
    (278, 8, 2),
    (1977, 8, 2),
]


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("n_tokens, n_experts, n_top_k", TEST_CONFIGS)
@pytest.mark.parametrize("do_padding", [False, True])
def test_moe_scatter(n_tokens, n_experts, n_top_k, do_padding):

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

    gate = RaggedTopKGating(DtypeEnum.fp16)

    # Gating outputs
    expert_counts = torch.zeros((n_experts, ), dtype=torch.int32, device=get_accelerator().current_device())
    scores = torch.empty((batch.tensor_toks, n_top_k), dtype=torch.float32, device=get_accelerator().current_device())
    expert_assignment = torch.empty((batch.tensor_toks, n_top_k),
                                    dtype=torch.int32,
                                    device=get_accelerator().current_device())
    expert_offset = torch.empty((batch.tensor_toks, n_top_k),
                                dtype=torch.int32,
                                device=get_accelerator().current_device())

    gate(expert_counts, scores, expert_assignment, expert_offset, logits, batch)

    # Scatter outputs
    moe_input = torch.empty((batch.tensor_toks * n_top_k, 4096),
                            dtype=torch.float16,
                            device=get_accelerator().current_device())
    expert_cumsum = torch.empty((n_experts, ), dtype=torch.int64, device=get_accelerator().current_device())
    mapped_slots = torch.empty((batch.tensor_toks, n_top_k),
                               dtype=torch.int32,
                               device=get_accelerator().current_device())

    scatter = MoEScatter(DtypeEnum.fp16, 4096)
    scatter(moe_input, expert_cumsum, mapped_slots, hidden_states, expert_counts, expert_assignment, expert_offset)
    get_accelerator().synchronize()
    assert torch.equal(expert_cumsum, torch.cumsum(expert_counts, dim=0).to(torch.int64))

    if not do_padding:
        assert torch.unique(mapped_slots).size(0) == n_top_k * n_tokens

    for token_idx in range(batch.tensor_toks):
        if token_idx < n_tokens:
            for k in range(n_top_k):
                expert_idx = expert_assignment[token_idx][k].item()
                if expert_idx == 0:
                    expert_cumsum_val = 0
                else:
                    expert_cumsum_val = expert_cumsum[expert_idx - 1]
                offset = expert_offset[token_idx][k]
                total_offset = offset + expert_cumsum_val

                assert total_offset == mapped_slots[token_idx][k].item()
                assert torch.equal(moe_input[total_offset], hidden_states[token_idx])
        else:
            for k in range(n_top_k):
                assert mapped_slots[token_idx][k].item() == -1

    assert expert_cumsum[-1] == n_tokens * n_top_k

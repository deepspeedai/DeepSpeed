# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import torch.nn.functional as F

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.inference_utils import DtypeEnum
from deepspeed.inference.v2.kernels.ragged_ops import RaggedTop1Gating
from .ragged_testing_utils import build_simple_batch
from ....v2.inference_test_utils import allclose


def _test_single_mapping_helper(n_tokens: int,
                                n_experts: int,
                                assigned_expert: int,
                                logit_fill: float = 0.0,
                                match_fill: float = 1.0) -> None:
    logits = torch.full((n_tokens, n_experts),
                        logit_fill,
                        dtype=torch.float16,
                        device=get_accelerator().current_device())

    logits[:, assigned_expert] = match_fill

    gate = RaggedTop1Gating(DtypeEnum.fp16)

    expert_counts = torch.zeros((n_experts, ), dtype=torch.int32, device=get_accelerator().current_device())
    scores = torch.empty((n_tokens, ), dtype=torch.float32, device=get_accelerator().current_device())
    expert_assignment = torch.empty((n_tokens, ), dtype=torch.int32, device=get_accelerator().current_device())
    expert_offset = torch.empty((n_tokens, ), dtype=torch.int32, device=get_accelerator().current_device())
    batch = build_simple_batch([n_tokens], padding=False)

    gate(expert_counts, scores, expert_assignment, expert_offset, logits, batch)

    assert expert_counts[assigned_expert] == n_tokens
    assert torch.all(expert_assignment == assigned_expert)
    assert torch.unique(expert_offset).shape[0] == n_tokens
    assert allclose(scores, F.softmax(logits.float(), dim=1)[:, assigned_expert])


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize('n_tokens, n_experts', [(1, 16), (17, 16), (32, 128), (89, 128), (433, 128)])
def test_single_mapping_gating(n_tokens: int, n_experts: int) -> None:
    """
    Evaluate our expert stacking behavior in complete isolation. This ensures all tokens
    mapped to the same expert are getting unique offsets and identical scores.
    """
    assigned_expert = 13
    _test_single_mapping_helper(n_tokens, n_experts, assigned_expert)


@pytest.mark.inference_v2_ops
def test_negative_logits():
    """
    Ensure that scores/values are propagated correctly when all the logits are negative. An
    earlier implementation of the scoring would return NaN for this case.
    """
    _test_single_mapping_helper(128, 32, 13, logit_fill=-2.0, match_fill=-1.0)


@pytest.mark.inference_v2_ops
def test_determinism():
    """
    Ensure that ties between two logits are broken deterministically. This is essential when
    the gating is distributed across multiple devices that need to map the same token to
    the same expert.
    """

    n_tokens = 512
    n_experts = 64

    logits = torch.zeros((n_tokens, n_experts), dtype=torch.float16, device=get_accelerator().current_device())
    batch = build_simple_batch([n_tokens], padding=False)

    logits[:, 19] = 1.0
    logits[:, 26] = 1.0

    gate = RaggedTop1Gating(DtypeEnum.fp16)

    for _ in range(1024):
        expert_counts = torch.zeros((n_experts, ), dtype=torch.int32, device=get_accelerator().current_device())
        scores = torch.empty((n_tokens, ), dtype=torch.float32, device=get_accelerator().current_device())
        expert_assignment = torch.empty((n_tokens, ), dtype=torch.int32, device=get_accelerator().current_device())
        expert_offset = torch.empty((n_tokens, ), dtype=torch.int32, device=get_accelerator().current_device())
        batch = build_simple_batch([n_tokens], padding=False)

        gate(expert_counts, scores, expert_assignment, expert_offset, logits, batch)

        assert expert_counts[19] == n_tokens
        assert expert_counts[26] == 0
        assert torch.all(expert_assignment == 19)
        assert torch.unique(expert_offset).shape[0] == n_tokens
        assert allclose(scores, F.softmax(logits.float(), dim=1)[:, 19])


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize('n_tokens, n_experts', [(1, 16), (17, 16), (32, 128), (89, 128), (433, 2)])
def test_score_accuracy(n_tokens: int, n_experts: int) -> None:
    """
    Validate expert scores are correct.
    """
    logits = torch.randn((n_tokens, n_experts), dtype=torch.float16, device=get_accelerator().current_device())
    batch = build_simple_batch([n_tokens], padding=False)

    gate = RaggedTop1Gating(DtypeEnum.fp16)

    expert_counts = torch.zeros((n_experts, ), dtype=torch.int32, device=get_accelerator().current_device())
    scores = torch.empty((n_tokens, ), dtype=torch.float32, device=get_accelerator().current_device())
    expert_assignment = torch.empty((n_tokens, ), dtype=torch.int32, device=get_accelerator().current_device())
    expert_offset = torch.empty((n_tokens, ), dtype=torch.int32, device=get_accelerator().current_device())

    ref_scores = F.softmax(logits.float(), dim=1).max(dim=1).values

    gate(expert_counts, scores, expert_assignment, expert_offset, logits, batch)
    assert allclose(scores, ref_scores)
    assert expert_counts.sum() == n_tokens

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""CPU-only tests for the vLLM rollout post-processing.

We can't run vLLM here, but the prompt/response stitching is pure tensor
manipulation and is the part most prone to silent index bugs.
"""

import pytest
import torch

from deepspeed.runtime.rollout import stitch_rollout
from deepspeed.runtime.rlhf.utils import build_response_mask


def test_stitch_basic_single_sample():
    prompt_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attn = torch.ones(2, 3, dtype=torch.long)
    responses = [[10, 11, 12], [20, 21]]
    out = stitch_rollout(prompt_ids, attn, responses, pad_id=0, n_samples_per_prompt=1)
    assert out.input_ids.shape == (2, 6)
    assert out.input_ids[0].tolist() == [1, 2, 3, 10, 11, 12]
    assert out.input_ids[1].tolist() == [4, 5, 6, 20, 21, 0]
    assert out.attention_mask[0].tolist() == [1, 1, 1, 1, 1, 1]
    assert out.attention_mask[1].tolist() == [1, 1, 1, 1, 1, 0]
    assert out.response_start_idx.tolist() == [3, 3]


def test_stitch_with_n_samples():
    prompt_ids = torch.tensor([[1, 2], [3, 4]])
    attn = torch.ones(2, 2, dtype=torch.long)
    responses = [[5, 6], [7, 8], [9, 10], [11, 12]]
    out = stitch_rollout(prompt_ids, attn, responses, pad_id=0, n_samples_per_prompt=2)
    assert out.input_ids.shape == (4, 4)
    # Prompts are repeat_interleaved: [P0, P0, P1, P1].
    assert out.input_ids[0].tolist() == [1, 2, 5, 6]
    assert out.input_ids[1].tolist() == [1, 2, 7, 8]
    assert out.input_ids[2].tolist() == [3, 4, 9, 10]
    assert out.input_ids[3].tolist() == [3, 4, 11, 12]
    assert out.response_start_idx.tolist() == [2, 2, 2, 2]


def test_stitch_left_padded_prompts():
    prompt_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    attn = torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.long)
    responses = [[6], [7]]
    out = stitch_rollout(prompt_ids, attn, responses, pad_id=0, n_samples_per_prompt=1)
    # Response begins at column T_p == 3 for both, regardless of prompt padding.
    assert out.response_start_idx.tolist() == [3, 3]
    # Prompt section keeps the caller's left-padding mask.
    assert out.attention_mask[:, :3].tolist() == [[0, 1, 1], [1, 1, 1]]


def test_stitch_mismatched_response_count_raises():
    prompt_ids = torch.tensor([[1, 2]])
    attn = torch.ones(1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="expected"):
        stitch_rollout(prompt_ids, attn, [[3], [4]], pad_id=0, n_samples_per_prompt=1)


def test_stitch_empty_responses_still_well_shaped():
    prompt_ids = torch.tensor([[1, 2], [3, 4]])
    attn = torch.ones(2, 2, dtype=torch.long)
    out = stitch_rollout(prompt_ids, attn, [[], []], pad_id=0, n_samples_per_prompt=1)
    # No response tokens means total length == prompt length.
    assert out.input_ids.shape == (2, 2)
    # Mask over the (zero) response section is empty; response_start_idx still
    # points at the end of the prompt.
    assert out.response_start_idx.tolist() == [2, 2]


def test_stitch_handles_variable_response_lengths():
    prompt_ids = torch.tensor([[1], [2], [3]])
    attn = torch.ones(3, 1, dtype=torch.long)
    responses = [[10], [20, 21, 22, 23], [30, 31]]
    out = stitch_rollout(prompt_ids, attn, responses, pad_id=99, n_samples_per_prompt=1)
    # Total length = T_p + max(response lengths) = 1 + 4 = 5.
    assert out.input_ids.shape == (3, 5)
    assert out.input_ids[0].tolist() == [1, 10, 99, 99, 99]
    assert out.input_ids[1].tolist() == [2, 20, 21, 22, 23]
    assert out.input_ids[2].tolist() == [3, 30, 31, 99, 99]
    assert out.attention_mask[0].tolist() == [1, 1, 0, 0, 0]
    assert out.attention_mask[1].tolist() == [1, 1, 1, 1, 1]
    assert out.attention_mask[2].tolist() == [1, 1, 1, 0, 0]


def test_stitch_output_feeds_build_response_mask():
    prompt_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    attn = torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.long)
    out = stitch_rollout(prompt_ids, attn, [[10, 11], [20]], pad_id=0, n_samples_per_prompt=1)
    mask = build_response_mask(out.response_start_idx, out.attention_mask)
    # Sample 0: T_p=3, response tokens at 3,4 (both attended).
    assert mask[0].tolist() == [0, 0, 0, 1, 1]
    # Sample 1: T_p=3, response token at 3 only (position 4 is pad).
    assert mask[1].tolist() == [0, 0, 0, 1, 0]

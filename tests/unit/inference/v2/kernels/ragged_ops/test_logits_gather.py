# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.kernels.ragged_ops import RaggedLogitsGather
from ....v2.inference_test_utils import allclose, get_dtypes
from .ragged_testing_utils import build_simple_batch


def baseline_implementation(hidden_states: torch.Tensor, seq_lens: List[int]) -> torch.Tensor:
    output = torch.empty((len(seq_lens), hidden_states.shape[1]),
                         dtype=hidden_states.dtype,
                         device=hidden_states.device)

    offset = 0
    for i, seq_len in enumerate(seq_lens):
        output[i] = hidden_states[offset + seq_len - 1]
        offset += seq_len

    return output


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize('dtype', get_dtypes())
def test_supported_dtypes(dtype: torch.dtype) -> None:
    """
    Validate support on nominally supported data types.
    """
    model_dim = 4096

    batch = build_simple_batch([256], padding=False)
    hidden_states = torch.randn((batch.current_tokens, model_dim),
                                dtype=dtype,
                                device=get_accelerator().current_device())

    reference_result = baseline_implementation(hidden_states, [256])

    kernel = RaggedLogitsGather(model_dim, dtype)
    output = torch.empty_like(reference_result)
    kernel(output, hidden_states, batch)

    assert allclose(output, reference_result)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize('seq_lens', [[128, 64, 192, 32], [57, 112, 63, 89, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1],
                                      [63, 27, 74, 83, 32, 17, 1, 1, 1, 1, 1]])
def test_multiple_sequences(seq_lens: List[int]) -> None:
    """
    Validate support on more multi-sequence inputs.
    """
    model_dim = 4096
    dtype = torch.float16

    batch = build_simple_batch(seq_lens, padding=False)
    hidden_states = torch.randn((batch.current_tokens, model_dim),
                                dtype=dtype,
                                device=get_accelerator().current_device())

    reference_result = baseline_implementation(hidden_states, seq_lens)

    kernel = RaggedLogitsGather(model_dim, dtype)
    output = torch.empty_like(reference_result)
    kernel(output, hidden_states, batch)

    assert allclose(output, reference_result)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("model_dim", [1024, 6144, 6784])
def test_problem_size_permutations(model_dim: int) -> None:
    """
    Validate for different embedding sizes.
    """
    dtype = torch.float16
    seq_lens = [128, 64, 192, 32]

    batch = build_simple_batch(seq_lens, padding=False)
    hidden_states = torch.randn((batch.current_tokens, model_dim),
                                dtype=dtype,
                                device=get_accelerator().current_device())

    reference_result = baseline_implementation(hidden_states, seq_lens)

    kernel = RaggedLogitsGather(model_dim, dtype)
    output = torch.empty_like(reference_result)
    kernel(output, hidden_states, batch)

    assert allclose(output, reference_result)

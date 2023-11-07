# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Tuple

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.kernels.core_ops import BlasLibLinear
from ....v2.inference_test_utils import allclose

# Note: only testing with FP16 and BF16 because we use TF32 on Ampere and we don't have a good
# set of tolerances. Since this is just on top of BLAS though, the test is more about
# making sure the stride/contiguity is correct and that's data type agnostic.


def reference_implementation(hidden_states, weights):
    return hidden_states @ weights.t()


problem_shapes = [
    (1, 1, 1024, 1024),
    (1, 1024, 1024, 1024),
    (2, 1024, 1024, 1024),
    (1, 128, 768, 3072),
    (1, 128, 3072, 768),
    (1, 1024, 8192, 8192),
    (1, 733, 8192, 32768),
    (1, 13, 32768, 8192),
]


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("fp_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("problem_shape", problem_shapes)
def test_blas_linear(fp_dtype: torch.dtype, problem_shape: Tuple[int, int, int, int]):
    batch, seq_len, in_features, out_features = problem_shape
    hidden_states = torch.randn(batch, seq_len, in_features, dtype=fp_dtype,
                                device=get_accelerator().current_device()) * 0.1
    weights = torch.randn(out_features, in_features, dtype=fp_dtype, device=get_accelerator().current_device()) * 0.01
    ds_output = torch.empty(batch, seq_len, out_features, dtype=fp_dtype, device=get_accelerator().current_device())

    ds_kernel = BlasLibLinear(fp_dtype)

    ds_output = ds_kernel(ds_output, hidden_states, weights)
    ref_output = reference_implementation(hidden_states, weights)

    assert allclose(ds_output, ref_output)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("fp_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("problem_shape", problem_shapes)
def test_blas_linear_t(fp_dtype: torch.dtype, problem_shape: Tuple[int, int, int, int]):
    batch, seq_len, in_features, out_features = problem_shape
    hidden_states = torch.randn(batch, seq_len, in_features, dtype=fp_dtype,
                                device=get_accelerator().current_device()) * 0.1
    weights = torch.randn(out_features, in_features, dtype=fp_dtype, device=get_accelerator().current_device()) * 0.01
    ds_output = torch.empty(batch, seq_len, out_features, dtype=fp_dtype, device=get_accelerator().current_device())

    ds_kernel = BlasLibLinear(fp_dtype)

    # Transpose the weights then revert to the format we expect.
    weights = weights.t().contiguous()
    weights = weights.t()
    ds_output = ds_kernel(ds_output, hidden_states, weights)

    ref_output = reference_implementation(hidden_states, weights)

    assert allclose(ds_output, ref_output)

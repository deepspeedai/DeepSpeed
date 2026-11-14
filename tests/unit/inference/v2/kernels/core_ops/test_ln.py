# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.kernels.core_ops import CUDAFPLN
from ....v2.inference_test_utils import (get_dtypes, allclose, allocate_output_with_canary, assert_rows_independent,
                                         norm_tolerances, CANARY_VALUE)


def reference_implementation(hidden_states: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                             epsilon: float) -> torch.Tensor:
    hidden_states_f = hidden_states.to(torch.float32)
    gamma_f = gamma.to(torch.float32)
    beta_f = beta.to(torch.float32)
    return torch.nn.functional.layer_norm(hidden_states_f, (hidden_states_f.size(-1), ),
                                          weight=gamma_f,
                                          bias=beta_f,
                                          eps=epsilon).to(hidden_states.dtype)


@pytest.mark.inference_v2_ops
# The shapes with <= 128 channels pack several rows into one thread block (16 to 256 rows per block, depending
# on dtype and channels) and have one row more than fits in a block, so the last block is partially filled and
# must not touch rows past the end. Together they cover every threads-per-row dispatch in fp32 and fp16.
@pytest.mark.parametrize("tokens, channels", [(1, 4096), (37, 2048), (112, 14432), (1024, 6144), (33, 64), (33, 128),
                                              (65, 32), (129, 16), (257, 8)])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_cuda_ln(tokens: int, channels: int, dtype: torch.dtype) -> None:
    if dtype == torch.float32 and 64 < channels <= 128:
        pytest.skip("fp32 with 65-128 channels launches no kernel, see #8827")

    # Input vals
    hidden_states = torch.randn((tokens, channels), dtype=dtype, device=get_accelerator().current_device_name())
    gamma = torch.randn((channels), dtype=dtype, device=get_accelerator().current_device_name())
    beta = torch.rand((channels), dtype=dtype, device=get_accelerator().current_device_name())
    epsilon = 1e-5

    # Reference output
    ref_output = reference_implementation(hidden_states, gamma, beta, epsilon)

    # New output
    ln_kernel = CUDAFPLN(hidden_states.size(-1), hidden_states.dtype)
    ds_output, canary = allocate_output_with_canary(tokens, channels, dtype)
    ln_kernel(ds_output, hidden_states, gamma, beta)

    def run_rows(start: int, end: int):
        output_single = torch.empty_like(hidden_states[start:end])
        ln_kernel(output_single, hidden_states[start:end], gamma, beta)
        return (output_single, )

    # Check
    assert allclose(ds_output, ref_output, norm_tolerances(dtype))
    assert (canary == CANARY_VALUE).all()
    assert_rows_independent(run_rows, (ds_output, ))

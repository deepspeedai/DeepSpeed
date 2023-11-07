# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.kernels.core_ops import CUDAFPPostLN
from ....v2.inference_test_utils import get_dtypes, allclose


def reference_implementation(residual: torch.Tensor, hidden_states: torch.Tensor, gamma: torch.Tensor,
                             beta: torch.Tensor, epsilon: float) -> torch.Tensor:
    residual_f = residual.to(torch.float32)
    hidden_states_f = hidden_states.to(torch.float32)
    gamma_f = gamma.to(torch.float32)
    beta_f = beta.to(torch.float32)
    return torch.nn.functional.layer_norm(residual_f + hidden_states_f, (hidden_states_f.size(-1), ),
                                          weight=gamma_f,
                                          bias=beta_f,
                                          eps=epsilon).to(hidden_states.dtype)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("tokens, channels", [(1, 4096), (37, 2048), (112, 14432), (1024, 6144)])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_cuda_post_ln(tokens: int, channels: int, dtype: torch.dtype) -> None:

    # Input vals
    hidden_states = torch.randn((tokens, channels), dtype=dtype, device=get_accelerator().current_device_name())
    residual = torch.randn((tokens, channels), dtype=dtype, device=get_accelerator().current_device_name())
    gamma = torch.randn((channels), dtype=dtype, device=get_accelerator().current_device_name())
    beta = torch.rand((channels), dtype=dtype, device=get_accelerator().current_device_name())
    epsilon = 1e-5

    # Reference output
    ref_output = reference_implementation(residual, hidden_states, gamma, beta, epsilon)

    # New output
    post_ln_kernel = CUDAFPPostLN(hidden_states.size(-1), residual.dtype)
    ds_output = torch.empty_like(residual)
    post_ln_kernel(ds_output, residual, hidden_states, gamma, beta)

    # Check
    assert allclose(ds_output, ref_output)

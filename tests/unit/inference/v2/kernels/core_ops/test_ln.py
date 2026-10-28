# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.kernels.core_ops import CUDAFPLN
from ....v2.inference_test_utils import get_dtypes, allclose


def reference_implementation(vals: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                             epsilon: float) -> torch.Tensor:
    return torch.nn.functional.layer_norm(vals.to(torch.float32), (vals.size(-1), ),
                                          weight=gamma.to(torch.float32),
                                          bias=beta.to(torch.float32),
                                          eps=epsilon).to(vals.dtype)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("tokens, channels", [(1, 4096), (37, 2048), (64, 96), (64, 128)])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_cuda_ln(tokens: int, channels: int, dtype: torch.dtype) -> None:
    device = get_accelerator().current_device_name()
    vals = torch.randn((tokens, channels), dtype=dtype, device=device)
    gamma = torch.randn((channels), dtype=dtype, device=device)
    beta = torch.rand((channels), dtype=dtype, device=device)
    epsilon = 1e-5

    ref_output = reference_implementation(vals, gamma, beta, epsilon)

    ln_kernel = CUDAFPLN(channels, dtype)
    ds_output = torch.full_like(vals, float('nan'))
    ln_kernel(ds_output, vals, gamma, beta)

    assert allclose(ds_output, ref_output)

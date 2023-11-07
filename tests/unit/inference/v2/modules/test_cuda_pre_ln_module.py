# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional, Tuple

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.modules import ConfigBundle
from deepspeed.inference.v2.modules.configs import DSNormConfig
from deepspeed.inference.v2.modules.interfaces import DSPreNormRegistry
from ...v2.inference_test_utils import get_dtypes, allclose


def reference_implementation(residual: torch.Tensor, hidden_states: Optional[torch.Tensor], gamma: torch.Tensor,
                             beta: torch.Tensor, epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = residual.dtype

    residual = residual.to(torch.float32)
    gamma = gamma.to(torch.float32)
    beta = beta.to(torch.float32)

    if hidden_states is not None:
        hidden_states = hidden_states.to(torch.float32)
        residual = residual + hidden_states
    hidden_states = torch.nn.functional.layer_norm(residual, (residual.size(-1), ),
                                                   weight=gamma,
                                                   bias=beta,
                                                   eps=epsilon)
    return residual.to(dtype), hidden_states.to(dtype)


def _pre_ln_test_helper(n_tokens: int, n_channels: int, dtype: torch.dtype, res_add: bool = False):
    config = DSNormConfig(max_tokens=2048,
                          type="layer_norm",
                          channels=n_channels,
                          residual_dtype=dtype,
                          input_dtype=dtype,
                          output_dtype=dtype,
                          eps=1e-5)
    bundle = ConfigBundle(name='cuda_pre_ln', config=config)

    # Input vals
    if res_add:
        hidden_states = torch.randn((n_tokens, n_channels),
                                    dtype=dtype,
                                    device=get_accelerator().current_device_name())
    else:
        hidden_states = None

    residual = torch.randn((n_tokens, n_channels), dtype=dtype, device=get_accelerator().current_device_name())
    gamma = torch.randn((n_channels), dtype=torch.float32, device=get_accelerator().current_device_name())
    beta = torch.rand((n_channels), dtype=torch.float32, device=get_accelerator().current_device_name())
    epsilon = 1e-5

    # Reference output
    ref_residual, ref_output = reference_implementation(residual, hidden_states, gamma, beta, epsilon)

    # New output
    pre_ln_module = DSPreNormRegistry.instantiate_config(bundle)
    gamma = pre_ln_module.transform_param(gamma)
    beta = pre_ln_module.transform_param(beta)

    ds_residual, ds_output = pre_ln_module(residual, hidden_states, gamma, beta)

    # Check
    assert allclose(ds_residual, ref_residual)
    assert allclose(ds_output, ref_output)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("tokens, channels", [(1, 2048), (37, 8192), (1280, 768), (2048, 5120)])
def test_token_channels(tokens: int, channels: int) -> None:
    _pre_ln_test_helper(tokens, channels, torch.float16)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("dtype", get_dtypes(include_float=False))
def test_dtype(dtype: torch.dtype) -> None:
    _pre_ln_test_helper(733, 2560, dtype)


@pytest.mark.inference_v2_ops
def test_no_res_add():
    _pre_ln_test_helper(733, 2560, torch.float16, res_add=False)

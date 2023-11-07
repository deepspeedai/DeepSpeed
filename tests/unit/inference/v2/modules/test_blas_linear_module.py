# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.inference_utils import ActivationType, DtypeEnum, is_gated
from deepspeed.inference.v2.modules import ConfigBundle
from deepspeed.inference.v2.modules.configs import DSLinearConfig
from deepspeed.inference.v2.modules.interfaces import DSLinearRegistry
from ...v2.inference_test_utils import allclose


def reference_implementation(hidden_states: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
                             act_type: ActivationType) -> torch.Tensor:
    dtype = hidden_states.dtype
    out_states = torch.nn.functional.linear(hidden_states, weight, bias)
    out_states.float()

    if is_gated(act_type):
        act_func_map = {
            ActivationType.ReGLU: torch.nn.functional.relu,
            ActivationType.GEGLU: lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
            ActivationType.SiGLU: torch.nn.functional.silu,
        }

        act_act = out_states[..., ::2]
        act_linear = out_states[..., 1::2]

        act_act = act_func_map[act_type](act_act)
        out_states = act_act * act_linear
    else:
        act_func_map = {
            ActivationType.RELU: torch.nn.functional.relu,
            ActivationType.GELU: torch.nn.functional.gelu,
            ActivationType.SILU: torch.nn.functional.silu,
            ActivationType.IDENTITY: lambda x: x,
        }

        out_states = act_func_map[act_type](out_states)
    return out_states.to(dtype)


def _blas_linear_helper(tokens: int,
                        in_channels: int,
                        out_channels: int,
                        dtype: DtypeEnum,
                        act_fn: ActivationType,
                        use_bias: bool = True) -> None:
    linear_config = DSLinearConfig(max_tokens=2048,
                                   in_channels=in_channels,
                                   out_channels=out_channels,
                                   activation=act_fn,
                                   input_dtype=dtype,
                                   output_dtype=dtype)

    bundle = ConfigBundle(name='blas_fp_linear', config=linear_config)

    module = DSLinearRegistry.instantiate_config(bundle)

    # Input vals
    hidden_states = torch.randn(
        (tokens, in_channels), dtype=dtype.value, device=get_accelerator().current_device_name()) * .01

    weight_out_channels = 2 * out_channels if is_gated(act_fn) else out_channels
    weight = torch.randn(
        (weight_out_channels, in_channels), dtype=dtype.value, device=get_accelerator().current_device_name()) * .01
    if use_bias:
        bias = torch.randn(
            (weight_out_channels), dtype=dtype.value, device=get_accelerator().current_device_name()) * .01
    else:
        bias = None

    # Reference output
    ref_output = reference_implementation(hidden_states, weight, bias, act_fn)

    # New output
    ds_output = module(hidden_states, weight, bias)

    # Check
    assert allclose(ds_output, ref_output)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("tokens, in_channels, out_channels", [(1, 4608, 1728), (37, 8192, 4096), (1280, 3072, 6144)])
def test_blas_linear_shapes(tokens: int, in_channels: int, out_channels: int) -> None:

    _blas_linear_helper(tokens, in_channels, out_channels, DtypeEnum.fp16, ActivationType.IDENTITY)


all_acts = [
    ActivationType.RELU,
    ActivationType.GELU,
    ActivationType.SILU,
    ActivationType.GEGLU,
    ActivationType.ReGLU,
    ActivationType.SiGLU,
]


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("act_fn", all_acts)
@pytest.mark.parametrize("use_bias", [True, False])
def test_blas_linear_act_fn(act_fn: ActivationType, use_bias: bool) -> None:

    _blas_linear_helper(283, 512, 4096, DtypeEnum.fp16, act_fn, use_bias=use_bias)

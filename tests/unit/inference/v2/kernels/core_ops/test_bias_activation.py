# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.inference_utils import ActivationType, DtypeEnum
from deepspeed.inference.v2.kernels.core_ops import CUDABiasActivation
from ....v2.inference_test_utils import get_dtypes, allclose


def reference_bias_act_implementation(input: torch.Tensor, bias: Optional[torch.Tensor],
                                      act_type: ActivationType) -> torch.Tensor:
    bias_func_map = {
        ActivationType.RELU: torch.nn.functional.relu,
        ActivationType.GELU: torch.nn.functional.gelu,
        ActivationType.SILU: torch.nn.functional.silu,
        ActivationType.IDENTITY: lambda x: x,
    }

    dtype = input.dtype
    input_f = input.to(torch.float32)
    if bias is not None:
        bias_f = bias.to(torch.float32)
        output_f = input_f + bias_f
    else:
        output_f = input_f
    output_f = bias_func_map[act_type](output_f)

    return output_f.to(dtype)


def _bias_activation_test_helper(tokens: int,
                                 channels: int,
                                 act_fn: ActivationType,
                                 dtype: DtypeEnum,
                                 use_bias: bool = True) -> None:
    """
    Fully parameterized testing entry point.
    """
    # Input vals
    input_tensor = torch.randn((tokens, channels), dtype=dtype.value, device=get_accelerator().current_device_name())
    if use_bias:
        bias = torch.randn((channels), dtype=dtype.value, device=get_accelerator().current_device_name())
    else:
        bias = None

    # Reference output
    ref_output = reference_bias_act_implementation(input_tensor, bias, act_fn)

    bias_act = CUDABiasActivation(channels, dtype, act_fn)

    # New output
    ds_tensor = input_tensor.clone()
    bias_act(ds_tensor, bias)

    # Check
    assert allclose(ds_tensor, ref_output)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("tokens, channels", [(1, 4096), (37, 2048), (112, 14432), (1024, 6144)])
@pytest.mark.parametrize("dtype", get_dtypes(include_float=False))
def test_token_channels_permutations(tokens: int, channels: int, dtype: torch.dtype) -> None:
    """
    Validate bias activation kernel with different token and channel permutations when using the RELU
    activation function.
    """
    act_fn = ActivationType.RELU
    dtype = DtypeEnum(dtype)
    _bias_activation_test_helper(tokens, channels, act_fn, dtype)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("act_fn",
                         [ActivationType.RELU, ActivationType.GELU, ActivationType.SILU, ActivationType.IDENTITY])
def test_act_fns(act_fn: ActivationType) -> None:
    """
    Validate bias activation kernel with different activation functions.
    """
    tokens = 223
    channels = 4096
    dtype = DtypeEnum.fp16
    _bias_activation_test_helper(tokens, channels, act_fn, dtype)


@pytest.mark.inference_v2_ops
def test_no_bias() -> None:
    """
    Validate bias activation kernel with no bias.
    """
    tokens = 223
    channels = 4096
    dtype = DtypeEnum.fp16
    act_fn = ActivationType.IDENTITY
    _bias_activation_test_helper(tokens, channels, act_fn, dtype, use_bias=False)

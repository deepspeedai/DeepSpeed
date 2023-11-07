# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Iterable, Optional

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.kernels.core_ops import CUDAGatedActivation
from deepspeed.inference.v2.inference_utils import ActivationType
from ....v2.inference_test_utils import get_dtypes, allclose


def reference_geglu_implementation(input: torch.Tensor,
                                   bias: Optional[torch.Tensor] = None,
                                   act_fn: Optional[ActivationType] = ActivationType.GEGLU) -> torch.Tensor:
    act_func_map = {
        ActivationType.ReGLU: torch.nn.functional.relu,
        ActivationType.GEGLU: lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
        ActivationType.SiGLU: torch.nn.functional.silu,
    }

    dtype = input.dtype
    input = input.to(torch.float32)

    if bias is not None:
        bias = bias.to(torch.float32)
        input = input + bias

    act_act = input[..., ::2]
    act_linear = input[..., 1::2]

    act_act = act_func_map[act_fn](act_act)

    return (act_act * act_linear).to(dtype)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("shape", [(1372, 16384), (2, 743, 22016)])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_dtypes(shape: Iterable[int], dtype: torch.dtype) -> None:
    input_tensor = torch.randn(shape, dtype=dtype, device=get_accelerator().current_device_name())

    # Reference output
    ref_output = reference_geglu_implementation(input_tensor, act_fn=ActivationType.GEGLU)

    # Build kernel
    geglu = CUDAGatedActivation(input_tensor.size(-1), input_tensor.dtype, ActivationType.GEGLU)

    # New output
    output_shape = list(input_tensor.shape)
    output_shape[-1] //= 2
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=get_accelerator().current_device_name())
    geglu(output_tensor, input_tensor)

    # Check
    assert allclose(output_tensor, ref_output)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("act_fn", [ActivationType.GEGLU, ActivationType.ReGLU, ActivationType.SiGLU])
def test_act_fn(act_fn: ActivationType) -> None:
    input_tensor = torch.randn(832, 4096, dtype=torch.float16, device=get_accelerator().current_device())

    # Reference output
    ref_output = reference_geglu_implementation(input_tensor, act_fn=act_fn)

    cuda_act = CUDAGatedActivation(4096, torch.float16, act_fn)

    # New output
    output_tensor = torch.empty(832, 2048, dtype=torch.float16, device=get_accelerator().current_device())
    cuda_act(output_tensor, input_tensor)

    assert allclose(output_tensor, ref_output)


@pytest.mark.inference_v2_ops
def test_act_with_bias():
    input_tensor = torch.randn(832, 4096, dtype=torch.float16, device=get_accelerator().current_device())
    bias = torch.randn(4096, dtype=torch.float16, device=get_accelerator().current_device())

    # Reference output
    ref_output = reference_geglu_implementation(input_tensor, bias=bias, act_fn=ActivationType.GEGLU)

    cuda_act = CUDAGatedActivation(4096, torch.float16, ActivationType.GEGLU)

    # New output
    output_tensor = torch.empty(832, 2048, dtype=torch.float16, device=get_accelerator().current_device())

    cuda_act(output_tensor, input_tensor, bias)

    assert allclose(output_tensor, ref_output)


@pytest.mark.inference_v2_ops
def test_max_channels():
    input_tensor = torch.randn(832, 48152, dtype=torch.float16, device=get_accelerator().current_device())

    ref_output = reference_geglu_implementation(input_tensor, act_fn=ActivationType.GEGLU)

    cuda_act = CUDAGatedActivation(48152, torch.float16, ActivationType.GEGLU)

    output_tensor = torch.empty(832, 24076, dtype=torch.float16, device=get_accelerator().current_device())
    cuda_act(output_tensor, input_tensor)

    assert allclose(output_tensor, ref_output)


@pytest.mark.inference_v2_ops
def test_bad_dtype() -> None:
    with pytest.raises(ValueError):
        CUDAGatedActivation(128, torch.int8, ActivationType.GEGLU)


@pytest.mark.inference_v2_ops
def test_bad_act_fn() -> None:
    with pytest.raises(ValueError):
        CUDAGatedActivation(128, torch.float16, ActivationType.RELU)


@pytest.mark.inference_v2_ops
def test_bad_alignment() -> None:
    with pytest.raises(ValueError):
        CUDAGatedActivation(127, torch.float16, ActivationType.GEGLU)


@pytest.mark.inference_v2_ops
def test_too_many_channels() -> None:
    with pytest.raises(ValueError):
        CUDAGatedActivation(49160, torch.float16, ActivationType.GEGLU)

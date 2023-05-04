# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.ops.op_builder import InferenceBuilder
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.types import ActivationFuncType
from .inference_test_utils import allclose, get_dtypes

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system", allow_module_level=True)

inference_module = None
torch_minor_version = None


def run_bias_geglu_reference(activations, bias):
    # Expected behavior is that of casting to float32 internally
    # Explicitly using the default GeLU
    activations = activations + bias.reshape(1, 1, -1)
    hidden_states, gate = activations.chunk(2, dim=-1)
    return hidden_states * torch.nn.functional.gelu(gate.to(torch.float32)).to(activations.dtype)


def run_bias_geglu_ds(activation, bias):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()
    return inference_module.gated_activation(activation, bias, ActivationFuncType.GATED_GELU)


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 128, 255])
@pytest.mark.parametrize("channels", [512, 1232, 4096])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_bias_geglu(batch, sequence, channels, dtype):
    activation = torch.randn((batch, sequence, channels * 2), dtype=dtype, device=get_accelerator().device_name())
    bias = torch.randn((channels * 2), dtype=dtype, device=get_accelerator().device_name())

    ds_out = run_bias_geglu_ds(activation, bias)
    ref_out = run_bias_geglu_reference(activation, bias)
    assert (allclose(ds_out, ref_out))


def run_gated_silu_reference(activations, bias):
    # Expected behavior is that of casting to float32 internally
    # Explicitly using the default GeLU
    activations = activations + bias.reshape(1, 1, -1)
    hidden_states, gate = activations.chunk(2, dim=-1)
    return hidden_states * torch.nn.functional.silu(gate.to(torch.float32)).to(activations.dtype)


def run_gated_silu_ds(activation, bias):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()
    return inference_module.gated_activation(activation, bias, ActivationFuncType.GATED_SILU)


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 128, 255])
@pytest.mark.parametrize("channels", [512, 1232, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_gated_silu(batch, sequence, channels, dtype):
    activation = torch.randn((batch, sequence, channels * 2), dtype=dtype, device=get_accelerator().device_name())
    bias = torch.randn((channels * 2), dtype=dtype, device=get_accelerator().device_name())

    ds_out = run_gated_silu_ds(activation, bias)
    ref_out = run_gated_silu_reference(activation, bias)
    assert (allclose(ds_out, ref_out))

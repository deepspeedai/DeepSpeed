# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
import torch
import pytest
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder  # type: ignore
from .inference_test_utils import allclose, get_dtypes

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system", allow_module_level=True)

inference_module = None


def ref_implementation(vals, gamma, epsilon):
    variance = vals.to(torch.float32).pow(2).mean(-1, keepdim=True)
    vals = vals * torch.rsqrt(variance + epsilon)

    if gamma.dtype in [torch.float16, torch.bfloat16]:
        vals = vals.to(gamma.dtype)

    return gamma * vals


def ds_implementation(vals, gamma, epsilon):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()
    return inference_module.rms_norm(vals, gamma, epsilon)


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("seq_len", [1, 128])
@pytest.mark.parametrize("channels", [384, 512, 768, 1024, 2048, 8192, 14432])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_rms_norm(batch, seq_len, channels, dtype):
    device = get_accelerator().current_device_name()
    vals = torch.randn((batch, seq_len, channels), dtype=dtype, device=device)
    gamma = torch.randn((channels), dtype=dtype, device=device)
    epsilon = 1e-5

    ref_output = ref_implementation(vals, gamma, epsilon)
    new_output = ds_implementation(vals, gamma, epsilon)

    assert allclose(new_output, ref_output)


def pre_ds_implementation(vals, residual, gamma, epsilon):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()
    return inference_module.pre_rms_norm(vals, residual, gamma, epsilon)


def pre_ref_implementation(vals, residual, gamma, epsilon):
    residual = vals.to(torch.float32) + residual.to(torch.float32)
    vals = residual

    variance = vals.to(torch.float32).pow(2).mean(-1, keepdim=True)
    vals = vals * torch.rsqrt(variance + epsilon)

    if gamma.dtype in [torch.float16, torch.bfloat16]:
        vals = vals.to(gamma.dtype)

    return gamma * vals, residual.to(gamma.dtype)


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("seq_len", [1, 128])
@pytest.mark.parametrize("channels", [384, 512, 768, 1024, 2048, 8192, 14432])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_pre_norm(batch, seq_len, channels, dtype):
    device = get_accelerator().current_device_name()
    vals = torch.randn((batch, seq_len, channels), dtype=dtype, device=device)
    residual = torch.randn((batch, seq_len, channels), dtype=dtype, device=device)
    gamma = torch.randn((channels), dtype=dtype, device=device)
    epsilon = 1e-5

    ref_output = pre_ref_implementation(vals, residual, gamma, epsilon)
    new_output = pre_ds_implementation(vals, residual, gamma, epsilon)

    assert allclose(new_output[0], ref_output[0])
    #assert allclose(new_output[1], ref_output[1])

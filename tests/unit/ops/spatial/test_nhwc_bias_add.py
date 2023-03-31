# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
from deepspeed.ops.transformer.inference.bias_add import nhwc_bias_add
from deepspeed.accelerator import get_accelerator


def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = {torch.float32: (5e-3, 5e-4), torch.float16: (3e-2, 2e-3), torch.int8: (1, 1)}[x.dtype]
    return torch.allclose(x, y, rtol=rtol, atol=atol)


def ref_bias_add(activations, bias):
    return activations + bias.reshape(1, -1, 1, 1)


channels_list = [192, 384, 320, 576, 640, 768, 960, 1152, 1280, 1536, 1600, 1920, 2240, 2560]


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 2, 10])
@pytest.mark.parametrize("image_size", [16, 32, 64])
@pytest.mark.parametrize("channels", channels_list)
def test_bias_add(batch, image_size, channels):
    activations = torch.randn((batch, channels, image_size, image_size),
                              dtype=torch.float16,
                              device=get_accelerator().device_name()).to(memory_format=torch.channels_last)
    bias = torch.randn((channels), dtype=torch.float16, device=get_accelerator().device_name())

    ref_vals = ref_bias_add(activations.clone().detach(), bias)
    ds_vals = nhwc_bias_add(activations, bias)

    assert allclose(ds_vals, ref_vals)


def ref_bias_add_add(activations, bias, other):
    return (activations + bias.reshape(1, -1, 1, 1)) + other


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 2, 10])
@pytest.mark.parametrize("image_size", [16, 32, 64])
@pytest.mark.parametrize("channels", channels_list)
def test_bias_add_add(batch, image_size, channels):
    activations = torch.randn((batch, channels, image_size, image_size),
                              dtype=torch.float16,
                              device=get_accelerator().device_name()).to(memory_format=torch.channels_last)
    other = torch.randn((batch, channels, image_size, image_size),
                        dtype=torch.float16,
                        device=get_accelerator().device_name()).to(memory_format=torch.channels_last)
    bias = torch.randn((channels), dtype=torch.float16, device=get_accelerator().device_name())

    ref_vals = ref_bias_add_add(activations.clone().detach(), bias, other)
    ds_vals = nhwc_bias_add(activations, bias, other=other)

    assert allclose(ds_vals, ref_vals)


def ref_bias_add_bias_add(activations, bias, other, other_bias):
    return (activations + bias.reshape(1, -1, 1, 1)) + (other + other_bias.reshape(1, -1, 1, 1))


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 2, 10])
@pytest.mark.parametrize("image_size", [16, 32, 64])
@pytest.mark.parametrize("channels", channels_list)
def test_bias_add_bias_add(batch, image_size, channels):
    activations = torch.randn((batch, channels, image_size, image_size),
                              dtype=torch.float16,
                              device=get_accelerator().device_name()).to(memory_format=torch.channels_last)
    other = torch.randn((batch, channels, image_size, image_size),
                        dtype=torch.float16,
                        device=get_accelerator().device_name()).to(memory_format=torch.channels_last)
    bias = torch.randn((channels), dtype=torch.float16, device=get_accelerator().device_name())
    other_bias = torch.randn((channels), dtype=torch.float16, device=get_accelerator().device_name())

    ref_vals = ref_bias_add_bias_add(activations.clone().detach(), bias, other, other_bias)
    ds_vals = nhwc_bias_add(activations, bias, other=other, other_bias=other_bias)

    assert allclose(ds_vals, ref_vals)

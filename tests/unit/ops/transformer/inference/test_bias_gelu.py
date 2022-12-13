"""
Copyright 2022 The Microsoft DeepSpeed Team
"""

import pytest
import torch
import deepspeed
from deepspeed.ops.op_builder import InferenceBuilder

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system",
                allow_module_level=True)

inference_module = None
torch_minor_version = None


def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = {torch.float32: (5e-4, 5e-5), torch.float16: (3e-2, 2e-3)}[x.dtype]
    return torch.allclose(x, y, rtol=rtol, atol=atol)


def version_appropriate_gelu(activations):
    global torch_minor_version
    if torch_minor_version is None:
        torch_minor_version = int(torch.__version__.split('.')[1])
    # If torch version = 1.12
    if torch_minor_version < 12:
        return torch.nn.functional.gelu(activations)
    else:
        return torch.nn.functional.gelu(activations, approximate='tanh')


def run_bias_gelu_reference(activations, bias):
    # Expected behavior is that of casting to float32 internally and using the tanh approximation
    return version_appropriate_gelu(
        activations.to(torch.float32) + bias.to(torch.float32)).to(activations.dtype)


def run_bias_gelu_ds(activations, bias):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()
    if activations.dtype == torch.float16:
        return inference_module.bias_gelu_fp16(activations, bias)
    else:
        return inference_module.bias_gelu_fp32(activations, bias)


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 128, 255])
@pytest.mark.parametrize("channels", [512, 1232, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_bias_gelu(batch, sequence, channels, dtype):
    activations_ds = torch.randn((batch, sequence, channels), dtype=dtype, device='cuda')
    bias_ds = torch.randn((channels), dtype=dtype, device='cuda')

    activations_ref = activations_ds.clone().detach()
    bias_ref = bias_ds.clone().detach()

    ds_out = run_bias_gelu_ds(activations_ds, bias_ds)
    ref_out = run_bias_gelu_reference(activations_ref, bias_ref)
    assert (allclose(ds_out, ref_out))

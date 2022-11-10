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
    rtol, atol = {torch.float32: (5e-3, 5e-4), torch.float16: (3e-2, 2e-3), torch.int8: (0, 0)}[x.dtype]
    return torch.allclose(x, y, rtol=rtol, atol=atol)


def run_bias_geglu_reference(activations, bias):
    # Expected behavior is that of casting to float32 internally
    # Explicitly using the default GeLU
    activations = activations + bias.reshape(1, 1, -1)
    hidden_states, gate = activations.chunk(2, dim=-1)
    return hidden_states * torch.nn.functional.gelu(gate.to(torch.float32)).to(
        activations.dtype)


def run_bias_geglu_ds(activation, bias):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()
    return inference_module.bias_geglu(activation, bias)


@pytest.mark.inference
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 128, 255])
@pytest.mark.parametrize("channels", [512, 1232, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_bias_geglu(batch, sequence, channels, dtype):
    activation = torch.randn((batch, sequence, channels * 2), dtype=dtype, device='cuda')
    bias = torch.randn((channels * 2), dtype=dtype, device='cuda')

    ds_out = run_bias_geglu_ds(activation, bias)
    ref_out = run_bias_geglu_reference(activation, bias)
    assert (allclose(ds_out, ref_out))

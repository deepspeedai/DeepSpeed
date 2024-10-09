# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.ops.op_builder import InferenceBuilder
from deepspeed.ops.transformer import DeepSpeedInferenceConfig
from deepspeed.ops.transformer.inference.op_binding.bias_gelu import BiasGeluOp

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system", allow_module_level=True)

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


def run_gelu_reference(activations):
    # Expected behavior is that of casting to float32 internally and using the tanh approximation
    return version_appropriate_gelu(activations.to(torch.float32)).to(activations.dtype)


def run_gelu_ds(activations, use_triton_ops=False):
    if use_triton_ops:
        from deepspeed.ops.transformer.inference.triton import gelu
        return gelu(activations)

    device = deepspeed.accelerator.get_accelerator().device_name()
    channels = activations.shape[-1]
    bias = torch.zeros((channels), dtype=activations.dtype, device=device)
    config = DeepSpeedInferenceConfig(dtype=activations.dtype)
    return BiasGeluOp(config)(activations, bias)


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 128, 255])
@pytest.mark.parametrize("channels", [512, 1232, 4096])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("use_triton_ops", [True, False])
def test_gelu(batch, sequence, channels, dtype, use_triton_ops):
    device = deepspeed.accelerator.get_accelerator().device_name()
    activations_ds = torch.randn((batch, sequence, channels), dtype=dtype, device=device)
    activations_ref = activations_ds.clone().detach()

    if not deepspeed.HAS_TRITON and use_triton_ops:
        pytest.skip("triton has to be installed for the test")
    ds_out = run_gelu_ds(activations_ds, use_triton_ops)
    ref_out = run_gelu_reference(activations_ref)
    assert (allclose(ds_out, ref_out))

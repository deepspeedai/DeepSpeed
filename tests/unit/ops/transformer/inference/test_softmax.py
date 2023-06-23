# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.ops.op_builder import InferenceBuilder

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system", allow_module_level=True)

inference_module = None
torch_minor_version = None


def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = {torch.float32: (5e-4, 5e-5), torch.float16: (3e-2, 2e-3)}[x.dtype]
    return torch.allclose(x, y, rtol=rtol, atol=atol)


def run_softmax_reference(input):
    return torch.nn.functional.softmax(input, dim=-1)


def run_softmax_ds(input, use_triton_ops=False):
    if use_triton_ops:
        from deepspeed.ops.transformer.inference.triton import softmax
        # return torch.empty_like(input)
        return softmax(input)

    assert use_triton_ops, "Only triton softmax is supported for now"


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 128, 255, 1232])
@pytest.mark.parametrize("channels", [512, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("use_triton_ops", [True])
def test_softmax(batch, sequence, channels, dtype, use_triton_ops):
    if not deepspeed.HAS_TRITON and use_triton_ops:
        pytest.skip("triton has to be installed for the test")
    input_ds = torch.randn((batch, sequence, channels), dtype=dtype, device='cuda')
    input_ref = input_ds.clone().detach()

    ds_out = run_softmax_ds(input_ds, use_triton_ops)
    ref_out = run_softmax_reference(input_ref)
    assert (allclose(ds_out, ref_out))

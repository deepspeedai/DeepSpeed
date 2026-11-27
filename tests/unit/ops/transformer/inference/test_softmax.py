# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.ops.op_builder import InferenceBuilder

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system", allow_module_level=True)


def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = {torch.float32: (5e-4, 5e-5), torch.float16: (3e-2, 2e-3)}[x.dtype]
    return torch.allclose(x, y, rtol=rtol, atol=atol)


def run_softmax_reference(input):
    return torch.nn.functional.softmax(input, dim=-1)


def run_softmax_ds(input, use_triton_ops=False):
    if use_triton_ops:
        from deepspeed.ops.transformer.inference.triton import softmax
        return softmax(input)

    assert use_triton_ops, "Only triton softmax is supported for now"


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 128, 255, 1232])
@pytest.mark.parametrize("channels", [512, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("use_triton_ops", [True])
def test_softmax(batch, sequence, channels, dtype, use_triton_ops):
    if not deepspeed.get_accelerator().is_triton_supported():
        pytest.skip("triton is not supported on this system")

    device = deepspeed.accelerator.get_accelerator().device_name()
    input_ds = torch.randn((batch, sequence, channels), dtype=dtype, device=device)
    input_ref = input_ds.clone().detach()

    ds_out = run_softmax_ds(input_ds, use_triton_ops)
    ref_out = run_softmax_reference(input_ref)
    assert (allclose(ds_out, ref_out))


@pytest.mark.inference_ops
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("mask_value", [-1000., -float('inf')])
def test_masked_softmax_when_max_is_masked(dtype, mask_value):
    if not deepspeed.get_accelerator().is_triton_supported():
        pytest.skip("triton is not supported on this system")

    from deepspeed.ops.transformer.inference.triton import softmax

    device = deepspeed.accelerator.get_accelerator().device_name()
    input_ds = torch.tensor([[0., -200.]], dtype=dtype, device=device)
    mask = torch.tensor([[mask_value, 0.]], dtype=dtype, device=device)

    ds_out = softmax(input_ds, mask)
    ref_out = torch.tensor([[0., 1.]], dtype=dtype, device=device)
    assert allclose(ds_out, ref_out)


@pytest.mark.inference_ops
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("mask_shape", ["shared", "per_batch", "per_batch_query", "full"])
def test_masked_softmax_broadcasts_mask(dtype, mask_shape):
    if not deepspeed.get_accelerator().is_triton_supported():
        pytest.skip("triton is not supported on this system")

    from deepspeed.ops.transformer.inference.triton import softmax

    device = deepspeed.accelerator.get_accelerator().device_name()
    B, H, Q, S = 3, 4, 5, 100
    shape = {
        "shared": (1, 1, 1, S),
        "per_batch": (B, 1, 1, S),
        "per_batch_query": (B, 1, Q, S),
        "full": (B, H, Q, S),
    }[mask_shape]
    input_ds = torch.randn((B, H, Q, S), dtype=dtype, device=device)
    mask = torch.zeros(shape, dtype=dtype, device=device)
    mask.masked_fill_(torch.rand(shape, device=device) < 0.3, -10000.)

    ds_out = softmax(input_ds, mask)
    ref_out = torch.nn.functional.softmax(input_ds.float() + mask.float(), dim=-1).to(dtype)
    assert allclose(ds_out, ref_out)

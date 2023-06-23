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
    rtol, atol = {torch.float32: (5e-4, 5e-5), torch.float16: (5e-2, 2e-3)}[x.dtype]
    return torch.allclose(x, y, rtol=rtol, atol=atol)


def run_matmul_ref(a, b):
    return torch.matmul(a, b)


def run_matmul_ds(a, b, use_triton_ops=False):
    if use_triton_ops:
        from deepspeed.ops.transformer.inference.triton import matmul_4d as matmul
        return matmul(a, b)

    assert use_triton_ops, "Only triton softmax is supported for now"


@pytest.mark.inference_ops
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("H", [1, 2, 16])
@pytest.mark.parametrize("M", [1, 7, 8, 128])
@pytest.mark.parametrize("K", [2, 5, 16, 128])
@pytest.mark.parametrize("N", [1, 2, 8, 512])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("use_triton_ops", [True])
def test_matmul_4d(B, H, M, K, N, dtype, use_triton_ops):
    if not deepspeed.HAS_TRITON and use_triton_ops:
        pytest.skip("triton has to be installed for the test")

    # skip autotune in testing
    from deepspeed.ops.transformer.inference.triton.matmul_ext import fp16_matmul
    fp16_matmul.skip_autotune()

    a_ds = torch.randn((B, H, M, K), dtype=dtype, device='cuda')
    b_ds = torch.randn((B, H, K, N), dtype=dtype, device='cuda')
    a_ref = a_ds.clone().detach()
    b_ref = b_ds.clone().detach()

    ds_out = run_matmul_ds(a_ds, b_ds, use_triton_ops)
    ref_out = run_matmul_ref(a_ref, b_ref)
    assert (allclose(ds_out, ref_out))

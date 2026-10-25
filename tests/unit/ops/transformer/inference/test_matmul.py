# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import importlib

import pytest
import torch
import deepspeed
from deepspeed.ops.op_builder import InferenceBuilder

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system", allow_module_level=True)


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
    if not deepspeed.get_accelerator().is_triton_supported():
        pytest.skip("triton is not supported on this system")
    if not deepspeed.HAS_TRITON:
        pytest.skip("triton is not installed")

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


@pytest.mark.inference_ops
@pytest.mark.parametrize("M,N,K,transposed", [(64, 128, 32, False), (17, 65, 33, True)])
@pytest.mark.parametrize("add_bias,activation", [(False, ""), (True, ""), (True, "gelu"), (True, "relu")])
def test_matmul_2d_autotune(monkeypatch, M, N, K, transposed, add_bias, activation):
    if not deepspeed.get_accelerator().is_triton_supported():
        pytest.skip("triton is not supported on this system")
    if not deepspeed.HAS_TRITON:
        pytest.skip("triton is not installed")

    from deepspeed.ops.transformer.inference.triton import matmul_ext, triton_matmul_kernel

    # Earlier skip_autotune() calls and cached choices must not bypass pruning.
    kernels = importlib.reload(triton_matmul_kernel)
    monkeypatch.setattr(matmul_ext.Fp16Matmul, "_2d_kernel", kernels._fp_matmul)

    device = deepspeed.get_accelerator().device_name()
    torch.manual_seed(20)
    a = torch.randn((M, K), dtype=torch.float16, device=device)
    b = torch.randn((N, K) if transposed else (K, N), dtype=torch.float16, device=device)
    if transposed:
        b = b.t()
    bias = torch.randn((N, ), dtype=torch.float16, device=device) if add_bias else None

    ref_out = torch.matmul(a.float(), b.float())
    if bias is not None:
        ref_out += bias.float()
    if activation == "gelu":
        ref_out = torch.nn.functional.gelu(ref_out)
    elif activation == "relu":
        ref_out = torch.nn.functional.relu(ref_out)

    ds_out = matmul_ext.matmul(a, b, bias=bias, activation=activation)
    torch.testing.assert_close(ds_out, ref_out.half(), rtol=5e-2, atol=2e-3)

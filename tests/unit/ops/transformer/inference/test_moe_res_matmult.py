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


def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = {torch.float32: (5e-4, 5e-5), torch.float16: (3e-2, 2e-3)}[x.dtype]
    return torch.allclose(x, y, rtol=rtol, atol=atol)


def run_moe_res_matmul_reference(residual, coef1, coef2, output):
    return residual * coef1 + output * coef2


def run_moe_res_matmul_ds(residual, coef, output):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()
    coef_t = coef.transpose(-1, -2).contiguous()
    return inference_module.moe_res_matmul(residual, coef_t, output)


@pytest.mark.inference
@pytest.mark.parametrize("hidden_dim", [16, 64])
@pytest.mark.parametrize("c", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_moe_residual_matmul(hidden_dim, c, dtype):
    residual_ds = torch.randn((c,
                               hidden_dim * c,
                               hidden_dim),
                              dtype=dtype,
                              device='cuda')
    coeff1 = torch.randn((1, 1, hidden_dim), dtype=dtype, device='cuda')
    coeff2 = torch.randn((1, 1, hidden_dim), dtype=dtype, device='cuda')
    out_ds = torch.randn((c, hidden_dim * c, hidden_dim), dtype=dtype, device='cuda')
    coeff_ds = torch.cat((coeff1, coeff2), dim=-1)
    residual_ref = residual_ds.clone().detach()
    coeff_ref = coeff_ds.clone().detach()
    out_ref = out_ds.clone().detach()

    ds_out = run_moe_res_matmul_ds(residual_ds, coeff_ds, out_ds)
    ref_out = run_moe_res_matmul_reference(residual_ref, coeff1, coeff2, out_ref)

    assert (allclose(ds_out, ref_out))

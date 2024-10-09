# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
import torch
import pytest
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder
from deepspeed.ops.transformer.inference.op_binding.layer_norm import LayerNormOp
from .inference_test_utils import allclose, get_dtypes, assert_almost_equal
try:
    import triton  # noqa: F401 # type: ignore
    from deepspeed.ops.transformer.inference.triton import (
        layer_norm,
        layer_norm_residual,
    )
except ImportError:
    print("triton import failed")

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system", allow_module_level=True)


def ref_implementation(vals, gamma, beta, epsilon, channels, dtype):
    vals_f = vals.to(torch.float32)
    gamma_f = gamma.to(torch.float32)
    beta_f = beta.to(torch.float32)
    return torch.nn.functional.layer_norm(vals_f, (channels, ), weight=gamma_f, bias=beta_f, eps=epsilon).to(dtype)


def ds_implementation(vals, gamma, beta, epsilon):
    return LayerNormOp()(vals, gamma, beta, epsilon)


def ds_triton_implementation(vals, gamma, beta, epsilon):
    return layer_norm(vals, gamma, beta, epsilon)


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("seq_len", [1, 128])
@pytest.mark.parametrize("channels", [384, 512, 768, 1024, 2048, 8192, 14432])
@pytest.mark.parametrize("dtype", get_dtypes())
@pytest.mark.parametrize("use_triton_ops", [False, True])
def test_layer_norm(batch, seq_len, channels, dtype, use_triton_ops):
    if not deepspeed.HAS_TRITON and use_triton_ops:
        pytest.skip("triton has to be installed for the test")

    vals = torch.randn((batch, seq_len, channels), dtype=dtype, device=get_accelerator().current_device_name())
    gamma = torch.randn((channels), dtype=dtype, device=get_accelerator().current_device_name())
    beta = torch.rand((channels), dtype=dtype, device=get_accelerator().current_device_name())
    epsilon = 1e-5

    ref_output = ref_implementation(vals, gamma, beta, epsilon, channels, dtype)
    if use_triton_ops:
        new_output = ds_triton_implementation(vals, gamma, beta, epsilon)
        if dtype != torch.float16:  # fp16 supported in triton
            return
    else:
        new_output = ds_implementation(vals, gamma, beta, epsilon)

    if not allclose(new_output, ref_output):
        #print(new_output - ref_output)
        assert allclose(new_output, ref_output)


def residual_ref_implementation(vals, bias, res, gamma, beta, epsilon, channels, dtype):
    vals_f = vals.to(torch.float32)
    bias_f = bias.to(torch.float32).reshape(1, 1, -1)
    res_f = res.to(torch.float32)
    gamma_f = gamma.to(torch.float32)
    beta_f = beta.to(torch.float32)
    return torch.nn.functional.layer_norm(vals_f + bias_f + res_f, (channels, ),
                                          weight=gamma_f,
                                          bias=beta_f,
                                          eps=epsilon).to(dtype)


def residual_ds_implementation(vals, bias, res, gamma, beta, epsilon):
    return LayerNormOp.layer_norm_residual(vals, bias, res, gamma, beta, epsilon)


def residual_ds_triton_implementation(vals, bias, res, gamma, beta, epsilon):
    return layer_norm_residual(vals, bias, res, gamma, beta, epsilon)


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("seq_len", [1, 128])
@pytest.mark.parametrize("channels", [384, 512, 768, 1024, 2048, 8192, 14432])
@pytest.mark.parametrize("dtype", get_dtypes())
@pytest.mark.parametrize("use_triton_ops", [False, True])
def test_layer_norm_residual(batch, seq_len, channels, dtype, use_triton_ops):
    if not deepspeed.HAS_TRITON and use_triton_ops:
        pytest.skip("triton has to be installed for the test")

    vals = torch.randn((batch, seq_len, channels), dtype=dtype, device=get_accelerator().current_device_name())
    residual = torch.randn((batch, seq_len, channels), dtype=dtype, device=get_accelerator().current_device_name())
    bias = torch.randn((channels), dtype=dtype, device=get_accelerator().current_device_name())
    gamma = torch.randn((channels), dtype=dtype, device=get_accelerator().current_device_name())
    beta = torch.rand((channels), dtype=dtype, device=get_accelerator().current_device_name())
    epsilon = 1e-5

    if use_triton_ops:
        new_output = residual_ds_triton_implementation(vals, bias, residual, gamma, beta, epsilon)
        if dtype != torch.float16:  # fp16 supported in triton
            return
    else:
        new_output = residual_ds_implementation(vals, bias, residual, gamma, beta, epsilon)

    ref_output = residual_ref_implementation(vals, bias, residual, gamma, beta, epsilon, channels, dtype)

    print((new_output - ref_output).abs().max())

    assert allclose(new_output, ref_output)


def residual_store_ref_implementation(vals, bias, res, gamma, beta, epsilon, channels, dtype):
    vals_f = vals.to(torch.float32)
    bias_f = bias.to(torch.float32).reshape(1, 1, -1)
    res_f = res.to(torch.float32)
    gamma_f = gamma.to(torch.float32)
    beta_f = beta.to(torch.float32)
    res_output = vals_f + bias_f + res_f
    norm_output = torch.nn.functional.layer_norm(res_output, (channels, ), weight=gamma_f, bias=beta_f,
                                                 eps=epsilon).to(dtype)
    return norm_output, res_output.to(dtype)


def residual_store_ds_implementation(vals, bias, res, gamma, beta, epsilon):
    return LayerNormOp.layer_norm_residual_store_pre_ln_res(vals, bias, res, gamma, beta, epsilon)


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("seq_len", [1, 128])
@pytest.mark.parametrize("channels", [384, 512, 768, 1024, 2048, 8192, 14432])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_layer_norm_residual_store_pre_ln_res(batch, seq_len, channels, dtype):
    vals = torch.randn((batch, seq_len, channels), dtype=dtype, device=get_accelerator().current_device_name())
    residual = torch.randn((batch, seq_len, channels), dtype=dtype, device=get_accelerator().current_device_name())
    bias = torch.randn((channels), dtype=dtype, device=get_accelerator().current_device_name())
    gamma = torch.randn((channels), dtype=dtype, device=get_accelerator().current_device_name())
    beta = torch.rand((channels), dtype=dtype, device=get_accelerator().current_device_name())
    epsilon = 1e-5

    # Need to run the reference first since there's an in-place component to ours
    ref_norm_output, norm_res_output = residual_store_ref_implementation(vals, bias, residual, gamma, beta, epsilon,
                                                                         channels, dtype)

    ds_norm_output, ds_res_output = residual_store_ds_implementation(vals, bias, residual, gamma, beta, epsilon)

    assert allclose(ds_res_output, norm_res_output)
    assert allclose(ds_norm_output, ref_norm_output)


@pytest.mark.inference_ops
@pytest.mark.parametrize("M", [4])
@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("input_bias", [True, False])
def test_triton_layer_norm(M, N, dtype, residual, input_bias, eps=1e-5, device='cuda'):
    if not deepspeed.HAS_TRITON:
        pytest.skip("triton has to be installed for the test")
    dev = get_accelerator().device_name()
    torch.manual_seed(0)
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=dev, requires_grad=False)
    bias = torch.rand(w_shape, dtype=dtype, device=dev, requires_grad=False)
    x_bias = torch.rand(w_shape, dtype=dtype, device=dev, requires_grad=False)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=dev)
    dy = .1 * torch.randn_like(x)
    if residual:
        res = torch.rand(x_shape, dtype=dtype, device=dev, requires_grad=False)
    else:
        res = torch.zeros(x_shape, dtype=dtype, device=dev, requires_grad=False)
    x.requires_grad_(True)
    # forward pass
    if residual or input_bias:
        y_tri = layer_norm_residual(x, x_bias if input_bias else None, res, weight, bias, eps)
    else:
        y_tri = layer_norm(x, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x + res + (x_bias if input_bias else 0), w_shape, weight, bias,
                                           eps).to(dtype)
    # compare
    assert_almost_equal(y_tri, y_ref)

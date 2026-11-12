# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.ops.op_builder import QuantizerBuilder
from deepspeed.accelerator import get_accelerator

from unit.util import torch_assert_close

if not deepspeed.ops.__compatible_ops__[QuantizerBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system", allow_module_level=True)

inference_module = None


def run_quantize_ds(activations, num_groups, q_bits, is_symmetric_quant):
    global inference_module
    if inference_module is None:
        inference_module = QuantizerBuilder().load()

    return inference_module.quantize(activations, num_groups, q_bits,
                                     inference_module.Symmetric if is_symmetric_quant else inference_module.Asymmetric)


def run_dequantize_ds(activations, params, num_groups, q_bits, is_symmetric_quant):
    global inference_module
    if inference_module is None:
        inference_module = QuantizerBuilder().load()
    return inference_module.dequantize(
        activations,
        params,
        num_groups,
        q_bits,
        inference_module.Symmetric if is_symmetric_quant else inference_module.Asymmetric,
    )


def run_swizzle_quant_ds(activations, num_groups, q_bits, is_symmetric_quant):
    global inference_module
    if inference_module is None:
        inference_module = QuantizerBuilder().load()
    # One node with one device keeps the groups in their original order.
    return inference_module.swizzle_quant(
        activations,
        num_groups,
        q_bits,
        inference_module.Symmetric if is_symmetric_quant else inference_module.Asymmetric,
        1,
        1,
        1,
    )


def get_q_props(q_bits):
    q_range = 2**q_bits
    q_min = -(2**(q_bits - 1))
    q_max = (2**(q_bits - 1) - 1)

    q_min = torch.IntTensor([q_min]).to(device=get_accelerator().device_name())
    q_max = torch.IntTensor([q_max]).to(device=get_accelerator().device_name())
    return q_range, q_max, q_min


def get_scale_zero_point(q_bits, is_symmetric_quant, max, min, absmax, scales=None, zero_points=None):

    q_range, q_max, q_min = get_q_props(q_bits)

    if is_symmetric_quant:
        scale = torch.empty_like(absmax)
        for i, x in enumerate(absmax):
            scale[i] = torch.ones_like(x) if x == 0 else q_range / (2 * x)
        zero_point = torch.zeros(scale.shape, dtype=torch.float32, device=get_accelerator().device_name())
    else:
        scale = torch.empty_like(max)
        for i, x in enumerate(max):
            scale[i] = torch.ones_like(x) if max[i] == min[i] else q_range / (max[i] - min[i])
        zero_point = q_min - (min * scale)

    return scale, zero_point


def int4x2to2xint4(int4X2tensor):
    high = int4X2tensor >> 4
    low = (int4X2tensor << 4) >> 4
    return torch.stack((high, low), dim=-1).flatten()


def run_float_quantize(q_bits, is_symmetric_quant, activations_ref, num_groups):

    # Reference implementation
    # https://pytorch.org/docs/stable/quantization-support.html

    activations_ref = activations_ref.reshape(num_groups, -1).to(dtype=torch.float32)

    max_abs_activations_ref = torch.amax(torch.abs(activations_ref), dim=-1).view(num_groups, -1)
    max_activations_ref = torch.amax(activations_ref, dim=-1).view(num_groups, -1)
    min_activations_ref = torch.amin(activations_ref, dim=-1).view(num_groups, -1)

    _, q_max, q_min = get_q_props(q_bits)

    scale, zero_point = get_scale_zero_point(q_bits, is_symmetric_quant, max_activations_ref, min_activations_ref,
                                             max_abs_activations_ref)

    data_f = activations_ref * scale

    if not is_symmetric_quant:
        data_f = data_f + zero_point

    data_i32 = torch.round(data_f).to(dtype=torch.int32)

    data_i32 = torch.minimum(torch.maximum(data_i32, q_min.expand_as(data_i32)), q_max.expand_as(data_i32))
    data_i8 = data_i32.to(dtype=torch.int8)

    scales = (1.0 / scale).reshape(-1, 1)
    offsets = zero_point.reshape(-1, 1)
    params = torch.cat((scales, offsets), dim=-1)

    return data_i8, params


def run_float_dequantize(q_bits, is_symmetric_quant, data_i8, params, num_groups):
    data_f = data_i8.reshape(num_groups, -1).to(dtype=torch.float32)

    scales = params[:, 0].reshape(-1, 1)
    offsets = params[:, 1].reshape(-1, 1)

    if not is_symmetric_quant:
        data_f = data_f - offsets
    else:
        assert offsets.allclose(torch.zeros_like(offsets))

    data_f = data_f * scales

    return data_f


@pytest.mark.inference_ops
@pytest.mark.parametrize("num_groups", [1, 13, 512])
@pytest.mark.parametrize("num_elems", [8, 16, 32, 64, 128, 256, 4096, 8192, 12288, 16384])
@pytest.mark.parametrize("is_symmetric_quant", [True, False])
@pytest.mark.parametrize("q_bits", [4, 8])
@pytest.mark.parametrize("directed_case", ["all_zeros", None])
def test_float_quantize(num_elems, num_groups, is_symmetric_quant, q_bits, directed_case):
    # fix seed
    torch.manual_seed(num_elems)

    if directed_case == "all_zeros":
        activations_ds = torch.zeros((num_groups, num_elems),
                                     dtype=torch.float16,
                                     device=get_accelerator().device_name())
    else:
        activations_ds = torch.randn((num_groups, num_elems),
                                     dtype=torch.float16,
                                     device=get_accelerator().device_name())
    activations_ref = activations_ds.clone().detach()

    ref_out_tensor, ref_params = run_float_quantize(q_bits, is_symmetric_quant, activations_ref, num_groups)
    ref_dequantized_tensor = run_float_dequantize(q_bits, is_symmetric_quant, ref_out_tensor, ref_params, num_groups)
    # we need to convert the tensor to float64 to avoid overflow
    ref_quantization_error = torch.sum(torch.abs((activations_ref - ref_dequantized_tensor).to(torch.float64)))

    ds_out_tensor, ds_out_params = run_quantize_ds(activations_ds, num_groups, q_bits, is_symmetric_quant)
    ds_dequantized_tensor = run_dequantize_ds(ds_out_tensor, ds_out_params, num_groups, q_bits, is_symmetric_quant)
    assert torch.all(torch.isfinite(ds_dequantized_tensor))

    ds_quantization_error = torch.sum(torch.abs((activations_ds - ds_dequantized_tensor).to(torch.float64)))

    assert (ds_quantization_error <= ref_quantization_error * 1.05)


def reference_quantize(x, groups, num_bits):
    # torch.round is half to even, matching the kernels' __float2int_rn.
    x = x.reshape(groups, -1).float()
    absmax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(absmax == 0, torch.ones_like(absmax), 2**num_bits / (2 * absmax))
    q = torch.round(x * scale).clamp(-2**(num_bits - 1), 2**(num_bits - 1) - 1)
    return q, 1.0 / scale


@pytest.mark.inference_ops
@pytest.mark.parametrize("num_bits", [4, 8])
def test_quantize_rounds_to_nearest(num_bits):
    # Pin round-half-to-even; truncation changes about half the codes.
    torch.manual_seed(0)
    groups = 16
    x = torch.randn(groups * 256, dtype=torch.float16).to(get_accelerator().device_name())
    _, scales = reference_quantize(x, groups, num_bits)

    if num_bits == 8:
        data, params = run_quantize_ds(x, groups, num_bits, True)
    else:
        data, params = run_swizzle_quant_ds(x, groups, num_bits, True)
    out = run_dequantize_ds(data, params, groups, num_bits, True)

    # Compare with codes from the kernel's own scale; reference-scale ULP
    # differences would flip values exactly on a rounding boundary.
    torch_assert_close(params, scales, rtol=1e-6, atol=1e-8)
    xg = x.reshape(groups, -1).float()
    pg = params.reshape(groups, 1)
    ratio = xg / pg
    q = torch.round(ratio).clamp(-2**(num_bits - 1), 2**(num_bits - 1) - 1)
    kernel_q = torch.round(out.view(groups, -1).float() / pg).clamp(-2**(num_bits - 1), 2**(num_bits - 1) - 1)

    # The kernel quantizes by multiplying with its reciprocal scale. At an exact
    # half-step boundary, one ULP in that reciprocal can legitimately choose
    # either neighboring code; allow that case, but reject truncation (which
    # would differ at non-boundary values) and any difference larger than one.
    mismatch = kernel_q != q
    if mismatch.any():
        frac = ratio - torch.floor(ratio)
        at_boundary = (frac - 0.5).abs() <= 1e-5
        assert torch.all(at_boundary[mismatch]), "quantizer rounded a non-boundary value incorrectly"
        assert torch.all((kernel_q[mismatch] - q[mismatch]).abs() == 1), "quantizer differed by more than one code"

"""
Copyright 2022 The Microsoft DeepSpeed Team
"""

import pytest
import torch
from deepspeed.ops import op_builder

inference_module = None


def run_quantize_ds(activations, num_groups, q_bits, is_symmetric_quant):
    global inference_module
    if inference_module is None:
        inference_module = op_builder.QuantizerBuilder().load()

    return inference_module.quantize(
        activations,
        num_groups,
        q_bits,
        inference_module.Symmetric
        if is_symmetric_quant else inference_module.Asymmetric)


def get_q_props(q_bits):
    q_range = 2**q_bits
    q_min = -(2**(q_bits - 1))
    q_max = (2**(q_bits - 1) - 1)

    q_min = torch.IntTensor([q_min]).to(device='cuda')
    q_max = torch.IntTensor([q_max]).to(device='cuda')
    return q_range, q_max, q_min


def get_scale_zero_point(q_bits,
                         is_symmetric_quant,
                         max,
                         min,
                         absmax,
                         scales=None,
                         zero_points=None):

    q_range, q_max, q_min = get_q_props(q_bits)

    if is_symmetric_quant:
        scale = torch.empty_like(absmax)
        for i, x in enumerate(absmax):
            scale[i] = torch.ones_like(x) if x == 0 else q_range / (2 * x)
        zero_point = torch.zeros(scale.shape, dtype=torch.float32, device='cuda')
    else:
        scale = torch.empty_like(max)
        for i, x in enumerate(max):
            scale[i] = torch.ones_like(x) if max[i] == min[i] else q_range / (max[i] -
                                                                              min[i])
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

    max_abs_activations_ref = torch.amax(torch.abs(activations_ref),
                                         dim=-1).view(num_groups,
                                                      -1)
    max_activations_ref = torch.amax(activations_ref, dim=-1).view(num_groups, -1)
    min_activations_ref = torch.amin(activations_ref, dim=-1).view(num_groups, -1)

    _, q_max, q_min = get_q_props(q_bits)

    scale, zero_point = get_scale_zero_point(q_bits, is_symmetric_quant, max_activations_ref, min_activations_ref, max_abs_activations_ref)

    data_f = activations_ref * scale

    if not is_symmetric_quant:
        data_f = data_f + zero_point

    data_i32 = torch.round(data_f).to(dtype=torch.int32)

    data_i32 = torch.minimum(torch.maximum(data_i32,
                                           q_min.expand_as(data_i32)),
                             q_max.expand_as(data_i32))
    data_i8 = data_i32.to(dtype=torch.int8)

    scales = (1.0 / scale).reshape(-1, 1)
    offsets = zero_point.reshape(-1, 1)
    params = torch.cat((scales, offsets), dim=-1)

    return data_i8, params


@pytest.mark.inference_ops
@pytest.mark.parametrize("num_groups", [1, 13, 512])
@pytest.mark.parametrize("num_elems",
                         [8,
                          16,
                          32,
                          64,
                          128,
                          256,
                          4096,
                          8192,
                          12288,
                          16384])
@pytest.mark.parametrize("is_symmetric_quant", [True, False])
@pytest.mark.parametrize("q_bits", [4, 8])
@pytest.mark.parametrize("directed_case", ["all_zeros", None])
def test_float_quantize(num_elems,
                        num_groups,
                        is_symmetric_quant,
                        q_bits,
                        directed_case):

    if directed_case == "all_zeros":
        activations_ds = torch.zeros((num_groups,
                                      num_elems),
                                     dtype=torch.float16,
                                     device='cuda')
    else:
        activations_ds = torch.randn((num_groups,
                                      num_elems),
                                     dtype=torch.float16,
                                     device='cuda')
    activations_ref = activations_ds.clone().detach()

    ref_out_tensor, ref_params = run_float_quantize(q_bits, is_symmetric_quant, activations_ref, num_groups)

    ds_out_tensor, ds_out_params = run_quantize_ds(activations_ds, num_groups, q_bits, is_symmetric_quant)

    if (q_bits == 4):
        ds_out_tensor = int4x2to2xint4(ds_out_tensor)

    # Allow a max difference of 1 to account for differences in rounding in pytorch implementation
    assert (torch.all(
        torch.lt(torch.abs(ds_out_tensor.flatten() - ref_out_tensor.flatten()),
                 2)))
    if is_symmetric_quant:
        assert (torch.allclose(ds_out_params.flatten(), ref_params[:, 0].flatten()))
    else:
        assert (torch.allclose(ds_out_params[:,
                                             0].flatten(),
                               ref_params[:,
                                          0].flatten()))
        assert (torch.allclose(ds_out_params[:,
                                             1].flatten(),
                               ref_params[:,
                                          1].flatten(),
                               atol=5e-5,
                               rtol=5e-5))


def run_integer_quantize_ds(activations, num_groups, q_bits):
    global inference_module
    if inference_module is None:
        inference_module = op_builder.QuantizerBuilder().load()

    return inference_module.quantize(activations,
                                     num_groups,
                                     q_bits,
                                     inference_module.IntegerSymmetric)


def run_integer_quantize(q_bits, activations_ref, num_groups):

    activations_ref = activations_ref.reshape(num_groups, -1).to(dtype=torch.float32)

    max_abs_activations_ref = torch.amax(torch.abs(activations_ref),
                                         dim=-1).view(num_groups,
                                                      -1)

    _, q_max, q_min = get_q_props(q_bits)
    print(max_abs_activations_ref)
    max_abs_activations_ref = (max_abs_activations_ref + 1).to(torch.int8).to(
        torch.float32)
    print(max_abs_activations_ref)
    numerator = activations_ref * q_max
    print(numerator.dtype)
    denominator = max_abs_activations_ref

    data_f = numerator / denominator

    data_i32 = torch.round(data_f).to(dtype=torch.int32)
    data_i32 = torch.minimum(torch.maximum(data_i32,
                                           q_min.expand_as(data_i32)),
                             q_max.expand_as(data_i32))
    data_i8 = data_i32.to(dtype=torch.int8)

    return data_i8, max_abs_activations_ref.to(torch.int32)


@pytest.mark.inference_ops
@pytest.mark.parametrize("num_groups", [1, 2, 4, 8, 16, 32, 64, 512])
@pytest.mark.parametrize("num_elems", [4096, 8192, 12288, 16384])
@pytest.mark.parametrize("q_bits", [4, 8])
def test_integer_quantize(num_elems, num_groups, q_bits):

    activations_ds = torch.ones((num_groups,
                                 num_elems),
                                dtype=torch.float16,
                                device='cuda') * 0.35
    activations_ref = activations_ds.clone().detach()

    ref_out_tensor, ref_params = run_integer_quantize(q_bits, activations_ref, num_groups)

    ds_out_tensor, ds_out_params = run_integer_quantize_ds(activations_ds, num_groups, q_bits)

    if (q_bits == 4):
        ds_out_tensor = int4x2to2xint4(ds_out_tensor)

    # Allow a max difference of 1 to account for differences in rounding in pytorch implementation
    assert (torch.all(
        torch.lt(torch.abs(ds_out_tensor.flatten() - ref_out_tensor.flatten()),
                 2)))
    assert (torch.allclose(ds_out_params.flatten(), ref_params.flatten()))

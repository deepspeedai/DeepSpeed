"""
Copyright 2022 The Microsoft DeepSpeed Team
"""

import pytest
import torch
from deepspeed.ops import op_builder

quantize_module = None


def int4x2to2xint4(int4X2tensor):
    high = int4X2tensor >> 4
    low = (int4X2tensor << 4) >> 4
    return torch.stack((high, low), dim=-1).flatten()


def run_quantize(data, num_groups, q_bits, is_symmetric_quant):
    global quantize_module
    if quantize_module is None:
        quantize_module = op_builder.QuantizerBuilder().load()

    return quantize_module.quantize(
        data,
        num_groups,
        q_bits,
        quantize_module.Symmetric if is_symmetric_quant else quantize_module.Asymmetric)


def run_dequantize(quantized_data, params, num_groups, q_bits, is_symmetric_quant):
    global quantize_module
    if quantize_module is None:
        quantize_module = op_builder.QuantizerBuilder().load()

    return quantize_module.dequantize(
        quantized_data,
        params,
        num_groups,
        q_bits,
        quantize_module.Symmetric if is_symmetric_quant else quantize_module.Asymmetric)


def run_ref_dequantize(quantized_data, params, num_groups, q_bits, is_symmetric_quant):

    if (q_bits == 4):
        quantized_data = int4x2to2xint4(quantized_data)

    quantized_data = quantized_data.reshape(num_groups, -1).to(torch.float32)

    if is_symmetric_quant:
        return (quantized_data * params).to(torch.float16)
    else:
        scales = params[:, 0].reshape(-1, 1)
        offsets = params[:, 1].reshape(-1, 1)
        return (quantized_data * scales + offsets).to(torch.float16)


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
def test_dequantize(num_elems, num_groups, is_symmetric_quant, q_bits):

    activations = torch.randn((num_groups,
                               num_elems),
                              dtype=torch.float16,
                              device='cuda')
    quantized_data, params = run_quantize(activations, num_groups, q_bits, is_symmetric_quant)

    ds_dequant = run_dequantize(quantized_data,
                                params,
                                num_groups,
                                q_bits,
                                is_symmetric_quant)
    ref_dequant = run_ref_dequantize(quantized_data,
                                     params,
                                     num_groups,
                                     q_bits,
                                     is_symmetric_quant)

    assert (torch.allclose(ds_dequant.flatten(),
                           ref_dequant.flatten(),
                           rtol=3e-2,
                           atol=2e-3))

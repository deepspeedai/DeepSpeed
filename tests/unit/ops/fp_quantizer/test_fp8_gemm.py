# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed

from deepspeed.ops.op_builder import FPQuantizerBuilder

if not deepspeed.ops.__compatible_ops__[FPQuantizerBuilder.NAME]:
    pytest.skip("FPQuantizer op is not available on this system", allow_module_level=True)

from deepspeed.ops.fp_quantizer import FP_Quantize, matmul_fp8

from deepspeed import get_accelerator
from deepspeed.linear import QuantizationConfig


def _quantizer(quantization_group_size):
    quant_config = QuantizationConfig(q_dtype=FPQuantizerBuilder.get_default_quant_dtype(),
                                      group_size=quantization_group_size)
    return FP_Quantize(quantization_config=quant_config)


def _dequantize_as(fpq, weight, dtype):
    # The GEMM rounds each decoded weight once, to the activation dtype, so the reference must too.
    # Going through bf16 first would add bf16 rounding that the fp16 kernel never performs.
    return fpq.dequantize(weight, fp_out=torch.empty(weight.shape, dtype=dtype, device=weight.device), scale=fpq.scale)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("q_bits", [8], ids=[
    "qbits8",
])
@pytest.mark.parametrize("M", [1, 2, 4, 8, 32, 64, 128, 256, 512, 1024, 2048])
def test_fp_quant(dtype, q_bits, M):
    device_name = get_accelerator().device_name()
    quantization_group_size = 128
    fpq = _quantizer(quantization_group_size)

    N = 8192
    H = 4096

    x = torch.randn(M, H, dtype=dtype, device=device_name)
    # The quantizer only accepts bf16 weights; fp16 covers fp16 activations.
    weight_bf16 = torch.randn(H, N, dtype=torch.bfloat16, device=device_name)

    weight, _ = fpq.quantize(weight_bf16.data, q_bits=q_bits, return_meta_tensor=True)
    scale = fpq.get_scales()
    out = matmul_fp8(x, weight, scale, quantization_group_size, fpq)

    out_q = torch.matmul(x, _dequantize_as(fpq, weight, dtype))

    # fp32 because the fp16 ratio overflows to inf wherever an output is near zero.
    out, out_q = out.float(), out_q.float()
    error = ((out - out_q).abs() / (out.abs() + 1e-5)).sum() / out.numel()
    assert 0.004 > error, f"failed on batch-size {M} with error {error}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("M", [1, 256])
def test_fp8_gemm_decodes_zero_and_subnormal_weights(dtype, M):
    device_name = get_accelerator().device_name()
    quantization_group_size = 128
    fpq = _quantizer(quantization_group_size)

    N = 8192
    H = 4096
    weight_bf16 = torch.randn(H, N, dtype=torch.bfloat16, device=device_name)
    # An anchor of 4 per group fixes the scale at 4 / 480, so the other entries of row 0 quantize to
    # exact zero and those of row 1 (about 0.006 of the scale) to subnormal codes (exponent field 0).
    anchors = torch.arange(N, device=device_name) % quantization_group_size == 0
    weight_bf16[0] = torch.where(anchors, 4.0, 0.0)
    weight_bf16[1] = torch.where(anchors, 4.0, weight_bf16[1] * 5e-5)

    weight, _ = fpq.quantize(weight_bf16.data, q_bits=8, return_meta_tensor=True)
    scale = fpq.get_scales()
    expected = _dequantize_as(fpq, weight, dtype)
    assert (expected[0][~anchors] == 0).all(), "dequantize did not restore exact zeros"
    assert (expected[1][~anchors] != 0).any(), "no subnormal codes were produced, so the check would be vacuous"

    # Each output row selects one weight row, so the GEMM must reproduce the decoded weights exactly.
    rows = torch.arange(M, device=device_name) % 2
    x = torch.nn.functional.one_hot(rows, H).to(dtype)
    out = matmul_fp8(x, weight, scale, quantization_group_size, fpq)

    assert torch.equal(out, expected[rows])

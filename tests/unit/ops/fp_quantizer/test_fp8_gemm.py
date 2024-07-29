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


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_bits", [8], ids=[
    "qbits8",
])
@pytest.mark.parametrize("M", [1, 2, 4, 8, 32, 64, 128, 256, 512, 1024, 2048])
def test_fp_quant(dtype, q_bits, M):
    quantization_group_size = 128
    fpq = FP_Quantize(group_size=quantization_group_size)

    N = 8192
    H = 4096

    x = torch.randn(M, H, dtype=dtype, device='cuda')
    weight_bf16 = torch.randn(H, N, dtype=dtype, device='cuda')

    weight, _ = fpq.quantize(weight_bf16.data, q_bits=8, return_meta_tensor=True)
    scale = fpq.get_scales()
    out = matmul_fp8(
        x,
        weight,
        scale,
        quantization_group_size,
    )

    out_q = torch.matmul(x, fpq.dequantize(weight, scale=fpq.scale))

    error = ((out - out_q).abs() / (out.abs() + 1e-5)).sum() / out.numel()
    assert 0.004 > error, f"failed on batch-size {M} with error {error}"

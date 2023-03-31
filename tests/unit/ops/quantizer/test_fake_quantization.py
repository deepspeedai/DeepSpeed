# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import pytest
from deepspeed.accelerator import get_accelerator
from deepspeed.ops import op_builder

quantizer_cuda_module = None


def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = {torch.float32: (2e-2, 5e-3), torch.float16: (2e-2, 5e-3)}[x.dtype]
    return torch.allclose(x, y, rtol=rtol, atol=atol)


def quantize_dequantize_ref(inputs, bit, num_groups=1):
    # quantize
    q_range = 2**bit
    input_flat = inputs.float().reshape(num_groups, -1).contiguous()
    input_flat = torch.nan_to_num(input_flat, nan=0.0)
    input_min = input_flat.amin(-1, keepdim=True)
    input_max = input_flat.amax(-1, keepdim=True)

    scale = q_range / (2 * torch.max(input_min.abs(), input_max.abs() + 1e-5))
    input_flat = (input_flat * scale).round().clamp(-q_range // 2, q_range // 2 - 1)
    # dequantize
    dequant_flat = torch.t(input_flat.to(torch.int8)) / scale.view(-1).to(torch.float16)
    return torch.t(dequant_flat).reshape(inputs.shape)


def run_quant_dequant(inputs, groups, bits):
    global quantizer_cuda_module

    if quantizer_cuda_module is None:
        quantizer_cuda_module = op_builder.QuantizerBuilder().load()
    return quantizer_cuda_module.ds_quantize_fp16(inputs, groups, bits)


@pytest.mark.inference_ops
@pytest.mark.parametrize("tensor_shape", [(16, 4096), (128, 256)])
# Test with two tensor shapes as (16, 4096) and (128, 256).
@pytest.mark.parametrize("groups", [1, 16])
# Test with number of quant groups as 1 and 16.
# Note that we have an explicit boundary for groups as ((size / groups) - 1) / 4096 + 1) <= MAX_REG.
def test_fake_quant_dequant(tensor_shape, groups):

    input_tensor = torch.rand((tensor_shape), dtype=torch.float16).to(get_accelerator().device_name())

    # 8-bit quantization.
    ref_input_8bit = input_tensor.clone().detach()
    ds_input_8bit = input_tensor.clone().detach()
    ref_out_8bit = quantize_dequantize_ref(ref_input_8bit, 8, groups)
    # run_quant_dequant will do quantize then dequantize, and return the dequantized value.
    ds_out_8bit = run_quant_dequant(ds_input_8bit, groups, 8)
    assert (allclose(ds_out_8bit, ref_out_8bit))

    # 4-bit quantization.
    ref_input_4bit = input_tensor.clone().detach()
    ds_input_4bit = input_tensor.clone().detach()
    ref_out_4bit = quantize_dequantize_ref(ref_input_4bit, 4, groups)
    ds_out_4bit = run_quant_dequant(ds_input_4bit, groups, 4)
    assert (allclose(ds_out_4bit, ref_out_4bit))

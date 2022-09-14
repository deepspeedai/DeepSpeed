import torch
import pytest
from deepspeed.ops import op_builder

quantizer_cuda_module = None


def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = {torch.float32: (2e-1, 5e-2), torch.float16: (2e-1, 5e-2)}[x.dtype]
    return torch.allclose(x, y, rtol=rtol, atol=atol)


def quantize_dequantize_ref(inputs, bit, num_groups=1):
    # quantize
    q_range = 2**bit
    input_flat = inputs.float().reshape(num_groups, -1).contiguous()
    input_flat = torch.nan_to_num(input_flat, nan=0.0)
    input_min = input_flat.amin(-1, keepdim=True)
    input_max = input_flat.amax(-1, keepdim=True)

    scale = q_range / (2 * torch.max(input_min.abs(), input_max.abs()))
    input_flat = (input_flat * scale).round().clamp(-q_range // 2, q_range // 2 - 1)
    # dequantize
    dequant_flat = torch.t(input_flat.to(torch.int8)) / scale.view(-1).to(torch.float16)
    return torch.t(dequant_flat).reshape(inputs.shape)


def run_quant_dequant(inputs, groups, bits):
    global quantizer_cuda_module

    if quantizer_cuda_module is None:
        quantizer_cuda_module = op_builder.QuantizerBuilder().load()
    return quantizer_cuda_module.ds_quantize_fp16(inputs, groups, bits)


@pytest.mark.inference
@pytest.mark.parametrize("tensor_shape", [(8, 8), (128, 256)])
def test_quant_dequant(tensor_shape):
    input_tensor = torch.rand((tensor_shape), dtype=torch.float16).cuda()

    # test 8bit quant/dequant on tensor partitioned in 1 group.
    ref_input_8bit_1group = input_tensor.clone().detach()
    ds_input_8bit_1group = input_tensor.clone().detach()
    ref_out_8bit_1group = quantize_dequantize_ref(ref_input_8bit_1group, 8)
    # run_quant_dequant will do quantize then dequantize and return the dequantized value.
    ds_out_8bit_1group = run_quant_dequant(ds_input_8bit_1group, 1, 8)
    assert (allclose(ds_out_8bit_1group, ref_out_8bit_1group))

    # test 4bit quant/dequant on tensor partitioned into 16 groups.
    # Note that we have an explicit boundary for groups as ((size / groups) - 1) / 4096 + 1) <= MAX_REG.
    ref_input_4bit_16group = input_tensor.clone().detach()
    ds_input_4bit_16group = input_tensor.clone().detach()
    ref_out_4bit_16group = quantize_dequantize_ref(ref_input_4bit_16group, 4, 16)
    ds_out_4bit_16group = run_quant_dequant(ds_input_4bit_16group, 16, 4)
    assert (allclose(ds_out_4bit_16group, ref_out_4bit_16group))

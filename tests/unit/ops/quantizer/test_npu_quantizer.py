# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import pytest
import torch

from deepspeed.ops.op_builder.npu.quantizer import NPUQuantizer


def reference_quantize(x, groups, num_bits):
    # The reference from run_float_quantize in test_quantize.py: torch.round is half to even,
    # like the CUDA kernel's __float2int_rn.
    x = x.reshape(groups, -1).float()
    absmax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(absmax == 0, torch.ones_like(absmax), 2**num_bits / (2 * absmax))
    q = torch.round(x * scale).clamp(-2**(num_bits - 1), 2**(num_bits - 1) - 1)
    return q, 1.0 / scale


@pytest.mark.parametrize("num_bits", [4, 8])
def test_npu_quantizer_rounds_to_nearest(num_bits):
    # NPUQuantizer is pure torch, so this runs on CPU. Truncating instead of rounding
    # changes about half the codes and doubles the quantization error.
    torch.manual_seed(0)
    groups = 16
    x = torch.randn(groups * 256, dtype=torch.float16)
    q, scales = reference_quantize(x, groups, num_bits)

    if num_bits == 8:
        data, params = NPUQuantizer.quantize(x, groups, num_bits, NPUQuantizer.Symmetric)
    else:
        # the qgZ int4 path; one node with one device keeps the groups in order
        data, params = NPUQuantizer.swizzle_quant(x, groups, num_bits, NPUQuantizer.Symmetric, 1, 1, 1)
    out = NPUQuantizer.dequantize(data, params, groups, num_bits, NPUQuantizer.Symmetric)

    assert torch.equal(params, scales)
    assert torch.equal(out.view(groups, -1), (q * scales).half())

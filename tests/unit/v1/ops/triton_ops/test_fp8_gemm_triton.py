# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for the FP8 GEMM dispatch and its Ascend variant.

The portable kernel decodes DeepSpeed's custom uint8 8-bit FP format with
integer bit manipulation; the Ascend variant performs the same decode with
PyTorch. Both are compared against the same hand-computed values, which pins
the format contract and the per-hardware dispatch without the CUDA-only FP
quantizer op.
"""

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.triton_ops import is_triton_available
from deepspeed.ops.triton_ops._triton import resolve_triton_ops_impl

if not is_triton_available():
    pytest.skip("Triton is not available", allow_module_level=True)

from deepspeed.ops.fp_quantizer.fp8_gemm_triton import matmul_fp8_triton


@pytest.fixture(autouse=True)
def _require_accelerator():
    if not get_accelerator().is_available():
        pytest.skip("FP8 GEMM requires an accelerator")


def test_fp8_gemm_dispatch_resolves_the_platform_module():
    impl = resolve_triton_ops_impl("fp8_gemm_triton", portable_module="deepspeed.ops.fp_quantizer.fp8_gemm_triton")
    if get_accelerator().triton_ops_dir() == "deepspeed.ops.triton_ops.npu":
        assert impl.__name__ == "deepspeed.ops.triton_ops.npu.fp8_gemm_triton"


def _decoded_weight_case(dtype):
    device = get_accelerator().current_device_name()
    # The custom layout is E4M3: exponent field 0 is zero or a subnormal. With a
    # scale of 128, 0x00 is 0, 0x01 is 2**-9 * 128 = 0.25, 0x40 is 2 * 128 = 256,
    # and 0xFF is -480 * 128. One quantization group per weight row makes the
    # scale layout the portable kernel's [K, 1]. The one-hot activation rows
    # select weight rows 0 and 1.
    k = n = 64
    group_size = 64
    weight = torch.zeros((k, n), dtype=torch.uint8, device=device)
    weight[0, 0], weight[0, 1], weight[0, 2] = 0x00, 0x01, 0x40
    weight[1, 0], weight[1, 1], weight[1, 2] = 0x80, 0x81, 0xFF
    scale = torch.full((k, 1), 128.0, device=device)
    inp = torch.zeros((2, k), dtype=dtype, device=device)
    inp[0, 0] = 1.0
    inp[1, 1] = 1.0
    expected = torch.zeros((2, n), dtype=dtype, device=device)
    expected[0, 1] = 0.25
    expected[0, 2] = 256.0
    expected[1, 1] = -0.25
    expected[1, 2] = -61440.0
    return inp, weight, scale, group_size, expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp8_gemm_matches_the_decoded_weight(dtype):
    inp, weight, scale, group_size, expected = _decoded_weight_case(dtype)
    out = matmul_fp8_triton(inp, weight, scale, group_size)
    torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_ascend_variant_matches_the_decoded_weight(dtype):
    from deepspeed.ops.triton_ops.npu.fp8_gemm_triton import matmul_fp8_triton as ascend_matmul_fp8_triton
    inp, weight, scale, group_size, expected = _decoded_weight_case(dtype)
    out = ascend_matmul_fp8_triton(inp, weight, scale, group_size)
    torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)

# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Ascend fallback for the custom FP8 GEMM.

The portable Triton kernel decodes DeepSpeed's 8-bit FP format with integer
bit manipulation and multiplies each weight element by its per-group scale.
The triton-ascend backend does not run that kernel on the current Ascend
generation, so this platform variant implements the same decode and GEMM with
PyTorch. The format is ``q_bits=8`` / ``q_mantissa_bits=3`` stored in a
``uint8`` tensor; it is not the hardware float8 dtype, so this fallback does
not require float8 support from the device.
"""

import torch


def _scale_per_element(scale: torch.Tensor, k: int, n: int, group_size: int) -> torch.Tensor:
    """Expand the flattened per-group scales to the ``[k, n]`` weight layout.

    The portable kernel indexes the scale pointer as
    ``k * (n // group_size) + n // group_size``, i.e. a row-major
    ``[k, n // group_size]`` view of the flattened scales.
    """
    groups = n // group_size
    if n % group_size or scale.numel() != k * groups:
        raise RuntimeError("FP8 GEMM expects the weight's column count to be divisible by the quantization "
                           f"group size and one scale per group, got weight {k}x{n}, group size {group_size}, "
                           f"and {scale.numel()} scales.")
    return scale.reshape(k, groups).repeat_interleave(group_size, dim=1)


def _decode_uint8_fp8(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Decode the uint8 8-bit FP layout to ``dtype`` with the portable kernel's arithmetic.

    The 7-bit field is laid out as fp16, whose 4 low exponent bits and subnormals
    line up with E4M3 (exponent field 0 is zero or subnormal); 2^(15 - 7) = 256
    then restores the bias.
    """
    if dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"FP8 GEMM supports bfloat16 and float16 activations, got {dtype}.")
    bits = weight.to(torch.int32)
    fp16_bits = ((bits & 0x80) << 8) | ((bits & 0x7F) << 7)
    return (fp16_bits.to(torch.uint16).view(torch.float16).float() * 256.0).to(dtype)


def matmul_fp8_triton(inp, weight, scale, quantization_group_size):
    """Ascend implementation of ``matmul_fp8_triton``: decode the uint8 FP8 weights and multiply in PyTorch.

    The public entry point in ``deepspeed.ops.fp_quantizer.fp8_gemm`` uses the
    same dispatch and reaches this variant on NPU. The argument and return
    contract is the portable kernel's: a 2-D ``[M, K]`` activation, a 2-D
    ``[K, N]`` uint8 weight, and one float32 scale per quantization group.
    """
    if inp.dim() != 2 or weight.dim() != 2:
        raise RuntimeError("FP8 GEMM expects 2-D inputs, got shapes "
                           f"{tuple(inp.shape)} and {tuple(weight.shape)}.")
    if inp.shape[1] != weight.shape[0]:
        raise RuntimeError(f"Incompatible dimensions (input: {tuple(inp.shape)}, "
                           f"weight: {tuple(weight.shape)}).")
    if weight.dtype != torch.uint8:
        raise RuntimeError(f"FP8 GEMM expects a uint8 quantized weight, got {weight.dtype}.")

    k, n = weight.shape
    weight_dequant = _decode_uint8_fp8(weight, inp.dtype)
    weight_dequant = (weight_dequant.float() * _scale_per_element(scale, k, n, quantization_group_size)).to(inp.dtype)
    return torch.matmul(inp, weight_dequant)

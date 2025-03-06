# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

######## Fused MoE kernel #########
# These kernels are implemented for
# fusing GeMM with dequantization of
# fp8 weight data when using bit-16
# activation.
###################################

import torch


def matmul_fp8(inp, weight, scale, quantization_group_size, quantizer):
    from deepspeed import get_accelerator

    if not get_accelerator().is_triton_supported():
        return matmul_fp8_fallback(inp, weight, scale, quantization_group_size, quantizer)
    else:
        # Import dynamically to prevent failures on systems without triton.
        from .fp8_gemm_triton import matmul_fp8_triton
        return matmul_fp8_triton(inp, weight, scale, quantization_group_size)


def matmul_fp8_fallback(inp, weight, scale, quantization_group_size, quantizer):
    return torch.matmul(inp, quantizer.dequantize(weight, scale=scale))

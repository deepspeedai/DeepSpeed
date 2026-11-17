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
    from deepspeed.ops.triton_ops._triton import is_triton_available

    if get_accelerator().is_triton_supported() and is_triton_available():
        from .fp8_gemm_triton import matmul_fp8_triton
        return matmul_fp8_triton(inp, weight, scale, quantization_group_size)
    return matmul_fp8_fallback(inp, weight, scale, quantization_group_size, quantizer)


def matmul_fp8_fallback(inp, weight, scale, quantization_group_size, quantizer):
    return torch.matmul(inp, quantizer.dequantize(weight, scale=scale))

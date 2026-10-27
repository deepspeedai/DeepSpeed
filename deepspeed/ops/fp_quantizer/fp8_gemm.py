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

    if get_accelerator().is_triton_supported() and is_triton_available() and _fp8_gemm_gated(
            inp, weight, scale, quantization_group_size):
        from .fp8_gemm_triton import matmul_fp8_triton
        return matmul_fp8_triton(inp, weight, scale, quantization_group_size)
    return matmul_fp8_fallback(inp, weight, scale, quantization_group_size, quantizer)


def _fp8_gemm_gated(inp, weight, scale, quantization_group_size) -> bool:
    """The current backend validates the FP8 Triton kernels for itself once per process."""
    from deepspeed.ops.triton_ops import _probes
    from deepspeed.ops.triton_ops._triton import is_triton_kernel_supported

    k, n = weight.shape
    k, n = min(k, 128), min(n, 128)
    if min(k, n) < quantization_group_size:
        # Too small to build a representative probe case; the backend-level answer stands.
        return True
    # The kernel name carries what the probe tested: the weight dtype and the group
    # size are compile-time constants, so distinct values keep distinct answers.
    return is_triton_kernel_supported(f"fp8_gemm:{weight.dtype}:{quantization_group_size}", inp.dtype, inp.device,
                                      _probes.fp8_gemm, inp.dtype, inp.device, weight.dtype, k, n,
                                      quantization_group_size)


def matmul_fp8_fallback(inp, weight, scale, quantization_group_size, quantizer):
    return torch.matmul(inp, quantizer.dequantize(weight, scale=scale))

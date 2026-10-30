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
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

# Largest num_stages that fits in shared memory, per device and tile configuration.
_fitting_num_stages = {}


@triton.jit
def matmul_kernel_fp8_bf16(inp_ptr, weight_ptr, out_ptr, scale_ptr, M, N, K, stride_am, stride_ak, stride_bk,
                           stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                           BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                           quantization_group_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    inp_data = inp_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    weight_data = weight_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    weight_ptrs_offset = offs_k[:, None] * (stride_bk // quantization_group_size) + (
        (pid_n * BLOCK_SIZE_N) // quantization_group_size)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        inp = tl.load(inp_data, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        weight = tl.load(weight_data, mask=offs_k[:, None] < K, other=0.0)
        scale = tl.load(scale_ptr + weight_ptrs_offset + ((k * BLOCK_SIZE_K * stride_bk) // quantization_group_size))
        # Dequantize weight (fp8 -> bf16)
        # The E4M3 bits are laid out as fp16, whose 4 low exponent bits and subnormals line up
        # with E4M3 (exponent field 0 is zero or subnormal); 2^(15 - 7) then restores the bias.
        w = (weight & 0x80).to(tl.uint16) << 8
        w = w | ((weight & 0x7f).to(tl.uint16) << 7)
        w = (w.to(tl.float16, bitcast=True).to(tl.float32) * (scale * 256.0)).to(tl.bfloat16)

        inp_data += BLOCK_SIZE_K * stride_ak
        weight_data += BLOCK_SIZE_K * stride_bk

        accumulator += tl.dot(inp, w)

    out = accumulator.to(tl.bfloat16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_data = out_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(out_data, out, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.jit
def matmul_kernel_fp8_fp16(inp_ptr, weight_ptr, out_ptr, scale_ptr, M, N, K, stride_am, stride_ak, stride_bk,
                           stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                           BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                           quantization_group_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    inp_data = inp_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    weight_data = weight_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    weight_ptrs_offset = offs_k[:, None] * (stride_bk // quantization_group_size) + (
        (pid_n * BLOCK_SIZE_N) // quantization_group_size)

    weight = tl.load(weight_data, mask=offs_k[:, None] < K, other=0.0)
    scale = tl.load(scale_ptr + weight_ptrs_offset)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        inp = tl.load(inp_data, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        # Dequantize weight (fp8 -> fp16)
        w = (weight & 0x80).to(tl.uint16) << 8
        w = w | ((weight & 0x7f).to(tl.uint16) << 7)
        w = (w.to(tl.float16, bitcast=True).to(tl.float32) * (scale * 256.0)).to(tl.float16)

        inp_data += BLOCK_SIZE_K * stride_ak
        weight_data += BLOCK_SIZE_K * stride_bk

        weight = tl.load(weight_data, mask=offs_k[:, None] < K - (k + 1) * BLOCK_SIZE_K, other=0.0)
        scale = tl.load(scale_ptr + (weight_ptrs_offset +
                                     (((k + 1) * BLOCK_SIZE_K * stride_bk) // quantization_group_size)))

        accumulator += tl.dot(inp, w)

    out = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_data = out_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(out_data, out, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def _launch_with_fitting_num_stages(kernel, grid, args, num_stages, num_warps, **tile_config):
    # Each pipeline stage keeps an input and a weight tile in shared memory, so the preferred
    # num_stages can exceed the per-block limit, which differs across GPUs (e.g. the large-M
    # tiles need 263168 bytes with 3 stages but H100/H200 allow 232448). Triton measures the
    # exact usage of the compiled kernel against the device limit, so fall back to fewer stages
    # when it does not fit.
    inp = args[0]
    cache_key = (inp.device, inp.dtype, num_warps, tuple(sorted(tile_config.items())))
    num_stages = _fitting_num_stages.get(cache_key, num_stages)
    while True:
        try:
            kernel[grid](*args, num_stages=num_stages, num_warps=num_warps, **tile_config)
            break
        except OutOfResources as e:
            if e.name != "shared memory" or num_stages == 1:
                raise
            num_stages -= 1
    _fitting_num_stages[cache_key] = num_stages


def matmul_fp8_triton(inp, weight, scale, quantization_group_size):

    assert inp.shape[1] == weight.shape[0], \
        f"Incompatible dimensions (input: {inp.shape}, weight: {weight.shape})"

    M, K = inp.shape
    K, N = weight.shape

    out = torch.empty((M, N), device=inp.device, dtype=inp.dtype)

    # GEMM tuning parameters!
    # TODO: Add a more configurable tuning for selecting the best GeMM
    BLOCK_SIZE_M = 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = max(64, quantization_group_size)
    GROUP_SIZE_M = 8
    num_stages = 4
    num_warps = 4
    if M >= 256:
        BLOCK_SIZE_M = 256
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = max(128, quantization_group_size)
        num_stages = 3
        num_warps = 8

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    kernel = matmul_kernel_fp8_bf16 if inp.dtype == torch.bfloat16 else matmul_kernel_fp8_fp16
    args = (inp, weight, out, scale, M, N, K, inp.stride(0), inp.stride(1), weight.stride(0), weight.stride(1),
            out.stride(0), out.stride(1))
    _launch_with_fitting_num_stages(kernel,
                                    grid,
                                    args,
                                    num_stages,
                                    num_warps,
                                    quantization_group_size=quantization_group_size,
                                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                                    GROUP_SIZE_M=GROUP_SIZE_M)
    return out

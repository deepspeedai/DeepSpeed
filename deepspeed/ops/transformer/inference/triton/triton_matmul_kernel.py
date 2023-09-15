# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import triton
import triton.language as tl
from .gelu import gelu_functor
import torch

AUTOTUNE_TOP_K = 10
SKIP_AUTOTUNE = False


def _triton_ops_matmul_early_config_prune(configs, named_args):
    device = torch.cuda.current_device()  #ignore-cuda
    capability = torch.cuda.get_device_capability()  #ignore-cuda
    # BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages
    dtsize = named_args['A'].element_size()
    dtype = named_args['A'].dtype

    # 1. make sure we have enough smem
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages = \
            kw['BLOCK_M'], kw['BLOCK_N'], kw['BLOCK_K'], config.num_stages

        max_shared_memory = triton.runtime.driver.utils.get_device_properties(device)["max_shared_mem"]
        required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory <= max_shared_memory:
            pruned_configs.append(config)

    return pruned_configs


def _fp16_matmul_prune_config(configs, named_args, skip_autotune=SKIP_AUTOTUNE):
    if skip_autotune:
        configs = [configs[0]]
    else:
        configs = _triton_ops_matmul_early_config_prune(configs, named_args)
    return configs


"""
fp16 matmul implementation is adapted from triton matmul:
https://github.com/openai/triton/blob/34817ecc954a6f4ca7b4dfb352fdde1f8bd49ca5/python/triton/ops/matmul.py
"""


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 256,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_M': 256,
            'BLOCK_N': 128,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_M': 256,
            'BLOCK_N': 64,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 256,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 128,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 64,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 128,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 32,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 32,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=5, num_warps=2),
    ],
    key=['CACHE_M', 'CACHE_N', 'CACHE_K'],
    prune_configs_by={
        'early_config_prune': _fp16_matmul_prune_config,
        'perf_model': None,
        'top_k': AUTOTUNE_TOP_K
    },
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def _fp_matmul(
    A,
    B,
    C,
    M,
    N,
    K,
    bias,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    CACHE_M,
    CACHE_N,
    CACHE_K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    BIAS_ADD: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K * SPLIT_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    # bias addition
    if BIAS_ADD:
        bias_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        bias_ptr = bias + bias_offset
        b = tl.load(bias_ptr, mask=bias_offset < N)
        acc = acc + b[None, :]
    # activation
    if ACTIVATION == "relu":
        acc = tl.where(acc >= 0, acc, 0)
    elif ACTIVATION == "leaky_relu":
        acc = tl.where(acc >= 0, acc, 0.01 * acc)
    elif ACTIVATION == "gelu":
        #acc = tl.sigmoid(1.702 * acc) * acc
        acc = gelu_functor(acc)
    elif ACTIVATION == "sigmoid":
        acc = tl.sigmoid(acc)  # sigmoid
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


def matmul_4d_prune_config(configs, named_args, skip_autotune=SKIP_AUTOTUNE):
    if skip_autotune:
        configs = [configs[0]]
    else:
        device = torch.cuda.current_device()  #ignore-cuda
        capability = torch.cuda.get_device_capability()  #ignore-cuda
        # BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages
        dtsize = named_args['a_ptr'].element_size()
        dtype = named_args['a_ptr'].dtype

        # make sure we have enough smem
        pruned_configs = []
        for config in configs:
            kw = config.kwargs
            BLOCK_M, BLOCK_N, BLOCK_K, num_stages = \
                kw['BLOCK_SIZE_M'], kw['BLOCK_SIZE_N'], kw['BLOCK_SIZE_K'], config.num_stages

            max_shared_memory = triton.runtime.driver.utils.get_device_properties(device)["max_shared_mem"]
            required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
            if required_shared_memory <= max_shared_memory:
                pruned_configs.append(config)
        configs = pruned_configs
    return configs


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8
            },
            num_stages=1,  # this is mainly for unit test, to minimize the share memory usage
            num_warps=8),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
    ],
    key=['CACHE_M', 'CACHE_N', 'CACHE_K'],
    prune_configs_by={
        'early_config_prune': matmul_4d_prune_config,
        'perf_model': None,
        'top_k': AUTOTUNE_TOP_K
    },
)
@triton.jit
def matmul_4d_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    CACHE_M,
    CACHE_N,
    CACHE_K,
    stride_ab,
    stride_ah,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bh,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_ch,
    stride_cm,
    stride_cn,
    scale,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MASK: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    head = tl.program_id(axis=1)
    batch = tl.program_id(axis=2)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if MASK:
        if (pid_m + 1) * BLOCK_SIZE_M - 1 < pid_n * BLOCK_SIZE_N:
            c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=c_ptr.dtype.element_ty) - float("inf")
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = (c_ptr + batch * stride_cb + head * stride_ch + stride_cm * offs_cm[:, None] +
                      stride_cn * offs_cn[None, :])
            tl.store(c_ptrs, c)
            return

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (a_ptr + batch * stride_ab + head * stride_ah +
              (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak))
    b_ptrs = (b_ptr + batch * stride_bb + head * stride_bh +
              (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] + k < K)
        b_mask = (offs_k[:, None] + k < K) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.)
        b = tl.load(b_ptrs, mask=b_mask, other=0.)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(c_ptr.dtype.element_ty)
    if scale > 0:
        c = c * scale.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if MASK:
        c += tl.where(offs_cm[:, None] >= offs_cn[None, :], 0, float("-inf"))
    c_ptrs = (c_ptr + batch * stride_cb + head * stride_ch + stride_cm * offs_cm[:, None] +
              stride_cn * offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

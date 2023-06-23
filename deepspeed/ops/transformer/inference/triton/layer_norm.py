# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import triton
import triton.language as tl
'''
layer-normalization
modified the triton kernel in
https://github.com/openai/triton/blob/34817ecc954a6f4ca7b4dfb352fdde1f8bd49ca5/python/tutorials/05-layer-norm.py
'''


@triton.jit
def layer_norm_kernel(
    Out,
    A,
    Weight,
    Bias,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # position of elements processed by this program
    row = tl.program_id(0)
    Out += row * stride
    A += row * stride
    # compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(A + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(A + cols, mask=cols < N, other=0.0).to(tl.float32)
        a = tl.where(cols < N, a - mean, 0.0)
        _var += a * a
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # multiply by weight and add bias
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        weight = tl.load(Weight + cols, mask=mask)
        bias = tl.load(Bias + cols, mask=mask)
        a = tl.load(A + cols, mask=mask, other=0.0).to(tl.float32)
        a_hat = (a - mean) * rstd
        out = a_hat * weight + bias
        # # write-back
        tl.store(Out + cols, out, mask=mask)


@triton.jit
def layer_norm_residual_kernel(
    Out,
    A,
    Residual,
    ln_input,
    Weight,
    Bias,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # position of elements processed by this program
    row = tl.program_id(0)
    Out += row * stride
    A += row * stride
    Residual += row * stride
    ln_input += row * stride
    # compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(A + cols, mask=cols < N, other=0.0).to(tl.float32)
        res = tl.load(Residual + cols, mask=cols < N, other=0.0).to(tl.float32)
        a = a + res
        tl.store(ln_input + cols, a, mask=cols < N)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(ln_input + cols, mask=cols < N, other=0.0).to(tl.float32)
        a = tl.where(cols < N, a - mean, 0.0)
        _var += a * a
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # multiply by weight and add bias
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        weight = tl.load(Weight + cols, mask=mask)
        bias = tl.load(Bias + cols, mask=mask)
        a = tl.load(ln_input + cols, mask=mask, other=0.0).to(tl.float32)
        a_hat = (a - mean) * rstd
        out = a_hat * weight + bias
        # write-back
        tl.store(Out + cols, out, mask=mask)


@triton.jit
def layer_norm_residual_bias_kernel(
    Out,
    A,
    Residual,
    InputBias,
    ln_input,
    Weight,
    Bias,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # position of elements processed by this program
    row = tl.program_id(0)
    Out += row * stride
    A += row * stride
    Residual += row * stride
    ln_input += row * stride
    # compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(A + cols, mask=cols < N, other=0.0).to(tl.float32)
        res = tl.load(Residual + cols, mask=cols < N, other=0.0).to(tl.float32)
        b = tl.load(InputBias + cols, mask=cols < N, other=0.0).to(tl.float32)
        a = a + b + res
        tl.store(ln_input + cols, a, mask=cols < N)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(ln_input + cols, mask=cols < N, other=0.0).to(tl.float32)
        a = tl.where(cols < N, a - mean, 0.0)
        _var += a * a
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # multiply by weight and add bias
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        weight = tl.load(Weight + cols, mask=mask)
        bias = tl.load(Bias + cols, mask=mask)
        a = tl.load(ln_input + cols, mask=mask, other=0.0).to(tl.float32)
        a_hat = (a - mean) * rstd
        out = a_hat * weight + bias
        # write-back
        tl.store(Out + cols, out, mask=mask)


def layer_norm(a, weight, bias, eps):
    assert a.is_contiguous()
    assert weight.is_contiguous()
    assert bias.is_contiguous()

    # allocate output
    out = torch.empty_like(a)
    # reshape input data into 2D tensor
    a_arg = a.view(-1, a.shape[-1])
    M, N = a_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // a.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    BLOCK_SIZE = BLOCK_SIZE if N <= 4096 else 8192
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    layer_norm_kernel[(M, )](
        out,
        a_arg,
        weight,
        bias,
        a_arg.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


def layer_norm_residual(a, input_bias, residual, weight, bias, eps):
    assert a.is_contiguous()
    assert weight.is_contiguous()
    assert bias.is_contiguous()
    assert residual.is_contiguous()

    # allocate output and scratch-pad for residual addition
    out = torch.empty_like(a)
    ln_input = torch.empty_like(a)
    # reshape input data into 2D tensor
    a_arg = a.view(-1, a.shape[-1])
    residual = residual.view(-1, residual.shape[-1])
    M, N = a_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // a.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    BLOCK_SIZE = BLOCK_SIZE if N <= 4096 else 8192
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    if input_bias is None:
        layer_norm_residual_kernel[(M, )](
            out,
            a_arg,
            residual,
            ln_input,
            weight,
            bias,
            a_arg.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        layer_norm_residual_bias_kernel[(M, )](
            out,
            a_arg,
            residual,
            input_bias,
            ln_input,
            weight,
            bias,
            a_arg.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    return out

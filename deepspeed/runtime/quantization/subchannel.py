# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Subchannel (block-scaled) FP8/INT8 matmuls for training.

Every linear layer runs three GEMMs: the forward ``y = x W^T``, the input gradient ``dx = dy W`` and the weight
gradient ``dW = dy^T x``. Each is quantized on its own terms (:class:`GemmQuantizationConfig`): both operands
are cut along that GEMM's reduction dimension into blocks with one FP32 scale each (or scaled as a whole,
``block_size`` 0), optionally rotated by a Hadamard transform first, and rounded to nearest or stochastically.

One autograd function, :func:`subchannel_grouped_matmul`, covers both module kinds. A dense ``nn.Linear`` is the
one-group case; AutoEP grouped experts pass the per-expert token boundaries. The weight-gradient reduction runs
over tokens, so its blocks restart at each expert's first token and never mix two experts.

GEMMs run on the Triton kernels of :mod:`deepspeed.ops.triton_ops.subchannel_quant` when the device supports
them, and otherwise on the PyTorch reference below, which quantizes the same way and multiplies the dequantized
operands in FP32.
"""

from __future__ import annotations

import fnmatch
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.quantization.config import GemmQuantizationConfig, SubchannelQuantizedMatmulConfig
from deepspeed.utils import logger
from deepspeed.utils.logging import warning_once

# format -> (largest magnitude, mantissa bits, smallest normal exponent, storage dtype)
_FORMAT_SPECS = {
    "e4m3": (448.0, 3, -6, torch.float8_e4m3fn),
    "e5m2": (57344.0, 2, -14, torch.float8_e5m2),
    "int8": (127.0, 0, 0, torch.int8),
}

# ---------------------------------------------------------------------------
# Reference numerics (PyTorch)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _hadamard_matrix(size: int, device: torch.device) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard matrix of order ``size``."""
    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.shape[0] < size:
        matrix = torch.cat([torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)], 0)
    return (matrix / math.sqrt(size)).to(device)


def _round_to_format(y: torch.Tensor, fmt: str, stochastic: bool) -> torch.Tensor:
    """Round ``y`` (already divided by its block scale) to the values ``fmt`` represents, as FP32."""
    qmax, mantissa_bits, min_exp, storage_dtype = _FORMAT_SPECS[fmt]
    if stochastic:
        noise = torch.rand_like(y)
        if fmt == "int8":
            y = torch.floor(y + noise)
        else:
            magnitude = y.abs()
            exponent = torch.clamp(torch.floor(torch.log2(magnitude)), min=min_exp)
            step = torch.exp2(exponent - mantissa_bits)
            low = torch.floor(magnitude / step) * step
            magnitude = low + torch.where(noise * step < magnitude - low, step, torch.zeros_like(step))
            y = torch.copysign(magnitude, y)
        return y.clamp(-qmax, qmax)
    y = y.clamp(-qmax, qmax)
    if fmt == "int8":
        # Half away from zero, as the kernel rounds.
        return torch.copysign(torch.floor(y.abs() + 0.5), y)
    return y.to(storage_dtype).float()


def reference_quantize(x: torch.Tensor, cfg: GemmQuantizationConfig) -> torch.Tensor:
    """Quantize and dequantize ``x`` along its last dimension, in the Hadamard-rotated basis when enabled.

    The last dimension is zero-padded to a multiple of ``cfg.block_size``; the result is FP32.
    """
    length = x.shape[-1]
    # block_size 0: the whole reduction dimension is one block.
    block = cfg.block_size or max(length, 1)
    padded = math.ceil(length / block) * block
    y = F.pad(x.float(), (0, padded - length))
    if cfg.hadamard_size:
        h = cfg.hadamard_size
        y = (y.reshape(*y.shape[:-1], padded // h, h) @ _hadamard_matrix(h, y.device)).reshape(y.shape)
    blocks = y.reshape(*y.shape[:-1], padded // block, block)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / _FORMAT_SPECS[cfg.format][0], torch.ones_like(amax))
    blocks = _round_to_format(blocks / scale, cfg.format, cfg.stochastic_rounding) * scale
    return blocks.reshape(y.shape)


def _group_bounds(group_ends: torch.Tensor):
    ends = group_ends.tolist()
    return list(zip([0] + ends[:-1], ends))


def _reference_rows_mm(a: torch.Tensor, b: torch.Tensor, group_ends: torch.Tensor,
                       cfg: GemmQuantizationConfig) -> torch.Tensor:
    a_q = reference_quantize(a, cfg)
    b_q = reference_quantize(b, cfg)
    out = a_q.new_zeros(a.shape[0], b.shape[1])
    for group, (start, end) in enumerate(_group_bounds(group_ends)):
        out[start:end] = a_q[start:end] @ b_q[group].transpose(0, 1)
    return out


def _reference_wgrad(dy: torch.Tensor, x: torch.Tensor, group_ends: torch.Tensor,
                     cfg: GemmQuantizationConfig) -> torch.Tensor:
    bounds = _group_bounds(group_ends)
    out = dy.new_zeros(len(bounds), dy.shape[1], x.shape[1], dtype=torch.float32)
    for group, (start, end) in enumerate(bounds):
        if end > start:
            dy_q = reference_quantize(dy[start:end].transpose(0, 1), cfg)
            x_q = reference_quantize(x[start:end].transpose(0, 1), cfg)
            out[group] = dy_q @ x_q.transpose(0, 1)
    return out


# ---------------------------------------------------------------------------
# GEMM dispatch
# ---------------------------------------------------------------------------


def _use_kernels(tensor: torch.Tensor, fmt: str) -> bool:
    accelerator = get_accelerator()
    if tensor.device.type != accelerator.device_name():
        return False
    return fmt in accelerator.supported_subchannel_quantization_formats()


def _warn_reference_fallback(fmt: str, device: torch.device) -> None:
    warning_once(f"Subchannel {fmt} GEMMs on {device} run on the PyTorch reference implementation, which "
                 "simulates the quantization and multiplies in FP32; this accelerator has no fused kernel for "
                 "the format (on CUDA they need Triton and compute capability 8.9+ for FP8, 8.0+ for INT8).")


def _random_seed(cfg: GemmQuantizationConfig) -> int:
    if not cfg.stochastic_rounding:
        return 0
    # Drawn from the CPU generator: reproducible under torch.manual_seed and free of device synchronization.
    return int(torch.randint(0, 2**31 - 1, (1, )).item())


def _high_precision_rows_mm(a: torch.Tensor, b: torch.Tensor, group_ends: torch.Tensor) -> torch.Tensor:
    if b.shape[0] == 1:
        return a @ b[0].transpose(0, 1)
    # torch._grouped_mm has no CPU kernel and takes BF16 only.
    on_device = a.device.type != "cpu" and get_accelerator().on_accelerator(a)
    if on_device and a.dtype == torch.bfloat16 and hasattr(torch, "_grouped_mm"):
        return torch._grouped_mm(a, b.transpose(-2, -1), offs=group_ends)
    out = a.new_zeros(a.shape[0], b.shape[1])
    for group, (start, end) in enumerate(_group_bounds(group_ends)):
        out[start:end] = a[start:end] @ b[group].transpose(0, 1)
    return out


def _high_precision_wgrad(dy: torch.Tensor, x: torch.Tensor, group_ends: torch.Tensor) -> torch.Tensor:
    if group_ends.shape[0] == 1:
        # Rows past the single group's end are padding; the dense caller has none.
        return (dy.transpose(0, 1) @ x).unsqueeze(0)
    # Not torch._grouped_mm: on torch 2.8 its 2D x 2D form never returns when an expert has no tokens.
    out = dy.new_zeros(group_ends.shape[0], dy.shape[1], x.shape[1])
    for group, (start, end) in enumerate(_group_bounds(group_ends)):
        out[group] = dy[start:end].transpose(0, 1) @ x[start:end]
    return out


def _quantized_rows_mm(a: torch.Tensor, b: torch.Tensor, group_ends: torch.Tensor,
                       cfg: GemmQuantizationConfig) -> torch.Tensor:
    if not _use_kernels(a, cfg.format):
        _warn_reference_fallback(cfg.format, a.device)
        return _reference_rows_mm(a, b, group_ends, cfg).to(a.dtype)

    if cfg.block_size == 0:
        from deepspeed.ops.triton_ops.subchannel_quant import quantize_rows, row_scaled_grouped_mm
        options = dict(fmt=cfg.format, stochastic=cfg.stochastic_rounding)
        a_q, a_amax = quantize_rows(a.unsqueeze(0), seed=_random_seed(cfg), **options)
        b_q, b_amax = quantize_rows(b, seed=_random_seed(cfg), **options)
        return row_scaled_grouped_mm(a_q[0],
                                     a_amax[0, :, 0],
                                     b_q,
                                     b_amax[:, :, 0],
                                     group_ends,
                                     a.dtype,
                                     fmt=cfg.format)

    from deepspeed.ops.triton_ops.subchannel_quant import quantize_blocks, block_scaled_grouped_mm
    options = dict(fmt=cfg.format,
                   block_size=cfg.block_size,
                   hadamard_size=cfg.hadamard_size,
                   stochastic=cfg.stochastic_rounding)
    a_q, a_scale, _ = quantize_blocks(a.unsqueeze(0), seed=_random_seed(cfg), **options)
    b_q, b_scale, _ = quantize_blocks(b, seed=_random_seed(cfg), **options)
    return block_scaled_grouped_mm(a_q[0], a_scale[0], b_q, b_scale, group_ends, cfg.block_size, a.dtype)


def _rows_mm(a: torch.Tensor, b: torch.Tensor, group_ends: torch.Tensor, cfg: GemmQuantizationConfig) -> torch.Tensor:
    """``out[m] = a[m] b[e]^T`` for the rows ``m`` of group ``e``; ``a`` is ``[M, K]``, ``b`` ``[E, N, K]``.

    Rows past the last group's end are zero.
    """
    if cfg.quantized:
        out = _quantized_rows_mm(a, b, group_ends, cfg)
    else:
        out = _high_precision_rows_mm(a, b, group_ends).to(a.dtype)
    if b.shape[0] > 1:
        # No GEMM writes the rows past the last group (AutoEP alignment padding); keep them finite.
        tail = torch.arange(a.shape[0], device=a.device) >= group_ends[-1]
        out = out.masked_fill(tail.unsqueeze(1), 0)
    return out


def _wgrad(dy: torch.Tensor, x: torch.Tensor, group_ends: torch.Tensor, cfg: GemmQuantizationConfig,
           out_dtype: torch.dtype) -> torch.Tensor:
    """``out[e] = dy[rows(e)]^T x[rows(e)]``: ``[E, N, K]`` from ``dy`` ``[M, N]`` and ``x`` ``[M, K]``."""
    if not cfg.quantized:
        return _high_precision_wgrad(dy, x, group_ends).to(out_dtype)
    if not _use_kernels(dy, cfg.format):
        _warn_reference_fallback(cfg.format, dy.device)
        return _reference_wgrad(dy, x, group_ends, cfg).to(out_dtype)

    if cfg.block_size == 0:
        from deepspeed.ops.triton_ops.subchannel_quant import quantize_rows, row_scaled_grouped_wgrad
        # A dense layer's single group spans every row; telling the kernels so keeps them on their fastest loop.
        spans = None if group_ends.shape[0] == 1 else group_ends
        options = dict(fmt=cfg.format, stochastic=cfg.stochastic_rounding, group_ends=spans)
        dy_q, dy_amax = quantize_rows(dy.transpose(0, 1).unsqueeze(0), seed=_random_seed(cfg), **options)
        x_q, x_amax = quantize_rows(x.transpose(0, 1).unsqueeze(0), seed=_random_seed(cfg), **options)
        return row_scaled_grouped_wgrad(dy_q[0], dy_amax[0], x_q[0], x_amax[0], spans, out_dtype, fmt=cfg.format)

    from deepspeed.ops.triton_ops.subchannel_quant import quantize_blocks, block_scaled_grouped_wgrad
    options = dict(fmt=cfg.format,
                   block_size=cfg.block_size,
                   hadamard_size=cfg.hadamard_size,
                   stochastic=cfg.stochastic_rounding,
                   group_ends=group_ends)
    dy_q, dy_scale, block_ends = quantize_blocks(dy.transpose(0, 1).unsqueeze(0), seed=_random_seed(cfg), **options)
    x_q, x_scale, _ = quantize_blocks(x.transpose(0, 1).unsqueeze(0), seed=_random_seed(cfg), **options)
    return block_scaled_grouped_wgrad(dy_q[0], dy_scale[0], x_q[0], x_scale[0], block_ends, cfg.block_size, out_dtype)


class _SubchannelGroupedMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, group_ends, recipe):
        ctx.save_for_backward(x, weight, group_ends)
        ctx.recipe = recipe
        return _rows_mm(x, weight, group_ends, recipe.forward)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, group_ends = ctx.saved_tensors
        recipe = ctx.recipe
        grad_out = grad_out.contiguous()
        grad_x = grad_weight = None
        if ctx.needs_input_grad[0]:
            # dx = dy W reduces over the output features, the last dimension of the transposed view.
            grad_x = _rows_mm(grad_out, weight.transpose(-2, -1), group_ends, recipe.dx)
        if ctx.needs_input_grad[1]:
            grad_weight = _wgrad(grad_out, x, group_ends, recipe.dw, weight.dtype)
        return grad_x, grad_weight, None, None


# Keep torch.compile from tracing the host code around the kernels: traced, the kernels return silently wrong results
# (row scaling most of all), so this function always runs eagerly inside a compiled model.
_run_eagerly = getattr(getattr(torch, "compiler", None), "disable", lambda fn: fn)


@_run_eagerly
def subchannel_grouped_matmul(x: torch.Tensor, weight: torch.Tensor, group_ends: torch.Tensor,
                              recipe: SubchannelQuantizedMatmulConfig) -> torch.Tensor:
    """``y[m] = x[m] weight[e]^T`` for the rows ``m`` of group ``e``, with quantized forward and backward GEMMs.

    Args:
        x: ``[M, K]`` activations, grouped contiguously by expert.
        weight: ``[E, N, K]`` expert weights, in the dtype of ``x``.
        group_ends: ``[E]`` int32 cumulative row counts; rows past ``group_ends[-1]`` produce zeros.
        recipe: The three GEMMs' quantization settings.
    """
    return _SubchannelGroupedMatmul.apply(x, weight, group_ends, recipe)


def subchannel_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                      recipe: SubchannelQuantizedMatmulConfig) -> torch.Tensor:
    """``F.linear`` with quantized forward and backward GEMMs."""
    rows = x.reshape(-1, x.shape[-1])
    group_ends = torch.full((1, ), rows.shape[0], dtype=torch.int32, device=x.device)
    out = subchannel_grouped_matmul(rows, weight.to(x.dtype).unsqueeze(0), group_ends, recipe)
    out = out.reshape(*x.shape[:-1], weight.shape[0])
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


# ---------------------------------------------------------------------------
# Module conversion
# ---------------------------------------------------------------------------


class SubchannelQuantizedLinear(nn.Linear):
    """An ``nn.Linear`` whose GEMMs run through :func:`subchannel_linear`.

    Produced in place from an existing ``nn.Linear`` by :func:`convert_to_subchannel_quantization`, which keeps
    its parameters, hooks and state-dict keys.
    """

    subchannel_recipe: SubchannelQuantizedMatmulConfig

    def forward(self, input):
        return subchannel_linear(input, self.weight, self.bias, self.subchannel_recipe)


def _selected(name: str, recipe: SubchannelQuantizedMatmulConfig) -> bool:
    if not any(fnmatch.fnmatchcase(name, pattern) for pattern in recipe.modules):
        return False
    return not any(fnmatch.fnmatchcase(name, pattern) for pattern in recipe.exclude_modules)


def convert_to_subchannel_quantization(model: nn.Module, recipe: SubchannelQuantizedMatmulConfig) -> list[str]:
    """Route the GEMMs of the selected ``nn.Linear`` modules and AutoEP grouped experts through ``recipe``.

    Modules are converted in place. Returns the qualified names of the converted modules; raises if the
    patterns select none, so a typo cannot silently train in full precision.
    """
    from deepspeed.moe.ep_experts import GroupedExperts

    converted = []
    for name, module in model.named_modules():
        if not _selected(name, recipe):
            continue
        # Exact type: subclasses such as MultiheadAttention's out_proj are used without calling forward().
        if type(module) is nn.Linear:
            module.__class__ = SubchannelQuantizedLinear
            module.subchannel_recipe = recipe
            converted.append(name)
        elif isinstance(module, GroupedExperts):
            if module.gate_up_impl == "fused":
                raise ValueError(f"quantization.subchannel runs the gate and up projections of '{name}' as two "
                                 'grouped GEMMs; leave the AutoEP gate_up_impl unset or "separate".')
            module.grouped_mm_fn = functools.partial(subchannel_grouped_matmul, recipe=recipe)
            converted.append(name)

    for name, module in model.named_modules():
        experts = getattr(module, "experts", None)
        if getattr(experts, "grouped_mm_fn", None) is not None and getattr(module, "overlap_weight_grad", False):
            raise ValueError(f"quantization.subchannel does not support AutoEP overlap_weight_grad, which '{name}' "
                             "enables. Disable overlap_weight_grad or exclude the layer's experts.")
        if getattr(experts, "grouped_mm_fn", None) is not None and getattr(module, "dw_staging", None) is not None:
            raise ValueError(f"quantization.subchannel does not support AutoEP deferred_dw_flush_period > 1, which "
                             f"'{name}' sets. Leave deferred_dw_flush_period unset or exclude the layer's experts.")

    if not converted:
        raise ValueError(f"quantization.subchannel is enabled but no nn.Linear or AutoEP expert module matches "
                         f"modules={recipe.modules} and exclude_modules={recipe.exclude_modules}.")
    logger.info(f"quantization.subchannel: converted {len(converted)} module(s)")
    return converted

# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Native BF16/FP16 grouped GEMM with explicit list-input autograd."""

import torch


def _native_mm(x, weight, offsets, group_type=0):
    import torch_npu

    return torch_npu.npu_grouped_matmul([x], [weight],
                                        group_list=offsets,
                                        group_type=group_type,
                                        group_list_type=0,
                                        split_item=3,
                                        output_dtype=x.dtype)[0]


class _GroupedMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, offsets, trans_b):
        ctx.trans_b = trans_b
        ctx.input_shape = x.shape
        ctx.weight_shape = weight.shape
        ctx.empty = x.shape[0] == 0
        saved = [offsets.to(torch.int64)]
        if ctx.needs_input_grad[0]:
            saved.append(weight)
        if ctx.needs_input_grad[1]:
            saved.append(x)
        ctx.save_for_backward(*saved)
        if ctx.empty:
            columns = weight.shape[-2] if trans_b else weight.shape[-1]
            return x.new_empty((0, columns))
        right = weight.transpose(-1, -2) if trans_b else weight
        return _native_mm(x, right, saved[0])

    @staticmethod
    def backward(ctx, grad):
        if ctx.empty:
            return (grad.new_zeros(ctx.input_shape) if ctx.needs_input_grad[0] else None,
                    grad.new_zeros(ctx.weight_shape) if ctx.needs_input_grad[1] else None, None, None)
        saved = iter(ctx.saved_tensors)
        offsets = next(saved)
        weight = next(saved) if ctx.needs_input_grad[0] else None
        x = next(saved) if ctx.needs_input_grad[1] else None
        grad = grad.contiguous()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            right = weight if ctx.trans_b else weight.transpose(-1, -2)
            dx = _native_mm(grad, right, offsets)
        if ctx.needs_input_grad[1]:
            # Produce dW directly in the parameter's packed storage layout.
            left, right = (grad.transpose(0, 1), x) if ctx.trans_b else (x.transpose(0, 1), grad)
            dw = _native_mm(left, right, offsets, group_type=2)
        return dx, dw, None, None


def grouped_mm(x, weight, offsets, *, trans_b=False):
    if not isinstance(trans_b, bool):
        raise TypeError("trans_b must be a boolean")
    if x.ndim != 2 or weight.ndim != 3 or offsets.ndim != 1:
        raise ValueError("Expected token matrix, packed expert weights and cumulative offsets")
    inner = weight.shape[-1] if trans_b else weight.shape[-2]
    if not offsets.numel() or weight.shape[0] != offsets.numel() or x.shape[1] != inner:
        raise ValueError("Grouped matmul shapes do not match")
    if x.device.type != "npu" or weight.device != x.device or offsets.device != x.device:
        raise ValueError("Grouped matmul inputs must share one NPU device")
    if offsets.dtype not in (torch.int32, torch.int64):
        raise TypeError("Grouped offsets must be int32 or int64")
    if x.dtype not in (torch.bfloat16, torch.float16) or weight.dtype != x.dtype:
        raise TypeError("Native grouped matmul requires matching BF16 or FP16 operands")
    return _GroupedMatmul.apply(x, weight, offsets, trans_b)

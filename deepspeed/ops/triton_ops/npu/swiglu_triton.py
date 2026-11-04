# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Ascend (NPU) variants of the fused Triton SwiGLU kernels.

The kernels are flat elementwise passes over the elements, in the style of
flash-linear-attention's ``fla/modules/activations/triton_ascend.py``: the
tile is budgeted from the Ascend unified buffer (see ``_ub``) instead of a
fixed constant, and the backward gets a smaller budget than the forward
because it keeps more live temporaries - the UB-derived forward tile lands at
the portable 2048, the backward would too, but its compiled footprint at 2048
overflows the unified buffer for a single output element on triton-ascend
(reported upstream), so the packed backward keeps the verified 1024-element
tile until the backend fix lands. The kernel arithmetic is the portable
module's unchanged, so outputs and gradients stay elementwise equal to eager.

The per-hardware dispatch (``_triton.dispatch``) picks this module's
``swiglu`` and ``swiglu_packed`` over the portable ones on Ascend. The steps
the platform does not change - validation, the runtime gate, the eager
fallback - are imported from the portable module, so the variants only fork
what actually differs.
"""

import torch

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, current_device, triton
from deepspeed.ops.triton_ops.swiglu_triton import (_swiglu_eager, _swiglu_packed_eager, _swiglu_packed_supported,
                                                    _swiglu_supported, _validate_packed_input)

if _TRITON_AVAILABLE:
    from deepspeed.ops.triton_ops.npu._ub import compute_activation_block_size
    from deepspeed.ops.triton_ops.swiglu_triton import (_swiglu_bwd_kernel, _swiglu_fwd_kernel,
                                                        _swiglu_packed_bwd_kernel, _swiglu_packed_fwd_kernel)

    # The packed backward's verified Ascend tile (see the module docstring).
    _PACKED_BWD_BLOCK_SIZE = 1024

    def _activation_block(n_elements: int, *, is_backward: bool) -> int:
        """FLA-style UB-budgeted tile for the flat elementwise kernels."""
        return compute_activation_block_size(n_elements, is_backward)

    class _SwiGLUFn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            gate = gate.contiguous()
            up = up.contiguous()
            out = torch.empty_like(gate)

            n_elements = gate.numel()
            if n_elements > 0:
                block = _activation_block(n_elements, is_backward=False)
                grid = (triton.cdiv(n_elements, block), )
                # Triton uses the current device and stream, which may differ from the input tensor's device.
                with current_device(gate):
                    _swiglu_fwd_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE=block)

            ctx.save_for_backward(gate, up)
            return out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            gate, up = ctx.saved_tensors
            grad_out = grad_out.contiguous()

            grad_gate = torch.empty_like(gate)
            grad_up = torch.empty_like(up)

            n_elements = gate.numel()
            if n_elements > 0:
                block = _activation_block(n_elements, is_backward=True)
                grid = (triton.cdiv(n_elements, block), )
                with current_device(gate):
                    _swiglu_bwd_kernel[grid](grad_out, gate, up, grad_gate, grad_up, n_elements, BLOCK_SIZE=block)

            return grad_gate, grad_up

    class _SwiGLUPackedFn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, gate_up: torch.Tensor) -> torch.Tensor:
            gate_up = gate_up.contiguous()
            inter = gate_up.shape[-1] // 2
            out = gate_up.new_empty(gate_up.shape[:-1] + (inter, ))

            n_elements = out.numel()
            if n_elements > 0:
                block = _activation_block(n_elements, is_backward=False)
                grid = (triton.cdiv(n_elements, block), )
                # Triton uses the current device and stream, which may differ from the input tensor's device.
                with current_device(gate_up):
                    _swiglu_packed_fwd_kernel[grid](gate_up, out, n_elements, INTER=inter, BLOCK_SIZE=block)

            ctx.save_for_backward(gate_up)
            return out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            # The Triton backward has no autograd graph, so create_graph would silently lose the second derivative.
            if torch.is_grad_enabled():
                raise RuntimeError("The packed SwiGLU kernel has no second derivative; "
                                   "backward with create_graph=True is not supported")
            gate_up, = ctx.saved_tensors
            grad_out = grad_out.contiguous()
            grad_gate_up = torch.empty_like(gate_up)

            n_elements = grad_out.numel()
            if n_elements > 0:
                grid = (triton.cdiv(n_elements, _PACKED_BWD_BLOCK_SIZE), )
                with current_device(gate_up):
                    _swiglu_packed_bwd_kernel[grid](grad_out,
                                                    gate_up,
                                                    grad_gate_up,
                                                    n_elements,
                                                    INTER=gate_up.shape[-1] // 2,
                                                    BLOCK_SIZE=_PACKED_BWD_BLOCK_SIZE)

            return grad_gate_up


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """The separate SwiGLU with the Ascend UB-budgeted tiles (see the module docstring)."""
    if gate.shape != up.shape:
        raise ValueError(f"swiglu expects gate and up to have the same shape, got {tuple(gate.shape)} "
                         f"and {tuple(up.shape)}")
    if gate.dtype != up.dtype:
        raise ValueError(f"swiglu expects gate and up to have the same dtype, got {gate.dtype} and {up.dtype}")

    if not _TRITON_AVAILABLE or not _swiglu_supported(gate):
        return _swiglu_eager(gate, up)

    return _SwiGLUFn.apply(gate, up)


def swiglu_packed(gate_up: torch.Tensor) -> torch.Tensor:
    """The packed SwiGLU with the Ascend packed-backward tile (see the module docstring)."""
    _validate_packed_input(gate_up)
    if not _TRITON_AVAILABLE or not _swiglu_packed_supported(gate_up):
        return _swiglu_packed_eager(gate_up)
    return _SwiGLUPackedFn.apply(gate_up)

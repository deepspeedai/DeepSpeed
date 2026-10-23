# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Ascend variant of the fused Triton SwiGLU kernels.

The packed backward kernel overflows the unified buffer at compile time for a
single output element with BLOCK_SIZE=2048 on the triton-ascend backend
(reported upstream); BLOCK_SIZE=1024 compiles and matches eager numerically
(verified on a 910B4). The portable kernel source is reused unchanged and only
the packed backward's block size is lowered, so the packed path - single
element included - runs on Ascend with no shape-specific gate in the shared
code. Remove this variant once the backend fix lands (or keep it if 1024 stays
the better Ascend tile).
"""

import torch

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, current_device, triton

# The public API stays the portable functions; they resolve the packed class
# through the dispatch, so the re-exports keep this module a drop-in for the
# portable one.
from deepspeed.ops.triton_ops.swiglu_triton import swiglu, swiglu_packed  # noqa: F401

if _TRITON_AVAILABLE:
    from deepspeed.ops.triton_ops.swiglu_triton import (_BLOCK_SIZE, _NUM_WARPS, _swiglu_packed_bwd_kernel,
                                                        _swiglu_packed_fwd_kernel)

    # Only the packed backward needs the smaller tile on Ascend: the forward
    # compiles and runs at 2048 there, and the separate (unpacked) kernels are
    # unchanged.
    _PACKED_BWD_BLOCK_SIZE = 1024

    class _SwiGLUPackedFn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, gate_up: torch.Tensor) -> torch.Tensor:
            gate_up = gate_up.contiguous()
            inter = gate_up.shape[-1] // 2
            out = gate_up.new_empty(gate_up.shape[:-1] + (inter, ))

            n_elements = out.numel()
            if n_elements > 0:
                grid = (triton.cdiv(n_elements, _BLOCK_SIZE), )
                # Triton uses the current device and stream, which may differ from the input tensor's device.
                with current_device(gate_up):
                    _swiglu_packed_fwd_kernel[grid](gate_up,
                                                    out,
                                                    n_elements,
                                                    INTER=inter,
                                                    BLOCK_SIZE=_BLOCK_SIZE,
                                                    num_warps=_NUM_WARPS)

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
                                                    BLOCK_SIZE=_PACKED_BWD_BLOCK_SIZE,
                                                    num_warps=_NUM_WARPS)

            return grad_gate_up

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""torch.library registration for the segKI ops.

Route A of glue fusion: registering the opbuilder methods as proper custom
ops (with fake kernels) makes them transparent boundaries for
torch.compile — inductor can then fuse the eager elementwise glue BETWEEN
our kernels instead of graph-breaking on pybind black boxes. The kernels
themselves stay hand-written; only the registration layer is new.

The active JIT kernel_op is installed per-model at injection time via
``bind_kernel_op``; the registered ops dispatch to it.
"""

from typing import List

import torch

_kernel_op = None


def bind_kernel_op(op) -> None:
    """Point the registered ops at this model's JIT extension."""
    global _kernel_op
    _kernel_op = op


def _op():
    assert _kernel_op is not None, "segKI torch.library ops used before bind_kernel_op"
    return _kernel_op


if hasattr(torch.library, "custom_op"):

    @torch.library.custom_op("segki::grouped_linear", mutates_args=())
    def grouped_linear(x: torch.Tensor, weights: List[torch.Tensor], glu_epilogue: bool) -> List[torch.Tensor]:
        return _op().grouped_linear(x, weights, glu_epilogue)

    @grouped_linear.register_fake
    def _(x, weights, glu_epilogue):
        # separate outputs per weight (the dispatcher's non-fused contract);
        # the glu variant folds the epilogue into the first output
        if glu_epilogue:
            return torch.empty(x.shape[0], weights[0].shape[0], dtype=x.dtype, device=x.device)
        return [
            torch.empty(x.shape[0], w.shape[0], dtype=x.dtype, device=x.device) for w in weights
        ]

    @torch.library.custom_op("segki::gdn_gates", mutates_args=())
    def gdn_gates(a: torch.Tensor, b: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor) -> List[torch.Tensor]:
        return list(_op().gdn_gates(a, b, a_log, dt_bias))

    @gdn_gates.register_fake
    def _(a, b, a_log, dt_bias):
        # beta tracks b's head dim, g tracks a's row count (fp32 accumulate)
        return [
            torch.empty_like(b, dtype=b.dtype),
            torch.empty(a.shape[0], a.shape[1] if a.dim() > 1 else a.numel(), dtype=a.dtype, device=a.device)
        ]

    @torch.library.custom_op("segki::gated_norm", mutates_args=("out", ))
    def gated_norm(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, eps: float) -> None:
        _op().gated_norm(x, gate, weight, out, eps)

    @gated_norm.register_fake
    def _(x, gate, weight, out, eps):
        return

    @torch.library.custom_op("segki::fused_add_norm", mutates_args=("residual", "out", ))
    def fused_add_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, out: torch.Tensor,
                       eps: float) -> None:
        _op().fused_add_norm(x, residual, weight, out, eps)

    @fused_add_norm.register_fake
    def _(x, residual, weight, out, eps):
        return

    TORCHLIB_READY = True
else:  # very old torch: registration is a no-op, calls stay direct
    TORCHLIB_READY = False

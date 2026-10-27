# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Probe library: one minimal risky usage per Triton kernel.

Every Triton call site validates its kernel through the accelerator's per-kernel
gate (``deepspeed.ops.triton_ops._triton.is_triton_kernel_supported``) with the
probe defined here. A probe runs the kernel's smallest representative usage -
forward and backward where the kernel has one - on the caller's device and
dtype, so the current backend answers for itself on first use and the answer is
cached per process. No platform keeps a blacklist, and a backend that gains or
fixes support flips the gate automatically.

Kernel modules are imported lazily inside each probe, so this module stays
import-light and free of import cycles. Probe names follow the module that owns
the kernel. Probe inputs are deterministic constants: a probe never draws
random numbers, so the caller's RNG state is never touched and the probe is
reproducible run over run.
"""

import torch


def _dispatched_class(feature: str, class_name: str, portable_module: str | None = None):
    """The kernel class the production path runs, through the per-hardware dispatch.

    A probe must exercise the class production actually runs - the gate
    protects that implementation - so the class resolves through the dispatch
    exactly like the dispatched public function does (the probe must not call
    the public function itself: that function checks the gate, and a probe
    running inside the gate would deadlock on the probe lock). Platforms
    without a variant get the portable class.
    """
    from deepspeed.ops.triton_ops._triton import resolve_triton_ops_impl

    return getattr(resolve_triton_ops_impl(feature, portable_module), class_name)


def _probe_tensor(size, dtype, device):
    """A deterministic probe input in [0, 1) with no RNG involvement.

    A probe's job is to make the backend compile and run the kernel's risky
    usage; the input values do not carry the risk (a compile crash or a device
    fault fires for any values), so no randomness is needed. Deterministic
    inputs mean the caller's generator is never touched - some backends cannot
    reliably save/restore their generator state - and the probe runs the same
    values every time.
    """
    numel = torch.Size(size).numel()
    return torch.arange(numel, dtype=torch.float32, device=device).div_(numel).reshape(size).to(dtype=dtype)


def _fwd_bwd(forward, device):
    """Wrap a forward callable in a grad-enabled, device-pinned forward + backward.

    The device pin makes the probe launch on the tensor's device rather than the
    caller's current device. The probe's inputs are deterministic (see
    ``_probe_tensor``), so the caller's RNG state is never touched.
    """

    def run():
        from contextlib import nullcontext

        from deepspeed.ops.triton_ops._triton import current_device

        # current_device takes a tensor or an index; pass the index so a concrete
        # torch.device works directly and index-less devices skip the switch.
        device_ctx = current_device(device.index) if device.index is not None else nullcontext()
        with torch.enable_grad(), device_ctx:
            y = forward()
            y.sum().backward()
            return y.detach()

    return run


def swiglu(dtype, device):
    _SwiGLUFn = _dispatched_class("swiglu_triton", "_SwiGLUFn")

    def forward():
        gate = _probe_tensor((32, 16), dtype, device).requires_grad_()
        up = _probe_tensor((32, 16), dtype, device).requires_grad_()
        return _SwiGLUFn.apply(gate, up)

    return _fwd_bwd(forward, device)


def _swiglu_packed_fn():
    """The packed SwiGLU class through the per-hardware dispatch (see ``_dispatched_class``)."""
    return _dispatched_class("swiglu_triton", "_SwiGLUPackedFn")


def swiglu_packed(dtype, device):
    _SwiGLUPackedFn = _swiglu_packed_fn()

    def forward():
        x = _probe_tensor((32, 32), dtype, device).requires_grad_()
        return _SwiGLUPackedFn.apply(x)

    return _fwd_bwd(forward, device)


def group_gemm(dtype, device):
    from deepspeed.ops.triton_ops.group_gemm_triton import _GroupGemmFn

    def forward():
        a = _probe_tensor((8, 16), dtype, device).requires_grad_()
        b = _probe_tensor((2, 16, 32), dtype, device).requires_grad_()
        offs = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
        return _GroupGemmFn.apply(a, b, offs, False)

    return _fwd_bwd(forward, device)


def fused_rms_norm(dtype, device):
    _FusedRMSNorm = _dispatched_class("fused_rms_norm", "_FusedRMSNorm")

    def forward():
        hidden = _probe_tensor((2, 256), dtype, device).requires_grad_()
        weight = _probe_tensor((256, ), dtype, device).requires_grad_()
        return _FusedRMSNorm.apply(hidden, weight, 1e-6)

    return _fwd_bwd(forward, device)


def fused_rotary(dtype, device):
    _FusedRotaryPosEmb = _dispatched_class("fused_rotary_pos_emb", "_FusedRotaryPosEmb")

    def forward():
        q = _probe_tensor((1, 4, 8, 64), dtype, device).requires_grad_()
        k = _probe_tensor((1, 4, 8, 64), dtype, device).requires_grad_()
        cos = _probe_tensor((1, 8, 64), dtype, device)
        sin = _probe_tensor((1, 8, 64), dtype, device)
        q_out, k_out = _FusedRotaryPosEmb.apply(q, k, cos, sin)
        return q_out + k_out

    return _fwd_bwd(forward, device)


def fused_weighted_restore(dtype, device):
    from deepspeed.ops.triton_ops.autoep_fused_token_ops import fused_weighted_restore

    def forward():
        bsz, seqlen, hidden, top_k = 1, 2, 64, 2
        combined_rows = _probe_tensor((bsz * seqlen * top_k, hidden), dtype, device).requires_grad_()
        top_scores = _probe_tensor((bsz * seqlen, top_k), torch.float32, device).requires_grad_()
        token_indices_sorted = torch.arange(bsz * seqlen * top_k, dtype=torch.int32, device=device)
        return fused_weighted_restore(combined_rows, top_scores, token_indices_sorted, top_k, (bsz, seqlen, hidden))

    return _fwd_bwd(forward, device)


def fused_row_weighting(dtype, device):
    from deepspeed.ops.triton_ops.autoep_fused_token_ops import _FusedRowWeighting

    def forward():
        rows = _probe_tensor((8, 64), dtype, device).requires_grad_()
        weights = _probe_tensor((8, 1), torch.float32, device).requires_grad_()
        return _FusedRowWeighting.apply(rows, weights)

    return _fwd_bwd(forward, device)


def chunked_cross_entropy(dtype, device):
    _TritonCrossEntropy = _dispatched_class("chunked_cross_entropy", "_TritonCrossEntropy",
                                            "deepspeed.runtime.chunked_cross_entropy")

    def forward():
        logits = _probe_tensor((2, 64), dtype, device).requires_grad_()
        target = torch.arange(2, dtype=torch.long, device=device)
        return _TritonCrossEntropy.apply(logits, target, -100)

    return _fwd_bwd(forward, device)


def fp8_gemm(dtype, device, weight_dtype, k, n, quantization_group_size):
    """FP8 GEMM has no autograd; the probe only runs the forward kernel."""
    from deepspeed.ops.fp_quantizer.fp8_gemm_triton import matmul_fp8_triton
    from deepspeed.ops.triton_ops._triton import current_device

    def run():
        from contextlib import nullcontext

        # current_device takes an index; index-less devices skip the switch.
        device_ctx = current_device(device.index) if device.index is not None else nullcontext()
        with device_ctx:
            inp = _probe_tensor((16, k), dtype, device)
            weight = torch.zeros((k, n), dtype=weight_dtype, device=device)
            scale = torch.ones((k // quantization_group_size, n // quantization_group_size), device=device)
            matmul_fp8_triton(inp, weight, scale, quantization_group_size)

    return run


def liger_vocab_parallel_cross_entropy(dtype, device, tp_group, local_vocab_size):
    from deepspeed.sequence.cross_entropy import _liger_vocab_cross_entropy

    def forward():
        logits = torch.zeros((2, local_vocab_size), dtype=dtype, device=device, requires_grad=True)
        target = torch.zeros((2, ), dtype=torch.long, device=device)
        return _liger_vocab_cross_entropy(logits, target, tp_group, -100)

    return _fwd_bwd(forward, device)

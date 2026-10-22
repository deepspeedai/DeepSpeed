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
the kernel.
"""

import torch


def _fwd_bwd(forward):
    """Wrap a forward callable in a grad-enabled forward + backward."""

    def run():
        with torch.enable_grad():
            y = forward()
            y.sum().backward()

    return run


def swiglu(dtype, device):
    from deepspeed.ops.triton_ops.swiglu_triton import _SwiGLUFn

    def forward():
        gate = torch.randn((32, 16), dtype=dtype, device=device, requires_grad=True)
        up = torch.randn((32, 16), dtype=dtype, device=device, requires_grad=True)
        return _SwiGLUFn.apply(gate, up)

    return _fwd_bwd(forward)


def swiglu_packed(dtype, device):
    from deepspeed.ops.triton_ops.swiglu_triton import _SwiGLUPackedFn

    def forward():
        x = torch.randn((32, 32), dtype=dtype, device=device, requires_grad=True)
        return _SwiGLUPackedFn.apply(x)

    return _fwd_bwd(forward)


def swiglu_packed_1elem(dtype, device):
    """The n_elements=1 packed backward overflows some backends' unified buffer at
    compile time (reported upstream), so this usage is gated separately."""
    from deepspeed.ops.triton_ops.swiglu_triton import _SwiGLUPackedFn

    def forward():
        x = torch.randn((2, ), dtype=dtype, device=device, requires_grad=True)
        return _SwiGLUPackedFn.apply(x)

    return _fwd_bwd(forward)


def group_gemm(dtype, device):
    from deepspeed.ops.triton_ops.group_gemm_triton import _GroupGemmFn

    def forward():
        a = torch.randn((8, 16), dtype=dtype, device=device, requires_grad=True)
        b = torch.randn((2, 16, 32), dtype=dtype, device=device, requires_grad=True)
        offs = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
        return _GroupGemmFn.apply(a, b, offs, False)

    return _fwd_bwd(forward)


def fused_rms_norm(dtype, device):
    from deepspeed.ops.triton_ops.fused_rms_norm import _FusedRMSNorm

    def forward():
        hidden = torch.randn((2, 256), dtype=dtype, device=device, requires_grad=True)
        weight = torch.randn((256, ), dtype=dtype, device=device, requires_grad=True)
        return _FusedRMSNorm.apply(hidden, weight, 1e-6)

    return _fwd_bwd(forward)


def fused_rotary(dtype, device):
    from deepspeed.ops.triton_ops.fused_rotary_pos_emb import _FusedRotaryPosEmb

    def forward():
        q = torch.randn((1, 4, 8, 64), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((1, 4, 8, 64), dtype=dtype, device=device, requires_grad=True)
        cos = torch.randn((1, 8, 64), dtype=dtype, device=device)
        sin = torch.randn((1, 8, 64), dtype=dtype, device=device)
        q_out, k_out = _FusedRotaryPosEmb.apply(q, k, cos, sin)
        return q_out + k_out

    return _fwd_bwd(forward)


def fused_weighted_restore(dtype, device):
    from deepspeed.ops.triton_ops.autoep_fused_token_ops import fused_weighted_restore

    def forward():
        bsz, seqlen, hidden, top_k = 1, 2, 64, 2
        combined_rows = torch.randn((bsz * seqlen * top_k, hidden), dtype=dtype, device=device, requires_grad=True)
        top_scores = torch.randn((bsz * seqlen, top_k), dtype=torch.float32, device=device, requires_grad=True)
        token_indices_sorted = torch.arange(bsz * seqlen * top_k, dtype=torch.int32, device=device)
        return fused_weighted_restore(combined_rows, top_scores, token_indices_sorted, top_k, (bsz, seqlen, hidden))

    return _fwd_bwd(forward)


def fused_row_weighting(dtype, device):
    from deepspeed.ops.triton_ops.autoep_fused_token_ops import _FusedRowWeighting

    def forward():
        rows = torch.randn((8, 64), dtype=dtype, device=device, requires_grad=True)
        weights = torch.randn((8, 1), dtype=torch.float32, device=device, requires_grad=True)
        return _FusedRowWeighting.apply(rows, weights)

    return _fwd_bwd(forward)


def chunked_cross_entropy(dtype, device):
    from deepspeed.runtime.chunked_cross_entropy import _TritonCrossEntropy

    def forward():
        logits = torch.randn((2, 64), dtype=dtype, device=device, requires_grad=True)
        target = torch.randint(0, 64, (2, ), device=device)
        return _TritonCrossEntropy.apply(logits, target, -100)

    return _fwd_bwd(forward)


def fp8_gemm(dtype, device, weight_dtype, k, n, quantization_group_size):
    """FP8 GEMM has no autograd; the probe only runs the forward kernel."""
    from deepspeed.ops.fp_quantizer.fp8_gemm_triton import matmul_fp8_triton

    def run():
        inp = torch.randn((16, k), dtype=dtype, device=device)
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

    return _fwd_bwd(forward)

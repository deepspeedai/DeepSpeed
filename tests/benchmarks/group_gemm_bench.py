# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Simple benchmark comparing grouped-GEMM backends for MoE experts.

Compares three implementations of the per-group GEMM ``out_e = A_e @ B_e^T``
(B in native ``[E, N, K]`` layout) on realistic MoE expert shapes with
unbalanced token counts per expert:

    * triton   : ``group_gemm_triton`` (this module's Triton kernels)
    * torch    : ``torch._grouped_mm``  (native; for-loop fallback on sm < 9.0)
    * for-loop : a plain Python loop of per-expert ``torch.mm``

Forward and forward+backward are timed with CUDA events.

Usage:
    python tests/benchmarks/group_gemm_bench.py            # experts_per_rank = 16, 32
    python tests/benchmarks/group_gemm_bench.py --experts 8 16 32 64
"""

import argparse

import torch

from deepspeed.moe.group_gemm_triton import group_gemm_triton, is_available


def _unbalanced_offs(num_experts, avg_tokens, device):
    """Cumulative row offsets with unbalanced (0.2x..1.8x avg) tokens per expert."""
    counts = torch.randint(int(avg_tokens * 0.2), int(avg_tokens * 1.8) + 1, (num_experts, ), device=device)
    counts = (counts // 8) * 8  # align to 8, as the real expert path does
    return torch.cumsum(counts, 0).to(torch.int32)


def _for_loop_grouped_mm(x, w, offs):
    """Per-expert loop: out_e = x_e @ w_e^T, w in [E, N, K]."""
    outs = []
    start = 0
    for e in range(offs.numel()):
        end = int(offs[e])
        outs.append(x[start:end] @ w[e].transpose(-2, -1))
        start = end
    return torch.cat(outs, 0)


_BACKENDS = {
    "triton": lambda x, w, offs: group_gemm_triton(x, w, offs, trans_b=True),
    "torch": lambda x, w, offs: torch._grouped_mm(x, w.transpose(-2, -1), offs=offs),
    "for-loop": _for_loop_grouped_mm,
}


def _time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)  #ignore-cuda
    torch.cuda.synchronize()  #ignore-cuda
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()  #ignore-cuda
    return start.elapsed_time(end) / iters


def benchmark(num_experts, dim=2048, hidden=768, avg_tokens=32, dtype=torch.bfloat16):
    dev = "cuda"
    offs = _unbalanced_offs(num_experts, avg_tokens, dev)
    m = int(offs[-1])
    x = torch.randn(m, dim, device=dev, dtype=dtype, requires_grad=True)
    w = torch.randn(num_experts, hidden, dim, device=dev, dtype=dtype, requires_grad=True)  # native [E, N, K]
    grad_out = torch.randn(m, hidden, device=dev, dtype=dtype)

    print(f"experts={num_experts:3d}  tokens={m:5d}")
    for name, op in _BACKENDS.items():
        fwd_ms = _time_ms(lambda op=op: op(x, w, offs))

        def fwd_bwd(op=op):
            x.grad = w.grad = None
            op(x, w, offs).backward(grad_out)

        fb_ms = _time_ms(fwd_bwd)
        print(f"    {name:8s}  forward={fwd_ms:.3f} ms  fwd+bwd={fb_ms:.3f} ms  backward={fb_ms - fwd_ms:.3f} ms")


if __name__ == "__main__":
    assert is_available() and torch.cuda.is_available(), "Triton + CUDA required"  #ignore-cuda
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, nargs="+", default=[16, 32], help="experts per rank to benchmark")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--avg-tokens", type=int, default=64, help="avg tokens per expert")
    args = parser.parse_args()

    torch.manual_seed(0)
    print(f"dim={args.dim} hidden={args.hidden} avg_tokens/expert={args.avg_tokens} dtype=bf16 (unbalanced)")
    for e in args.experts:
        benchmark(e, dim=args.dim, hidden=args.hidden, avg_tokens=args.avg_tokens)

# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Simple benchmark for the fused Triton SwiGLU activation.

Compares the fused Triton ``swiglu(gate, up)`` against the eager PyTorch
baseline ``F.silu(gate) * up`` on MoE-expert-sized activations. The kernels are
memory-bound and very fast, so each variant is captured into a CUDA graph and
replayed to remove launch overhead from the measurement.

Reports forward, forward+backward, and (derived) backward replay latency.

Usage:
    python tests/benchmarks/swiglu_bench.py
    python tests/benchmarks/swiglu_bench.py --tokens 4096 16384 --hidden 768
"""

import argparse

import torch
import torch.nn.functional as F

from deepspeed.ops.triton_ops import is_triton_available, swiglu


def _baseline(gate, up):
    return F.silu(gate) * up


_VARIANTS = {
    "triton_fused": swiglu,
    "eager": _baseline,
}


def _graph_time(work, iters=100):
    """Capture ``work()`` into a CUDA graph and return avg replay time (us)."""
    # Warmup on a side stream (required before graph capture).
    s = torch.cuda.Stream()  #ignore-cuda
    s.wait_stream(torch.cuda.current_stream())  #ignore-cuda
    with torch.cuda.stream(s):  #ignore-cuda
        for _ in range(3):
            work()
    torch.cuda.current_stream().wait_stream(s)  #ignore-cuda

    g = torch.cuda.CUDAGraph()  #ignore-cuda
    with torch.cuda.graph(g):  #ignore-cuda
        work()

    torch.cuda.synchronize()  #ignore-cuda
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)  #ignore-cuda
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()  #ignore-cuda
    return start.elapsed_time(end) / iters * 1e3  # us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 16384, 65536])
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    dev = "cuda"

    print(f"SwiGLU benchmark | hidden={args.hidden} dtype={args.dtype}\n")
    header = f"{'tokens':>8} {'variant':>14} {'fwd(us)':>9} {'fwd+bwd(us)':>12} {'bwd(us)':>9}"
    print(header)
    print("-" * len(header))

    for tokens in args.tokens:
        gate = torch.randn(tokens, args.hidden, device=dev, dtype=dtype, requires_grad=True)
        up = torch.randn(tokens, args.hidden, device=dev, dtype=dtype, requires_grad=True)
        grad_out = torch.randn(tokens, args.hidden, device=dev, dtype=dtype)

        for name, fn in _VARIANTS.items():
            t_fwd = _graph_time(lambda: fn(gate, up))
            # fwd+bwd recomputes forward each replay to rebuild the autograd graph.
            t_fwdbwd = _graph_time(lambda: torch.autograd.grad(fn(gate, up), (gate, up), grad_out))
            print(f"{tokens:>8} {name:>14} {t_fwd:>9.2f} {t_fwdbwd:>12.2f} {t_fwdbwd - t_fwd:>9.2f}")
        print()


if __name__ == "__main__":
    assert torch.cuda.is_available() and is_triton_available(), "Triton + CUDA required"  #ignore-cuda
    main()

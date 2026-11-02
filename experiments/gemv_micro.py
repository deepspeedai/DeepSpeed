# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Micro-bench: triple_gemv kernel vs cuBLAS GEMV alternatives at b=1.

Compares, on identical weight shapes (Qwen3.5-4B attention projections):
  A. triple_gemv custom kernel (reads the three original weights)
  C. on-the-fly concat + one F.linear (GEMM)
  D. three separate F.linear calls (cuBLAS GEMV/GEMM path)
  B. pre-concatenated weight copy + one F.linear (upper bound, breaks
     co-location: stale after optimizer steps — data point only)
"""

import torch

from deepspeed.accelerator import get_accelerator

dev = "cuda"
H = 2560        # hidden (Qwen3.5-4B)
NQ = 32 * 128 * 2   # q_proj out (2x-wide query|gate), 32 heads x 128 dim
NK = 8 * 128        # kv heads x dim
ITERS, WARMUP = 200, 50

torch.manual_seed(0)
h = torch.randn(H, dtype=torch.bfloat16, device=dev)
wq = torch.randn(NQ, H, dtype=torch.bfloat16, device=dev)
wk = torch.randn(NK, H, dtype=torch.bfloat16, device=dev)
wv = torch.randn(NK, H, dtype=torch.bfloat16, device=dev)
w_cat = torch.cat([wq, wk, wv], dim=0).contiguous()
total_bytes = (wq.numel() + wk.numel() + wv.numel()) * 2

op = None
try:
    from deepspeed.ops.module_inject import get_fused_glu_op
    op = get_fused_glu_op()
except Exception:
    op = None
assert op is not None and hasattr(op, "triple_gemv"), "triple_gemv op unavailable"


def bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(ITERS):
        fn()
    b.record()
    torch.cuda.synchronize()
    us = a.elapsed_time(b) * 1000 / ITERS
    return us, total_bytes / (us * 1e-6) / 1e9  # us, effective GB/s


def run_a():
    q, k, v = op.triple_gemv(h, wq, wk, wv)


def run_c():
    w = torch.cat([wq, wk, wv], dim=0)
    out = torch.nn.functional.linear(h, w)


def run_d():
    q = torch.nn.functional.linear(h, wq)
    k = torch.nn.functional.linear(h, wk)
    v = torch.nn.functional.linear(h, wv)


def run_b():
    out = torch.nn.functional.linear(h, w_cat)


with torch.no_grad():
    for name, fn in [("A triple_gemv (3 orig weights)", run_a), ("C cat-then-linear (per step)", run_c),
                     ("D 3x F.linear (cuBLAS)", run_d), ("B pre-cat copy + linear", run_b)]:
        us, gbs = bench(fn)
        print(f"MICRO {name:32s}: {us:7.1f} us/iter | {gbs:6.0f} GB/s effective", flush=True)
print("MICRO-DONE", flush=True)

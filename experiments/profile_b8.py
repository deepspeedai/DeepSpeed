# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Profile the b=8 2K decode step: per-kernel breakdown via torch.profiler."""

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig

MODEL = "Qwen/Qwen3.5-4B-Base"
BS, N = 8, 128
dev = "cuda"


class Shim:

    def __init__(self, m):
        self.module = m


def classify(name):
    n = name.lower()
    if 'decode_attn' in n:
        return 'decode_attn (ours)'
    if 'gemm' in n or 'cutlass' in n or 'nvjet' in n or 's16816' in n or 'bf16' in n and 'gemv' in n:
        return 'GEMM (cuBLAS)'
    if 'gemv' in n or 'dual_gemv' in n or 'quad_gemv' in n or 'triple_gemv' in n:
        return 'GEMV (ours)'
    if 'flash' in n or 'mem_efficient' in n or 'fmha' in n:
        return 'attention (SDPA)'
    if 'reduce_kernel' in n or 'softmax' in n:
        return 'reduce/softmax'
    if 'elementwise' in n or 'vectorized' in n or 'vectorized_elementwise' in n:
        return 'elementwise'
    if 'memcpy' in n or 'memset' in n:
        return 'memcpy/memset'
    return 'other'


tok = AutoTokenizer.from_pretrained(MODEL)
object.__setattr__(tok, "eos_token_id", -1)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
para = ("The Eiffel Tower is located in Paris, France. It was built in 1889 "
        "and stands 330 meters tall. The Louvre museum houses the Mona Lisa. ")
ids = tok(para * 120, return_tensors="pt").input_ids[:, :2048].to(dev)
req = RolloutRequest(prompt_ids=ids.repeat(BS, 1), prompt_attention_mask=torch.ones_like(ids).repeat(BS, 1))

for label, cfg in [("segki", HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True)),
                   ("graphonly", HybridEngineRolloutConfig(use_graph_capture=True))]:
    ro = HybridEngineRollout(Shim(m), tok, cfg)
    ro.generate(req, SamplingConfig(max_new_tokens=32, temperature=0.0))  # warmup
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        ro.generate(req, SamplingConfig(max_new_tokens=N, temperature=0.0))
        torch.cuda.synchronize()
    evs = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0]
    evs.sort(key=lambda e: -e.self_device_time_total)
    print(f"\n===== {label}: top kernels (generate {N} tok, b={BS}, 2K) =====", flush=True)
    print(f"{'kernel':64s} {'calls':>8s} {'total ms':>10s} {'ms/step':>9s}", flush=True)
    for e in evs[:18]:
        print(f"{e.key[:64]:64s} {e.count:8d} {e.self_device_time_total / 1000:10.1f}"
              f" {e.self_device_time_total / 1000 / (N - 1):9.3f}", flush=True)
    tot = sum(e.self_device_time_total for e in evs)
    print(f"TOTAL CUDA: {tot / 1000:.0f} ms -> {tot / 1000 / (N - 1):.2f} ms/step (upper bound incl. prefill/capture)",
          flush=True)
    agg = {}
    for e in evs:
        agg[classify(e.key)] = agg.get(classify(e.key), 0.0) + e.self_device_time_total
    print("--- by class (ms/step): ---", flush=True)
    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"{k:24s} {v / 1000 / (N - 1):8.3f}", flush=True)
    del ro
    torch.cuda.empty_cache()
print("PROFILE-DONE", flush=True)

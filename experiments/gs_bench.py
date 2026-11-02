# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Greedy vs sampling on the graph path (slope method), short + 2K, b=1.

eos_token_id is redirected to -1 so both configs always generate the full
length — required for valid slope arithmetic; the sampling path itself is
the real implementation (no library patches)."""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig

MODEL = "Qwen/Qwen3.5-4B-Base"
dev = "cuda"


class Shim:

    def __init__(self, m):
        self.module = m


def timed_gen(ro, req, n, temp, top_p=1.0):
    s = SamplingConfig(max_new_tokens=n, temperature=temp, top_p=top_p)
    out = ro.generate(req, s)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = ro.generate(req, s)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    got = out.input_ids.shape[1] - req.prompt_ids.shape[1]
    assert got == n, f"early stop: {got}/{n}"
    return dt


tok = AutoTokenizer.from_pretrained(MODEL)
tok.eos_token_id = -1  # keep full-length generations for slope validity
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
para = ("The Eiffel Tower is located in Paris, France. It was built in 1889 "
        "and stands 330 meters tall. The Louvre museum houses the Mona Lisa. ")
ids_short = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
ids_2k = tok(para * 120, return_tensors="pt").input_ids[:, :2048].to(dev)

CASES = [("greedy", 0.0, 1.0), ("sample t1.0", 1.0, 1.0), ("sample t1.0 p0.9", 1.0, 0.9)]

for ctx, ids in [("short", ids_short), ("2k", ids_2k)]:
    req = RolloutRequest(prompt_ids=ids, prompt_attention_mask=torch.ones_like(ids))
    ro = HybridEngineRollout(Shim(m), tok, HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))
    for label, temp, top_p in CASES:
        t_s = timed_gen(ro, req, 256, temp, top_p)
        t_l = timed_gen(ro, req, 1024, temp, top_p)
        print(f"GS ours {ctx:5s} {label:18s}: decode-only {768 / (t_l - t_s):7.1f} tok/s"
              f" (T256 {t_s:.2f}s T1024 {t_l:.2f}s)", flush=True)
    del ro
    torch.cuda.empty_cache()
print("GS-DONE", flush=True)

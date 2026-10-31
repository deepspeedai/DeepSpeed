# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Slope-method benchmark: e2e and fixed-overhead-free decode-only rates.

decode-only tok/s = (T_long - T_short) / (N_long - N_short): differencing two
generation lengths removes every per-call fixed cost (prefill, graph capture,
cache setup) without estimating it.
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig

MODEL = "Qwen/Qwen3.5-4B-Base"
N_SHORT, N_LONG = 256, 1024
dev = "cuda"


class Shim:

    def __init__(self, m):
        self.module = m


def timed_gen(ro, req, n):
    s = SamplingConfig(max_new_tokens=n, temperature=0.0)
    ro.generate(req, s)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = ro.generate(req, s)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    got = out.input_ids.shape[1] - req.prompt_ids.shape[1]
    return dt, got


tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)

para = ("The Eiffel Tower is located in Paris, France. It was built in 1889 "
        "and stands 330 meters tall. The Louvre museum houses the Mona Lisa. ")
ids_short = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
ids_2k = tok(para * 120, return_tensors="pt").input_ids[:, :2048].to(dev)
print("prompts:", ids_short.shape[1], ids_2k.shape[1], flush=True)

CONFIGS = [("eager", HybridEngineRolloutConfig()),
           ("graph", HybridEngineRolloutConfig(use_graph_capture=True)),
           ("segki", HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))]

for ctx, ids in [("short", ids_short), ("2k", ids_2k)]:
    req = RolloutRequest(prompt_ids=ids, prompt_attention_mask=torch.ones_like(ids))
    for name, cfg in CONFIGS:
        ro = HybridEngineRollout(Shim(m), tok, cfg)
        t_s, n_s = timed_gen(ro, req, N_SHORT)
        t_l, n_l = timed_gen(ro, req, N_LONG)
        decode_tps = (n_l - n_s) / (t_l - t_s)
        e2e_tps = n_s / t_s
        print(f"RESULT {ctx:5s} {name:5s}: e2e {e2e_tps:7.1f} tok/s | decode-only {decode_tps:7.1f} tok/s"
              f" | (T256 {t_s:.2f}s T1024 {t_l:.2f}s)", flush=True)
        del ro
        torch.cuda.empty_cache()
print("DONE", flush=True)

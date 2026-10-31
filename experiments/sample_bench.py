# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Measure the cost of sampling outside the graph (route a).

Three configs, slope method, b=1 short context, Qwen3.5-4B:
  full-step greedy      (argmax inside the captured graph)   — today's fast path
  graph + py argmax     (step kernel skipped)
  graph + py sampling   (step kernel skipped, multinomial per step)
"""

import os
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


def timed_gen(ro, req, n):
    s = SamplingConfig(max_new_tokens=n, temperature=0.0)
    ro.generate(req, s)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = ro.generate(req, s)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt, out.input_ids.shape[1] - req.prompt_ids.shape[1]


tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
ids = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
req = RolloutRequest(prompt_ids=ids, prompt_attention_mask=torch.ones_like(ids))

CASES = [
    ("full-step greedy", "", ""),
    ("graph+py argmax", "step", ""),
    ("graph+py sampling", "step", "s"),
]

for label, skip, sample in CASES:
    os.environ["DS_SEGKI_SKIP"] = skip
    os.environ["DS_SAMPLE_MODE"] = sample
    ro = HybridEngineRollout(Shim(m), tok, HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))
    t_s, n_s = timed_gen(ro, req, 256)
    t_l, n_l = timed_gen(ro, req, 1024)
    decode_tps = (n_l - n_s) / (t_l - t_s)
    print(f"SAMPLE {label:18s}: decode-only {decode_tps:7.1f} tok/s (T256 {t_s:.2f}s T1024 {t_l:.2f}s)", flush=True)
    del ro
    torch.cuda.empty_cache()
print("SAMPLE-DONE", flush=True)

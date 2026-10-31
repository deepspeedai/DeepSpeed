# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Long-context decode benchmark: does segki's gain grow with KV length?"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig

MODEL = "Qwen/Qwen3.5-4B-Base"
N_NEW = 256
dev = "cuda"


class Shim:

    def __init__(self, m):
        self.module = m


tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)

# ~2K-token prompt: factual filler the model must attend across
para = ("The Eiffel Tower is located in Paris, France. It was built in 1889 "
        "and stands 330 meters tall. The Louvre museum houses the Mona Lisa. ")
ids_long = tok(para * 120, return_tensors="pt").input_ids[:, :2048].to(dev)
print("prompt tokens:", ids_long.shape[1])

req = RolloutRequest(prompt_ids=ids_long, prompt_attention_mask=torch.ones_like(ids_long))
s = SamplingConfig(max_new_tokens=N_NEW, temperature=0.0)

with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    m(ids_long)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
print(f"prefill-only forward: {t_prefill:.2f}s (subtracted below)")

for label, cfg in [("segki+graph", HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True)),
                   ("graph only", HybridEngineRolloutConfig(use_graph_capture=True)),
                   ("HF eager", HybridEngineRolloutConfig())]:
    ro = HybridEngineRollout(Shim(m), tok, cfg)
    ro.generate(req, s)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = ro.generate(req, s)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0 - t_prefill
    n = out.input_ids.shape[1] - ids_long.shape[1]
    print(f"BENCH-2K {label:12s}: {n / dt:7.1f} decode tok/s ({dt:.2f}s for {n} tok)")
    del ro
    torch.cuda.empty_cache()

# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""bs=8 investigation: graph-only vs segKI vs (vLLM separately), b=8 greedy.

Same prompt repeated 8x (uniform rows, no padding effects), eos disabled so
slope arithmetic sees full-length generations."""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig

MODEL = "Qwen/Qwen3.5-4B-Base"
BS = 8
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
    got = out.input_ids.shape[1] - req.prompt_ids.shape[1]
    assert got == n, f"early stop {got}/{n}"
    return dt


tok = AutoTokenizer.from_pretrained(MODEL)
object.__setattr__(tok, "eos_token_id", -1)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
para = ("The Eiffel Tower is located in Paris, France. It was built in 1889 "
        "and stands 330 meters tall. The Louvre museum houses the Mona Lisa. ")
ids_short = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
ids_2k = tok(para * 120, return_tensors="pt").input_ids[:, :2048].to(dev)

for ctx, ids in [("short", ids_short), ("2k", ids_2k)]:
    req = RolloutRequest(prompt_ids=ids.repeat(BS, 1), prompt_attention_mask=torch.ones_like(ids).repeat(BS, 1))
    for label, cfg in [("graphonly", HybridEngineRolloutConfig(use_graph_capture=True)),
                       ("segki", HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))]:
        ro = HybridEngineRollout(Shim(m), tok, cfg)
        t_s = timed_gen(ro, req, 256)
        t_l = timed_gen(ro, req, 1024)
        print(f"BS8 ours {ctx:5s} {label:10s}: decode-only {768 * BS / (t_l - t_s):7.1f} tok/s total"
              f" | {768 / (t_l - t_s):6.2f} ms/step (T256 {t_s:.2f}s T1024 {t_l:.2f}s)", flush=True)
        del ro
        torch.cuda.empty_cache()
print("BS8-DONE", flush=True)

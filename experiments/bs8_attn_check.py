# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""b=8 E2E: batched decode_attn correctness (token equality vs graph-only) + payoff."""

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


def run(ro, req, n):
    s = SamplingConfig(max_new_tokens=n, temperature=0.0)
    ro.generate(req, s)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = ro.generate(req, s)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


tok = AutoTokenizer.from_pretrained(MODEL)
object.__setattr__(tok, "eos_token_id", -1)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
para = ("The Eiffel Tower is located in Paris, France. It was built in 1889 "
        "and stands 330 meters tall. The Louvre museum houses the Mona Lisa. ")
ids_short = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
ids_2k = tok(para * 120, return_tensors="pt").input_ids[:, :2048].to(dev)

for ctx, ids in [("short", ids_short), ("2k", ids_2k)]:
    req = RolloutRequest(prompt_ids=ids.repeat(BS, 1), prompt_attention_mask=torch.ones_like(ids).repeat(BS, 1))
    # correctness: batched-kernel segki output must equal graph-only output
    ro_a = HybridEngineRollout(Shim(m), tok, HybridEngineRolloutConfig(use_graph_capture=True))
    _, out_a = run(ro_a, req, 64)
    ro_b = HybridEngineRollout(Shim(m), tok, HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))
    _, out_b = run(ro_b, req, 64)
    print(f"BS8-ATTN {ctx:5s} tokens-equal-to-graphonly:", bool(torch.equal(out_a.input_ids, out_b.input_ids)),
          flush=True)
    del ro_a, ro_b
    torch.cuda.empty_cache()

    # payoff: slope bench
    for label, cfg in [("graphonly", HybridEngineRolloutConfig(use_graph_capture=True)),
                       ("segki", HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))]:
        ro = HybridEngineRollout(Shim(m), tok, cfg)
        t_s, _ = run(ro, req, 256)
        t_l, _ = run(ro, req, 1024)
        print(f"BS8-ATTN {ctx:5s} {label:10s}: decode-only {768 * BS / (t_l - t_s):7.1f} tok/s total"
              f" ({768 / (t_l - t_s):6.2f} ms/step)", flush=True)
        del ro
        torch.cuda.empty_cache()
print("BS8-ATTN-DONE", flush=True)

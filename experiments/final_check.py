# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Final check on the full 19-commit state: throughput + output sanity."""

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


def make(rollout_cfg):
    return lambda m, tok: HybridEngineRollout(Shim(m), tok, rollout_cfg)


tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
ids = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
req = RolloutRequest(prompt_ids=ids, prompt_attention_mask=torch.ones_like(ids))
s256 = SamplingConfig(max_new_tokens=N_NEW, temperature=0.0)

# --- throughput on the final state ---
for label, cfg in [("segki+graph (main path)", HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True)),
                   ("graph only (no segki)", HybridEngineRolloutConfig(use_graph_capture=True)),
                   ("HF eager generate", HybridEngineRolloutConfig())]:
    ro = make(cfg)(m, tok)
    ro.generate(req, s256)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = ro.generate(req, s256)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n = out.input_ids.shape[1] - ids.shape[1]
    print(f"BENCH {label:28s}: {n / dt:7.1f} tok/s ({dt:.2f}s)")
    if label.startswith("HF"):
        eager_out = out
    del ro
    torch.cuda.empty_cache()

# --- output sanity: segki+graph vs HF eager, same greedy decode ---
s32 = SamplingConfig(max_new_tokens=32, temperature=0.0)
ro_fast = make(HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))(m, tok)
fast_a = ro_fast.generate(req, s32)
fast_b = ro_fast.generate(req, s32)  # determinism across runs
eager = eager_out.input_ids[:, ids.shape[1]:ids.shape[1] + 32]

fa = fast_a.input_ids[0, ids.shape[1]:]
fb = fast_b.input_ids[0, ids.shape[1]:]
print("TEXT fast :", tok.decode(fa))
print("TEXT eager:", tok.decode(eager))
print("DETERMINISTIC (fast vs fast):", bool(torch.equal(fa, fb)))
print("MATCH (fast vs eager):", bool(torch.equal(fa.cpu(), eager[0].cpu())))
neq = (fa.cpu() != eager[0].cpu()).nonzero().flatten()
print("FIRST-DIVERGE-POS:", int(neq[0]) if neq.numel() else None, "/ 32")

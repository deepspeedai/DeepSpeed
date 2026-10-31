# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Decode throughput benchmark: segki+graph vs graph-only vs HF eager."""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig

MODEL = "Qwen/Qwen3.5-0.8B-Base"
N_NEW = 256
dev = "cuda"


class Shim:

    def __init__(self, m):
        self.module = m


def bench(label, model, tok, segki, graph):
    rollout = HybridEngineRollout(Shim(model),
                                  tok,
                                  HybridEngineRolloutConfig(use_graph_capture=graph,
                                                            use_segki=segki))
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
    req = RolloutRequest(prompt_ids=ids, prompt_attention_mask=torch.ones_like(ids))
    s = SamplingConfig(max_new_tokens=N_NEW, temperature=0.0)
    rollout.generate(req, s)  # warmup (JIT/capture/cudnn autotune)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = rollout.generate(req, s)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n = out.input_ids.shape[1] - ids.shape[1]
    print(f"{label:28s}: {n / dt:7.1f} tok/s  ({dt:.2f}s for {n} tok)")


tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)

bench("segki+graph (main path)", m, tok, True, True)
bench("graph only (no segki)", m, tok, False, True)
bench("HF eager generate", m, tok, False, False)

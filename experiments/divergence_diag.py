# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Diagnose b=8 segki-vs-graphonly divergence: near-tie drift or corruption?"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig

MODEL = "Qwen/Qwen3.5-4B-Base"
BS, dev = 8, "cuda"


class Shim:

    def __init__(self, m):
        self.module = m


tok = AutoTokenizer.from_pretrained(MODEL)
object.__setattr__(tok, "eos_token_id", -1)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
ids = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
req = RolloutRequest(prompt_ids=ids.repeat(BS, 1), prompt_attention_mask=torch.ones_like(ids).repeat(BS, 1))
s = SamplingConfig(max_new_tokens=24, temperature=0.0)

outs = {}
for label, cfg in [("graphonly", HybridEngineRolloutConfig(use_graph_capture=True)),
                   ("segki", HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))]:
    ro = HybridEngineRollout(Shim(m), tok, cfg)
    outs[label] = ro.generate(req, s).input_ids[:, ids.shape[1]:]
    del ro
    torch.cuda.empty_cache()

a, b = outs["graphonly"][0], outs["segki"][0]
print("graphonly:", repr(tok.decode(a))[:120])
print("segki    :", repr(tok.decode(b))[:120])
diff = (a != b).nonzero().flatten()
print("first-diff pos:", int(diff[0]) if diff.numel() else None, "/ 24")
same_rows = sum(1 for r in range(BS) if torch.equal(outs["graphonly"][r], outs["segki"][r]))
print(f"identical rows: {same_rows}/{BS}")

# near-tie evidence: rerun graphonly logits in fp32 for the divergence step
m32 = m.float()
with torch.no_grad():
    fp32_logits = m32(ids.to(dev)).logits[0, -1].float()
top = fp32_logits.topk(4)
print("step-0 fp32 top4:", [(tok.decode([i]), round(v.item(), 3)) for v, i in zip(top.values, top.indices)])
print("step-0 choices:", int(a[0]), int(b[0]))

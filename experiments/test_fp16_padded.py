# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Targeted GPU regression: fp16 model (C13) and padded prompt (C4/C7)."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig

MODEL = "Qwen/Qwen3.5-0.8B"
dev = "cuda"

tok = AutoTokenizer.from_pretrained(MODEL)
greedy = SamplingConfig(max_new_tokens=16, temperature=0.0)


class Shim:

    def __init__(self, m):
        self.module = m


# --- 1) fp16 model: must generate without crashing (C13) ---
m16 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev)
ro16 = HybridEngineRollout(Shim(m16), tok, HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))
ids = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
out16 = ro16.generate(RolloutRequest(prompt_ids=ids, prompt_attention_mask=torch.ones_like(ids)), greedy)
print("FP16_OK:", tok.decode(out16.input_ids[0, ids.shape[1]:]))
print("segki_report:", ro16._segki_report)
del m16, ro16
torch.cuda.empty_cache()

# --- 2) bf16 + right-padded prompt: decode must fall back (C4); prefill
#        mask mapping (C7) is still unfixed, so record the divergence ---
mb = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
rob = HybridEngineRollout(Shim(mb), tok, HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))
ids = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
attn = torch.ones_like(ids)
clean = rob.generate(RolloutRequest(prompt_ids=ids, prompt_attention_mask=attn), greedy)

pad_len = 8
padded_ids = torch.cat([ids, torch.full((1, pad_len), tok.pad_token_id or tok.eos_token_id, device=dev)], dim=1)
padded_attn = torch.cat([attn, torch.zeros(1, pad_len, dtype=attn.dtype, device=dev)], dim=1)
padded = rob.generate(RolloutRequest(prompt_ids=padded_ids, prompt_attention_mask=padded_attn), greedy)
print("CLEAN :", tok.decode(clean.input_ids[0, ids.shape[1]:]))
print("PADDED:", tok.decode(padded.input_ids[0, ids.shape[1]:]))
print("PADDED_MATCH:", bool(torch.equal(clean.input_ids, padded.input_ids)))
print("PROMPT_PADDED_FLAG:", getattr(mb, "_ki_prompt_padded", "unset"))

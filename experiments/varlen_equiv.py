# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Varlen equivalence: injected GDN forward must match the native forward."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspeed.module_inject.segment_ki import apply_segment_ki

dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Base", dtype=torch.bfloat16).to(dev)

a = tok("The capital of France is Paris", return_tensors="pt").input_ids.to(dev)
b = tok("Two plus two equals four", return_tensors="pt").input_ids.to(dev)
la, lb = a.shape[1], b.shape[1]
packed = torch.cat([a, b], dim=1)
cu = torch.tensor([0, la, la + lb], dtype=torch.int32, device=dev)
seq_idx = torch.cat([torch.zeros(1, la, dtype=torch.int32), torch.ones(1, lb, dtype=torch.int32)], dim=1).to(dev)

with torch.no_grad():
    native = m(packed, cu_seq_lens_q=cu, seq_idx=seq_idx).logits.float()

report = apply_segment_ki(m)
print("INJECT:", report["fused_gdn"])
with torch.no_grad():
    injected = m(packed, cu_seq_lens_q=cu, seq_idx=seq_idx).logits.float()

diff = (native - injected).abs().max().item()
print("max |native - injected|:", diff)
print("argmax equal:", bool(torch.equal(native.argmax(-1), injected.argmax(-1))))
print("EQUIVALENT:", diff < 0.1)

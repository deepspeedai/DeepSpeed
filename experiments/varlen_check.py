# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Varlen consistency: packed sequences with cu_seq_lens_q must match separate runs."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspeed.module_inject.segment_ki import apply_segment_ki

dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Base", dtype=torch.bfloat16).to(dev)
report = apply_segment_ki(m)
print("INJECT:", report["fused_gdn"])

a = tok("The capital of France is Paris", return_tensors="pt").input_ids.to(dev)
b = tok("Two plus two equals four", return_tensors="pt").input_ids.to(dev)
la, lb = a.shape[1], b.shape[1]

with torch.no_grad():
    out_a = m(a).logits.float()
    out_b = m(b).logits.float()
    packed = torch.cat([a, b], dim=1)
    cu = torch.tensor([0, la, la + lb], dtype=torch.int32, device=dev)
    seq_idx = torch.cat([torch.zeros(1, la, dtype=torch.int32), torch.ones(1, lb, dtype=torch.int32)],
                        dim=1).to(dev)
    out_p = m(packed, cu_seq_lens_q=cu, seq_idx=seq_idx).logits.float()

pa, pb = out_p[0, :la], out_p[0, la:]
print("max |packed_A - separate_A|:", (pa - out_a[0]).abs().max().item())
print("max |packed_B - separate_B|:", (pb - out_b[0]).abs().max().item())
print("argmax A equal:", bool(torch.equal(pa.argmax(-1), out_a[0].argmax(-1))))
print("argmax B equal:", bool(torch.equal(pb.argmax(-1), out_b[0].argmax(-1))))

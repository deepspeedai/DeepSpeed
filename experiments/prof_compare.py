# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Profiler comparison harness for segKI vs HF (see segment-ki-proto.md)."""

import os
import time

import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed.accelerator import get_accelerator

USE_KI = os.environ.get("USE_KI", "0") == "1"
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B-Base')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B-Base',
                                             dtype=torch.bfloat16).to(get_accelerator().device_name()).eval()
if USE_KI:
    from deepspeed.module_inject.segment_ki import apply_segment_ki
    print('KI_REPORT', apply_segment_ki(model, kernel="all", backend="auto"), flush=True)

dev = get_accelerator().device_name()
ids = tok('The capital of France is', return_tensors='pt').input_ids.to(dev)
mask = torch.ones_like(ids)
out = model.generate(ids, attention_mask=mask, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)
get_accelerator().synchronize()
t0 = time.perf_counter()
out = model.generate(ids, attention_mask=mask, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
get_accelerator().synchronize()
wall_ms = (time.perf_counter() - t0) * 1e3

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    out = model.generate(ids, attention_mask=mask, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)

STEPS = 32
evs = prof.key_averages()


def cat(k):
    kl = k.lower()
    if 'nccl' in kl: return 'comm'
    if 'gemm' in kl or 'cutlass' in kl or 'matmul' in kl or 'nvjet' in kl or 'sgemm' in kl: return 'gemm'
    if 'fla' in kl or 'chunk' in kl or 'delta' in kl or 'recurrent' in kl or 'conv' in kl: return 'gdn_core'
    if 'elementwise' in kl or 'silu' in kl or 'sigmoid' in kl or 'softplus' in kl or 'gdn_gates' in kl or 'fused_silu' in kl:
        return 'elementwise'
    if 'memcpy' in kl or 'copy' in kl or 'cat_' in kl or '_cat' in kl: return 'copy/cat'
    if 'norm' in kl: return 'norm'
    return 'other'


cats = defaultdict(lambda: [0, 0.0])
total_launches = 0
rows = []
for e in evs:
    if e.self_device_time_total <= 0:
        continue
    c = cat(e.key)
    cats[c][0] += e.count
    cats[c][1] += e.self_device_time_total
    total_launches += e.count
    rows.append((e.key[:70], e.count, e.self_device_time_total))
rows.sort(key=lambda r: -r[2])
print(f'== USE_KI={USE_KI} wall_ms(64tok)={wall_ms:.0f} tok/s={64/(wall_ms/1e3):.1f} ==', flush=True)
print(f'launches_total={total_launches} per_step={total_launches/STEPS:.0f}', flush=True)
print('-- category breakdown (count, device_us) --', flush=True)
for c, (n, t) in sorted(cats.items(), key=lambda x: -x[1][1]):
    print(f'  {c:12s} n={n:6d} ({n/STEPS:5.1f}/step) t={t:9.0f}us ({t/STEPS:6.0f}us/step)', flush=True)
print('-- top 12 kernels --', flush=True)
for k, n, t in rows[:12]:
    print(f'  {t/STEPS:8.0f}us/step x{n//max(STEPS,1):3d}/step  {k}', flush=True)

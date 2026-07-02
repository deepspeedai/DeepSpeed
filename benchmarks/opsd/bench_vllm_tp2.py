# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Benchmark vLLM TP=2 on 14B, 1P1R.

Launched as a subprocess wrapper to avoid CUDA fork issues.
"""
import os
import subprocess
import sys

script = '''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import time
from vllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen2.5-14B-Instruct", tensor_parallel_size=2,
          gpu_memory_utilization=0.85, dtype="bfloat16", enforce_eager=True)
sp = SamplingParams(max_tokens=256, temperature=0.8, top_p=0.95, n=1)
prompt = "def fibonacci(n):"

# warmup
llm.generate([prompt], sp)

times = []
for i in range(5):
    t0 = time.time()
    out = llm.generate([prompt], sp)
    times.append(time.time() - t0)

t_avg = sum(times[1:]) / len(times[1:])
total_toks = sum(len(o.token_ids) for r in out for o in r.outputs)
print(f"vLLM TP=2 14B 1P1R: {total_toks} toks, {t_avg*1000:.1f}ms, {total_toks/t_avg:.1f} tok/s")
print(f"Per-run: {[f'{t*1000:.0f}ms' for t in times]}")
'''

# Write to temp file and exec in a fresh process with no prior CUDA init
tmp = "/tmp/bench_vllm_inner.py"
with open(tmp, "w") as f:
    f.write(script)

env = os.environ.copy()
env.pop("CUDA_VISIBLE_DEVICES", None)
proc = subprocess.run([sys.executable, tmp], env=env)
sys.exit(proc.returncode)

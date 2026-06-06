# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Micro-benchmark for 1p1r HybridEngineRollout decode.

Measures time breakdown of each decode step:
  - model forward (attention + FFN)
  - sampling (softmax + multinomial)
  - Python overhead (mask concat, state update, etc.)

Usage:
  python examples/opsd/bench_decode_1p1r.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspeed.accelerator import get_accelerator

from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig


def bench_decode_raw(model, tokenizer, device, prompt_len=64, max_new_tokens=64, num_warmup=3, num_iters=10):
    """Raw decode loop benchmark — measures each component separately."""
    model.eval()
    model_dtype = next(model.parameters()).dtype

    input_ids = torch.randint(10, 1000, (1, prompt_len), device=device)
    attn_mask = torch.ones(1, prompt_len, dtype=torch.long, device=device)

    results = {
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "model_dtype": str(model_dtype),
    }

    timings = {"prefill": [], "decode_forward": [], "sampling": [], "overhead": [], "total": []}

    for _ in range(num_warmup + num_iters):
        with torch.no_grad():
            t0 = time.perf_counter()
            out = model(input_ids, attention_mask=attn_mask, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1:, :]
            t_prefill = time.perf_counter()

            generated = []
            cur_token = logits.argmax(dim=-1)
            generated.append(cur_token)
            cur_mask = attn_mask

            decode_times = []
            sample_times = []
            overhead_times = []

            for step in range(max_new_tokens):
                t_step = time.perf_counter()
                cur_mask = torch.cat([cur_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
                pos_ids = torch.tensor([[prompt_len + step]], device=device)

                t_fwd = time.perf_counter()
                out = model(cur_token,
                            attention_mask=cur_mask,
                            position_ids=pos_ids,
                            past_key_values=past,
                            use_cache=True)
                past = out.past_key_values
                t_fwd_end = time.perf_counter()

                next_logits = out.logits[:, -1, :]
                probs = torch.softmax(next_logits / 1.0, dim=-1)
                cur_token = torch.multinomial(probs, 1)
                t_sample = time.perf_counter()

                generated.append(cur_token)
                t_overhead = time.perf_counter()

                decode_times.append(t_fwd_end - t_fwd)
                sample_times.append(t_sample - t_fwd_end)
                overhead_times.append(t_overhead - t_sample)

            t_total = time.perf_counter()

        timings["prefill"].append(t_prefill - t0)
        timings["decode_forward"].append(decode_times)
        timings["sampling"].append(sample_times)
        timings["overhead"].append(overhead_times)
        timings["total"].append(t_total - t0)

    import numpy as np

    def avg_last_n(lst, n):
        return np.mean(lst[-n:])

    def avg_of_avg(list_of_lists, n):
        arrs = [np.array(ls[-n:]) for ls in list_of_lists]
        return np.mean([a.mean() for a in arrs])

    results["prefill_ms"] = avg_last_n(timings["prefill"], num_iters) * 1000
    results["decode_forward_ms_per_step"] = avg_of_avg(timings["decode_forward"], num_iters) * 1000
    results["sampling_ms_per_step"] = avg_of_avg(timings["sampling"], num_iters) * 1000
    results["overhead_ms_per_step"] = avg_of_avg(timings["overhead"], num_iters) * 1000
    results["total_ms"] = avg_last_n(timings["total"], num_iters) * 1000
    results["decode_steps_total_ms"] = results["decode_forward_ms_per_step"] * max_new_tokens
    results["sampling_total_ms"] = results["sampling_ms_per_step"] * max_new_tokens
    results["overhead_total_ms"] = results["overhead_ms_per_step"] * max_new_tokens

    return results


def bench_hybrid_rollout(rollout, tokenizer, device, prompt_len=64, max_new_tokens=64, num_warmup=3, num_iters=10):
    """Benchmark the full HybridEngineRollout.generate() path."""
    input_ids = torch.randint(10, 1000, (1, prompt_len), device=device)
    attn_mask = torch.ones(1, prompt_len, dtype=torch.long, device=device)
    sampling = SamplingConfig(max_new_tokens=max_new_tokens, temperature=1.0, top_p=1.0)
    request = RolloutRequest(prompt_ids=input_ids, prompt_attention_mask=attn_mask)

    times = []
    for _ in range(num_warmup + num_iters):
        get_accelerator().synchronize()  #ignore-cuda
        t0 = time.perf_counter()
        with torch.no_grad():
            rollout.generate(request, sampling)
        get_accelerator().synchronize()  #ignore-cuda
        times.append(time.perf_counter() - t0)

    import numpy as np
    avg = np.mean(times[-num_iters:]) * 1000
    return {"rollout_total_ms": avg, "prompt_len": prompt_len, "max_new_tokens": max_new_tokens}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=10)
    args = parser.parse_args()

    device = get_accelerator().current_device()  #ignore-cuda

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)

    print(f"=== Raw decode loop benchmark (model={args.model}) ===")
    raw = bench_decode_raw(model, tokenizer, device, args.prompt_len, args.max_new_tokens, args.num_warmup,
                           args.num_iters)
    print(f"  Prefill:              {raw['prefill_ms']:.2f} ms")
    print(
        f"  Decode forward/step:  {raw['decode_forward_ms_per_step']:.3f} ms  (total: {raw['decode_steps_total_ms']:.1f} ms)"
    )
    print(f"  Sampling/step:        {raw['sampling_ms_per_step']:.3f} ms  (total: {raw['sampling_total_ms']:.1f} ms)")
    print(f"  Overhead/step:        {raw['overhead_ms_per_step']:.3f} ms  (total: {raw['overhead_total_ms']:.1f} ms)")
    print(f"  Total:                {raw['total_ms']:.1f} ms")

    print(f"\n=== HybridEngineRollout benchmark ===")
    rollout = HybridEngineRollout(model, tokenizer)
    rr = bench_hybrid_rollout(rollout, tokenizer, device, args.prompt_len, args.max_new_tokens, args.num_warmup,
                              args.num_iters)
    print(f"  Rollout generate:     {rr['rollout_total_ms']:.1f} ms")

    print(f"\n=== Summary ===")
    print(f"  Raw decode loop:      {raw['total_ms']:.1f} ms")
    print(f"  HybridEngine rollout: {rr['rollout_total_ms']:.1f} ms")
    print(f"  Overhead (rollout - raw): {rr['rollout_total_ms'] - raw['total_ms']:.1f} ms")
    print(
        f"  Bottleneck: decode forward = {raw['decode_forward_ms_per_step']:.3f} ms/step x {args.max_new_tokens} steps = {raw['decode_steps_total_ms']:.1f} ms"
    )


if __name__ == "__main__":
    main()

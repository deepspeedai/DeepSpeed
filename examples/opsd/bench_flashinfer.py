# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Benchmark HybridEngineRollout with FlashInfer kernels enabled.

Usage:
    deepspeed --num_gpus 2 bench_flashinfer.py
"""

import argparse
import os
import time

import deepspeed
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--no-flashinfer", action="store_true")
    parser.add_argument("--graph-capture", action="store_true")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    args = parser.parse_args()

    local_rank = args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    deepspeed.init_distributed()

    if local_rank == 0:
        print(f"=== HybridEngineRollout Benchmark ===")
        print(f"  Model:         {args.model}")
        print(f"  TP size:       {world_size}")
        print(f"  FlashInfer:    {not args.no_flashinfer}")
        print(f"  Graph capture: {args.graph_capture}")
        print()

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )

    ds_config = {
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 0
        },
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": world_size,
        "gradient_accumulation_steps": 1,
        "tensor_parallel": {
            "autotp_size": world_size,
            "preset_model": "qwen2",
        },
    }

    engine, *_ = deepspeed.initialize(
        model=model,
        optimizer=None,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    if local_rank == 0:
        param_count = sum(p.numel() for p in engine.parameters()) / 1e9
        alloc = get_accelerator().memory_allocated(local_rank) / 1e9
        print(f"  Parameters (local):  {param_count:.2f}B")
        print(f"  GPU mem allocated:   {alloc:.1f} GB")
        print()

    use_flashinfer = not args.no_flashinfer
    rollout = HybridEngineRollout(engine,
                                  tokenizer,
                                  use_flashinfer=use_flashinfer,
                                  use_graph_capture=args.graph_capture)

    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(42)
    input_ids = torch.randint(10, 1000, (1, args.prompt_len), device=device)
    attn_mask = torch.ones(1, args.prompt_len, dtype=torch.long, device=device)
    sampling = SamplingConfig(max_new_tokens=args.max_new_tokens, temperature=1.0, top_p=1.0)
    request = RolloutRequest(prompt_ids=input_ids, prompt_attention_mask=attn_mask)

    times = []
    for i in range(args.num_warmup + args.num_iters):
        get_accelerator().synchronize()  #ignore-cuda
        t0 = time.perf_counter()
        with torch.no_grad():
            result = rollout.generate(request, sampling)
        get_accelerator().synchronize()  #ignore-cuda
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if local_rank == 0:
            label = "warmup" if i < args.num_warmup else "iter"
            n_tokens = result.input_ids.shape[-1] - args.prompt_len
            print(f"  [{label}] {elapsed*1000:.1f} ms, tokens={n_tokens}")

    if local_rank == 0:
        avg = np.mean(times[-args.num_iters:]) * 1000
        per_step = avg / args.max_new_tokens
        throughput = 1000.0 / per_step
        print()
        mode = "FlashInfer" if use_flashinfer else "Baseline (SDPA)"
        print(f"=== Results ({mode}) ===")
        print(f"  Total generate:   {avg:.1f} ms")
        print(f"  Per decode step:  {per_step:.2f} ms")
        print(f"  Throughput:       {throughput:.1f} tokens/s")


if __name__ == "__main__":
    main()

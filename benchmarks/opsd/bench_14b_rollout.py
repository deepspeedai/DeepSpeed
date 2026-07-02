# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Comprehensive 14B rollout benchmark: Naive, GC, TP=2 GC, TP=4 GC."""
import os
import time

import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout import HybridEngineRollout, RolloutRequest, SamplingConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-14B-Instruct"
MAX_NEW_TOKENS = 256
N_SAMPLES = 1
CB_SIZE = 1
N_RUNS = 5
PROMPT = "def fibonacci(n):"


def bench_rollout(engine, tokenizer, use_graph_capture, cb_size, label):
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(get_accelerator().device_name(local_rank))

    rollout = HybridEngineRollout(
        engine=engine,
        tokenizer=tokenizer,
        continuous_batching_size=cb_size,
        use_graph_capture=use_graph_capture,
    )

    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    req = RolloutRequest(prompt_ids=ids, prompt_attention_mask=torch.ones_like(ids))
    sampling = SamplingConfig(max_new_tokens=MAX_NEW_TOKENS,
                              temperature=0.8,
                              top_p=0.95,
                              n_samples_per_prompt=N_SAMPLES)

    # Warmup
    torch.manual_seed(42)
    engine.eval()
    rollout.generate(req, sampling)
    engine.train()

    # Benchmark
    times = []
    total_toks = 0
    for i in range(N_RUNS):
        torch.manual_seed(42 + i)
        engine.eval()
        get_accelerator().synchronize()
        t0 = time.time()
        batch = rollout.generate(req, sampling)
        get_accelerator().synchronize()
        times.append(time.time() - t0)
        engine.train()

    # Count tokens from last run
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    for i in range(batch.input_ids.shape[0]):
        resp = batch.input_ids[i, batch.response_start_idx[i]:]
        total_toks += (resp != pad_id).sum().item()

    t_avg = sum(times[1:]) / len(times[1:])

    if rank == 0:
        print(f"[{label}] {total_toks} toks, {t_avg*1000:.0f}ms, {total_toks/t_avg:.1f} tok/s  "
              f"runs={[f'{t*1000:.0f}' for t in times]}")

    return total_toks, t_avg


def main():
    deepspeed.init_distributed()
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    get_accelerator().set_device(local_rank)

    world_size = torch.distributed.get_world_size()
    tp_size = world_size  # all GPUs used for TP

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, trust_remote_code=True)

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
        "hybrid_engine": {
            "enabled": True,
            "max_out_tokens": 512,
            "inference_tp_size": 1,
            "release_inference_cache": False,
            "pin_parameters": True,
            "tp_gather_partition_size": 8,
        },
    }

    if tp_size > 1:
        ds_config["tensor_parallel"] = {
            "autotp_size": tp_size,
            "preset_model": "qwen2",
            "tp": {
                "tp_size": tp_size
            },
        }

    engine, *_ = deepspeed.initialize(model=model, config=ds_config)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Model: {MODEL}, TP={tp_size}, n={N_SAMPLES}, cb={CB_SIZE}, max_new={MAX_NEW_TOKENS}")
        print(f"{'='*60}")

    # 1P1R without graph capture (CB=1, no GC)
    try:
        bench_rollout(engine, tokenizer, use_graph_capture=False, cb_size=CB_SIZE, label=f"TP{tp_size} CB={CB_SIZE}")
    except Exception as e:
        if rank == 0:
            print(f"[TP{tp_size} CB={CB_SIZE}] FAILED: {e}")
            import traceback
            traceback.print_exc()

    # 1P1R with CUDA graph capture
    try:
        bench_rollout(engine, tokenizer, use_graph_capture=True, cb_size=CB_SIZE, label=f"TP{tp_size} CB={CB_SIZE}+GC")
    except Exception as e:
        if rank == 0:
            print(f"[TP{tp_size} CB={CB_SIZE}+GC] FAILED: {e}")
            import traceback
            traceback.print_exc()

    if rank == 0:
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

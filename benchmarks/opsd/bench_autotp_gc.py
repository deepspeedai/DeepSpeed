# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Benchmark rollout with AutoTP + graph capture on 14B model."""
import time
import torch
import deepspeed
from deepspeed.runtime.rollout import HybridEngineRollout, RolloutRequest, SamplingConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    deepspeed.init_distributed()
    rank = torch.distributed.get_rank()
    local_rank = int(torch.distributed.get_rank()) % torch.cuda.device_count()  #ignore-cuda
    torch.cuda.set_device(local_rank)  #ignore-cuda
    device = torch.device(f"cuda:{local_rank}")

    model_name = "Qwen/Qwen2.5-14B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, trust_remote_code=True)

    ds_config = {
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 0
        },
        "tensor_parallel": {
            "autotp_size": 2,
            "preset_model": "qwen2",
            "tp": {
                "tp_size": 2
            },
        },
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": 2,
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

    engine, *_ = deepspeed.initialize(model=model, config=ds_config)

    rollout = HybridEngineRollout(
        engine=engine,
        tokenizer=tokenizer,
        continuous_batching_size=2,
        use_graph_capture=True,
    )

    # Prepare prompt
    prompt = "def fibonacci(n):"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    req = RolloutRequest(prompt_ids=ids, prompt_attention_mask=torch.ones_like(ids))
    sampling = SamplingConfig(max_new_tokens=256, temperature=0.8, top_p=0.95, n_samples_per_prompt=4)

    # Warmup
    torch.manual_seed(42)
    engine.eval()
    rollout.generate(req, sampling)
    engine.train()

    # Benchmark
    times = []
    for i in range(5):
        torch.manual_seed(42)
        engine.eval()
        torch.cuda.synchronize()  #ignore-cuda
        t0 = time.time()
        batch = rollout.generate(req, sampling)
        torch.cuda.synchronize()  #ignore-cuda
        times.append(time.time() - t0)
        engine.train()

    t_avg = sum(times[1:]) / len(times[1:])
    # Count tokens
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    total_toks = 0
    for i in range(batch.input_ids.shape[0]):
        resp = batch.input_ids[i, batch.response_start_idx[i]:]
        total_toks += (resp != pad_id).sum().item()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"TP=2, n=8, cb=4, graph_capture=True, max_new_tokens=256")
        print(f"Avg latency (excl warmup): {t_avg*1000:.1f}ms")
        print(f"Total response tokens: {total_toks}")
        print(f"Throughput: {total_toks/t_avg:.1f} tok/s")
        print(f"Per-run times: {[f'{t*1000:.0f}ms' for t in times]}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

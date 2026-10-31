# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""TP=2 b=8 benchmark: DeepSpeed rollout vs vLLM, same prompts, greedy.

DeepSpeed side: AutoTP to shard the model across 2 GPUs, then the graph
capture decode path. vLLM side: tensor_parallel_size=2, same everything."""

import os
import time

import torch

MODEL = "Qwen/Qwen3.5-4B-Base"
BS, N = 8, 128
PARA = "tp2"


def bench_ds():
    """DeepSpeed rollout with AutoTP sharding + graph capture."""
    import deepspeed
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
    from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig

    tok = AutoTokenizer.from_pretrained(MODEL)
    object.__setattr__(tok, "eos_token_id", -1)

    ds_config = {
        "train_batch_size": 2,
        "tensor_parallel": {
            "autotp_size": 2,
        },
        "bf16": {
            "enabled": True,
        },
    }

    # Use deepspeed.initialize for TP sharding
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, torch_dtype=torch.bfloat16)
    engine, _, _, _ = deepspeed.initialize(model=m, config=ds_config)

    class Shim:

        def __init__(self, m):
            self.module = m

    # AutoTP wraps module.config as a dict; restore the model's config object
    shim = Shim(engine)
    if isinstance(getattr(shim.module, "config", None), dict):
        shim.module.config = m.config
    import sys as _sys
    use_graph = "--graph" in _sys.argv
    rollout = HybridEngineRollout(shim, tok, HybridEngineRolloutConfig(use_graph_capture=use_graph,
                                                                      use_segki=use_graph))

    para = ("The Eiffel Tower is located in Paris, France. It was built in 1889 "
            "and stands 330 meters tall. The Louvre museum houses the Mona Lisa. ")
    ids_short = tok("The capital of France is", return_tensors="pt").input_ids
    ids_2k = tok(para * 120, return_tensors="pt").input_ids[:, :2048]

    for ctx, ids in [("short", ids_short), ("2k", ids_2k)]:
        ids = ids.to("cuda")
        req = RolloutRequest(prompt_ids=ids.repeat(BS, 1), prompt_attention_mask=torch.ones_like(ids).repeat(BS, 1))
        s = SamplingConfig(max_new_tokens=32, temperature=0.0)
        rollout.generate(req, s)  # warmup
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        rollout.generate(req, SamplingConfig(max_new_tokens=256, temperature=0.0))
        torch.cuda.synchronize()
        t_256 = time.perf_counter() - t0
        t0 = time.perf_counter()
        rollout.generate(req, SamplingConfig(max_new_tokens=1024, temperature=0.0))
        torch.cuda.synchronize()
        t_1024 = time.perf_counter() - t0
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"TP2-DS {ctx:5s}: decode-only {768 * BS / (t_1024 - t_256):7.1f} tok/s total", flush=True)
    if int(os.environ.get("RANK", "0")) == 0:
        print("TP2-DS-DONE")


def bench_vllm():
    """vLLM with tensor_parallel_size=2."""
    from vllm import LLM, SamplingParams

    para = ("The Eiffel Tower is located in Paris, France. It was built in 1889 "
            "and stands 330 meters tall. The Louvre museum houses the Mona Lisa. ")
    SHORT = "The capital of France is"
    LONG = para * 120

    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.92, max_model_len=8192,
              tensor_parallel_size=2)
    for ctx, text in [("short", SHORT), ("2k", LONG)]:
        prompts = [text] * BS
        sp = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)
        llm.generate(prompts, sp)
        t0 = time.perf_counter()
        llm.generate(prompts, sp)
        t_256 = time.perf_counter() - t0
        sp_long = SamplingParams(temperature=0.0, max_tokens=1024, ignore_eos=True)
        llm.generate(prompts, sp_long)
        t0 = time.perf_counter()
        llm.generate(prompts, sp_long)
        t_1024 = time.perf_counter() - t0
        print(f"TP2-VLLM {ctx:5s}: decode-only {768 * BS / (t_1024 - t_256):7.1f} tok/s total", flush=True)
    print("TP2-VLLM-DONE")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("ds", "both"):
        bench_ds()
    if mode in ("vllm", "both"):
        bench_vllm()

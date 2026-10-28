# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Measure continuous rollout latency before and after a training update."""

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=positive_int, default=8)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--prompt-len", type=positive_int, default=32)
    parser.add_argument("--max-new-tokens", type=positive_int, default=32)
    parser.add_argument("--iterations", type=positive_int, default=10)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this file")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 2:
        parser.error("--max-new-tokens must be at least 2 to exercise graph replay")
    if args.prompt_len < 2:
        parser.error("--prompt-len must be at least 2 for the next-token training loss")
    if args.requests <= args.batch_size:
        parser.error("--requests must exceed --batch-size to exercise continuous slot refills")
    return args


def measure_calls(rollout, request, sampling, iterations, expected):
    accelerator = get_accelerator()
    latencies = []
    generated_tokens = 0
    for _ in range(iterations):
        accelerator.synchronize()
        start = time.perf_counter()
        output = rollout.generate(request, sampling)
        accelerator.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)

        # Keep correctness checks and device-to-host copies outside the timed interval.
        actual = (output.input_ids.cpu(), output.attention_mask.cpu(), output.response_start_idx.cpu())
        for observed, reference in zip(actual, expected):
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        response_mask = torch.arange(output.seq_len)[None, :] >= actual[2][:, None]
        generated_tokens += int((actual[1] * response_mask).sum())

    return {
        "latencies_ms": latencies,
        "median_ms": statistics.median(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "generated_tokens": generated_tokens,
        "tokens_per_second": generated_tokens / (sum(latencies) / 1000.0),
    }


def training_step(model, request):
    accelerator = get_accelerator()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    accelerator.synchronize()
    start = time.perf_counter()
    loss = model(input_ids=request.prompt_ids,
                 attention_mask=request.prompt_attention_mask,
                 labels=request.prompt_ids,
                 use_cache=False).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    accelerator.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if not torch.isfinite(loss):
        raise RuntimeError("The training update produced a non-finite loss")
    model.eval()
    return {"latency_ms": elapsed_ms, "loss": loss.item(), "optimizer": "SGD", "learning_rate": 0.01}


@torch.no_grad()
def reference_output(model, request, sampling):
    output_ids = model.generate(
        input_ids=request.prompt_ids,
        attention_mask=request.prompt_attention_mask,
        max_new_tokens=sampling.max_new_tokens,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=0,
    )
    return (output_ids.cpu(), torch.ones_like(output_ids, device="cpu"),
            torch.full((output_ids.shape[0], ), request.prompt_ids.shape[1], dtype=torch.long))


def run_benchmark(args):
    accelerator = get_accelerator()
    if accelerator.device_name() != "cuda" or not accelerator.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    accelerator.set_device(0)
    torch.manual_seed(args.seed)
    device = torch.device(accelerator.current_device_name())
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=(args.prompt_len + args.max_new_tokens) * args.requests,
        pad_token_id=0,
        eos_token_id=None,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).to(device=device, dtype=getattr(torch, args.dtype)).eval()
    prompt_ids = torch.randint(1, config.vocab_size, (args.requests, args.prompt_len), device=device)
    request = RolloutRequest(prompt_ids, torch.ones_like(prompt_ids))
    sampling = SamplingConfig(max_new_tokens=args.max_new_tokens, temperature=0, continuous_batch_size=args.batch_size)
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None)
    engine = SimpleNamespace(module=model)
    rollouts = {
        mode: HybridEngineRollout(engine, tokenizer, HybridEngineRolloutConfig(use_graph_capture=(mode == "graph")))
        for mode in ("eager", "graph")
    }
    results = {mode: {} for mode in rollouts}

    # Warm CUDA/model libraries without touching either rollout's graph state.
    with torch.no_grad():
        model(input_ids=prompt_ids, attention_mask=request.prompt_attention_mask, use_cache=False)
    accelerator.synchronize()

    update = None
    for first_phase, steady_phase in (("first_call", "steady_state"), ("first_call_after_update",
                                                                       "steady_state_after_update")):
        if first_phase == "first_call_after_update":
            update = training_step(model, request)
        expected = reference_output(model, request, sampling)
        for mode, rollout in rollouts.items():
            results[mode][first_phase] = measure_calls(rollout, request, sampling, 1, expected)
            results[mode][steady_phase] = measure_calls(rollout, request, sampling, args.iterations, expected)

    graph = results["graph"]
    return {
        "environment": {
            "gpu": torch.cuda.get_device_name(0),  #ignore-cuda: accelerator has no GPU model-name API
            "total_memory_bytes": accelerator.total_memory(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "workload": {
            "model": "random tiny Llama (no download)",
            "model_config": config.to_dict(),
            "dtype": args.dtype,
            "seed": args.seed,
            "requests": args.requests,
            "continuous_batch_size": args.batch_size,
            "prompt_len": args.prompt_len,
            "max_new_tokens": args.max_new_tokens,
            "iterations": args.iterations,
            "eos_disabled": True,
            "profiling_enabled": False,
        },
        "results":
        results,
        "training_update":
        update,
        "graph_first_call_premium_ms":
        graph["first_call"]["median_ms"] - graph["steady_state"]["median_ms"],
        "graph_post_update_premium_ms":
        graph["first_call_after_update"]["median_ms"] - graph["steady_state_after_update"]["median_ms"],
        "outputs_match":
        True,
    }


def main():
    args = parse_args()
    report = run_benchmark(args)
    serialized = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()

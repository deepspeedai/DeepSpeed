# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Benchmark: paged KV cache across every generation path that can drive it.

Runs the same greedy rollout under three cache/kernel configurations and, for each
one, under every generation path of ``HybridEngineRollout``:

  variants   dense          the engine's default dense static cache (baseline)
             paged          block-table cache + the blocked-attention kernel
             paged_split_kv block-table cache + FlashInfer tensor-core split-KV decode

  paths      batched            ``module.generate()``; dense only, since HuggingFace
                                owns that loop and never advances paged write positions
             graph              one captured decode graph, no continuous batching
             continuous         eager continuous batching
             continuous_graph   continuous batching with a captured decode graph

The table reports wall time, tokens/s and peak device memory per cell, plus the token
stream check: every paged cell must emit the same greedy tokens as the dense cell of the
same path. Paged cells also report the cache statistics (blocks in use, block
utilization, prefix-reuse counters) from ``get_last_continuous_stats()``.

Launch (from repo root):

    # tiny random Qwen3, one GPU: shape only, no weights
    python benchmarks/rollout/bench_paged_attention.py --prompt-len 2048 --max-new-tokens 64

    # a real model
    python benchmarks/rollout/bench_paged_attention.py --model /path/to/Qwen3-4B \\
        --prompt-len 3072 --max-new-tokens 1024 --batch-size 4 --dtype bf16

Arguments:
    --model                 HuggingFace id or path; omit to build a small random Qwen3
    --prompt-len            prompt tokens per row (default: 1024)
    --max-new-tokens        response tokens per request (default: 64)
    --batch-size            prompt rows, and the continuous-batching slot count (default: 4)
    --paths                 comma separated subset of batched,graph,continuous,continuous_graph
    --variants              comma separated subset of dense,paged,paged_split_kv
    --paged-prefill         prefill prompts straight into pages (default: on)
    --prefix-reuse          duplicate half the prompts so paged prefill can claim their blocks
    --flashinfer-prefill    add a fourth variant that also routes paged prefill through FlashInfer
    --warmup / --trials     unmeasured / measured repetitions per cell (default: 1 / 3)
    --dtype                 fp32 | fp16 | bf16 (default: bf16)
    --layers/--hidden-size/--heads/--kv-heads/--head-dim   shape of the random model

The paged variants need a CUDA device with compute capability >= 8.0 and build the
ragged-device ops on first use (a few minutes); the dense baseline runs anywhere.

    python benchmarks/rollout/bench_paged_attention.py ...            (installed deepspeed)
    PYTHONPATH=$PWD python benchmarks/rollout/bench_paged_attention.py ...  (repo checkout)
"""

import argparse
import time
from types import SimpleNamespace

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# (name, use_graph_capture, continuous batching)
PATHS = (
    ("batched", False, False),
    ("graph", True, False),
    ("continuous", False, True),
    ("continuous_graph", True, True),
)

VARIANTS = ("dense", "paged", "paged_split_kv")


def parse_args():
    parser = argparse.ArgumentParser(description="paged KV cache rollout benchmark")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--paths", type=str, default=",".join(name for name, _, _ in PATHS))
    parser.add_argument("--variants", type=str, default=",".join(VARIANTS))
    parser.add_argument("--paged-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefix-reuse", action="store_true")
    parser.add_argument("--flashinfer-prefill", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--dtype", type=str, default="bf16", choices=list(DTYPES))
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_model(args, device, dtype):
    if args.model is None:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        config = Qwen3Config(vocab_size=3072,
                             hidden_size=args.hidden_size,
                             intermediate_size=2 * args.hidden_size,
                             num_hidden_layers=args.layers,
                             num_attention_heads=args.heads,
                             num_key_value_heads=args.kv_heads,
                             head_dim=args.head_dim,
                             max_position_embeddings=args.prompt_len + args.max_new_tokens,
                             pad_token_id=0,
                             eos_token_id=None)
        return Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa")
    return model.to(device=device).eval()


def build_workload(args, model, device):
    vocab_size = int(getattr(model.config, "vocab_size", 3072))
    prompts = torch.randint(3, vocab_size, (args.batch_size, args.prompt_len), device=device)
    if args.prefix_reuse:
        # Duplicate rows give the paged prefill complete blocks to claim.
        prompts[1::2] = prompts[::2]
    return RolloutRequest(prompts, torch.ones_like(prompts))


def build_variants(args):
    split_kv = {"use_paged_cache": True, "use_flashinfer_decode": True}
    variants = {
        "dense": {},
        "paged": {
            "use_paged_cache": True
        },
        "paged_split_kv": dict(split_kv),
        "paged_flashinfer": dict(split_kv, use_flashinfer_prefill=True),
    }
    selected = [name.strip() for name in args.variants.split(",") if name.strip()]
    if args.flashinfer_prefill and "paged_flashinfer" not in selected:
        selected.append("paged_flashinfer")
    unknown = [name for name in selected if name not in variants]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {sorted(variants)}")
    built = []
    for name in selected:
        kwargs = dict(variants[name])
        if kwargs.get("use_paged_cache"):
            kwargs["enable_paged_prefill"] = args.paged_prefill
            kwargs["enable_prefix_reuse"] = args.prefix_reuse
        built.append((name, kwargs))
    return built


def build_config(args, variant_kwargs, continuous):
    kwargs = dict(variant_kwargs)
    if continuous:
        # One capacity for every cell, so the paged and dense tables compare like for like.
        kwargs["continuous_cache_capacity"] = args.prompt_len + args.max_new_tokens
    return HybridEngineRolloutConfig(**kwargs)


def run_cell(args, model, workload, path, variant, device):
    name, graph_capture, continuous = path
    variant_name, variant_kwargs = variant
    sampling = SamplingConfig(max_new_tokens=args.max_new_tokens,
                              temperature=0.0,
                              continuous_batch_size=args.batch_size if continuous else None)
    accelerator = get_accelerator()
    try:
        rollout = HybridEngineRollout(
            SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=None),
            build_config(args, dict(variant_kwargs, use_graph_capture=graph_capture), continuous))
        for _ in range(max(0, args.warmup)):
            with torch.no_grad():
                output = rollout.generate(workload, sampling)
        accelerator.synchronize()
        accelerator.reset_peak_memory_stats()
        best_ms = None
        for _ in range(max(1, args.trials)):
            accelerator.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                output = rollout.generate(workload, sampling)
            accelerator.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            best_ms = elapsed_ms if best_ms is None else min(best_ms, elapsed_ms)
    except Exception as error:  # a cell that cannot run is reported, not silent
        return {"error": f"{type(error).__name__}: {error}"}

    # max_new_tokens is a cap: a pretrained model's generation config can stop a row at
    # EOS, so count the valid response positions instead of the requested budget.
    generated = int(output.attention_mask[:, args.prompt_len:].sum().item())
    return {
        "ids": output.input_ids.cpu(),
        "ms": best_ms,
        "tokens/s": generated / (best_ms / 1000.0) if best_ms > 0 else 0.0,
        "peak GB": accelerator.max_memory_allocated() / 2**30,
        "stats": rollout.get_last_continuous_stats() or {},
    }


def summarize(path_name, variant_name, run):
    stats = run.get("stats", {})
    return {
        "path": path_name,
        "variant": variant_name,
        "ms": run.get("ms", 0.0),
        "tokens/s": run.get("tokens/s", 0.0),
        "peak GB": run.get("peak GB", 0.0),
        "match": run.get("match", "n/a"),
        "blocks": stats.get("blocks_in_use", "-"),
        "block util": stats.get("block_utilization", 0.0),
        "reuse blk": stats.get("prefix_reuse_blocks", "-"),
        "note": run.get("error", ""),
    }


def print_table(rows):
    print(f"{'path':<18} {'variant':<16} {'ms':>10} {'tokens/s':>10} {'peak GB':>9} "
          f"{'match':>8} {'blocks':>8} {'block util':>11} {'reuse blk':>10}")
    print("-" * 112)
    for row in rows:
        if row["note"]:
            print(f"{row['path']:<18} {row['variant']:<16} {'n/a':>10} {'n/a':>10} {'n/a':>9} "
                  f"{'n/a':>8} {'-':>8} {'-':>11} {'-':>10}  {row['note']}")
            continue
        blocks = row["blocks"] if row["blocks"] != "-" else "-"
        reuse = row["reuse blk"] if row["reuse blk"] != "-" else "-"
        print(f"{row['path']:<18} {row['variant']:<16} {row['ms']:>10.1f} {row['tokens/s']:>10.2f} "
              f"{row['peak GB']:>9.2f} {row['match']:>8} {blocks!s:>8} {row['block util']:>11.4f} {reuse!s:>10}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    accelerator = get_accelerator()
    if not accelerator.is_available():
        raise SystemExit("this benchmark needs an accelerator")
    device = torch.device(accelerator.current_device_name())
    dtype = DTYPES[args.dtype]

    graph_capture = accelerator.is_graph_capture_supported() and accelerator.create_graph() is not None
    paths = [path for path in PATHS if path[0] in [name.strip() for name in args.paths.split(",") if name.strip()]]
    if not graph_capture:
        print("NOTE: graph capture is unavailable on this accelerator; the graph paths are skipped.\n")
        paths = [path for path in paths if not path[1]]

    model = build_model(args, device, dtype)
    workload = build_workload(args, model, device)
    variants = build_variants(args)
    if args.dtype == "fp32":
        # The blocked-attention kernel has no fp32 path, so the paged cells would only
        # report that rejection; the dense baseline is what fp32 can measure here.
        print("NOTE: the paged cache needs fp16/bf16 K/V; skipping the paged variants for --dtype fp32.\n")
        variants = [variant for variant in variants if not variant[1].get("use_paged_cache")]
    print(f"device={device} dtype={dtype} model={args.model or 'random-qwen3'} "
          f"batch={args.batch_size} prompt_len={args.prompt_len} new_tokens={args.max_new_tokens} "
          f"paged_prefill={args.paged_prefill} prefix_reuse={args.prefix_reuse}\n")

    rows = []
    for path in paths:
        baseline = None
        for variant in variants:
            run = run_cell(args, model, workload, path, variant, device)
            if "ids" in run and baseline is not None:
                run["match"] = "MATCH" if torch.equal(run["ids"], baseline) else "MISMATCH"
            elif "ids" in run:
                run["match"] = "base"
                baseline = run["ids"]
            rows.append(summarize(path[0], variant[0], run))

    print_table(rows)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Benchmark: one wide decode graph vs a ladder of decode graphs, on a multi-turn rollout.

A multi-turn rollout re-submits every conversation with the tokens it just generated,
and conversations end their turns at different times, so the number of rows decoding at
once changes from turn to turn. With one captured graph sized to ``max_batch_size``,
every decode step replays that width; with a ladder, each step replays the smallest
captured width that covers the rows doing work.

The bench drives the same multi-turn workload twice - once with
``graph_capture_batch_sizes=(max_batch_size,)`` and once with the default ladder - and
prints the token streams, the work each configuration feeds through the decode graph,
and the timing.

  row-steps   sum over decode steps of the width that step replayed. Attention and MLP
              cost scale with rows, so this is the decode work the graph does; the
              ladder is meant to move it, and padding that cannot be removed shows up
              as row-steps above the live rows.
  pad rows    rows a step carried that were not decoding, as reported by the rollout.
  tokens/s    the throughput price of the remaining padding.

Launch (from repo root):

    # offline dispatch run: no GPU, the graph is mocked, so only the accounting is real
    python benchmarks/rollout/bench_graph_capture_ladder.py --mock-capture

    # a real model on one GPU
    python benchmarks/rollout/bench_graph_capture_ladder.py --model meta-llama/Llama-3.2-1B \\
        --conversations 16 --turns 4 --max-batch-size 16 --max-new-tokens 32 --dtype bf16

Arguments:
    --model           HuggingFace id or path; omit to build a small random model
    --conversations   conversations in the rollout (default: 8)
    --turns           longest conversation, in turns; budgets are staggered (default: 3)
    --prompt-len      tokens in each conversation's first prompt (default: 64)
    --max-new-tokens  response tokens per turn (default: 16)
    --max-batch-size  decode slots, i.e. rows that can decode at once (default: 8)
    --device          auto | cpu | cuda | mps (default: auto)
    --dtype           fp32 | fp16 | bf16 (default: bf16 on cuda, fp32 otherwise)
    --graph           auto | on | off capture of the decode graph (default: auto)
    --mock-capture    exercise the captured path with a mocked accelerator, so the ladder
                      runs without a GPU; the timing is then meaningless (default: off)
    --trials          measured repetitions per configuration, best is reported (default: 3)
    --warmup          unmeasured repetitions per configuration (default: 1)
    --layers/--hidden-size/--heads/--vocab-size   shape of the random model

    python benchmarks/rollout/bench_graph_capture_ladder.py ...            (installed deepspeed)
    PYTHONPATH=$PWD python benchmarks/rollout/bench_graph_capture_ladder.py ...  (repo checkout)
"""

import argparse
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args():
    parser = argparse.ArgumentParser(description="one wide decode graph vs a ladder of decode graphs")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--conversations", type=int, default=8)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", type=str, default=None, choices=list(DTYPES))
    parser.add_argument("--graph", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--mock-capture", action="store_true")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def build_model(args, device, dtype):
    """Load a checkpoint, or build a small random model so the script runs offline."""
    if args.model is None:
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(vocab_size=args.vocab_size,
                             hidden_size=args.hidden_size,
                             intermediate_size=4 * args.hidden_size,
                             num_hidden_layers=args.layers,
                             num_attention_heads=args.heads,
                             num_key_value_heads=args.heads,
                             max_position_embeddings=4096,
                             pad_token_id=0,
                             eos_token_id=1)
        model = LlamaForCausalLM(config)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    return model.to(device=device, dtype=dtype).eval()


def build_conversations(args, device):
    """One growing token stream per conversation, with staggered turn budgets.

    Conversation ``i`` gets ``1 + i % turns`` turns, so some conversations retire after
    every turn and the live batch drains the way a multi-turn rollout drains.
    """
    turns = max(1, args.turns)
    return [{
        "ids": torch.randint(2, args.vocab_size, (args.prompt_len, ), dtype=torch.long, device=device),
        "turns": 1 + index % turns,
    } for index in range(args.conversations)]


def build_request(conversations, device, pad_token_id=0):
    """Left-pad the live conversations to a common width, the layout the continuous path wants."""
    width = max(conversation["ids"].numel() for conversation in conversations)
    prompt_ids = torch.full((len(conversations), width), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(prompt_ids)
    for row, conversation in enumerate(conversations):
        length = conversation["ids"].numel()
        prompt_ids[row, width - length:] = conversation["ids"]
        attention[row, width - length:] = 1
    return RolloutRequest(prompt_ids, attention)


def run_once(args, model, conversations, ladder, graph_capture, device):
    """One multi-turn rollout: generate a turn, append it, drop the conversations that ended."""
    cfg = HybridEngineRolloutConfig(use_graph_capture=graph_capture, graph_capture_batch_sizes=ladder)
    rollout = HybridEngineRollout(SimpleNamespace(module=model),
                                  SimpleNamespace(pad_token_id=0, eos_token_id=None),
                                  cfg=cfg)
    sampling = SamplingConfig(max_new_tokens=args.max_new_tokens,
                              temperature=0,
                              continuous_batch_size=args.max_batch_size)
    live = [dict(conversation) for conversation in conversations]
    turns = []
    stream = []
    widths = {}
    generated = steps = row_steps = padded_rows = captures = calls = 0
    accelerator = get_accelerator()
    accelerator.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        while live:
            output = rollout.generate(build_request(live, device), sampling)
            stats = rollout.get_last_continuous_stats() or {}
            turns.append(len(live))
            stream.append(output.input_ids.cpu())
            calls += 1
            generated += len(live) * args.max_new_tokens
            steps += stats.get("decode_steps", 0)
            padded_rows += stats.get("decode_padded_rows", 0)
            captures += stats.get("graph_captures", 0)
            for width, count in stats.get("decode_bucket_widths", {}).items():
                widths[width] = widths.get(width, 0) + count
                row_steps += width * count
            live = [{
                "ids": output.input_ids[row].clone(),
                "turns": conversation["turns"] - 1,
            } for row, conversation in enumerate(live) if conversation["turns"] > 1]
    accelerator.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "stream": torch.cat([turn.reshape(-1) for turn in stream]),
        "turns": turns,
        "elapsed_ms": elapsed_ms,
        "calls": calls,
        "generated": generated,
        "steps": steps,
        "row_steps": row_steps,
        "padded_rows": padded_rows,
        "captures": captures,
        "widths": widths,
    }


def measure(args, model, conversations, ladder, graph_capture, device):
    """Best of `--trials` runs after `--warmup` unmeasured ones.

    A fresh rollout is built per run, because the graph cache and the captured buffers
    are per rollout: timing then reflects steady-state generation, not first-call
    initialization.
    """
    for _ in range(max(0, args.warmup)):
        run_once(args, model, conversations, ladder, graph_capture, device)
    best = None
    for _ in range(max(1, args.trials)):
        run = run_once(args, model, conversations, ladder, graph_capture, device)
        if best is None or run["elapsed_ms"] < best["elapsed_ms"]:
            best = run
    return best


def summarize(name, run, args):
    widths = ",".join(f"{width}x{count}" for width, count in sorted(run["widths"].items()))
    return {
        "config": name,
        "calls": run["calls"],
        "gen tokens": run["generated"],
        "steps": run["steps"],
        "row-steps": run["row_steps"],
        "pad rows": run["padded_rows"],
        "caps": run["captures"],
        "widths": widths or "-",
        "total ms": run["elapsed_ms"],
        "tokens/s": run["generated"] / (run["elapsed_ms"] / 1000.0) if run["elapsed_ms"] > 0 else 0.0,
    }


def print_table(rows):
    columns = list(rows[0])
    numeric = ["calls", "gen tokens", "steps", "row-steps", "pad rows", "caps", "total ms", "tokens/s"]
    print("  ".join(f"{column:>12}" if column in numeric else f"{column:<16}" for column in columns))
    print("-" * (len(columns) * 15))
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            if column in numeric:
                cells.append(f"{value:>12.2f}" if isinstance(value, float) else f"{value:>12}")
            else:
                cells.append(f"{value:<16}")
        print("  ".join(cells))


def mock_graph_accelerator():
    """Stand in for a graph-capable accelerator: capture records, replay is a no-op."""
    stream = MagicMock()
    accelerator = MagicMock()
    accelerator.Stream.return_value = stream
    accelerator.current_stream.return_value = stream
    accelerator.stream.side_effect = lambda _stream: nullcontext()
    accelerator.capture_to_graph.side_effect = lambda _graph: nullcontext()
    return patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator", return_value=accelerator)


def main():
    args = parse_args()
    accelerator = get_accelerator()
    device = torch.device(args.device if args.device != "auto" else (
        accelerator.current_device_name() if accelerator.is_available() else "cpu"))
    dtype = DTYPES[args.dtype] if args.dtype else (torch.bfloat16 if device.type == "cuda" else torch.float32)
    graph_capture = device.type == "cuda" and accelerator.create_graph() is not None
    if args.graph == "on":
        graph_capture = True
    elif args.graph == "off":
        graph_capture = False
    if args.mock_capture and args.graph == "off":
        raise SystemExit("--mock-capture and --graph off are mutually exclusive")
    graph_capture = graph_capture or args.mock_capture
    if not graph_capture:
        print("NOTE: graph capture is unavailable on this accelerator, so both configurations run the eager "
              "continuous path and the ladder is not exercised. Use --mock-capture for the dispatch "
              "accounting, or run on a device with graph capture for the timing.\n")

    model = build_model(args, device, dtype)
    conversations = build_conversations(args, device)
    print(f"device={device} dtype={dtype} graph_capture={graph_capture} mocked={args.mock_capture} "
          f"conversations={args.conversations} slots={args.max_batch_size} prompt={args.prompt_len} "
          f"new_tokens={args.max_new_tokens} turn_budgets={[c['turns'] for c in conversations]}\n")

    configs = [("one wide graph", (args.max_batch_size, )), ("ladder", None)]
    context = mock_graph_accelerator() if args.mock_capture else nullcontext()
    with context:
        results = {
            name: measure(args, model, conversations, ladder, graph_capture, device)
            for name, ladder in configs
        }

    print_table([summarize(name, run, args) for name, run in results.items()])
    baseline, ladder = results["one wide graph"], results["ladder"]
    print(f"\nlive rows per turn: {baseline['turns']}")
    if baseline["row_steps"] and ladder["row_steps"]:
        print(f"row-steps: {baseline['row_steps']} -> {ladder['row_steps']} "
              f"({ladder['row_steps'] / baseline['row_steps']:.2f}x), "
              f"padded rows: {baseline['padded_rows']} -> {ladder['padded_rows']}")
    else:
        print("row-steps: not reported (the eager path decodes rows without a captured width)")
    if args.mock_capture:
        print("token streams: not checked (the mocked replay returns the capture's logits)")
    else:
        print(f"token streams match: {torch.equal(baseline['stream'], ladder['stream'])}")


if __name__ == "__main__":
    main()

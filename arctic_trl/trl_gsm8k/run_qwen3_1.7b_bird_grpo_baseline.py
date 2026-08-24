#!/usr/bin/env python
# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TRL async-GRPO baseline for BIRD (txt2sql) -- trainer only (no server launch).

BIRD twin of ``run_qwen3_1.7b_gsm8k_grpo_baseline.py``, but designed to run under ``accelerate launch`` across
multiple trainer GPUs (DDP) against an *externally* launched ``vllm serve`` (see ``run_bird_baseline.sh``). It uses
TRL's native backends -- ``LocalTrainingClient`` (real model fwd/bwd on the trainer GPUs), ``AsyncRolloutWorker``
(generation via the external ``vllm serve``), ``WeightTransferClient`` (train -> sampler over NCCL) -- so the only
difference from the Arctic runs is which backend computes what. Data + reward come from ``bird_task`` (shared).

This process must NOT launch the server: under ``accelerate launch --num_processes N`` there are N copies of it, so
server lifecycle is owned by the launcher shell. ``AsyncGRPOTrainer`` gates rollout + weight transfer to the main
process, and averages grads across ``accelerator.num_processes``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback

# Make bird_task importable in the spawned rollout child (AsyncRolloutWorker pickles reward_funcs by module path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3-1.7B"))
    ap.add_argument("--data-path", default=os.environ.get(
        "BIRD_TRAIN_PARQUET", "/data/snowflakesql/txt2sql/train.parquet"))
    ap.add_argument("--vllm-base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "30")))
    ap.add_argument("--num-generations", type=int, default=int(os.environ.get("NUM_GEN", "8")))
    ap.add_argument("--per-device-bsz", type=int, default=int(os.environ.get("PER_DEVICE_BSZ", "8")))
    ap.add_argument("--grad-accum", type=int, default=int(os.environ.get("GRAD_ACCUM", "1")))
    ap.add_argument("--max-completion-length", type=int, default=int(os.environ.get("MAX_COMPLETION_LEN", "1024")))
    ap.add_argument("--num-prompts", type=int, default=int(os.environ.get("NUM_PROMPTS", "128")))
    ap.add_argument("--learning-rate", type=float, default=float(os.environ.get("LR", "1e-6")))
    ap.add_argument("--metrics-out", default=os.environ.get("METRICS_OUT",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_bird_baseline.json")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    import bird_task
    from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer

    is_main = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0

    # R1 txt2sql prompts instruct the model to reason inside <think> tags, so keep thinking ENABLED.
    chat_template_kwargs: dict = {}

    ok = False
    out_dir = tempfile.mkdtemp(prefix="trl_bird_base_out_")
    trainer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = bird_task.load_bird_dataset(args.data_path, args.num_prompts)
        if is_main:
            print(f"[base] BIRD dataset ready: {len(dataset)} prompts (from {args.data_path})", flush=True)

        config = AsyncGRPOConfig(
            output_dir=out_dir,
            save_strategy="no",
            seed=args.seed,
            per_device_train_batch_size=args.per_device_bsz,
            gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing=True,  # long SQL prompts: trade compute for memory
            learning_rate=args.learning_rate,
            bf16=True,  # real GPU training (the Arctic server also trains bf16)
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            temperature=1.0,
            chat_template_kwargs=chat_template_kwargs,
            max_steps=args.max_steps,
            weight_sync_steps=1,
            token_budget=0,  # FixedCountBatcher: per_device_train_batch_size samples/step
            logging_steps=1,
            report_to="none",
            log_completions=True,
            vllm_server_base_url=args.vllm_base_url,
        )

        # Native backends: pass reward_funcs -> LocalTrainingClient + AsyncRolloutWorker + WeightTransferClient.
        trainer = AsyncGRPOTrainer(
            model=args.model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            reward_funcs=bird_task.sql_reward,
        )

        if is_main:
            print(f"[base] starting trainer.train() (max_steps={args.max_steps}) ...", flush=True)
        trainer.train()

        if is_main:
            history = [h for h in trainer.state.log_history if "loss" in h]
            print(f"[base] completed {trainer.state.global_step} optimizer steps", flush=True)
            for h in history[-10:]:
                keep = {k: h[k] for k in ("loss", "reward", "ratio", "kl", "entropy", "completions/mean_length") if k in h}
                print(f"[base]   step {h.get('step')}: {keep}", flush=True)

        if trainer.state.global_step >= args.max_steps:
            ok = True
            if is_main:
                print("[base] BASELINE PASSED: native TRL async-GRPO ran end-to-end on BIRD", flush=True)
        elif is_main:
            print(f"[base] BASELINE INCOMPLETE: only {trainer.state.global_step}/{args.max_steps} steps", flush=True)
    except Exception:
        print("[base] BASELINE FAILED with exception:", flush=True)
        traceback.print_exc()
    finally:
        # Only the main process writes metrics (log_history is identical across ranks).
        if is_main and trainer is not None:
            try:
                with open(args.metrics_out, "w") as f:
                    json.dump(trainer.state.log_history, f, indent=2)
                print(f"[base] wrote metrics -> {args.metrics_out} ({len(trainer.state.log_history)} records)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[base] metrics dump raised (ignored): {e}", flush=True)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

# On-Policy Distillation (OPSD) on DeepSpeed

A DeepSpeed-native port of [HJSang/OPSD_OnPolicyDistillation](https://github.com/HJSang/OPSD_OnPolicyDistillation),
removing the verl dependency and building directly on DeepSpeed primitives
(ZeRO-3, hybrid engine, `deepspeed.initialize`).

On-policy distillation trains a small **student** model to imitate a large
frozen **teacher** on the student's *own* generated rollouts. Each training
step has three phases:

```
┌────────────┐   prompts   ┌──────────────────┐   prompt+response   ┌────────────┐
│ Dataloader │ ──────────▶ │ Student rollout  │ ──────────────────▶ │  Teacher   │
└────────────┘             │ (hybrid / vLLM)  │                     │  forward   │
                           └──────────────────┘                     └─────┬──────┘
                                                                          │ logits → CPU cache
                                                                          ▼
                                                              ┌─────────────────────┐
                                                              │ Student forward +   │
                                                              │ streamed KL / JSD + │
                                                              │ backward / step     │
                                                              └─────────────────────┘
```

Loss = per-token divergence (`forward_kl` | `reverse_kl` | `jsd`) between
student and teacher distributions on the student's generated tokens, chunked
over the sequence axis so the full `[B, T, V]` teacher tensor never
co-resides with the student logits on the training device.

## Layout

```
examples/opsd/
├── main.py                            # entry point (deepspeed launcher)
├── opsd/
│   ├── config.py                      # OPSDConfig dataclass + JSON loader
│   ├── losses.py                      # chunked / streamed KL & JSD
│   ├── teacher.py                     # frozen teacher + CPU logit cache
│   ├── trainer.py                     # three-phase training loop
│   ├── data.py                        # JSONL prompt dataset + left-pad collator
│   ├── utils.py                       # response-mask + shift helpers
│   ├── rollout/
│   │   ├── base.py                    # RolloutEngine ABC, request/batch dataclasses
│   │   ├── hybrid_engine.py           # DeepSpeed hybrid-engine rollout
│   │   └── vllm.py                    # vLLM rollout on disjoint GPUs
│   └── weight_bridge/
│       ├── base.py                    # ParallelKind + per-rank slicer
│       ├── qwen2.py                   # Qwen2 / Qwen2.5 TP mapping
│       └── qwen3.py                   # Qwen3 dense (adds q_norm/k_norm)
├── configs/
│   ├── ds_zero3.json                  # base DeepSpeed ZeRO-3 + hybrid engine
│   ├── opsd_hybrid_engine.json        # production-ish hybrid-engine OPSD config
│   ├── opsd_vllm_disjoint.json        # vLLM rollout on a disjoint GPU group
│   ├── smoke_hybrid.json              # 5-step smoke test with Qwen2.5-0.5B / 1.5B
│   ├── smoke_vllm.json                # same but with vLLM rollout
│   └── smoke_ds_zero3.json            # ZeRO-3 config tuned for smoke runs
├── scripts/
│   ├── train_opsd_hybrid.sh           # launch hybrid-engine training
│   └── train_opsd_vllm.sh             # launch vLLM training
└── tests/                             # CPU-only unit tests (run with pytest)
```

## Quick start

### Install

```
pip install deepspeed transformers datasets accelerate
# Optional, only for the vLLM rollout backend:
pip install 'vllm>=0.6.4'
```

### Hybrid-engine training (single-node, no vLLM)

```
cd examples/opsd
NUM_GPUS=8 bash scripts/train_opsd_hybrid.sh configs/opsd_hybrid_engine.json
```

The hybrid engine path lives entirely within DeepSpeed: the student engine
both trains and generates, sharing weights without a copy step. Easiest to
get running; slower generation than vLLM.

### vLLM training (disjoint GPU group)

```
cd examples/opsd
# Train on GPUs 0..5, run vLLM on 6,7 (matches default config)
NUM_TRAIN_GPUS=6 INCLUDE_GPUS=0,1,2,3,4,5 \
    bash scripts/train_opsd_vllm.sh configs/opsd_vllm_disjoint.json
```

vLLM gets dedicated GPUs (`rollout.gpus` in the config). Training rank 0
constructs the `LLM` handle; other training ranks receive generated token
ids via NCCL broadcast.

### Smoke tests (5 steps, small models)

The `smoke_*.json` configs run on 2 GPUs in a few minutes with Qwen2.5-0.5B
(student) and Qwen2.5-1.5B (teacher), so the full pipeline can be validated
end-to-end before scaling up.

```
cd examples/opsd
deepspeed --num_gpus 2 main.py --config configs/smoke_hybrid.json
# For vLLM (uses GPUs 0,1 for training and 2,3 for vLLM):
NUM_TRAIN_GPUS=2 INCLUDE_GPUS=0,1 deepspeed --num_gpus 2 --include localhost:0,1 \
    main.py --config configs/smoke_vllm.json
```

## Unit tests

The CPU-runnable test suite exercises the loss math, teacher caching, rollout
contract, weight-bridge TP slicing, and vLLM stitch logic. Run with:

```
cd examples/opsd
python -m pytest tests/ -v
```

## Configuration

`OPSDConfig` is a plain dataclass loaded from JSON (no Hydra). The schema:

```json
{
  "student":    { "model_name_or_path": "...", "dtype": "bfloat16", "arch": "qwen2" },
  "teacher":    { "model_name_or_path": "...", "dtype": "bfloat16", "offload_to_cpu": true },
  "rollout":    { "engine": "hybrid_engine | vllm", ... },
  "distillation": { "loss_type": "reverse_kl", "temperature": 1.0, "chunk_size": 512 },
  "training":   { "train_batch_size": 8, "learning_rate": 1e-6, ... },
  "data":       { "path": "data/prompts.jsonl", "prompt_field": "prompt" },
  "deepspeed_config": "configs/ds_zero3.json"
}
```

See `configs/opsd_hybrid_engine.json` and `configs/opsd_vllm_disjoint.json`
for fully-populated examples.

## Adding a new model architecture

To support a model the bridge doesn't recognise yet:

1. Add `opsd/weight_bridge/<arch>.py` subclassing `Qwen2WeightBridge` (or
   `WeightBridge` directly) and override `parallel_kind` / `_extra_layer_kind`
   for any parameters not in Qwen2's table.
2. Register the new arch in `opsd/weight_bridge/__init__.py::get_bridge`.
3. Add a test in `tests/test_weight_bridge.py` covering parallel-kind dispatch
   and a slice-then-gather round trip for one layer of realistic shapes.

## Design notes

* **Why CPU-cache the teacher logits?** Holding both student and teacher
  `[B, T, V]` tensors on GPU at once doubles memory pressure. Staging the
  teacher to host between the teacher forward and the student backward halves
  the worst-case GPU footprint of the loss path. The streamed loss
  (`losses.streamed_distillation_loss`) pulls teacher chunks back to GPU
  one sequence slice at a time so the full tensor never re-materialises.

* **Why an abstract `RolloutEngine`?** The hybrid-engine and vLLM backends
  have very different lifecycles (hybrid engine reads student weights live;
  vLLM holds its own copy and must be synced) but the trainer should not
  care. The ABC keeps the trainer engine-agnostic so additional backends
  (e.g. a future colocated-vLLM-with-`sleep_mode`) drop in without touching
  the loop.

* **vLLM topology = disjoint, not colocated (v1).** The disjoint topology is
  simpler to debug — failures in vLLM don't take down training and vice
  versa. A colocated topology using vLLM 0.6.4+'s `sleep_mode` is planned as
  a follow-up.

* **Weight bridge does not pre-fuse QKV / gate-up.** vLLM's per-model loader
  already knows how to fuse these from the standard HuggingFace layout, so
  the bridge only handles per-rank slicing.

## vLLM status

The vLLM rollout (`opsd/rollout/vllm.py`) is **written and unit-tested but
not yet usable under the DeepSpeed launcher**. During live validation on
4× H200 we hit a blocking issue:

> vLLM's worker init calls `new_group(...)` on the global process group as
> a collective. Under `deepspeed --num_gpus N`, the world is all `N`
> training ranks but only rank 0 calls into vLLM, so the constructor hangs
> waiting on the other ranks. Reproduced with vllm 0.6.6 + deepspeed 0.15.4 +
> torch 2.5.1. Standalone vLLM (world size 1) works in seconds.

The fix requires running vLLM in a **separate top-level Python process**
with its own world, accessed over HTTP/RPC from the trainer — the pattern
used by TRL and OpenRLHF. That's a larger refactor than fits in this PR;
the current `VLLMRollout` will be the basis for it once landed.

What's verified for the vLLM path today:
* `tests/test_vllm_stitch.py` — prompt + response stitching (CPU unit test)
* `tests/test_weight_bridge.py` — TP-slice math for Qwen2 / Qwen3 (CPU)
* `vllm.LLM` itself runs fine standalone on Qwen2.5-0.5B (validated)

What's **not** verified:
* End-to-end training loop with `rollout.engine = "vllm"` in `OPSDConfig`
* `LLM.collective_rpc("load_weights", ...)` weight sync at training time

The hybrid-engine path (`rollout.engine = "hybrid_engine"`) is validated
end-to-end on the same hardware.

## Other known limitations (v1)

* **vLLM weight sync (when it works) goes through pickle** —
  `LLM.collective_rpc("load_weights", args=((name, tensor_on_cpu),))`.
  Expect several seconds per sync on a 7B model. A faster v2 would broadcast
  tensors via NCCL on a shared trainer↔vLLM process group — see verl's
  `bucketed_weight_transfer.py` for a reference design.
* **vLLM `tensor_parallel_size > 1` is untested.** The weight bridge's
  slicing math is unit-tested but no live run exists.
* **Reward-weighted distillation** (OPSD's `opd.reward_beta` knob) is not
  ported. Easy to add: scale `per_tok` by a reward weight in the loss path.
* **GRPO and other on-policy RL recipes** are out of scope. The
  `RolloutEngine` / `WeightBridge` abstractions are reusable, but a GRPO
  trainer would add its own advantage / KL-to-reference logic on top.
* **Qwen3-MoE** is not covered. Add `weight_bridge/qwen3_moe.py` when needed.
* **Hybrid engine on Qwen-family models uses a ZeRO-3 fallback** (no
  hybrid-engine inference acceleration), since DeepSpeed's inference policy
  list only covers GPT2/GPT-NeoX/OPT/BLOOM/LLAMA/LLAMA2/InternLM as of 0.15.
  The fallback gathers params via `GatheredParameters` and calls the HF
  model's `generate` directly — correct, just ~3-5x slower than the
  accelerated path.

## References

* OPSD reference repo: <https://github.com/HJSang/OPSD_OnPolicyDistillation>
* DeepSpeed hybrid engine: `deepspeed/runtime/hybrid_engine.py`
* verl rollout / weight-sync design (used as a cross-check):
  <https://github.com/volcengine/verl/tree/main/verl/workers/rollout/vllm_rollout>

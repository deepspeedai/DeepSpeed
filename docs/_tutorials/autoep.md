---
title: "Automatic Expert Parallelism (AutoEP) with ZeRO-3"
tags: moe autoep zero
---

AutoEP lets DeepSpeed automatically shard MoE expert layers across
Expert Parallel (EP) ranks while keeping the rest of the model under
ZeRO-3 data parallelism.  Expert parameters are **never** DP-partitioned;
each EP rank owns its local expert weights in full, which eliminates the
all-gather overhead that standard ZeRO-3 would otherwise impose on
expert tensors at every forward pass.

## When to use AutoEP

* Your model uses MoE layers (e.g. Mixtral, DeepSeek-MoE).
* You want ZeRO-3 memory savings for non-expert parameters.
* You have ≥ 2 GPUs and `num_experts` is divisible by `autoep_size`.

## Quick-start configuration

```json
{
  "train_batch_size": 64,
  "bf16": { "enabled": true },
  "expert_parallel": {
    "enabled": true,
    "autoep_size": 8,
    "preset_model": "mixtral"
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "none" }
  }
}
```

| Field | Description |
|---|---|
| `autoep_size` | Number of GPUs per EP group. Must divide `world_size` and `num_experts`. |
| `preset_model` | Optional hint (`"mixtral"`, `"deepseek"`). Sets sensible defaults for layer detection. |

## How it works

1. **Detection** — `auto_ep.py` scans `model.named_modules()` for
   `GroupedExperts` (or layers matching the preset pattern) and records
   each layer's `num_experts`, `dim`, and `ffn_dim`.
2. **Tagging** — Every expert weight tensor receives
   `param._autoep_expert = True` and `param.allreduce = False`.
3. **ZeRO-3 exemption** — `partition_parameters.py` skips `_autoep_expert`
   params during `_zero_init_param`, setting `ds_persist = True` so the
   full tensor stays resident on the owning rank.
4. **Gradient reduction** — `stage3.py` routes expert param gradients
   through `_reduce_expert_grad` (EP all-reduce) instead of the standard
   DP reduce-scatter bucket.
5. **Optimizer step** — `_step_expert_params` runs a dedicated optimizer
   instance for expert params with the same hyperparameters (lr,
   weight_decay, etc.) as the main optimizer.
6. **Checkpointing** — Expert state is saved under `ds_autoep_layers` in
   the ZeRO checkpoint; old checkpoints without this key are loaded
   without error (backward compatible).

## Launching

```bash
deepspeed --num_gpus 8 train.py \
    --deepspeed_config ds_autoep_config.json
```

No code changes are required in your training script beyond the config.

## Running the unit tests

The testing suite is split into multiple files (e.g. smoke tests and zero3 integration tests). All 84 unit tests can run seamlessly both in multi-GPU environments and on CPU-only setups without `dist.init_process_group`:

```bash
python -m pytest tests/unit/moe/test_autoep_*.py -v -m autoep
```

## Limitations

* End-to-end multi-GPU integration tests with real EP communication
  require 8 × H100 (or equivalent) hardware and are not yet included.
* `autoep_size` must evenly divide both `world_size` and `num_experts`.
* Gradient checkpointing inside expert layers is not yet supported.

---
title: "Reflow"
tags: training IO large-model
---
Reflow is an asynchronous CPU-offload optimizer for ZeRO stage 3. Like [ZeRO-Offload](/tutorials/zero-offload/), it keeps the FP32 master weights and the optimizer state on the CPU and runs the optimizer there. The difference is *when* that CPU work runs: instead of a serial optimizer phase after backward, Reflow overlaps it with GPU compute, so the iteration spends less time waiting on the CPU.

We recommend that you read the tutorials on [ZeRO](/tutorials/zero/) and [ZeRO-Offload](/tutorials/zero-offload/) before this one.

## How Reflow works

Adding a `reflow` block to `zero_optimization` turns on three cooperating behaviors:

1. **Per-bucket optimizer during backward.** As each gradient bucket is reduced, its CPU optimizer update is submitted to a background worker, so the optimizer overlaps the rest of backward and the gradient reduce-scatter.
2. **Half-precision gradient transfer.** Gradients are copied to CPU in FP16/BF16 and promoted to FP32 inside the AVX kernel. This halves the GPU-to-CPU traffic and removes the CPU-side FP32 gradient buffer.
3. **Asynchronous state commit.** The foreground only produces the new FP16/BF16 parameters. The FP32 master weights and optimizer state are committed by a background worker after the gradient clipping and overflow checks, overlapping the next forward.

Checkpoint APIs (`state_dict`, `load_state_dict`, and the `safe_get_*`/`safe_set_*` helpers) wait for that background commit, so they always see the fully applied step.

## Configuration

Reflow needs no model code changes. Enable it in the DeepSpeed configuration:

```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "reflow": {
      "enable_cpu_affinity": true,
      "main_thread_cores": 2,
      "bucketwise_cores_per_worker": 8,
      "state_update_cores": 2
    }
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-5
    }
  }
}
```

DeepSpeed builds `ReflowCPUAdam` (or `ReflowCPULion` for Lion) for this configuration. A client `DeepSpeedCPUAdam`/`DeepSpeedCPULion` passed to `deepspeed.initialize` is remapped automatically. An empty `"reflow": {}` runs with the defaults; see the [configuration reference](/docs/config-json/#reflow) for every option.

Launch with `--bind_cores_to_rank` so each rank runs on the CPU cores local to its GPU:

```bash
deepspeed --bind_cores_to_rank train.py --deepspeed_config ds_config.json
```

With `enable_cpu_affinity`, Reflow reserves `main_thread_cores` for the forward/backward thread and splits the rest into workers of `bucketwise_cores_per_worker` cores. While the state commit overlaps the forward it uses only `state_update_cores` cores: every busy core lowers the CPU's turbo frequency, which would slow the launch-bound forward.

## Gradient clipping

The bucketwise workers apply the update before the global gradient norm is known. When a step actually needs clipping, Reflow recomputes that step's update with the clipped scale inside `step()`, so the step pays the full optimizer cost. DeepSpeed's `gradient_clipping` defaults to `1.0`; set it to `0.0` when you do not want clipping. With or without clipping, the loss matches ZeRO-Offload bit for bit in the unit tests.

## Performance

OPT-350m fine-tuning on one NVIDIA RTX 5080 (batch 4, sequence length 2048, bf16, activation checkpointing) with `gradient_clipping: 0.0`, averaged over 50 steady-state iterations:

| | ZeRO-Offload (ZeRO-3 + CPU optimizer offload) | Reflow |
| --- | --- | --- |
| Optimizer step | 264 ms | 58 ms |
| Iteration | 792 ms | 525 ms |

With every step clipped, Reflow's `step` time rises to about 139 ms because of the recomputation described above, and the iteration to 608 ms.

## Limitations

- ZeRO stage 3 with `offload_optimizer.device` set to `cpu`; any other stage or device is rejected when the configuration is parsed. NVMe optimizer offload is accepted but is not covered by tests.
- The CPU Adam/Lion extension must be built with AVX2 or AVX-512; Reflow raises an error when it creates the optimizer on other builds.
- Not supported with `super_offload`, ZenFlow, DeepCompile, the Muon optimizer, or `managed_gradient_accumulation: false`. The first four are rejected at initialization.
- Muon support needs to be implemented in the future. Reflow's CPU step applies its Adam/Lion kernels to every subgroup, so Muon's orthogonalized update needs its own path through the two-phase step.
- Per-parameter ZeRO-3 partition groups (for example AutoEP expert parallelism) run, but this combination is not validated and may produce wrong results. Reflow logs a warning when it sees such parameters.
- Each subgroup keeps a double half-precision gradient buffer on the CPU by default. A single-buffer mode (`REFLOW_SINGLE_GRAD_BUFFER=1`) exists but is not fully implemented yet; runs it does not support fall back to the double buffer with a warning.

## Questions and maintenance

Reflow is maintained by [@st-bang97](https://github.com/st-bang97). Please tag me on GitHub issues about it.

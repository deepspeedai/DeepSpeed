---
title: "Reflow"
tags: training IO large-model
---
Reflow is an asynchronous CPU-offload optimizer for ZeRO stage 3. Like [ZeRO-Offload](/tutorials/zero-offload/), it keeps the FP32 master weights and the optimizer state on the CPU and runs the optimizer there. The difference is *when* that CPU work runs: instead of a serial optimizer phase after backward, Reflow overlaps it with GPU compute, so the iteration spends less time waiting on the CPU.

We recommend that you read the tutorials on [ZeRO](/tutorials/zero/) and [ZeRO-Offload](/tutorials/zero-offload/) before this one.

## How Reflow works

Adding a `reflow` block to `zero_optimization` turns on three cooperating behaviors:

1. **Per-bucket optimizer during backward.** As each gradient bucket is reduced, its CPU optimizer update is submitted to a background worker, so the optimizer overlaps the rest of backward and the gradient reduce-scatter.
2. **Half-precision gradient transfer.** Gradients are copied to CPU in BF16 and promoted to FP32 inside the AVX kernel. This halves the GPU-to-CPU traffic and removes the CPU-side FP32 gradient buffer.
3. **Asynchronous state commit.** The foreground only produces the new BF16 parameters. The FP32 master weights and optimizer state are committed by a background worker after the gradient clipping and overflow checks, overlapping the next forward.

Checkpoint and FP32 parameter/state access APIs wait for that background commit. If a worker fails, `engine.save_checkpoint()` aborts on every rank before writing checkpoint files, and direct state reads report the failure. A full FP32 parameter or optimizer-state read uses an additional collective over the parameter partition group so all participating ranks see a failed commit; the training path adds no collective. Local gradient inspection and modification before `step()` use Reflow’s CPU buffers for the current accumulation window. Reads return FP32 values; writes are stored in the model’s FP16/BF16 gradient dtype and rounded accordingly. Changing a gradient requires Ready to regenerate the affected update.

After backward and before `engine.step()`, all ranks in a parameter’s partition group can call `safe_update_full_grad_vectorized(params, update_func)` in the same order. The callback receives the full accumulated gradient in FP32, in the parameter’s original shape and at the current loss scale. Each rank writes its slice of the result into the FP16/BF16 gradient buffer used by Ready and Persist. If a callback fails on one rank, its parameter keeps its previous gradient on every owner. Earlier successful edits are retained. Individual full-gradient reads and writes through `safe_get_full_grad` and `safe_set_full_grad` remain unsupported.

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
      "main_thread_cores": 2
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

DeepSpeed builds `ReflowCPUAdam` (or `ReflowCPULion` for Lion) for this configuration. A client `DeepSpeedCPUAdam`/`DeepSpeedCPULion` or PyTorch `Adam`/`AdamW` passed to `deepspeed.initialize` is remapped automatically. PyTorch optimizers require `"zero_force_ds_cpu_optimizer": false`. If you pass a client optimizer and an LR scheduler, pass a scheduler factory that accepts the remapped optimizer, or configure the scheduler in the DeepSpeed config. A scheduler already bound to the original optimizer is rejected. Passing a `ReflowCPUAdam` or `ReflowCPULion` without the `reflow` block is rejected. An empty `"reflow": {}` runs with the defaults; see the [configuration reference](/docs/config-json/#reflow) for every option. Partial optimizer offload (`offload_optimizer.ratio` below `1.0`) and a `data_types.grad_accum_dtype` different from the model dtype are not supported.

Launch with `--bind_cores_to_rank` so each rank runs on the CPU cores local to its GPU:

```bash
deepspeed --bind_cores_to_rank train.py --deepspeed_config ds_config.json
```

By default Reflow pins the initializing thread to `main_thread_cores` CPU IDs and gives the remaining CPUs assigned to the rank to one optimizer worker. Set `bucketwise_cores_per_worker` to split each NUMA-local group among multiple workers. Local ranks sharing a CPU set divide it before reserving their main CPUs; partially overlapping rank affinities are rejected. If the requested reservation leaves no worker CPU, Reflow reduces it to leave at least one CPU each for the main thread and workers. If only one CPU is available, they share it; ranks also share CPUs when there are fewer CPUs than ranks. The state commit may use all worker CPUs unless `state_update_cores` or `state_update_backward_cores` sets a limit. `enable_cpu_affinity` additionally restricts the main process to its NUMA-local CPU slice.

Reflow copies each contiguous range of Ready weights to the accelerator in chunks of at most 16 MiB. Set `zero_optimization.reflow.h2d_chunk_size_mib` to a positive integer to change this limit. CPU computation can overlap the transfer of an earlier chunk. Weights needed by checkpoint recomputation wait for their normal publication point. With one optimizer worker using multiple CPU threads and pinned gradients, large direct CUDA transfers share an OpenMP team. Each writer can start its portion of the next independent chunk after fencing its own stores. The last writer submits the completed chunk to the copy stream.

## Gradient clipping

The bucketwise workers apply the update before the global gradient norm is known. When a step actually needs clipping, Reflow recomputes that step's update with the clipped scale inside `step()`, so the step pays the full optimizer cost. DeepSpeed's `gradient_clipping` defaults to `1.0`; set it to `0.0` when you do not want clipping. With or without clipping, the loss matches ZeRO-Offload bit for bit in the unit tests.

## Performance

OPT-350m fine-tuning on one NVIDIA RTX 5080 with an Intel Xeon w7-3445 (batch 4, sequence length 2048, bf16, activation checkpointing, Alpaca 10%, `gradient_clipping: 0.0`). These measurements use the Reflow example's `finetune_zero3.py`, `--bind_cores_to_rank`, and `overlap_comm: false`. Times are the mean of steps 11–60 after 10 warmup steps (September 2026):

| | ZeRO-Offload (ZeRO-3 + CPU optimizer offload) | Reflow |
| --- | --- | --- |
| Optimizer step | 268 ms | 57 ms |
| Iteration | 796 ms | 523 ms |

With `gradient_clipping: 1.0` in the same setup, Reflow's optimizer step averaged 138 ms and the iteration 606 ms because clipping recomputes the update as described above.

## Limitations

- Reflow uses FP16/BF16 model parameters and gradients with FP32 master weights and optimizer states. Enable `fp16.enabled` or `bf16.enabled`; FP32 model parameters, `torch_autocast`, low-precision master weights/states, and `fp32_optimizer_states: false` are rejected.
- Use `engine.step()` after `deepspeed.initialize()`. Direct `ReflowCPUAdam.step()` and `ReflowCPULion.step()` calls are rejected.
- Reflow requires ZeRO stage 3 with optimizer offload; unsupported stages and offload devices are rejected when the configuration is parsed.
- The CPU Adam/Lion extension must be built with AVX2 or AVX-512; Reflow raises an error when it creates the optimizer on other builds.
- Not supported with `super_offload`, ZenFlow, DeepCompile, the Muon optimizer, or `managed_gradient_accumulation: false`. The first four are rejected at initialization.
- Muon support needs to be implemented in the future. Reflow's CPU step applies its Adam/Lion kernels to every subgroup, so Muon's orthogonalized update needs its own path through the two-phase step.
- Per-parameter ZeRO-3 partition groups (for example AutoEP expert parallelism) run, but this combination is not validated and may produce wrong results. Reflow logs a warning when it sees such parameters.
- Each subgroup keeps a double half-precision gradient buffer on the CPU by default. A single-buffer mode (`REFLOW_SINGLE_GRAD_BUFFER=1`) exists but is not fully implemented yet; runs it does not support fall back to the double buffer with a warning.

## Questions and maintenance

Reflow is maintained by [@st-bang97](https://github.com/st-bang97). Please tag me on GitHub issues about it.

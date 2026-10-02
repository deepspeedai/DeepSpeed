# Reflow — asynchronous CPU-offload optimizer for ZeRO-3

Reflow is an asynchronous CPU-offload optimizer for ZeRO Stage 3, driven by `ReflowCPUAdam` (and
`ReflowCPULion`). Like ZeRO-Offload it keeps the optimizer state (Adam's `exp_avg`/`exp_avg_sq`, or
Lion's single `exp_avg`) and the FP32 master weights on the CPU, but instead of running the CPU
optimizer as a separate serial phase *after* backward, it hides almost all of that cost *inside* the
backward pass. The result is ZeRO-3 CPU offloading where the CPU optimizer is largely overlapped with
GPU compute and the inter-GPU gradient reduction.

**Optimizer support:** Adam/AdamW (`ReflowCPUAdam`) and Lion (`ReflowCPULion`). Lion keeps a single
momentum tensor (`exp_avg`) — no second moment, no `eps`, no bias correction — so its offloaded state
is half of Adam's; everything else in the pipeline below is identical.

**Precision support:** BF16 model parameters and gradients with FP32 master weights and optimizer
states. FP16/FP32 model parameters, `torch_autocast`, low-precision master weights/states, and
`fp32_optimizer_states=False` are rejected at initialization. Use `engine.step()` after
`deepspeed.initialize()`; direct `ReflowCPUAdam.step()` and `ReflowCPULion.step()` calls are rejected.

This directory is the implementation. A fine-tuning example is planned for
[DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples); until it lands, the tutorial
(`docs/_tutorials/reflow.md`) shows the configuration a training script needs.

**Maintainer:** [@st-bang97](https://github.com/st-bang97) — please tag me on issues or questions about Reflow.

---

## What Reflow adds to ZeRO-3 (the `reflow` block)

A `reflow` block in `zero_optimization` turns on three cooperating behaviors:

1. **`cpu_bucketwise` — per-bucket optimizer during backward.**
   As each gradient bucket becomes ready during backward, the CPU-Adam update for that bucket is
   submitted to a background worker immediately, rather than waiting for the whole backward to
   finish. The CPU optimizer therefore overlaps the remaining backward compute and the inter-GPU
   reduce-scatter.

2. **`cpu_conversion` — half-precision gradient transfer.**
   Gradients are copied to the CPU in their native BF16 precision and promoted to FP32 *inside*
   the AVX CPU-Adam kernel. This halves the GPU→CPU traffic with no extra host-side cast, and removes
   the CPU-side FP32 gradient buffer entirely — only the half-precision BF16 gradient is kept
   on the CPU — so CPU memory usage drops.

3. **`async_state` — asynchronous state update.**
   The foreground step only generates the new BF16 parameters (produced just in time for the
   next step); updating the FP32 master parameters and the optimizer state (Adam's
   `exp_avg`/`exp_avg_sq`, or Lion's `exp_avg`) is split off to a background CPU worker. That state
   update runs after the gradient-clipping and non-finite gradient checks, so it can overlap the *next*
   iteration's forward.

A **NUMA-aware core binding** underlies all three: worker threads are pinned to the CPU cores local
to each GPU's NUMA node for higher CPU↔GPU bandwidth.

---

## Per-iteration pipeline

```
backward:   reduce bucket ──► D2H grad copy (half precision, copy_grad_stream)
                              └► submit per-bucket CPU-Adam to a worker
                                   ├► worker: FP32-promote + generate updated BF16 params
                                   └► worker: H2D copy updated BF16 weights (bucketwise_h2d_stream)
step():     drain bucket workers ─► non-finite gradient check ─► grad-norm / gradient clipping
                              └► submit the deferred FP32-master + exp_avg/exp_avg_sq update to the
                                 async-state worker (the param generation already ran during
                                 backward, so this is a thin tail)
next fwd:   the async-state worker runs concurrently with the forward; the param all-gather reads the
            already-updated GPU weights (with overlap_comm it waits on the bucketwise H2D writes,
            wait_for_external_allgather_dependencies)
```

Streams: `copy_grad_stream` (D2H grads), `_bucketwise_h2d_stream` (H2D updated weights, least
priority so it never preempts compute), and the standard ZeRO-3 `reduce_and_partition_stream` /
`__allgather_stream`. The CPU side uses a `ThreadPoolExecutor` of bucketwise workers (each pinned to
a disjoint NUMA-local core slice) plus a single chained async-state worker.

---

## Components

| File | Responsibility |
|---|---|
| `reflow_stage3.py` | `ReflowOptimizer_Stage3` (ZeRO-3 subclass): the bucketwise pipeline, the grad-copy / H2D streams, the split `step()`, the worker pools, and the affinity wiring. |
| `reflow_cpu_adam.py` | `ReflowCPUAdam` (extends `DeepSpeedCPUAdam`): the half-grad / state-only / per-range Adam variants. Registers its native optimizer in a **separate** Reflow C++ registry with its own `opt_id` space, isolating it from any plain `DeepSpeedCPUAdam` in the process. |
| `reflow_cpu_lion.py` | `ReflowCPULion` (extends `DeepSpeedCPULion`): the same half-grad / state-only / per-range variants for Lion's single-momentum update. Its own Reflow C++ registry in the `cpu_lion` extension. |
| `reflow_utils.py` | `plan_cpu_core_layout()` — NUMA-aware affinity planning (reads each GPU's PCIe→NUMA node) — plus contiguous-range helpers for the bucketwise pipeline. Pure helpers, no optimizer state. |
| `csrc/adam/reflow_cpu_adam_impl.cpp`, `csrc/includes/reflow_cpu_adam.h` | The AVX CPU-Adam kernels and the `reflow_*` C++ entry points (half-grad param generation, half-grad state commit, and half-precision gradient accumulation). |
| `csrc/lion/reflow_cpu_lion_impl.cpp`, `csrc/includes/reflow_cpu_lion.h` | The matching AVX CPU-Lion kernels and `reflow_lion_*` C++ entry points (single-momentum, sign-based update). |
| `csrc/includes/reflow_cpu_affinity.h` | Applies each worker CPU mask to its retained OpenMP helper threads when the worker moves or the state commit widens for backward. |
| `csrc/includes/reflow_simd.h` | Reflow-only SIMD conversions and streaming stores, layered on the shared `simd.h`. |
| `csrc/adam/reflow_cpu_adam_bindings.cpp`, `csrc/lion/reflow_cpu_lion_bindings.cpp` | Reflow Python bindings and GIL-released worker wrappers, registered through `csrc/includes/reflow_bindings.h` in the existing CPU optimizer modules. |

A client that passes a plain `DeepSpeedCPUAdam` / `DeepSpeedCPULion` is auto-remapped to
`ReflowCPUAdam` / `ReflowCPULion` when the `reflow` block is set (see
`engine._maybe_remap_to_reflow_cpu_optimizer`). PyTorch `Adam` and `AdamW` are also remapped.
Pass client LR schedulers as factories or configure them in DeepSpeed; a scheduler already bound
to the original optimizer cannot follow the remapping and is rejected.

---

## Configuration

```jsonc
"zero_optimization": {
    "stage": 3,
    "overlap_comm": true,            // recommended: overlaps the grad reduce with backward
    "reduce_bucket_size": 5e8,
    "sub_group_size": 1e9,
    "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true
    },
    // The block enables Reflow; {} runs with the defaults. Its options are NUMA / affinity tuning
    // (recommended for best throughput):
    "reflow": {
        "enable_cpu_affinity": true,        // restrict the main process to NUMA-local cores; workers are always pinned
        "main_thread_cores": 2,             // cores reserved for the forward/backward (main) thread
        "bucketwise_cores_per_worker": 7,   // cores per bucketwise CPU-Adam worker; worker count =
                                            //   (rank worker cores) / this value
        "state_update_cores": 2             // cores for the background optimizer-state commit while it
                                            //   overlaps the next forward (default 2)
        // "num_threads": <int>             // CPU-Adam threads (default: all available cores)
    }
    // optional: also offload PARAMETERS to CPU (orthogonal to Reflow; frees the per-rank param
    // shard of training memory). See "Parameter offload" below.
    // "offload_param": { "device": "cpu", "pin_memory": true }
}
```

Reflow keeps the optimizer state in CPU RAM (`offload_optimizer.device: "cpu"`). It is mutually
exclusive with `super_offload` and `zenflow` (the config validator raises a clear error otherwise;
see `config.reflow_compat_check`), and with DeepCompile, whose ZeRO-3 backward reduces gradients
without `partition_grads`, where Reflow launches its workers (rejected in `DeepSpeedConfig`). The
Muon optimizer is rejected too, from the config or as a client `MuonWithAuxAdam`: Reflow's CPU step
runs Adam/Lion kernels on every subgroup and would drop Muon's orthogonalized update. Muon support
needs to be implemented in the future. The
Reflow kernels need a CPU Adam/Lion build with AVX2 or AVX-512; other builds raise an error when the
optimizer is created. To fall back to plain ZeRO-Offload, drop the `reflow` block.

Per-parameter ZeRO-3 partition groups (e.g. AutoEP expert parallelism) run, and the grad-norm
reduction follows each subgroup's partition and expert-parallel groups like the base optimizer, but
this combination is not validated and may produce wrong results; Reflow logs a warning when it sees
such parameters.

Pick the core split for your CPU: with `C` cores visible per rank, reserve a few for the main thread
and give the rest to the workers (e.g. on a 9-core-per-rank box, `main_thread_cores: 2` +
`bucketwise_cores_per_worker: 7`). Launch with `deepspeed --bind_cores_to_rank` so each rank is
pinned to the cores physically attached to its GPU.

By default, counts refer to logical CPUs in the rank's affinity mask. `main_thread_cores` excludes
that many CPU IDs from workers; SMT siblings remain available and the main thread is not pinned
to the reservation. The following options allow other placements without machine-specific CPU IDs:

| Option | Default | Behavior |
|---|---|---|
| `main_thread_core_type` | `"logical"` | `"physical"` counts physical cores and reserves all their available SMT siblings. |
| `pin_main_thread` | `false` | Pins the initializing thread to the reserved main CPUs; `engine.destroy()` restores its original affinity. |
| `worker_core_type` | `"logical"` | `"physical"` uses one logical CPU per remaining physical core. |
| `bucketwise_cores_per_worker` | `8` | Maximum logical CPUs per bucketwise worker; the last mask within a NUMA node may be smaller. |
| `bucketwise_worker_affinity` | `"task"` | `"task"` assigns CPU masks round-robin; `"thread"` fixes each pool thread to a mask at startup, reducing migrations and preserving its OpenMP team placement. |
| `state_update_backward_cores` | `null` | Caps the number of worker CPU IDs used by the state commit during backward; `null` uses all worker CPUs. `state_update_cores` still controls its forward mask. |

For example, this reserves one physical main core, uses its SMT siblings for the main thread,
and keeps optimizer pool threads on fixed CPU masks:

```json
"reflow": {
    "main_thread_cores": 1,
    "main_thread_core_type": "physical",
    "pin_main_thread": true,
    "worker_core_type": "logical",
    "bucketwise_cores_per_worker": 4,
    "bucketwise_worker_affinity": "thread",
    "state_update_cores": 4
}
```

Physical placement requires readable Linux CPU topology. Automatic rank assignment keeps SMT
siblings together; an existing taskset/launcher affinity remains authoritative. Main reservations
are capped to leave a worker core, and physical main reservation raises an error if a rank has fewer
than two physical cores. Counts for worker masks and state commits remain logical CPU counts, even
when main reservations count physical cores. Tune these values for the workload and CPU topology.

Reflow applies each worker's CPU mask to its retained OpenMP helpers, changing their affinity only
when their current mask differs. Its uniform SIMD loops use contiguous `schedule(static)` work
sharing, keeping the parallel-region join without an extra loop barrier. Gradient accumulation
also caps its team size to the worker mask and the caller's OpenMP thread limit. OpenMP waiting behavior
can be tuned before launching the process with `OMP_WAIT_POLICY=ACTIVE` or `PASSIVE`; Reflow does
not change the application's environment. `OMP_SCHEDULE` affects only `schedule(runtime)` loops,
so it does not override Reflow's static loops. See the [OpenMP scheduling specification](https://openmp.org/spec-html/5.0/openmpse49.html)
and [waiting policy](https://openmp.org/spec-html/5.0/openmpse55.html).

---

## Parameter offload (`offload_param`)

Reflow composes with ZeRO-3 parameter offload (`"offload_param": {"device": "cpu"}`) — the two are
orthogonal and no Reflow code change is needed. Reflow offloads the *optimizer*; `offload_param`
additionally keeps the *parameter* partitions on CPU, fetched per layer by the ZeRO-3 coordinator.
With it on, `fp16_partitioned_groups_flat` lives on CPU, so Reflow's post-step weight writeback (a
CPU→CPU copy into that very buffer) lands exactly where the coordinator H2D-gathers from.

Measured against standard ZeRO-3 + `offload_param` (the baseline) on OPT-30B, steady-state training
GPU memory **matches the baseline** and drops by the per-rank parameter shard — ~7 GB/rank on 8 GPUs,
~27 GB on 2 GPUs, and ~55 GB on 1 GPU (where the whole model would otherwise stay resident).

The saving is in the steady-state training footprint; the one-time full-parameter materialization
spike at init is unchanged (orthogonal to Reflow — use `zero.Init()` to avoid that).

---

## Gradient buffers (double by default; single is not fully implemented)

Each subgroup keeps a **double** half-precision grad buffer on the CPU: an active slot the next
backward writes into, and a pending slot the previous step's async state commit is still reading.
`step()` flips the two, so the background commit never sees gradients from a later iteration. This is
the default and the only fully supported path.

`REFLOW_SINGLE_GRAD_BUFFER=1` enables a **single** grad buffer prototype that is not fully implemented
yet. With `offload_param: cpu`, one slot only duplicates the CPU parameter source, so the prototype
writes the updated BF16 params straight into that source and saves ~2 bytes/param. It has been
checked only with `offload_param: cpu` and `gradient_accumulation_steps: 1`; any other run (gradient
accumulation, params on GPU, NVMe offload) logs a warning and falls back to the double buffer.

---

## Performance characteristics

- **The optimizer is hidden in the backward.** On a saturated multi-GPU run the DeepSpeed `step`
  timer is a few ms — the per-rank optimizer shard drains within the backward window. The backward
  wall-clock is also shorter than plain ZeRO-Offload because the grad D2H offload overlaps it.
- **The forward parameter all-gather is the inherent ZeRO-3 cost**, not a Reflow cost. The forward
  blocks gathering the partitioned parameters across GPUs (interconnect-bound); a plain ZeRO-Offload
  run, and even a no-offload ZeRO-3 run, show the same forward fetch wait. Reflow does not add to it
  and cannot remove it — it is the ZeRO-3 structural floor (and the first forward layers' weights are
  finalized last, since backward runs in reverse, so there is no boundary window to prime them into).
- **Single-GPU is the stress case.** With one GPU the optimizer is the *full* (unpartitioned) model
  and must hide inside a single backward, so the bucketwise CPU-Adam workers must run truly in
  parallel across the NUMA-local cores (see the mutex note under Correctness).
- **The state commit narrows its cores while it overlaps the forward.** The previous step's
  optimizer-state commit overlaps the forward, which is launch-bound on the main thread. Every busy
  core lowers the turbo frequency the CPU allows, so a commit spread over all worker cores slowed the
  forward (OPT-350m: ~25%, main core ~3.6 GHz vs ~4.2 GHz). The commit therefore runs in slices on
  `state_update_cores` cores (default 2) until the next backward starts, then on every worker core,
  so a large model's commit still finishes before the next `step()` waits on it.
- **Gradient clipping re-runs the update inside `step()`.** The bucketwise workers apply the update
  during backward, before the global grad norm is known. When a step needs clipping
  (`gradient_clipping > 0` and the grad norm exceeds it), the in-flight work is cancelled and every
  range is recomputed with the clipped scale in `step()`, so that step pays the full optimizer cost.
  DeepSpeed defaults `gradient_clipping` to `1.0`; set `"gradient_clipping": 0.0` when clipping is not
  wanted. On OPT-350m with one RTX 5080, `step` took ~58 ms unclipped vs ~139 ms with every step clipped.

Illustrative (opt-30B, B200): 8-GPU step ~5 ms with the optimizer hidden; 1-GPU optimizer-step tail
fell from ~1700 ms to ~120 ms once the bucketwise workers ran concurrently.

---

## Correctness

- **Non-finite gradients**: a step with non-finite gradients is detected with an on-GPU `isfinite` scan over
  the offloaded grads and discarded exactly like the standard optimizer — the FP32 master is
  untouched and the GPU weights are regenerated from it.
- **Async grad-copy aliveness**: the D2H grad-copy source is recorded against `copy_grad_stream`
  (`record_stream`) so the caching allocator cannot recycle it before the async copy runs (this was a
  NaN source on large models).
- **Shared-scalar guard**: concurrent bucketwise workers share one `param_opt` per `opt_id`.
  Each worker updates and copies its step and group hyperparameters under a short mutex, then runs
  the SIMD kernel against that private snapshot. Disjoint parameter buffers still run concurrently,
  even when optimizer groups use different learning rates or weight decay.
- **overlap_comm ordering**: with `overlap_comm: true` the next forward's all-gather runs on a
  separate stream, so the all-gather stream is made to wait on the bucketwise H2D weight-copy events
  (`wait_for_external_allgather_dependencies`), ordering the param reads after the H2D writes.
- **Checkpoints see the committed state**: a step's FP32 master and optimizer state are committed in
  the background after `step()` returns, so `state_dict` / `load_state_dict` and the `safe_get_*` /
  `safe_set_*` hp-param APIs wait for that commit first. Only these entry points wait; the training
  step is unchanged.

---

## Diagnostics / debug environment variables

| Env var | Effect |
|---|---|
| `REFLOW_SINGLE_GRAD_BUFFER=1` | Use the single grad buffer prototype (not fully implemented; see "Gradient buffers"). Unsupported runs fall back to the double buffer with a warning. Default off (double buffer). |
| `DS_REFLOW_SYNC_H2D_BEFORE_FORWARD=0` | Skip the host sync on the bucketwise H2D weight copies in `step()`. The sync keeps the next forward's parameter all-gather from reading a partition before its write-back lands, so leave it on except for measurement. Default on. |

---

## Design history (by theme)

The implementation was built and hardened across many commits; the major themes:

- **Bucketwise overlap** — run the CPU-Adam per gradient bucket during backward; decouple the
  per-shard size from `reduce_bucket_size` (a ~50M-element cap) so subgroups flush to the worker
  mid-backward instead of waiting for the whole reduce bucket.
- **Half-precision transfer & async state** — send grads to CPU in FP16/BF16 (promote in-kernel, so
  there is no CPU-side FP32 grad buffer); the foreground only generates the new FP16/BF16 params
  while the FP32-master + optimizer-state update runs on a background worker after the
  clipping/overflow checks.
- **NUMA affinity** — bind the main thread and per-NUMA-node worker groups to GPU-local cores; pin
  the main process to the full NUMA-local slice (not only the reserved main cores) so it does not
  throttle the launch-bound forward.
- **Stream scheduling** — least-priority H2D stream (so weight copies never preempt compute); move
  the top-of-step device sync after the CPU step setup; allow more in-flight reduce events; order
  overlap_comm all-gathers after the H2D weight writes.
- **Concurrency** — narrow the param CPU-Adam mutex to the scalar update so the bucketwise workers
  run concurrently across the rank's cores (the key to hiding the optimizer when the per-rank shard
  is large, e.g. single-GPU).
- **Correctness & robustness** — FP16-overflow detect-and-discard; async grad-copy aliveness;
  auto-remap a client `DeepSpeedCPUAdam` to `ReflowCPUAdam`; isolated C++ `opt_id` registry.

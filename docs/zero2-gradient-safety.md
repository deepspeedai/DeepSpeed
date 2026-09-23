# ZeRO-2 Gradient Safety

These experimental changes originated on v0.18.4
(`b35d9eb01bc04e774cc05dc43713f2a41423da5c`). The
`jeffra/913-fixes-master-sync` branch merges master at `4ea3b47b`, retaining
upstream accumulation and gradient-stream fixes alongside the offload protections.

## Automatic CPU Offload Protections

Gradient storage and stream protections are always enabled for ZeRO-2 with
CPU optimizer offload, without ZenFlow, pipeline parallelism, or configured
DeepCompile. CPU offload enforces contiguity even with
`contiguous_gradients=false`; `overlap_comm` may be either value.
Non-offloaded paths and ZeRO-0/1/3 do not enable these protections.

The `copy_oversized_gradients` and `track_gradient_streams` configuration
fields and optimizer constructor arguments have been removed. Neither
protection can be independently enabled or disabled. Remove both keys from
existing configurations, including keys set to `null` or `false`; they are
now rejected as unknown fields.
`check_offload_gradients` and `accumulate_offload_gradients` still default
to false and retain their existing restrictions.

The optimizer derives its policy from the offload, stage, ZenFlow, model,
and compilation context. The engine logs whether protections are enabled on
rank zero. Integrations constructing ZeRO-1/2 optimizers directly must pass
`pipeline_parallel` and `deepcompile` when applicable; both default to false.

Configure DeepCompile before engine initialization. Resolved optimizer
settings remain fixed across compile success, failure, and eager fallback,
so pending copies and gradients cannot outlive a change in tracking policy.
An engine configured for DeepCompile keeps automatic protections disabled
even if compilation falls back; recreate it without DeepCompile to enable
automatic protections. Late DeepCompile activation is rejected when the
optimizer already has protections enabled. Ordinary `torch.compile` is not
excluded.

## Ownership and Ordering

On supported CPU-offload paths, the optimizer clones an oversized gradient on its producer
stream before storing it in the pending bucket. Both reduction and CPU
offload use the replacement storage. The bucket threshold, gradient dtype,
FP32 communication configuration, and partition/reduction grouping remain
unchanged. The extra copy costs roughly 170 MiB for an 89,128,960-element
BF16 weight; larger weights require more memory. No model-specific parameter
name or shape is hardcoded.

Stream tracking records the latest ready event for every stream
that contributes to a bucket, including the oversized path. Reduction waits
on those events even without communication overlap. It separately records
storage usage on consumer streams and consumer completion before bucket
reuse. Every producer waits before writing to a reused buffer, and successive
CPU-gradient writes from different streams are ordered before overwriting
the destination. Neither a Python reference nor `record_stream` alone
establishes producer/consumer ordering.

Tracked offload copies must finish before the CPU optimizer consumes their
results. CPU accumulator reloads wait on the host because pageable H2D
sources may be staged before a GPU-side wait executes. These protections introduce
synchronization and can alter timing; they do not prove the original NaN's
cause merely by making a replay finite.

## Optimizer Safety

`check_offload_gradients` waits for pending offload writes, then tests the
actual master-gradient CPU buffers with scalar min/max reductions. It does
not allocate a whole extra FP32 tensor and several elementwise masks.
Missing gradients are treated as invalid optimizer input.

All ranks agree on overflow before any optimizer update. If gradients pass,
group norms are calculated and validated before the update as well, so a
negative invalid-norm sentinel cannot become a positive outer norm. Norm
failure is also agreed across ranks. This option forces existing overflow
checking on, including for BF16. Rejected steps use the existing skip path;
they are not counted as successful training.

## Accumulation

`accumulate_offload_gradients` uses a per-parameter seen set for the current
optimizer window rather than inferring prior contributions from GAS or
the global microstep count. The first contribution is always stored,
including a nonzero first backward already marked as a boundary. Later
contributions restore/add/copy back each time.

The live master-gradient fragment and its norm are refreshed after every
backward, including non-boundary backwards. A parameter used earlier but
absent from the last backward retains its contribution. Successful/skipped
steps reset the seen set and zero the master-gradient buffers, preventing
unused parameters from inheriting a prior window's gradients.

This is separate from the NaN candidate changes. Master already accumulates
across non-boundary backwards by default. The opt-in option additionally
tracks each parameter's first contribution and refreshes master gradients
on every backward; keep it disabled in the initial storage/ordering comparison.

The accumulator retains DeepSpeed's existing low-precision accumulation
dtype and does not alter engine loss normalization. Optimizer checkpoint
loading discards the pending window. Gradient accumulation is not included
in checkpoint state. External model-only weight replacement must explicitly
discard pending gradients with `optimizer.reset_cpu_buffers()`.
Master's unmanaged accumulation API is retained. Its deferred CPU accumulator
reloads also wait for tracked offload copies.

## QA Comparisons

Keep the model, frames, historical operation sequence, offload settings and
BF16 overflow guard constant. In the DSS recipe, configuration goes under
`training_config.ds_config.zero_optimization`.
Storage and ordering protections are now inseparable on supported offload
paths. Comparisons against the unprotected behavior require a baseline build,
not a configuration opt-out. Compare the 50M bucket with the prior 100M
workaround while leaving the two remaining opt-in options disabled.

Then test `check_offload_gradients` separately, followed by accumulation
equivalence tests and a combined configuration. Compare complete wire
inputs as well as returned outputs; generated samples and post-update old
logprobs can differ between runs.

Example combined configuration for later validation:

```json
{
  "bf16": {
    "enabled": true,
    "check_grad_overflow": true,
    "bf16_master_weights_and_grads": false,
    "bf16_optimizer_states": false
  },
  "zero_optimization": {
    "stage": 2,
    "reduce_bucket_size": 50000000,
    "overlap_comm": false,
    "offload_optimizer": {"device": "cpu", "pin_memory": false},
    "check_offload_gradients": true,
    "accumulate_offload_gradients": true
  }
}
```

This is a config fragment, not a replacement for the existing training recipe.

## Tests and Build Handoff

```bash
python -m pytest \
  tests/unit/v1/zero/test_zero2_gradient_safety.py \
  tests/unit/runtime/zero/test_zero_config.py \
  tests/unit/runtime/test_ds_config_model.py \
  tests/unit/compile/test_zero3_grad_dtype.py
```

CPU-offload-only policy verification on 2026-09-22: **150 passed, 2 skipped**
using the command above with `DS_ACCELERATOR=cpu`. Hardware: Apple M2 Max
(12 CPU cores, 64 GiB RAM), macOS 26.6.2 arm64, Python 3.11.6, PyTorch 2.9.1.
Real two-rank Gloo forward/backward/SGD updates matched an independent
full-batch reference for offloaded and non-offloaded ZeRO-2, overlap on/off,
forced contiguity, gradient accumulation, and ZeRO-1 with/without offload.
Direct optimizer construction also covered pipeline and DeepCompile
exclusions. The two CUDA stream-ordering tests were skipped; no GPU/NCCL
validation was performed for this change.

Historical compatibility-default verification on 2026-09-18: **182 passed, 4 skipped**
with Python 3.11.6, PyTorch 2.9.1, and `DS_ACCELERATOR=cpu` on macOS arm64.
The two-rank Gloo integration ran real forward/backward/SGD updates against
an independent full-batch reference, including oversized and bucketed
gradients, accumulation, overlap toggles, CPU optimizer offload, explicit
opt-outs, and ZeRO-1. It also constructed ZeRO optimizers directly.
The same integration supports two CUDA devices, but that path was not run.
DeepCompile lifecycle policy tests isolate compiler collaborators; they do
not substitute for native DeepCompile execution.

In that historical run, the excluded noncontiguous ZeRO-2 path with GAS=2 showed a finite gradient
and update mismatch against the CPU reference on both the unmodified branch
and the default-policy patch, with identical error magnitudes. Its integration
check compared automatic-versus-explicit-disabled behavior, not numerical
correctness. Those opt-out cases have been removed with the flags.
That pre-existing discrepancy needs separate investigation.

GPU/NCCL execution, native DeepCompile/ZenFlow/pipeline integration, accelerator
coverage, and GPU memory/throughput measurements remain release gates for
this policy. The earlier passing image replay with explicit flags does not
validate a newly built image with automatic CPU-offload protections.

Original v0.18.4 verification: **100 passed, 4 skipped**, using Python 3.12.12 and
PyTorch 2.9.1+cu130 on a CPU-only host. `git diff --check` also passes.

Tests use the actual modified ZeRO methods with small tensor fixtures.
They cover engine configuration forwarding, routing/storage independence,
producer waits and reuse with stream fakes, empty/oversized bucket events,
cross-stream CPU writes, CPU-copy completion, accumulation/boundary patterns,
partial ownership, late first use, reset/restore, and real Adam state preservation.
A two-process Gloo test exercises real overflow consensus for both corrupt
gradients and invalid norms through DeepSpeed's communication wrapper, with
the optional shared-memory communication extension disabled.

Two CUDA cases exercise an oversized gradient produced on a delayed stream
and consumed on a different stream, with overlap enabled and disabled.
They skip on CPU-only hosts. CUDA/NCCL, CPUAdam, full-engine multi-GPU
equivalence, memory/performance, and Sean's replay still require image tests.
The initial changes do not add per-parameter numerical tracing inside the
reducer; that remains a further diagnostic if the ablations are inconclusive.

Changed production files are `deepspeed/runtime/zero/stage_1_and_2.py`,
`deepspeed/runtime/zero/config.py`, and `deepspeed/runtime/engine.py`.
The image must install this patched DeepSpeed package: changing a monorepo
pointer alone does not replace the published `deepspeed==0.18.4` wheel.
Verify that `DeepSpeedZeroConfig.model_fields` contains the two remaining
offload options and no longer contains the removed copy/tracking keys inside
the built image before submitting tests.

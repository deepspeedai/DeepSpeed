# ZeRO-2 Gradient Safety: Experimental Changes

These experimental changes originated on v0.18.4
(`b35d9eb01bc04e774cc05dc43713f2a41423da5c`). The
`jeffra/913-fixes-master-sync` branch merges master at `4ea3b47b`, retaining
upstream accumulation and gradient-stream fixes alongside the opt-in options.
The all-flags-disabled control now uses master's behavior, not v0.18.4's.

## Ownership and Ordering

`copy_oversized_gradients` clones an oversized gradient on its producer
stream before storing it in the pending bucket. Both reduction and CPU
offload use the replacement storage. The bucket threshold, gradient dtype,
FP32 communication configuration, and partition/reduction grouping remain
unchanged. The extra copy costs roughly 170 MiB for an 89,128,960-element
BF16 weight; larger weights require more memory. No model-specific parameter
name or shape is hardcoded.

`track_gradient_streams` records the latest ready event for every stream
that contributes to a bucket, including the oversized path. Reduction waits
on those events even without communication overlap. It separately records
storage usage on consumer streams and consumer completion before bucket
reuse. Every producer waits before writing to a reused buffer, and successive
CPU-gradient writes from different streams are ordered before overwriting
the destination. Neither a Python reference nor `record_stream` alone
establishes producer/consumer ordering.

Tracked offload copies must finish before the CPU optimizer consumes their
results. CPU accumulator reloads wait on the host because pageable H2D
sources may be staged before a GPU-side wait executes. The flags introduce
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

## Initial QA Matrix

Keep the model, frames, historical operation sequence, offload settings and
BF16 overflow guard constant. In the DSS recipe, these flags go under
`training_config.ds_config.zero_optimization`.

| Arm | Bucket | Copy | Track Streams | Check Offload | Accumulate Offload |
| --- | --- | --- | --- | --- | --- |
| Control | 50M | false | false | false | false |
| Owned storage | 50M | true | false | false | false |
| Ordering | 50M | false | true | false | false |
| Both | 50M | true | true | false | false |
| Prior workaround | 100M | false | false | false | false |

Then test `check_offload_gradients` separately, followed by accumulation
equivalence tests and a combined configuration. Compare complete wire
inputs as well as returned outputs; generated samples and post-update old
logprobs can differ between runs.

Example combined configuration for later validation, not the initial
isolation matrix:

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
    "copy_oversized_gradients": true,
    "track_gradient_streams": true,
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
  tests/unit/runtime/test_ds_config_model.py
```

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

Four CUDA cases exercise an oversized gradient produced on a delayed stream
and consumed on a different stream, with copy and overlap toggles.
They skip on CPU-only hosts. CUDA/NCCL, CPUAdam, full-engine multi-GPU
equivalence, memory/performance, and Sean's replay still require image tests.
The initial changes do not add per-parameter numerical tracing inside the
reducer; that remains a further diagnostic if the ablations are inconclusive.

Changed production files are `deepspeed/runtime/zero/stage_1_and_2.py`,
`deepspeed/runtime/zero/config.py`, and `deepspeed/runtime/engine.py`.
The image must install this patched DeepSpeed package: changing a monorepo
pointer alone does not replace the published `deepspeed==0.18.4` wheel.
Verify the four fields in `DeepSpeedZeroConfig.model_fields` inside the
built image before submitting tests.

# DeepSpeed Benchmarks

If you are looking for DeepSpeed benchmarks, please see the following resources:

1. [Communication Benchmarking Suite](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/communication)
2. [Inference Benchmarks](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/inference)

## Continuous rollout CUDA Graph first-call costs

Run from the repository root on one CUDA GPU with PyTorch, Transformers, and
DeepSpeed installed:

```bash
python -m benchmarks.rollout.bench_cuda_graph_first_call \
    --requests 8 --batch-size 4 --prompt-len 32 --max-new-tokens 32 \
    --iterations 10 --output rollout-first-call.json
```

This offline benchmark uses a randomly initialized two-layer Llama, native
Transformers attention, greedy decoding, and more requests than physical slots
to exercise continuous refills. EOS is disabled for a fixed token workload.
It compares eager and CUDA Graph rollouts sharing the same model before and
after a real next-token-loss backward pass and in-place SGD update.

The JSON report includes synchronized wall-clock latency samples, median/min/max,
generated-token throughput, model/workload configuration, library versions, and
GPU name/memory. Each mode reports `first_call`, `steady_state`,
`first_call_after_update`, and `steady_state_after_update`. The training update
is timed separately; output tokens, masks, and response offsets must match an
independent native Hugging Face `generate()` reference for every measured call,
both before and after the update.

There is **no rollout warmup before either first-call measurement**. A non-cached
model forward and native reference generation initialize CUDA/model libraries
before measurement; eager is measured before graph in each phase. First graph
calls include cache allocation, capture warmup, graph capture, and generation.
Post-update calls include any
invalidation/recapture required by the rollout implementation. Stable calls reuse
the same rollout, shapes, and weights. Internal profiling is disabled to avoid
per-stage synchronization overhead.

`graph_first_call_premium_ms` and `graph_post_update_premium_ms` subtract the
corresponding graph steady-state median from the first-call latency. These are
observed end-to-end premiums, **not isolated capture timings**; they can be
negative under measurement noise. First calls are single samples, not medians
across independent captures. Repeat the command in fresh processes for
first-call distributions. This small synthetic model measures rollout overhead,
not pretrained-model quality or production-model speedup.

Example measured with the command above on an NVIDIA RTX A4500 (20 GiB),
PyTorch 2.9.1+cu128 / CUDA 12.8, Transformers 5.20.1, FP16:

| Phase | Eager latency (ms) | CUDA Graph latency (ms) |
|---|---:|---:|
| First call | 240.182 | 61.933 |
| Steady state (10-call median) | 228.014 | 45.424 |
| First call after SGD update | 226.304 | 58.800 |
| Post-update steady state (10-call median) | 234.919 | 45.281 |

The observed graph first-call premium was 16.509 ms; the post-update premium was
13.519 ms. The separately timed training update took 55.849 ms. All measured
outputs matched native generation before and after the update. These are one
run's illustrative results, not performance thresholds.

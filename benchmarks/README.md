# DeepSpeed Benchmarks

If you are looking for DeepSpeed benchmarks, please see the following resources:

1. [Communication Benchmarking Suite](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/communication)
2. [Inference Benchmarks](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/benchmarks/inference)

## OPSD HybridEngine rollout profiling

The OPSD rollout benchmark drives the stage-level instrumentation in
``HybridEngineRollout`` with exact synthetic prompt lengths. The initial
benchmark is single-accelerator and ZeRO stage 0. From the repository root:

```bash
torchrun --nproc_per_node=1 benchmarks/opsd/benchmark_hybrid_engine_rollout.py \
    --model facebook/opt-6.7b \
    --batch-sizes 1 \
    --samples-per-prompt 1 4 \
    --prompt-lengths 128 512 \
    --response-lengths 32 128 \
    --warmup 5 \
    --iterations 20 \
    --output opsd_rollout_profile.json
```

Run the same command with ``--release-inference-cache`` and a different output
path to measure the cost of releasing and reacquiring the inference workspace.
The output JSON contains the raw profiles and mean, p50, and p95 values for each
stage and workload.

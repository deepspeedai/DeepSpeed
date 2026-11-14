# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Benchmark the complete non-ZeRO mixed-precision FusedAdam wrapper step.

The baseline is the existing FP32-gradient path, selected by disabling the
optional mixed-precision kernel on one otherwise identical FusedAdam instance.
"""

import argparse
import gc
import math
from pathlib import Path
from statistics import median

import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def initialize_distributed():
    if dist.is_initialized():
        return
    get_accelerator().set_device(0)
    deepspeed.init_distributed(dist_backend=get_accelerator().communication_backend_name(),
                               auto_mpi_discovery=False,
                               init_method="tcp://127.0.0.1:29500",
                               rank=0,
                               world_size=1)


def create_optimizer(numel, dtype, mode, seed):
    torch.manual_seed(seed)
    parameter = torch.nn.Parameter(torch.randn(numel, device=get_accelerator().current_device_name(), dtype=dtype))
    optimizer = FusedAdam([parameter], lr=2e-3, weight_decay=0.01, adam_w_mode=mode == "adamw")
    optimizer = FP16_Optimizer(optimizer,
                               low_precision_dtype=dtype,
                               static_loss_scale=128.0,
                               dynamic_loss_scale=False,
                               clip_grad=0.5,
                               verbose=False)
    torch.manual_seed(seed + 1)
    gradient = torch.randn_like(parameter)
    return parameter, optimizer, gradient


def run_step(parameter, optimizer, gradient):
    parameter.grad = gradient
    optimizer.step()


def assert_matches(numel, dtype, mode):
    candidate_param, candidate, candidate_grad = create_optimizer(numel, dtype, mode, seed=1234)
    baseline_param, baseline, baseline_grad = create_optimizer(numel, dtype, mode, seed=1234)
    baseline.optimizer.multi_tensor_adam_mixed_precision = None

    run_step(candidate_param, candidate, candidate_grad)
    run_step(baseline_param, baseline, baseline_grad)
    torch.cuda.synchronize()  #ignore-cuda

    torch.testing.assert_close(candidate_param, baseline_param, rtol=0, atol=torch.finfo(dtype).eps)
    candidate_master = candidate.fp32_groups_flat[0]
    baseline_master = baseline.fp32_groups_flat[0]
    torch.testing.assert_close(candidate_master, baseline_master, rtol=1e-6, atol=1e-7)
    candidate_state = candidate.optimizer.state[candidate_master]
    baseline_state = baseline.optimizer.state[baseline_master]
    assert candidate_state["step"] == baseline_state["step"]
    torch.testing.assert_close(candidate_state["exp_avg"], baseline_state["exp_avg"], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(candidate_state["exp_avg_sq"], baseline_state["exp_avg_sq"], rtol=1e-6, atol=1e-7)
    assert candidate._global_grad_norm == baseline._global_grad_norm

    del candidate_param, candidate, candidate_grad, baseline_param, baseline, baseline_grad
    gc.collect()
    torch.cuda.empty_cache()  #ignore-cuda


def percentile(samples, fraction):
    return sorted(samples)[max(0, math.ceil(fraction * len(samples)) - 1)]


def measure(numel, dtype, mode, baseline, warmup, iters):
    parameter, optimizer, gradient = create_optimizer(numel, dtype, mode, seed=4321)
    if baseline:
        optimizer.optimizer.multi_tensor_adam_mixed_precision = None

    for _ in range(warmup):
        run_step(parameter, optimizer, gradient)
    torch.cuda.synchronize()  #ignore-cuda

    persistent_bytes = torch.cuda.memory_allocated()  #ignore-cuda
    torch.cuda.reset_peak_memory_stats()  #ignore-cuda
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))  #ignore-cuda
        for _ in range(iters)
    ]
    for start, end in events:
        parameter.grad = gradient
        start.record()
        optimizer.step()
        end.record()
    torch.cuda.synchronize()  #ignore-cuda

    samples = [start.elapsed_time(end) for start, end in events]
    peak_bytes = torch.cuda.max_memory_allocated()  #ignore-cuda
    result = {
        "median_ms": median(samples),
        "p95_ms": percentile(samples, 0.95),
        "peak_bytes": peak_bytes,
        "temporary_bytes": peak_bytes - persistent_bytes,
    }

    del parameter, optimizer, gradient, events
    gc.collect()
    torch.cuda.empty_cache()  #ignore-cuda
    return result


def profile_step(path, numel, dtype, mode, baseline):
    parameter, optimizer, gradient = create_optimizer(numel, dtype, mode, seed=9876)
    arm = "baseline" if baseline else "fused"
    if baseline:
        optimizer.optimizer.multi_tensor_adam_mixed_precision = None
    run_step(parameter, optimizer, gradient)
    torch.cuda.synchronize()  #ignore-cuda

    output = Path(f"{path}.{numel}.{str(dtype).split('.')[-1]}.{mode}.{arm}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True,
                                profile_memory=True) as profiler:
        run_step(parameter, optimizer, gradient)
    torch.cuda.synchronize()  #ignore-cuda
    profiler.export_chrome_trace(str(output))
    print(f"PROFILE path={output}")

    del parameter, optimizer, gradient, profiler
    gc.collect()
    torch.cuda.empty_cache()  #ignore-cuda


def print_result(numel, dtype_name, mode, baseline, fused):
    speedup = (baseline["median_ms"] / fused["median_ms"] - 1.0) * 100.0
    peak_saved = (baseline["peak_bytes"] - fused["peak_bytes"]) / 2**20
    temporary_saved = (baseline["temporary_bytes"] - fused["temporary_bytes"]) / 2**20
    print(f"RESULT numel={numel} dtype={dtype_name} mode={mode} "
          f"baseline_median_ms={baseline['median_ms']:.6f} fused_median_ms={fused['median_ms']:.6f} "
          f"speedup_pct={speedup:.2f} baseline_p95_ms={baseline['p95_ms']:.6f} fused_p95_ms={fused['p95_ms']:.6f} "
          f"baseline_peak_mib={baseline['peak_bytes'] / 2**20:.2f} fused_peak_mib={fused['peak_bytes'] / 2**20:.2f} "
          f"peak_saved_mib={peak_saved:.2f} baseline_temporary_mib={baseline['temporary_bytes'] / 2**20:.2f} "
          f"fused_temporary_mib={fused['temporary_bytes'] / 2**20:.2f} temporary_saved_mib={temporary_saved:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numel", type=int, nargs="+", required=True)
    parser.add_argument("--dtype", choices=DTYPES, nargs="+", required=True)
    parser.add_argument("--mode", choices=["adam", "adamw"], nargs="+", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if min(args.numel) <= 0 or args.warmup < 0 or args.iters <= 0:
        parser.error("numel and iters must be positive, and warmup must be non-negative")

    initialize_distributed()
    for numel in args.numel:
        for dtype_name in args.dtype:
            dtype = DTYPES[dtype_name]
            if dtype not in get_accelerator().supported_dtypes():
                print(f"SKIP numel={numel} dtype={dtype_name} reason=unsupported")
                continue
            for mode in args.mode:
                if args.verify:
                    assert_matches(numel, dtype, mode)
                baseline = measure(numel, dtype, mode, baseline=True, warmup=args.warmup, iters=args.iters)
                fused = measure(numel, dtype, mode, baseline=False, warmup=args.warmup, iters=args.iters)
                print_result(numel, dtype_name, mode, baseline, fused)
                if args.profile:
                    profile_step(args.profile, numel, dtype, mode, baseline=True)
                    profile_step(args.profile, numel, dtype, mode, baseline=False)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required"  #ignore-cuda
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

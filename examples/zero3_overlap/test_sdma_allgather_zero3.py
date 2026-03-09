#!/usr/bin/env python3
"""
Unit test for SDMA allgather in the ZeRO-3 code path.

Simulates exactly how ZeRO-3's _all_gather_dtype calls _dist_allgather_fn:
  1. Creates a flat_tensor and partitions (same as partition_parameters.py)
  2. Each rank fills its partition with known data
  3. Calls _dist_allgather_fn on a dedicated allgather stream (same as coordinator)
  4. Rebuilds partitions from transit buffer (zero-copy path)
  5. handle.wait() + stream sync (same as fetch_sub_module)
  6. Verifies correctness and measures algorithm bandwidth

Usage:
    cd /root/wuyl/DeepSpeed/examples/zero3_overlap
    deepspeed --num_gpus 8 test_sdma_allgather_zero3.py
    deepspeed --num_gpus 8 test_sdma_allgather_zero3.py --partition_sz 4194304 --iterations 50
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.distributed as torch_dist
import deepspeed
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator

import deepspeed.runtime.zero.partition_parameters as pp


def verify_allgather(partitions, world_size, partition_sz, rank, dtype):
    """Verify that each rank's partition contains the expected fill pattern."""
    passed = True
    for r in range(world_size):
        chunk = partitions[r].narrow(0, 0, partition_sz).float().cpu()
        expected_val = float(r + 1)
        if not torch.allclose(chunk, torch.full_like(chunk, expected_val)):
            unique_vals = chunk.unique()
            print(f"  [rank {rank}] FAIL: partition[{r}] expected all {expected_val}, "
                  f"got unique values: {unique_vals[:10]}")
            passed = False
    return passed


def run_single_allgather(rank, world_size, dtype, partition_sz, ag_stream):
    """Execute one allgather call following the exact ZeRO-3 _all_gather_dtype path."""
    device = get_accelerator().current_device_name()

    flat_tensor = torch.empty(
        partition_sz * world_size, dtype=dtype, device=device, requires_grad=False
    )
    partitions = []
    for i in range(world_size):
        partitions.append(flat_tensor.narrow(0, partition_sz * i, partition_sz))

    partitions[rank].fill_(float(rank + 1))

    with get_accelerator().stream(ag_stream):
        handle = pp._dist_allgather_fn(partitions[rank], flat_tensor)

    if pp._sdma_allgather_enabled() and not pp._sdma_allgather_handle._copy:
        transit_buf_u32 = pp._sdma_allgather_handle.get_output_transit_buffer()
        transit_buf = transit_buf_u32.view(dtype)
        partitions = []
        for i in range(world_size):
            partitions.append(transit_buf.narrow(0, partition_sz * i, partition_sz))

    with get_accelerator().stream(ag_stream):
        handle.wait()
    get_accelerator().current_stream().wait_stream(ag_stream)

    return partitions


def run_correctness_test(rank, world_size, dtype, partition_sz, ag_stream):
    """Run a single correctness test."""
    partitions = run_single_allgather(rank, world_size, dtype, partition_sz, ag_stream)
    return verify_allgather(partitions, world_size, partition_sz, rank, dtype)


def run_bandwidth_test(rank, world_size, dtype, partition_sz, ag_stream,
                       iterations, warmup):
    """Measure allgather bandwidth following the ZeRO-3 overlap pattern."""
    device = get_accelerator().current_device_name()
    elem_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = partition_sz * elem_size * world_size

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    times_ms = []

    for i in range(warmup + iterations):
        flat_tensor = torch.empty(
            partition_sz * world_size, dtype=dtype, device=device, requires_grad=False
        )
        partitions = []
        for r in range(world_size):
            partitions.append(flat_tensor.narrow(0, partition_sz * r, partition_sz))
        partitions[rank].fill_(float(rank + 1))

        dist.barrier()

        ev_start.record(ag_stream)
        with get_accelerator().stream(ag_stream):
            handle = pp._dist_allgather_fn(partitions[rank], flat_tensor)
        with get_accelerator().stream(ag_stream):
            handle.wait()
        ev_end.record(ag_stream)

        ag_stream.synchronize()

        elapsed_ms = ev_start.elapsed_time(ev_end)
        if i >= warmup:
            times_ms.append(elapsed_ms)

    return times_ms, total_bytes


def main():
    parser = argparse.ArgumentParser(description="SDMA allgather unit test (ZeRO-3 style)")
    parser.add_argument("--partition_sz", type=int, default=1024 * 1024,
                        help="Elements per rank per allgather call")
    parser.add_argument("--max_numel", type=int, default=4 * 1024 * 1024,
                        help="Max uint32 elements for SDMA transit buffer")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of measurement iterations")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations")
    parser.add_argument("--local_rank", type=int,
                        default=int(os.environ.get("LOCAL_RANK", 0)))
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    deepspeed.init_distributed(dist_backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    get_accelerator().set_device(args.local_rank)

    if rank == 0:
        print(f"\n{'=' * 65}")
        print(f"  SDMA Allgather Unit Test (ZeRO-3 code path)")
        print(f"  world_size    : {world_size}")
        print(f"  partition_sz  : {args.partition_sz:,} elements")
        print(f"  iterations    : {args.iterations}  (warmup {args.warmup})")
        print(f"{'=' * 65}")

    pp._init_sdma_allgather(max_numel=args.max_numel)

    if rank == 0:
        if pp._sdma_allgather_enabled():
            mode = "zero-copy transit buffer" if not pp._sdma_allgather_handle._copy else "copy-to-user"
            print(f"  backend       : SDMA ({mode})")
        else:
            print(f"  backend       : RCCL (SDMA not available, handle is None)")
        print()

    ag_stream = get_accelerator().Stream()

    test_dtypes = [
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("float32", torch.float32),
    ]

    # ── 1. Correctness ────────────────────────────────────────────────
    if rank == 0:
        print("--- Correctness ---")

    all_correct = True
    for dtype_name, dtype in test_dtypes:
        dist.barrier()
        passed = run_correctness_test(rank, world_size, dtype, args.partition_sz, ag_stream)

        passed_t = torch.tensor([1 if passed else 0], dtype=torch.int32)
        torch_dist.all_reduce(passed_t, op=torch_dist.ReduceOp.MIN)
        ok = passed_t.item() == 1

        if rank == 0:
            elem_bytes = torch.tensor([], dtype=dtype).element_size()
            data_mb = args.partition_sz * elem_bytes * world_size / (1024 ** 2)
            status = "PASSED" if ok else "FAILED"
            print(f"  {dtype_name:10s}  data={data_mb:8.2f} MB  {status}")
        if not ok:
            all_correct = False

    # ── 2. Bandwidth ──────────────────────────────────────────────────
    if rank == 0:
        print(f"\n--- Bandwidth (iterations={args.iterations}, warmup={args.warmup}) ---")
        print(f"  {'dtype':10s}  {'data_MB':>10s}  {'avg_ms':>9s}  {'min_ms':>9s}  {'max_ms':>9s}  {'algo_BW':>12s}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*12}")

    for dtype_name, dtype in test_dtypes:
        dist.barrier()
        times_ms, total_bytes = run_bandwidth_test(
            rank, world_size, dtype, args.partition_sz, ag_stream,
            args.iterations, args.warmup,
        )

        avg_ms = np.mean(times_ms)
        min_ms = np.min(times_ms)
        max_ms = np.max(times_ms)

        avg_t = torch.tensor([avg_ms], dtype=torch.float64)
        min_t = torch.tensor([min_ms], dtype=torch.float64)
        max_t = torch.tensor([max_ms], dtype=torch.float64)
        torch_dist.all_reduce(avg_t, op=torch_dist.ReduceOp.SUM)
        torch_dist.all_reduce(min_t, op=torch_dist.ReduceOp.MIN)
        torch_dist.all_reduce(max_t, op=torch_dist.ReduceOp.MAX)

        if rank == 0:
            g_avg_ms = avg_t.item() / world_size
            g_min_ms = min_t.item()
            g_max_ms = max_t.item()
            data_mb = total_bytes / (1024 ** 2)
            algo_bw_gbs = total_bytes / (g_avg_ms / 1000) / (1024 ** 3)
            print(f"  {dtype_name:10s}  {data_mb:10.2f}  {g_avg_ms:9.3f}  "
                  f"{g_min_ms:9.3f}  {g_max_ms:9.3f}  {algo_bw_gbs:9.2f} GB/s")

    # ── Summary ───────────────────────────────────────────────────────
    dist.barrier()
    if rank == 0:
        print()
        if all_correct:
            print("Result: All correctness tests PASSED")
        else:
            print("Result: Some correctness tests FAILED")
        print(f"{'=' * 65}\n")

    get_accelerator().synchronize()
    dist.barrier()
    if pp._sdma_allgather_enabled():
        import mori.shmem as shmem
        shmem.shmem_finalize()


if __name__ == "__main__":
    main()

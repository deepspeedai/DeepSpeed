#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Unit test for the transparent SDMA allgather path in deepspeed.comm.

After ``deepspeed.init_distributed()`` returns, ``dist.all_gather_into_tensor``
on the WORLD process group is transparently routed through
``mori_cpp.AllGatherIntoTensor`` on AMD MI300 when mori is available, with
RCCL/NCCL as a fallback.  This test exercises that path the same way
ZeRO-3's ``_all_gather_dtype`` does (flat output / per-rank shard input
with ``async_op=True``) and verifies correctness and algorithm bandwidth
for the common dtypes.

Usage:
    cd examples/sdma_allgather
    deepspeed --num_gpus 8 test_sdma_allgather_zero3.py
    deepspeed --num_gpus 8 test_sdma_allgather_zero3.py --partition_sz 4194304 --iterations 50
"""

import argparse
import os

import numpy as np
import torch

import deepspeed
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.comm import mori as _mori


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
    """Execute one allgather call following the ZeRO-3 ``_all_gather_dtype`` path."""
    device = get_accelerator().current_device_name()

    flat_tensor = torch.empty(partition_sz * world_size, dtype=dtype, device=device, requires_grad=False)
    partitions = [flat_tensor.narrow(0, partition_sz * i, partition_sz) for i in range(world_size)]
    partitions[rank].fill_(float(rank + 1))

    with get_accelerator().stream(ag_stream):
        handle = dist.allgather_fn(flat_tensor, partitions[rank], async_op=True)

    with get_accelerator().stream(ag_stream):
        handle.wait()
    get_accelerator().current_stream().wait_stream(ag_stream)

    return partitions


def run_bandwidth_test(rank, world_size, dtype, partition_sz, ag_stream, iterations, warmup):
    """Measure allgather bandwidth following the ZeRO-3 overlap pattern."""
    device = get_accelerator().current_device_name()
    elem_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = partition_sz * elem_size * world_size

    ev_start = get_accelerator().Event(enable_timing=True)
    ev_end = get_accelerator().Event(enable_timing=True)
    times_ms = []

    for i in range(warmup + iterations):
        flat_tensor = torch.empty(partition_sz * world_size, dtype=dtype, device=device, requires_grad=False)
        partitions = [flat_tensor.narrow(0, partition_sz * r, partition_sz) for r in range(world_size)]
        partitions[rank].fill_(float(rank + 1))

        dist.barrier()

        ev_start.record(ag_stream)
        with get_accelerator().stream(ag_stream):
            handle = dist.allgather_fn(flat_tensor, partitions[rank], async_op=True)
        with get_accelerator().stream(ag_stream):
            handle.wait()
        ev_end.record(ag_stream)

        ag_stream.synchronize()

        elapsed_ms = ev_start.elapsed_time(ev_end)
        if i >= warmup:
            times_ms.append(elapsed_ms)

    return times_ms, total_bytes


def main():
    parser = argparse.ArgumentParser(description="Transparent SDMA allgather unit test")
    parser.add_argument("--partition_sz", type=int, default=1024 * 1024, help="Elements per rank per allgather call")
    parser.add_argument("--iterations", type=int, default=20, help="Number of measurement iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    deepspeed.init_distributed(dist_backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    get_accelerator().set_device(args.local_rank)

    if rank == 0:
        backend = "SDMA (mori)" if _mori.is_enabled() else "RCCL/NCCL (mori unavailable or disabled)"
        print(f"\n{'=' * 65}")
        print(f"  Transparent SDMA Allgather Unit Test")
        print(f"  world_size    : {world_size}")
        print(f"  partition_sz  : {args.partition_sz:,} elements")
        print(f"  iterations    : {args.iterations}  (warmup {args.warmup})")
        print(f"  backend       : {backend}")
        print(f"{'=' * 65}\n")

    ag_stream = get_accelerator().Stream()

    test_dtypes = [
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("float32", torch.float32),
    ]

    if rank == 0:
        print("--- Correctness ---")

    all_correct = True
    for dtype_name, dtype in test_dtypes:
        dist.barrier()
        partitions = run_single_allgather(rank, world_size, dtype, args.partition_sz, ag_stream)
        passed = verify_allgather(partitions, world_size, args.partition_sz, rank, dtype)

        passed_t = torch.tensor([1 if passed else 0], dtype=torch.int32)
        dist.all_reduce(passed_t, op=dist.ReduceOp.MIN)
        ok = passed_t.item() == 1

        if rank == 0:
            elem_bytes = torch.tensor([], dtype=dtype).element_size()
            data_mb = args.partition_sz * elem_bytes * world_size / (1024**2)
            status = "PASSED" if ok else "FAILED"
            print(f"  {dtype_name:10s}  data={data_mb:8.2f} MB  {status}")
        if not ok:
            all_correct = False

    if rank == 0:
        print(f"\n--- Bandwidth (iterations={args.iterations}, warmup={args.warmup}) ---")
        print(f"  {'dtype':10s}  {'data_MB':>10s}  {'avg_ms':>9s}  "
              f"{'min_ms':>9s}  {'max_ms':>9s}  {'algo_BW':>12s}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*12}")

    for dtype_name, dtype in test_dtypes:
        dist.barrier()
        times_ms, total_bytes = run_bandwidth_test(rank, world_size, dtype, args.partition_sz, ag_stream,
                                                   args.iterations, args.warmup)

        avg_ms = np.mean(times_ms)
        min_ms = np.min(times_ms)
        max_ms = np.max(times_ms)

        avg_t = torch.tensor([avg_ms], dtype=torch.float64)
        min_t = torch.tensor([min_ms], dtype=torch.float64)
        max_t = torch.tensor([max_ms], dtype=torch.float64)
        dist.all_reduce(avg_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(min_t, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_t, op=dist.ReduceOp.MAX)

        if rank == 0:
            g_avg_ms = avg_t.item() / world_size
            g_min_ms = min_t.item()
            g_max_ms = max_t.item()
            data_mb = total_bytes / (1024**2)
            algo_bw_gbs = total_bytes / (g_avg_ms / 1000) / (1024**3)
            print(f"  {dtype_name:10s}  {data_mb:10.2f}  {g_avg_ms:9.3f}  "
                  f"{g_min_ms:9.3f}  {g_max_ms:9.3f}  {algo_bw_gbs:9.2f} GB/s")

    dist.barrier()
    if rank == 0:
        print()
        print(f"Result: {'All correctness tests PASSED' if all_correct else 'Some correctness tests FAILED'}")
        print(f"{'=' * 65}\n")

    get_accelerator().synchronize()
    dist.barrier()
    if _mori.is_enabled():
        import mori.shmem as shmem
        shmem.shmem_finalize()


if __name__ == "__main__":
    main()

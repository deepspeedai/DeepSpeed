# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch

try:
    from torch._subclasses.fake_tensor import unset_fake_temporarily
except ImportError:
    # Unsupported torch version
    pass

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator


def sync_all():
    get_accelerator().synchronize()
    dist.barrier()


def get_bw(comm_op, size, duration):
    n = dist.get_world_size()
    tput = 0
    busbw = 0

    if duration == 0:
        raise ValueError("Error. Duration is 0.")

    if comm_op == "all_to_all":
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_gather":
        size *= n
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_reduce":
        tput = (size * 2 / duration)
        busbw = (size / duration) * (2 * (n - 1) / n)
    elif comm_op == "pt2pt" or comm_op == "broadcast":
        tput = (size / duration)
        busbw = tput
    else:
        raise ValueError("wrong comm_op specified")

    return tput, busbw


# Run all_gather and print metrics
def timed_all_gather(device, input, output, start_event, end_event, warmup, trials, async_op):
    sync_all()
    # Warmups, establish connections, etc.
    for i in range(warmup):
        dist.all_gather_into_tensor(output, input, async_op=async_op)
    sync_all()

    # time the actual comm op trials times and average it
    start_event.record()
    for i in range(trials):
        dist.all_gather_into_tensor(output, input, async_op=async_op)
    end_event.record()
    sync_all()
    duration = start_event.elapsed_time(end_event) / 1000

    # maintain and clean performance data
    avg_duration = duration / trials
    size = input.element_size() * input.nelement() * dist.get_world_size()
    # tput, busbw = get_bw('all_gather', size, avg_duration)

    avg_duration_ten = torch.tensor([avg_duration], device=device)
    if dist.get_world_size() > 1:
        dist.all_reduce(avg_duration_ten, dist.ReduceOp.AVG)

    return size, avg_duration_ten.item()


def run_all_gather(device, dtype, maxsize, warmup=5, trials=10, async_op=False):

    # Prepare benchmark header
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    start_event = get_accelerator().Event(enable_timing=True)
    end_event = get_accelerator().Event(enable_timing=True)

    # Create list of message sizes
    M_LIST = []
    for x in (2**p for p in range(1, maxsize)):
        m = x // world_size
        if m > 0:
            M_LIST.append(m)

    results = [(0, 0)]
    sync_all()
    # loop over various tensor sizes
    for M in M_LIST:
        global_rank = dist.get_rank()
        try:
            mat = torch.ones(M, dtype=dtype, device=device)
            sync_all()
            input = ((mat.mul_(float(global_rank))).view(-1))
            # Delete original mat to avoid OOM
            del mat
            get_accelerator().empty_cache()
            output = torch.zeros(input.nelement() * world_size, dtype=dtype, device=device)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if dist.get_rank() == 0:
                    print('WARNING: Ran out of GPU memory. Exiting comm op.')
                sync_all()
                break
            else:
                raise e
        sync_all()
        results.append(timed_all_gather(device, input, output, start_event, end_event, warmup, trials, async_op))

    return results


profile_results = None


def create_predictor():
    global profile_results
    if profile_results is None:
        with unset_fake_temporarily():
            device = get_accelerator().current_device()
            profile_results = run_all_gather(device, torch.bfloat16, 31)
        if dist.get_rank() == 0:
            for size, avg_duration in profile_results:
                print(f"size: {size}, avg_duration: {avg_duration}")

    # Extract size and avg_duration from results
    sizes = [result[0] for result in profile_results]
    durations = [result[1] for result in profile_results]

    try:
        from scipy.interpolate import interp1d
    except ImportError:
        raise RuntimeError("Please install scipy to use communication profiler in DeepCompile")

    predictor = interp1d(sizes, durations, kind='linear', fill_value="extrapolate")

    def f(size):
        if size == 0:
            return 0
        return predictor(size)

    # Create an interpolation function
    return f


if __name__ == "__main__":
    local_rank = int(os.environ['LOCAL_RANK'])
    get_accelerator().set_device(local_rank)
    print(f"local_rank={local_rank}")

    deepspeed.init_distributed(dist_backend='nccl')

    # Create predictor function
    predictor = create_predictor()

    # Predict time for a specific data size
    example_size = 1e9
    predicted_time = predictor(example_size)
    print(f"Predicted time for size {example_size}: {predicted_time:.6f} seconds")

    dist.destroy_process_group()

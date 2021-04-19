"""
Copyright 2020 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import os

GIGABYTE = 1024**3
LOG_TIDS = [0]


def task_log(tid, msg):
    if tid in LOG_TIDS:
        print(f'tid {tid}: {msg}')


def task_barrier(barrier, num_parties):
    assert barrier.parties == num_parties
    barrier.wait()
    assert barrier.broken == False


def report_results(args, read_op, pool_results):
    #print(f'pool_results = {pool_results}')
    io_string = 'Read' if read_op else 'Write'
    if None in pool_results:
        print(f'Failure in one of {args.threads} {io_string} processes')
        return

    total_bytes = sum([num_bytes for _, _, num_bytes in pool_results])

    task_latency_sec = max([sec for _, sec, _ in pool_results])
    task_speed_GB = total_bytes / task_latency_sec / GIGABYTE
    print(f'Task {io_string} Latency = {task_latency_sec} sec')
    print(f'Task {io_string} Speed = {task_speed_GB} GB/sec')

    e2e_latency_sec = max([sec for sec, _, _ in pool_results])
    e2e_speed_GB = total_bytes / e2e_latency_sec / GIGABYTE
    print(f'E2E {io_string} Latency = {e2e_latency_sec} sec')
    print(f'E2E {io_string} Speed = {e2e_speed_GB} GB/sec')

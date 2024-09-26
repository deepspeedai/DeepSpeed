# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import os
from .ds_aio_job import Job, run_job

BYTES_PER_GB = 1024**3
BYTES_PER_MB = 1024**2
BYTES_PER_KB = 1024
LOG_TIDS = [0]


def task_log(tid, msg, force=False):
    if force or tid in LOG_TIDS:
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
    task_speed_GB = 0 if task_latency_sec == 0 else total_bytes / task_latency_sec / BYTES_PER_GB
    print(f'Task {io_string} Latency = {task_latency_sec} sec')
    print(f'Task {io_string} Speed = {task_speed_GB} GB/sec')

    e2e_latency_sec = max([sec for sec, _, _ in pool_results])
    e2e_speed_GB = 0 if e2e_latency_sec == 0 else total_bytes / e2e_latency_sec / BYTES_PER_GB
    print(f'E2E {io_string} Latency = {e2e_latency_sec} sec')
    print(f'E2E {io_string} Speed = {e2e_speed_GB} GB/sec')


def get_block_size_and_count(io_bytes):
    if io_bytes > BYTES_PER_MB and io_bytes % BYTES_PER_MB == 0:
        block_size = BYTES_PER_MB
        block_size_string = '1M'
    else:
        assert io_bytes % BYTES_PER_KB == 0
        block_size = BYTES_PER_KB
        block_size_string = '1K'
    block_count = io_bytes / block_size

    return block_size_string, int(block_count)


def refine_integer_value(value):
    unit_dict = {'K': 1024, 'M': 1024**2, 'G': 1024**3}

    if value[-1] in list(unit_dict.keys()):
        int_value = int(value[:-1]) * unit_dict[value[-1]]
        return int_value
    return int(value)


def create_filename(folder, read_op, size, tid):
    io_string = "read" if read_op else "write"
    return os.path.join(folder, f'_aio_{io_string}_{size}.pt.{tid}')


def create_file(filename, num_bytes):
    block_size, block_count = get_block_size_and_count(num_bytes)
    dd_job = Job(cmd_line=[f'dd if=/dev/urandom of={filename} bs={block_size} count={block_count}'])
    print(f'[Start] Create {filename} of {num_bytes} bytes by running {dd_job.cmd()} ....')
    run_job(dd_job)
    print(f'[Done] Create read file of {num_bytes} bytes by running {dd_job.cmd()} ....')

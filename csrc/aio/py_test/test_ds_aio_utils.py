# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

BYTES_PER_GB = 1024**3
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
    task_speed_GB = total_bytes / task_latency_sec / BYTES_PER_GB
    print(f'Task {io_string} Latency = {task_latency_sec} sec')
    print(f'Task {io_string} Speed = {task_speed_GB} GB/sec')

    e2e_latency_sec = max([sec for sec, _, _ in pool_results])
    e2e_speed_GB = total_bytes / e2e_latency_sec / BYTES_PER_GB
    print(f'E2E {io_string} Latency = {e2e_latency_sec} sec')
    print(f'E2E {io_string} Speed = {e2e_speed_GB} GB/sec')


def refine_integer_value(value):
    unit_dict = {'K': 1024, 'M': 1024**2, 'G': 1024**3}

    if value[-1] in list(unit_dict.keys()):
        int_value = int(value[:-1]) * unit_dict[value[-1]]
        return int_value
    return int(value)


def refine_args(args):
    if args.write_size and type(args.write_size) == str:
        args.write_size = refine_integer_value(args.write_size)

    if args.block_size and type(args.block_size) == str:
        args.block_size = refine_integer_value(args.block_size)

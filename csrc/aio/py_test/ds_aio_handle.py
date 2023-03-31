# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import torch
import os
import time
from multiprocessing import Pool, Barrier
from test_ds_aio_utils import report_results, task_log, task_barrier
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import AsyncIOBuilder


def pre_handle(args, tid, read_op):
    io_string = "Read" if read_op else "Write"
    num_bytes = os.path.getsize(args.read_file) if read_op else args.write_size
    file = args.read_file if read_op else f'{args.write_file}.{tid}'

    io_parallel = args.io_parallel if args.io_parallel else 1
    handle = AsyncIOBuilder().load().aio_handle(args.block_size, args.queue_depth, args.single_submit,
                                                args.overlap_events, io_parallel)
    task_log(tid, f'Created deepspeed aio handle')

    if args.gpu:
        buffer = torch.empty(num_bytes, dtype=torch.uint8, device=get_accelerator().device_name())
    else:
        if args.use_accelerator_pin_memory:
            buffer = get_accelerator().pin_memory(torch.empty(num_bytes, dtype=torch.uint8, device='cpu'))
        else:
            buffer = handle.new_cpu_locked_tensor(num_bytes, torch.empty(0, dtype=torch.uint8))

    task_log(tid, f'Allocate tensor of size {num_bytes} bytes')

    ctxt = {}
    ctxt['file'] = file
    ctxt['num_bytes'] = num_bytes
    ctxt['handle'] = handle
    ctxt['buffer'] = buffer
    ctxt['elapsed_sec'] = 0

    task_log(tid, f'{io_string} file {file} of size {num_bytes} bytes from buffer on device {buffer.device}')

    return ctxt


def pre_handle_read(pool_params):
    args, tid = pool_params
    ctxt = pre_handle(args, tid, True)
    return ctxt


def pre_handle_write(pool_params):
    args, tid = pool_params
    ctxt = pre_handle(args, tid, False)
    return ctxt


def post_handle(pool_params):
    _, _, ctxt = pool_params
    ctxt["buffer"].detach()
    ctxt["buffer"] = None
    return ctxt


def main_parallel_read(pool_params):
    args, tid, ctxt = pool_params
    handle = ctxt['handle']

    start_time = time.time()
    ret = handle.pread(ctxt['buffer'], ctxt['file'], args.validate, True)
    assert ret != -1
    handle.wait()
    end_time = time.time()
    ctxt['elapsed_sec'] += end_time - start_time

    return ctxt


def main_parallel_write(pool_params):
    args, tid, ctxt = pool_params
    handle = ctxt['handle']
    start_time = time.time()
    ret = handle.pwrite(ctxt['buffer'], ctxt['file'], args.validate, True)
    assert ret != -1
    handle.wait()
    end_time = time.time()
    ctxt['elapsed_sec'] += end_time - start_time

    return ctxt


def main_handle_read(pool_parms):
    args, tid, ctxt = pool_parms
    handle = ctxt['handle']

    start_time = time.time()
    ret = handle.read(ctxt['buffer'], ctxt['file'], args.validate)
    assert ret != -1
    end_time = time.time()
    ctxt['elapsed_sec'] += end_time - start_time

    return ctxt


def main_handle_write(pool_parms):
    args, tid, ctxt = pool_parms
    handle = ctxt['handle']
    start_time = time.time()
    ret = handle.write(ctxt['buffer'], ctxt['file'], args.validate)
    assert ret != -1
    end_time = time.time()
    ctxt['elapsed_sec'] += end_time - start_time

    return ctxt


def get_schedule(args, read_op):
    schedule = {}
    if read_op:
        schedule['pre'] = pre_handle_read
        schedule['post'] = post_handle
        schedule['main'] = main_parallel_read if args.io_parallel else main_handle_read
    else:
        schedule['pre'] = pre_handle_write
        schedule['post'] = post_handle
        schedule['main'] = main_parallel_write if args.io_parallel else main_handle_write

    return schedule


def _aio_handle_tasklet(pool_params):
    args, tid, read_op = pool_params

    # Create schedule
    schedule = get_schedule(args, read_op)
    task_log(tid, f'schedule = {schedule}')
    task_barrier(aio_barrier, args.threads)

    # Run pre task
    task_log(tid, f'running pre-task')
    ctxt = schedule["pre"]((args, tid))
    task_barrier(aio_barrier, args.threads)

    # Run main tasks in a loop
    ctxt["main_task_sec"] = 0
    for i in range(args.loops):
        task_log(tid, f'running main task {i}')
        start_time = time.time()
        ctxt = schedule["main"]((args, tid, ctxt))
        task_barrier(aio_barrier, args.threads)
        stop_time = time.time()
        ctxt["main_task_sec"] += stop_time - start_time

    # Run post task
    task_log(tid, f'running post-task')
    ctxt = schedule["post"]((args, tid, ctxt))
    task_barrier(aio_barrier, args.threads)

    return ctxt["main_task_sec"], ctxt["elapsed_sec"], ctxt["num_bytes"] * args.loops


def _init_tasklet(b):
    global aio_barrier
    aio_barrier = b


def aio_handle_multiprocessing(args, read_op):
    b = Barrier(args.threads)
    pool_params = [(args, p, read_op) for p in range(args.threads)]
    with Pool(processes=args.threads, initializer=_init_tasklet, initargs=(b, )) as p:
        pool_results = p.map(_aio_handle_tasklet, pool_params)

    report_results(args, read_op, pool_results)

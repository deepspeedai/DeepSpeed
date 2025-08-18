# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
from multiprocessing import Pool, Barrier

from .ds_aio_constants import AIO_BASIC, TORCH_FAST_IO, TORCH_IO
from .test_ds_aio_utils import report_results, task_log, task_barrier
from .ds_aio_handle import AIOHandle_Engine
from .ds_aio_basic import AIOBasic_Engine
from .torch_io import TorchIO_Engine
from .torch_fastio_engine import Torch_FastIO_Engine


def prepare_operation(args, tid, read_op):
    if args.engine == TORCH_IO:
        io_engine = TorchIO_Engine(args, tid, read_op)
    elif args.engine == AIO_BASIC:
        io_engine = AIOBasic_Engine(args, tid, read_op)
    elif args.engine == TORCH_FAST_IO:
        io_engine = Torch_FastIO_Engine(args, tid, read_op)
    else:
        io_engine = AIOHandle_Engine(args, tid, read_op)

    return io_engine


def prepare_read(pool_params):
    args, tid = pool_params
    return prepare_operation(args, tid, True)


def prepare_write(pool_params):
    args, tid = pool_params
    return prepare_operation(args, tid, False)


def post_operation(pool_params):
    _, _, io_engine = pool_params
    io_engine.fini()


def read_operation(pool_params):
    args, tid, loop_id, io_engine = pool_params
    return io_engine.read(args, tid, loop_id)


def write_operation(pool_params):
    args, tid, loop_id, io_engine = pool_params
    return io_engine.write(args, tid, loop_id)


def get_schedule(args, read_op):
    schedule = {}
    if read_op:
        schedule['pre'] = prepare_read
        schedule['post'] = post_operation
        schedule['main'] = read_operation
    else:
        schedule['pre'] = prepare_write
        schedule['post'] = post_operation
        schedule['main'] = write_operation

    return schedule


def io_engine_tasklet(pool_params):
    args, tid, read_op = pool_params
    num_processes = len(args.mapping_dict)

    # Create schedule
    schedule = get_schedule(args, read_op)
    task_log(tid, f'schedule = {schedule}')
    task_barrier(aio_barrier, num_processes)

    # Run pre task
    task_log(tid, 'running pre-task')
    io_engine = schedule["pre"]((args, tid))
    task_barrier(aio_barrier, num_processes)

    # Run main tasks in a loop
    io_engine.ctxt["main_task_sec"] = []
    for i in range(args.total_loops):
        task_log(tid, f'running main task {i}')
        start_time = time.time()
        schedule["main"]((args, tid, i, io_engine))
        task_barrier(aio_barrier, num_processes)
        stop_time = time.time()
        io_engine.ctxt["main_task_sec"].append(stop_time - start_time)

    # Run post task
    task_log(tid, 'running post-task')
    schedule["post"]((args, tid, io_engine))
    task_barrier(aio_barrier, num_processes)

    ctxt = io_engine.ctxt
    # return ctxt["main_task_sec"], ctxt["elapsed_sec"], ctxt["num_bytes"] * args.loops
    if args.include_warmup_time:
        e2e_latency_sec = sum(ctxt["main_task_sec"])
        task_latency_sec = sum(ctxt["elapsed_sec"])
        actual_loops = args.total_loops
    else:
        e2e_latency_sec = sum(ctxt["main_task_sec"][args.warmup_loops:])
        task_latency_sec = sum(ctxt["elapsed_sec"][args.warmup_loops:])
        actual_loops = args.loops

    l = ctxt["elapsed_sec"]
    task_log(tid, f'task_latency_sec = {l}')
    return e2e_latency_sec, task_latency_sec, ctxt["num_bytes"] * actual_loops


def _init_takslet(b):
    global aio_barrier
    aio_barrier = b


def io_engine_multiprocessing(args, read_op):
    num_processes = len(args.mapping_dict)
    b = Barrier(num_processes)
    pool_params = [(args, p, read_op) for p in range(num_processes)]
    with Pool(processes=num_processes, initializer=_init_takslet, initargs=(b, )) as p:
        pool_results = p.map(io_engine_tasklet, pool_params)

    report_results(args, read_op, pool_results)

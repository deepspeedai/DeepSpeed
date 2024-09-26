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
from deepspeed.ops.aio import AsyncIOBuilder
from deepspeed.ops.op_builder import GDSBuilder
from deepspeed.accelerator import get_accelerator
from .test_ds_aio_utils import report_results, task_log, task_barrier, create_filename, create_file

BUFFER = 'buffer'
BOUNCE_BUFFER = 'bounce_buffer'


def pre_handle(args, tid, read_op):
    io_string = "Read" if read_op else "Write"
    gds = True if args.use_gds else False
    device_id, folder = args.mapping_list[tid]
    filename = create_filename(folder, args.read, args.io_size, tid)
    if args.read and not (os.path.isfile(filename) and os.path.getsize(filename) == args.io_size):
        create_file(filename, args.io_size)

    task_log(tid, f'Allocate tensor of size {args.io_size} bytes')
    bounce_buffer = None
    if args.gpu:
        device_name = get_accelerator().device_name(device_id)
        buffer = torch.randint(high=128, size=(args.io_size, ), dtype=torch.uint8, device=device_name)
        if not (args.slow_bounce_buffer or gds):
            bounce_buffer = torch.randint(high=128, size=(args.io_size, ), dtype=torch.uint8,
                                          device='cpu').pin_memory()
    else:
        buffer = torch.randint(high=128, size=(args.io_size, ), dtype=torch.uint8, device='cpu').pin_memory()
    task_log(tid,
             f'{io_string} file {filename} of size {args.io_size} bytes from buffer on device {buffer.device}',
             force=True)

    io_parallel = args.io_parallel if args.io_parallel else 1
    if gds:
        handle = GDSBuilder().load().gds_handle(args.block_size, args.queue_depth, args.single_submit,
                                                not args.sequential_requests, io_parallel)
        handle.pin_device_tensor(buffer)
    else:
        handle = AsyncIOBuilder().load().aio_handle(args.block_size, args.queue_depth, args.single_submit,
                                                    not args.sequential_requests, io_parallel)
    task_log(tid, f'created deepspeed aio handle')

    ctxt = {}
    ctxt['file'] = filename
    ctxt['num_bytes'] = args.io_size
    ctxt['handle'] = handle
    ctxt['gds'] = gds
    ctxt[BUFFER] = buffer
    ctxt[BOUNCE_BUFFER] = bounce_buffer
    ctxt['elapsed_sec'] = 0

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
    for buf in [BUFFER, BOUNCE_BUFFER]:
        if ctxt[buf] is not None:
            if ctxt['gds']:
                ctxt['handle'].unpin_device_tensor(ctxt[buf])
            ctxt[buf].detach()
            ctxt[buf] = None
    return ctxt


def main_parallel_read(pool_params):
    args, tid, ctxt = pool_params
    handle = ctxt['handle']

    start_time = time.time()
    dest_buffer = BOUNCE_BUFFER if ctxt[BOUNCE_BUFFER] is not None else BUFFER
    ret = handle.pread(ctxt[dest_buffer], ctxt['file'], args.validate, True)
    assert ret != -1
    handle.wait()
    if dest_buffer == BOUNCE_BUFFER:
        ctxt[BUFFER].data.copy_(ctxt[BOUNCE_BUFFER].data)
    end_time = time.time()
    ctxt['elapsed_sec'] += end_time - start_time
    return ctxt


def main_parallel_write(pool_params):
    args, tid, ctxt = pool_params
    # Avoid overwriting existing files as it could be artificially faster
    if os.path.isfile(ctxt['file']):
        os.remove(ctxt['file'])

    handle = ctxt['handle']
    start_time = time.time()
    if ctxt[BOUNCE_BUFFER] is not None:
        source_buffer = BOUNCE_BUFFER
        ctxt[BOUNCE_BUFFER].data.copy_(ctxt[BUFFER].data)
    else:
        source_buffer = BUFFER
    ret = handle.pwrite(ctxt[source_buffer], ctxt['file'], args.validate, True)
    assert ret != -1
    handle.wait()
    end_time = time.time()
    ctxt['elapsed_sec'] += end_time - start_time

    return ctxt


def main_handle_read(pool_parms):
    args, tid, ctxt = pool_parms
    handle = ctxt['handle']

    start_time = time.time()
    dest_buffer = BOUNCE_BUFFER if ctxt[BOUNCE_BUFFER] is not None else BUFFER
    ret = handle.read(ctxt[dest_buffer], ctxt['file'], args.validate)
    assert ret != -1
    if dest_buffer == BOUNCE_BUFFER:
        ctxt[BUFFER].data.copy_(ctxt[BOUNCE_BUFFER].data)
    end_time = time.time()
    ctxt['elapsed_sec'] += end_time - start_time

    return ctxt


def main_handle_write(pool_parms):
    args, tid, ctxt = pool_parms
    # Avoid overwriting existing files as it could be artificially faster
    if os.path.isfile(ctxt['file']):
        os.remove(ctxt['file'])

    handle = ctxt['handle']
    start_time = time.time()
    if ctxt[BOUNCE_BUFFER] is not None:
        source_buffer = BOUNCE_BUFFER
        ctxt[BOUNCE_BUFFER].data.copy_(ctxt[BUFFER].data)
    else:
        source_buffer = BUFFER
    ret = handle.write(ctxt[source_buffer], ctxt['file'], args.validate)
    assert ret != -1
    end_time = time.time()
    ctxt['elapsed_sec'] += end_time - start_time

    return ctxt


def get_schedule(args, read_op):
    schedule = {}
    if read_op:
        schedule['pre'] = pre_handle_read
        schedule['post'] = post_handle
        schedule['main'] = main_parallel_read
    else:
        schedule['pre'] = pre_handle_write
        schedule['post'] = post_handle
        schedule['main'] = main_parallel_write

    return schedule


def _aio_handle_tasklet(pool_params):
    args, tid, read_op = pool_params
    num_processes = len(args.mapping_dict)

    # Create schedule
    schedule = get_schedule(args, read_op)
    task_log(tid, f'schedule = {schedule}')
    task_barrier(aio_barrier, num_processes)

    # Run pre task
    task_log(tid, f'running pre-task')
    ctxt = schedule["pre"]((args, tid))
    task_barrier(aio_barrier, num_processes)

    # Run main tasks in a loop
    ctxt["main_task_sec"] = 0
    for i in range(args.loops):
        task_log(tid, f'running main task {i}')
        start_time = time.time()
        ctxt = schedule["main"]((args, tid, ctxt))
        task_barrier(aio_barrier, num_processes)
        stop_time = time.time()
        ctxt["main_task_sec"] += stop_time - start_time

    # Run post task
    task_log(tid, f'running post-task')
    ctxt = schedule["post"]((args, tid, ctxt))
    task_barrier(aio_barrier, num_processes)

    return ctxt["main_task_sec"], ctxt["elapsed_sec"], ctxt["num_bytes"] * args.loops


def _init_tasklet(b):
    global aio_barrier
    aio_barrier = b


def aio_handle_multiprocessing(args, read_op):
    num_processes = len(args.mapping_dict)
    b = Barrier(num_processes)
    pool_params = [(args, p, read_op) for p in range(num_processes)]
    with Pool(processes=num_processes, initializer=_init_tasklet, initargs=(b, )) as p:
        pool_results = p.map(_aio_handle_tasklet, pool_params)

    report_results(args, read_op, pool_results)

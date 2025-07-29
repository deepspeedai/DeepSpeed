# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import torch
import os
import time
from deepspeed.ops.aio import AsyncIOBuilder
from deepspeed.ops.op_builder import GDSBuilder
from deepspeed.accelerator import get_accelerator
from .test_ds_aio_utils import task_log, create_filename, create_file, create_page_locked_tensor
from .ds_aio_constants import *


class AIOHandle_Engine(object):

    def __init__(self, args, tid, read_op):
        self.ctxt = self._create_context(args, tid, read_op)

    def fini(self):
        for buf in [BUFFER, BOUNCE_BUFFER]:
            if self.ctxt[buf] is not None:
                if self.ctxt[USE_CPU_LOCKED_TENSOR]:
                    self.ctxt[HANDLE].free_cpu_locked_tensor(self.ctxt[buf])

                self.ctxt[buf].detach()
                self.ctxt[buf] = None

    def read(self, args, tid, loop_id):
        handle = self.ctxt[HANDLE]

        start_time = time.time()
        dest_buffer = BOUNCE_BUFFER if self.ctxt[BOUNCE_BUFFER] is not None else BUFFER
        ret = handle.pread(self.ctxt[dest_buffer], self.ctxt[FILE][loop_id], args.validate, True)
        assert ret != -1
        handle.wait()
        if dest_buffer == BOUNCE_BUFFER:
            self.ctxt[BUFFER].data.copy_(self.ctxt[BOUNCE_BUFFER].data)
        end_time = time.time()
        self.ctxt[ELAPSED_SEC].append(end_time - start_time)

    def write(self, args, tid, loop_id):
        # Avoid overwriting existing files as it could be artificially faster
        # if os.path.isfile(self.ctxt[FILE]):
        #     os.remove(self.ctxt[FILE])

        handle = self.ctxt[HANDLE]
        start_time = time.time()
        if self.ctxt[BOUNCE_BUFFER] is not None:
            source_buffer = BOUNCE_BUFFER
            self.ctxt[BOUNCE_BUFFER].data.copy_(self.ctxt[BUFFER].data)
        else:
            source_buffer = BUFFER
        ret = handle.pwrite(self.ctxt[source_buffer], self.ctxt[FILE][loop_id], args.validate, True)
        assert ret != -1
        handle.wait()
        end_time = time.time()
        self.ctxt[ELAPSED_SEC].append(end_time - start_time)

    def _create_files(self, args, folder, tid):
        if args.different_file_each_iteration:
            filenames = [
                create_filename(folder, args.read, args.io_size, f'{tid}_{l}') for l in range(args.total_loops)
            ]
        else:
            filenames = [
                create_filename(folder, args.read, args.io_size, f'{tid}_{0}') for _ in range(args.total_loops)
            ]

        if args.read:
            for f in filenames:
                if not (os.path.isfile(f) and os.path.getsize(f) == args.io_size):
                    create_file(f, args.io_size)
        else:
            for f in filenames:
                if os.path.isfile(f):
                    os.remove(f)

        return filenames

    def _create_context(self, args, tid, read_op):
        io_string = "Read" if read_op else "Write"
        device_id, folder = args.mapping_list[tid]
        filenames = self._create_files(args, folder, tid)

        gds = True if args.use_gds else False
        io_parallel = args.io_parallel if args.io_parallel else 1
        if gds:
            handle = GDSBuilder().load().gds_handle(args.block_size, args.queue_depth, args.single_submit,
                                                    not args.sequential_requests, io_parallel)
        else:
            handle = AsyncIOBuilder().load().aio_handle(args.block_size, args.queue_depth, args.single_submit,
                                                        not args.sequential_requests, io_parallel)
        task_log(tid, 'Created DeepNVMe handle engine')

        bounce_buffer = None
        if args.gpu:
            device_name = get_accelerator().device_name(device_id)
            buffer = torch.randint(high=128, size=(args.io_size, ), dtype=torch.uint8, device=device_name)
            if gds:
                handle.pin_device_tensor(buffer)
            elif not args.slow_bounce_buffer:
                bounce_buffer = create_page_locked_tensor(args.io_size, args.use_accelerator_pin_memory, handle)
        else:
            buffer = create_page_locked_tensor(args.io_size, args.use_accelerator_pin_memory, handle)
        task_log(tid, f'Allocate tensor of size {args.io_size} bytes')

        ctxt = {}
        ctxt[FILE] = filenames
        ctxt[NUM_BYTES] = args.io_size
        ctxt[HANDLE] = handle
        ctxt[USE_GDS] = gds
        ctxt[BUFFER] = buffer
        ctxt[BOUNCE_BUFFER] = bounce_buffer
        ctxt[ELAPSED_SEC] = []
        ctxt[USE_CPU_LOCKED_TENSOR] = not args.use_accelerator_pin_memory

        task_log(tid,
                 f'{io_string} file {filenames} of size {args.io_size} bytes from buffer on device {buffer.device}',
                 force=True)

        return ctxt

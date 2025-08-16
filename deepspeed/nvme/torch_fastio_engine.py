# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import os
import time
from deepspeed.ops.aio import AsyncIOBuilder
from .test_ds_aio_utils import task_log, create_filename, create_file, create_page_locked_tensor
from .ds_aio_constants import *
from deepspeed.io import FastFileWriter


class Torch_FastIO_Engine(object):

    def __init__(self, args, tid, read_op):
        assert read_op is False, 'Read operation is not currently supported'
        self.ctxt = self._create_context(args, tid, read_op)
        self.zipfile_serialization = not args.torch_legacy_save

    def fini(self):
        if self.ctxt[USE_CPU_LOCKED_TENSOR]:
            for buf in [BUFFER, FAST_IO_BUFFER]:
                self.ctxt[HANDLE].free_cpu_locked_tensor(self.ctxt[buf])

        self.ctxt[BUFFER].detach()
        self.ctxt[BUFFER] = None

    def read(self, args, tid):
        start_time = time.time()
        torch.load(f=self.ctxt[FILE], map_location=self.ctxt[BUFFER].device)
        end_time = time.time()
        self.ctxt[ELAPSED_SEC] += end_time - start_time

    def write(self, args, tid):
        # Avoid overwriting existing files as it could be artificially faster
        if os.path.isfile(self.ctxt[FILE]):
            os.remove(self.ctxt[FILE])

        ds_file_writer = FastFileWriter(file_path=self.ctxt[FILE],
                                        aio_handle=self.ctxt[HANDLE],
                                        pinned_tensor=self.ctxt[FAST_IO_BUFFER])

        start_time = time.time()
        torch.save(obj=self.ctxt[BUFFER], f=ds_file_writer, _use_new_zipfile_serialization=self.zipfile_serialization)
        ds_file_writer.close()  # Force flush to storage
        end_time = time.time()
        self.ctxt[ELAPSED_SEC] += end_time - start_time
        ds_file_writer._dump_state()

    def _create_context(self, args, tid, read_op):
        io_string = "Read" if read_op else "Write"
        device_id, folder = args.mapping_list[tid]
        filename = create_filename(folder, args.read, args.io_size, tid)
        if args.read and not (os.path.isfile(filename) and os.path.getsize(filename) == args.io_size):
            create_file(filename, args.io_size)

        io_parallel = args.io_parallel if args.io_parallel else 1
        aio_handle = AsyncIOBuilder().load().aio_handle(args.block_size, args.queue_depth, args.single_submit,
                                                        not args.sequential_requests, io_parallel)

        if args.gpu:
            buffer = torch.randint(high=128, size=(args.io_size, ), dtype=torch.uint8, device=f'cuda:{device_id}')
        else:
            buffer = create_page_locked_tensor(args.io_size, args.use_accelerator_pin_memory, aio_handle)

        task_log(tid, f'Allocate tensor of size {args.io_size} bytes')

        fast_io_buffer = create_page_locked_tensor(args.fast_io_size, args.use_accelerator_pin_memory, aio_handle)

        task_log(tid, 'created torch_fastio engine')

        ctxt = {}
        ctxt[FILE] = filename
        ctxt[NUM_BYTES] = args.io_size
        ctxt[BUFFER] = buffer
        ctxt[HANDLE] = aio_handle
        ctxt[FAST_IO_BUFFER] = fast_io_buffer
        ctxt[ELAPSED_SEC] = 0
        ctxt[USE_CPU_LOCKED_TENSOR] = not args.use_accelerator_pin_memory

        task_log(tid,
                 f'{io_string} file {filename} of size {args.io_size} bytes from buffer on device {buffer.device}',
                 force=True)

        return ctxt

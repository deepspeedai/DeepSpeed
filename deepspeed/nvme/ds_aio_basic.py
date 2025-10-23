# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import os
import time
from deepspeed.ops.aio import AsyncIOBuilder
from .test_ds_aio_utils import task_log, create_filename, create_file, create_page_locked_tensor
from .ds_aio_constants import *


class AIOBasic_Engine(object):

    def __init__(self, args, tid, read_op):
        self.ctxt = self._create_context(args, tid, read_op)

    def fini(self):
        self.ctxt[BUFFER].detach()
        self.ctxt[BUFFER] = None

    def read(self, args, tid, loop_id):
        start_time = time.time()
        AsyncIOBuilder().load().aio_read(self.ctxt[BUFFER], self.ctxt[FILE], args.block_size, args.queue_depth,
                                         args.single_submit, not args.sequential_requests, args.validate)
        end_time = time.time()
        self.ctxt[ELAPSED_SEC] += end_time - start_time

    def write(self, args, tid, loop_id):
        # Avoid overwriting existing files as it could be artificially faster
        if os.path.isfile(self.ctxt[FILE]):
            os.remove(self.ctxt[FILE])

        start_time = time.time()
        AsyncIOBuilder().load().aio_write(self.ctxt[BUFFER], self.ctxt[FILE], args.block_size, args.queue_depth,
                                          args.single_submit, not args.sequential_requests, args.validate)
        end_time = time.time()
        self.ctxt[ELAPSED_SEC] += end_time - start_time

    def _create_context(self, args, tid, read_op):
        io_string = "Read" if read_op else "Write"
        device_id, folder = args.mapping_list[tid]
        filename = create_filename(folder, args.read, args.io_size, tid)
        if args.read and not (os.path.isfile(filename) and os.path.getsize(filename) == args.io_size):
            create_file(filename, args.io_size)

        task_log(tid, f'Allocate tensor of size {args.io_size} bytes')

        buffer = create_page_locked_tensor(args.io_size, True)

        task_log(tid,
                 f'{io_string} file {filename} of size {args.io_size} bytes from buffer on device {buffer.device}')

        task_log(tid, 'created deepspeed aio basic engine')

        ctxt = {}
        ctxt[FILE] = filename
        ctxt[NUM_BYTES] = args.io_size
        ctxt[BUFFER] = buffer
        ctxt[ELAPSED_SEC] = 0
        return ctxt

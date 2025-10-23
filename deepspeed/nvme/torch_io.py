# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import os
import time
from .test_ds_aio_utils import task_log, create_filename, create_file, create_page_locked_tensor
from .ds_aio_constants import *


class TorchIO_Engine(object):

    def __init__(self, args, tid, read_op):
        self.ctxt = self._create_context(args, tid, read_op)
        self.zipfile_serialization = not args.torch_legacy_save

    def fini(self):
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

        start_time = time.time()
        torch.save(obj=self.ctxt[BUFFER], f=self.ctxt[FILE], _use_new_zipfile_serialization=self.zipfile_serialization)
        end_time = time.time()
        self.ctxt[ELAPSED_SEC] += end_time - start_time

    def _create_context(self, args, tid, read_op):
        io_string = "Read" if read_op else "Write"
        device_id, folder = args.mapping_list[tid]
        filename = create_filename(folder, args.read, args.io_size, tid)
        if args.read and not (os.path.isfile(filename) and os.path.getsize(filename) == args.io_size):
            create_file(filename, args.io_size)

        task_log(tid, f'Allocate tensor of size {args.io_size} bytes')

        if args.gpu:
            buffer = torch.randint(high=128, size=(args.io_size, ), dtype=torch.uint8, device=f'cuda:{device_id}')
        else:
            buffer = create_page_locked_tensor(args.io_size, True)

        task_log(tid,
                 f'{io_string} file {filename} of size {args.io_size} bytes from buffer on device {buffer.device}',
                 force=True)

        task_log(tid, 'created torch_io engine')

        ctxt = {}
        ctxt[FILE] = filename
        ctxt[NUM_BYTES] = args.io_size
        ctxt[BUFFER] = buffer
        ctxt[ELAPSED_SEC] = 0
        return ctxt

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.ops.op_builder import AsyncIOBuilder, GDSBuilder
from deepspeed.io import MockFileWriter, PyFileWriter, FastFileWriter, FastFileWriterConfig
from deepspeed.runtime.swap_tensor.constants import *
from .constants import *
from deepspeed.accelerator import get_accelerator


class CheckpointWriterFactory(object):

    def __init__(self, writer_config, aio_config, dp_writer_config):
        self._type = writer_config[CHECKPOINT_WRITER_TYPE]
        self._io_buffer_size = writer_config[CHECKPOINT_IO_BUFFER_SIZE]
        self._io_buffer_double = writer_config[CHECKPOINT_IO_BUFFER_DOUBLE]
        self._data_parallel_writer = dp_writer_config
        self._io_multiplier = writer_config[CHECKPOINT_IO_MULTIPLIER]
        if self._data_parallel_writer.pure_dp:
            self._show_statistics = writer_config[CHECKPOINT_IO_STATISTICS] and self._data_parallel_writer is not None
        else:
            self._show_statistics = writer_config[CHECKPOINT_IO_STATISTICS] and self._data_parallel_writer is not None
        self._io_buffer = None
        self._dnvme_handle = None
        self._writer = None
        self._use_gds = False

        if self._type == CheckpointWriterType.FAST:
            self._use_gds = aio_config[AIO_USE_GDS]
            if self._use_gds:
                self._setup_for_gds(aio_config)
            else:
                self._setup_for_aio(aio_config)
        print(
            f'WriterFactory: self._data_parallel_writer={self._data_parallel_writer} self._show_statistics={self._show_statistics}'
        )

    def create_writer(self, file_path, optimize_dp_state):
        assert self._writer is None, \
            f'Cannot create checkpoint writer for {file_path} because writer is currently used for {self._writer.file_path()}.\
            Must call writer.release() before reusing to avoid this error.'

        if self._type == CheckpointWriterType.MOCK:
            self._writer = MockFileWriter(file_path)
        elif self._type == CheckpointWriterType.PYTHON:
            self._writer = PyFileWriter(file_path)
        else:
            if optimize_dp_state:
                num_parallel_writers = self._data_parallel_writer.world_size * self._io_multiplier
                writer_rank = self._data_parallel_writer.rank
                file_path = f'{file_path}-{writer_rank}.{num_parallel_writers}'
                # print(f'create_dp_writer: {self._data_parallel_writer.global_rank=} {writer_rank=} {num_parallel_writers=} {file_path=}')
            else:
                num_parallel_writers = 1
                writer_rank = 0
                # print(f'create_rank0_writer: {self._data_parallel_writer.global_rank=} {writer_rank=} {num_parallel_writers=} {file_path=}')

            config = FastFileWriterConfig(dnvme_handle=self._dnvme_handle,
                                          pinned_tensor=self._io_buffer,
                                          double_buffer=self._io_buffer_double,
                                          num_parallel_writers=num_parallel_writers,
                                          writer_rank=writer_rank,
                                          global_rank=self._data_parallel_writer.global_rank)
            self._writer = FastFileWriter(file_path=file_path, config=config)

        return self._writer

    def release_writer(self):
        self._writer.close()
        if self._show_statistics:
            self._writer._dump_state()
        self._writer = None

    def _setup_for_aio(self, aio_config):
        self._io_buffer = torch.zeros(self._io_buffer_size, dtype=torch.uint8, device='cpu').pin_memory()
        self._dnvme_handle = AsyncIOBuilder().load().aio_handle(
            block_size=aio_config[AIO_BLOCK_SIZE],
            queue_depth=aio_config[AIO_QUEUE_DEPTH],
            single_submit=aio_config[AIO_SINGLE_SUBMIT],
            overlap_events=aio_config[AIO_OVERLAP_EVENTS],
            intra_op_parallelism=aio_config[AIO_INTRA_OP_PARALLELISM])

    def _setup_for_gds(self, aio_config):
        self._io_buffer = torch.zeros(self._io_buffer_size,
                                      dtype=torch.uint8,
                                      device=get_accelerator().current_device_name())
        self._dnvme_handle = GDSBuilder().load().gds_handle(block_size=aio_config[AIO_BLOCK_SIZE],
                                                            queue_depth=aio_config[AIO_QUEUE_DEPTH],
                                                            single_submit=aio_config[AIO_SINGLE_SUBMIT],
                                                            overlap_events=aio_config[AIO_OVERLAP_EVENTS],
                                                            intra_op_parallelism=aio_config[AIO_INTRA_OP_PARALLELISM])
        self._dnvme_handle.pin_device_tensor(self._io_buffer)

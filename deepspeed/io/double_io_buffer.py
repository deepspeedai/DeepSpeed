# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from .base_io_buffer import Base_IO_Buffer

NUM_BUFFERS = 2
INVALID_BUFFER_INDEX = -1


class Double_IO_Buffer(Base_IO_Buffer):

    def __init__(self, pinned_tensor, dnvme_handle):
        super(Double_IO_Buffer, self).__init__(pinned_tensor, dnvme_handle)
        assert self._pinned_tensor.numel() % (NUM_BUFFERS * self._dnvme_handle.get_alignment()) == 0
        self._buffers = self._split_buffer()
        self._fill_index = 0
        self._drain_index = INVALID_BUFFER_INDEX
        self._buffer_offset = 0

    def fill(self, src_tensor, src_offset):
        self._validate_buffer_index(self._fill_index)
        copy_bytes = Base_IO_Buffer.fill_buffer(src_tensor, src_offset, self._buffers[self._fill_index],
                                                self._buffer_offset)
        self._buffer_offset += copy_bytes
        return copy_bytes

    def drain(self, num_bytes, fd, file_offset):
        self._validate_buffer_index(self._fill_index)
        self.complete_ongoing_drain()
        assert self._drain_index == INVALID_BUFFER_INDEX
        self._drain(num_bytes, fd, file_offset, blocking=False)
        self._drain_index = self._fill_index
        self._fill_index = (self._fill_index + 1) % NUM_BUFFERS
        self._buffer_offset = 0

    def get_buffer(self):
        self._validate_buffer_index(self._fill_index)
        return self._buffers[self._fill_index]

    def get_offset(self):
        self._validate_buffer_index(self._fill_index)
        return self._buffer_offset

    def get_aligned_num_bytes(self):
        self._validate_buffer_index(self._fill_index)
        aligned_size = self._dnvme_handle.get_alignment()
        return (self._buffer_offset // aligned_size) * aligned_size

    def get_unaligned_num_bytes(self):
        self._validate_buffer_index(self._fill_index)
        return self._buffer_offset % self._dnvme_handle.get_alignment()

    def is_full(self):
        self._validate_buffer_index(self._fill_index)
        return self._buffer_offset == self._buffers[self._fill_index].numel()

    def is_empty(self):
        self._validate_buffer_index(self._fill_index)
        return self._buffer_offset == 0 and not self._is_ongoing_drain()

    def reset(self):
        self._buffer_offset = 0

    def complete_ongoing_drain(self):
        if self._is_ongoing_drain():
            self._wait_for_drain()

    def _split_buffer(self):
        buffer_size = self._pinned_tensor.numel() // NUM_BUFFERS
        return [torch.narrow(self._pinned_tensor, 0, (i * buffer_size), buffer_size) for i in range(NUM_BUFFERS)]

    def _validate_buffer_index(self, index):
        assert index in [0, 1]

    def _wait_for_drain(self):
        self._validate_buffer_index(self._drain_index)
        assert 1 == self._dnvme_handle.wait()
        self._drain_index = INVALID_BUFFER_INDEX

    def _is_ongoing_drain(self):
        return self._drain_index != INVALID_BUFFER_INDEX

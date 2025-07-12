# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base_io_buffer import Base_IO_Buffer


class Single_IO_Buffer(Base_IO_Buffer):

    def __init__(self, pinned_tensor, dnvme_handle):
        super(Single_IO_Buffer, self).__init__(pinned_tensor, dnvme_handle)
        self._pinned_offset = 0

    def fill(self, src_tensor, src_offset):
        copy_bytes = Base_IO_Buffer.fill_buffer(src_tensor, src_offset, self._pinned_tensor, self._pinned_offset)
        self._pinned_offset += copy_bytes
        return copy_bytes

    def drain(self, num_bytes, fd, file_offset):
        self._drain(num_bytes, fd, file_offset, blocking=True)
        self._pinned_offset = 0

    def get_buffer(self):
        return self._pinned_tensor

    def get_offset(self):
        return self._pinned_offset

    def get_aligned_num_bytes(self):
        aligned_size = self._dnvme_handle.get_alignment()
        return (self._pinned_offset // aligned_size) * aligned_size

    def get_unaligned_num_bytes(self):
        return self._pinned_offset % self._dnvme_handle.get_alignment()

    def is_full(self):
        return self._pinned_offset == self._pinned_tensor.numel()

    def is_empty(self):
        return self._pinned_offset == 0

    def reset(self):
        self._pinned_offset = 0

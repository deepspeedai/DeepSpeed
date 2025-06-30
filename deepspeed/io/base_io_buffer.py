# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch


class Base_IO_Buffer(object):

    def __init__(self, pinned_tensor, dnvme_handle):
        assert pinned_tensor.numel() % dnvme_handle.get_alignment() == 0
        self._dnvme_handle = dnvme_handle
        self._pinned_tensor = pinned_tensor

    def fill(self, src_tensor, src_offset):
        pass

    def drain(self, num_bytes, fd, file_offset):
        pass

    def is_empty(self):
        pass

    def is_full(self):
        pass

    def get_buffer(self):
        pass

    def get_offset(self):
        pass

    def get_aligned_num_bytes(self):
        pass

    def get_unaligned_num_bytes(self):
        pass

    def reset(self):
        pass

    def complete_ongoing_drain(self):
        pass

    def _drain(self, num_bytes, fd, file_offset, blocking=False):
        assert num_bytes <= self.get_offset()
        assert num_bytes % self._dnvme_handle.get_alignment() == 0
        buffer = self.get_buffer()
        r = self._dnvme_handle.async_pwrite(torch.narrow(buffer, 0, 0, num_bytes), fd, file_offset)
        assert 0 == r
        if blocking:
            assert 1 == self._dnvme_handle.wait()

    @staticmethod
    def fill_buffer(src_tensor, src_offset, buffer_tensor, buffer_offset):
        src_bytes = src_tensor.numel() - src_offset
        assert src_bytes > 0

        dst_bytes = buffer_tensor.numel() - buffer_offset
        copy_bytes = min(src_bytes, dst_bytes)
        assert (buffer_offset + copy_bytes) <= buffer_tensor.numel()

        if copy_bytes > 0:
            src_slice = torch.narrow(src_tensor, 0, src_offset, copy_bytes)
            dst_slice = torch.narrow(buffer_tensor, 0, buffer_offset, copy_bytes)
            dst_slice.data.copy_(src_slice.data)

        return copy_bytes

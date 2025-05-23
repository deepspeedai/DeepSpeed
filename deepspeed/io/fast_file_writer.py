# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
import time
from dataclasses import dataclass

from .constants import *
from .base_file_writer import BaseFileWriter
from .single_io_buffer import Single_IO_Buffer
from .double_io_buffer import Double_IO_Buffer
from deepspeed.ops.op_builder import UtilsBuilder
from deepspeed.accelerator import get_accelerator

from .utils import (tensor_to_bytes, bytes_to_tensor, obj_serialization_details)

FASTIO_STAT_KEYS = [
    AIO_WRITE_SEC_KEY,
    AIO_WRITE_BYTES_KEY,
    AIO_SPEED_KEY,
    SLOW_WRITE_BYTES_KEY,
    SLOW_WRITE_SEC_KEY,
    AIO_FILL_BUFFER_COUNT_KEY,
    AIO_FILL_BUFFER_SEC_KEY,
    AIO_FILL_BUFFER_SPEED_KEY,
    SAVE_STORAGE_KEY,
    SAVE_STORAGE_BYTES_KEY,
]


@dataclass
class FastFileWriterConfig:
    dnvme_handle: object
    pinned_tensor: torch.Tensor
    double_buffer: bool = True
    num_parallel_writers: int = 1
    writer_rank: int = 0
    global_rank: int = 0


class FastFileWriter(BaseFileWriter):

    def __init__(self, file_path, config):
        super(FastFileWriter, self).__init__(file_path)
        self._aio_fd = os.open(self._file_path, flags=os.O_DIRECT | os.O_CREAT | os.O_WRONLY)
        self._dnvme_handle = config.dnvme_handle
        self._file_offset = 0
        io_buffer_type = Double_IO_Buffer if config.double_buffer else Single_IO_Buffer
        self._io_buffer = io_buffer_type(config.pinned_tensor, self._dnvme_handle)
        self._cast_to_byte_tensor = UtilsBuilder().load().cast_to_byte_tensor
        self._get_serialization_details = obj_serialization_details()
        self._num_parallel_writers = config.num_parallel_writers
        self._writer_rank = config.writer_rank
        self._global_rank = config.global_rank

        for k in FASTIO_STAT_KEYS:
            self._stats[k] = 0

    def write(self, buffer):
        assert self._file_offset % self._dnvme_handle.get_alignment() == 0
        buffer_num_bytes = len(buffer)
        num_written_bytes = self._write_from_tensor(bytes_to_tensor(buffer))
        assert buffer_num_bytes == num_written_bytes
        return buffer_num_bytes

    def split_index_list(self, storage_obj_list, num_splits):
        assert num_splits > 0
        split_list = [-1] * num_splits
        # t[0] is data, t[1] is data_type
        tensor_bytes_list = [len(t[0]) for t in storage_obj_list]
        print(tensor_bytes_list)
        total_bytes = sum(tensor_bytes_list)
        bytes_per_group = total_bytes / num_splits
        split_counter = 0
        tmp_size = 0
        for i in range(len(tensor_bytes_list)):
            tmp_size += tensor_bytes_list[i]
            if tmp_size > bytes_per_group:
                split_list[split_counter] = i
                tmp_size = 0
                split_counter += 1
        if split_list[num_splits - 1] == -1:
            split_list[num_splits - 1] = len(tensor_bytes_list)
        return split_list

    def save_torch_storage_object_list(self, storage_obj_list, save_size):
        assert self._file_offset % self._dnvme_handle.get_alignment() == 0
        num_bytes_written = self._save_storage_list(storage_obj_list, save_size)
        return num_bytes_written

    def close(self):
        self._fini()
        self._incr_stats(CLOSE_COUNT_KEY)

    def fileno(self):
        self._incr_stats(FILENO_COUNT_KEY)
        return INVALID_FD  # self._aio_fd

    def flush(self):
        self._incr_stats(FLUSH_COUNT_KEY)

    def __del__(self):
        self._fini()
        assert self._aio_fd == INVALID_FD
        assert self._io_buffer.get_offset() == 0, \
            f'__del__ assert: pinned_offset {self._io_buffer.get_offset()} != 0'
        assert self._file_offset == self._stats[WRITE_BYTES_KEY], \
            f'__del__ assert: file_offset != write_bytes - {self._file_offset} != {self._stats[WRITE_BYTES_KEY]}'

    def _fini(self):
        if not self._io_buffer_is_empty():
            self._force_drain()
        self._io_buffer.reset()
        self._aio_fd = INVALID_FD

    def _fill_io_buffer(self, src_tensor, src_offset):
        st = time.time()
        copy_bytes = self._io_buffer.fill(src_tensor, src_offset)
        self._incr_stats(AIO_FILL_BUFFER_SEC_KEY, time.time() - st)
        self._incr_stats(AIO_FILL_BUFFER_COUNT_KEY)
        return copy_bytes

    def _drain_io_buffer(self, num_bytes):
        st = time.time()
        self._io_buffer.drain(num_bytes, self._aio_fd, self._file_offset)
        self._incr_stats(AIO_WRITE_SEC_KEY, time.time() - st)
        self._incr_stats(AIO_WRITE_BYTES_KEY, num_bytes)
        self._file_offset += num_bytes

    def _io_buffer_is_full(self):
        return self._io_buffer.is_full()

    def _io_buffer_is_empty(self):
        return self._io_buffer.is_empty()

    def _force_drain(self):
        st = time.time()
        aligned_num_bytes = self._io_buffer.get_aligned_num_bytes()
        # Important to retrieve unaligned drain bytes and tensor before doing aligned drain because of the side effects.
        # TODO: Need to eliminate this dependency
        unaligned_num_bytes = self._io_buffer.get_unaligned_num_bytes()
        unaligned_tensor = torch.narrow(self._io_buffer.get_buffer(), 0, aligned_num_bytes, unaligned_num_bytes)

        if aligned_num_bytes > 0:
            self._drain_io_buffer(aligned_num_bytes)

        self._io_buffer.complete_ongoing_drain()
        self._incr_stats(AIO_WRITE_SEC_KEY, time.time() - st)

        if unaligned_num_bytes > 0:
            self._unaligned_drain(unaligned_tensor)
        self._incr_stats(WRITE_SEC_KEY, time.time() - st)

    def _unaligned_drain(self, unaligned_tensor):
        os.close(self._aio_fd)
        st = time.time()
        fp = open(self._file_path, 'ab')
        fp.write(tensor_to_bytes(unaligned_tensor.cpu()))
        fp.close()
        self._file_offset += unaligned_tensor.numel()
        self._incr_stats(SLOW_WRITE_SEC_KEY, time.time() - st)
        self._incr_stats(SLOW_WRITE_BYTES_KEY, unaligned_tensor.numel())
        self._aio_fd = os.open(self._file_path, flags=os.O_DIRECT | os.O_WRONLY | os.O_APPEND)

    def _dump_state(self):
        if self._stats[AIO_WRITE_SEC_KEY] > 0:
            self._stats[AIO_SPEED_KEY] = (self._stats[AIO_WRITE_BYTES_KEY] / self._stats[AIO_WRITE_SEC_KEY] /
                                          (1024**3))
        if self._stats[AIO_FILL_BUFFER_SEC_KEY] > 0:
            self._stats[AIO_FILL_BUFFER_SPEED_KEY] = (self._stats[AIO_WRITE_BYTES_KEY] /
                                                      self._stats[AIO_FILL_BUFFER_SEC_KEY] / (1024**3))
        super()._dump_state()

    def _update_write_stats(self, num_bytes, secs_latency):
        self._incr_stats(WRITE_COUNT_KEY)
        self._incr_stats(WRITE_BYTES_KEY, num_bytes)
        self._incr_stats(WRITE_SEC_KEY, secs_latency)

    def _write_from_tensor(self, buffer_tensor):
        st = time.time()
        buffer_offset = 0
        while (buffer_offset < buffer_tensor.numel()):
            num_copied_bytes = self._fill_io_buffer(buffer_tensor, buffer_offset)
            if self._io_buffer_is_full():
                self._drain_io_buffer(self._io_buffer.get_offset())
            buffer_offset += num_copied_bytes

        self._update_write_stats(buffer_offset, time.time() - st)

        return buffer_offset

    def _save_storage_list(self, obj_list, save_size):
        byte_tensor_list, byte_tensor_nbytes = self._convert_to_byte_tensors(obj_list, save_size)
        if self._num_parallel_writers > 1:
            my_byte_tensor_list = self._partition_byte_tensors(byte_tensor_list, byte_tensor_nbytes,
                                                               self._num_parallel_writers, self._writer_rank)
        else:
            my_byte_tensor_list = byte_tensor_list

        num_object_bytes_written = 0
        for byte_tensor in my_byte_tensor_list:
            num_object_bytes_written += self._write_from_tensor(byte_tensor)

        self._incr_stats(SAVE_STORAGE_KEY, len(obj_list))
        self._incr_stats(SAVE_STORAGE_BYTES_KEY, num_object_bytes_written)
        return num_object_bytes_written

    # Convert list of storage objects into list of byte tensors of object and size bytes
    def _convert_to_byte_tensors(self, obj_list, save_size):
        tensor_list = []
        num_bytes = 0
        for storage_obj in obj_list:
            details = self._get_serialization_details(storage_obj)
            if save_size:
                tensor_list.append(
                    torch.tensor(
                        details.size,
                        dtype=torch.int64,
                    ).to(get_accelerator().device_name()))
            tensor_list.append(torch.empty(0, dtype=details.dtype, device=details.obj.device).set_(details.obj))
            num_bytes += details.nbytes
        if save_size:
            num_bytes += STORAGE_OBJ_SIZE * len(obj_list)

        return self._cast_to_byte_tensor(tensor_list), num_bytes

    def _partition_byte_tensors(self, byte_tensor_list, byte_tensor_nbytes, num_ranks, my_rank):
        assert my_rank >= 0, f'Invalid for rank number to be negative: {my_rank}'
        assert num_ranks > my_rank, f'Number of ranks {num_ranks} must be greater than rank {my_rank}'

        partition_size = int(byte_tensor_nbytes // num_ranks)
        num_remainder_bytes = byte_tensor_nbytes % num_ranks
        if num_remainder_bytes == 0:
            partition_start = partition_size * my_rank
        else:
            # Spread extra bytes evenly among early ranks
            if num_remainder_bytes > my_rank:
                partition_size += 1
                partition_start = partition_size * my_rank
            else:
                # Account for allocation of extra bytes to earlier ranks
                partition_start = (partition_size * my_rank) + num_remainder_bytes

        partition_end = min(partition_start + partition_size, byte_tensor_nbytes)
        partition_tensor_list = []
        current_offset = 0
        for byte_tensor in byte_tensor_list:
            byte_tensor_end = current_offset + byte_tensor.numel()
            if current_offset < partition_end and byte_tensor_end > partition_start:
                fragment_start = max(current_offset, partition_start)
                fragment_end = min(byte_tensor_end, partition_end)
                assert fragment_start < fragment_end, \
                    f'fragment start {fragment_start} should be < fragment_end {fragment_end}'

                fragment_numel = fragment_end - fragment_start
                partition_tensor_list.append(byte_tensor.narrow(0, fragment_start - current_offset, fragment_numel))

            current_offset += byte_tensor.numel()

        actual_partition_nbytes = sum([t.numel() for t in partition_tensor_list])
        assert actual_partition_nbytes == partition_size, \
        f'Incorrect partition bytes for rank {my_rank}, expected = {partition_size} actual = {actual_partition_nbytes}'

        return partition_tensor_list

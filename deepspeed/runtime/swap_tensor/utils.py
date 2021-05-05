"""
Copyright 2020 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality of swapping tensors to/from (NVMe) storage devices.
"""

import os
import torch
from deepspeed.utils.logging import logger

from deepspeed.runtime.swap_tensor.constants import AIO_BLOCK_SIZE, AIO_QUEUE_DEPTH, \
    AIO_THREAD_COUNT, AIO_SINGLE_SUBMIT, AIO_OVERLAP_EVENTS

MIN_AIO_BYTES = 1024**2
AIO_ALIGNED_BYTES = 1024


def swap_in_tensors(swap_handle, tensor_buffers, swap_paths):
    for buffer, path in zip(tensor_buffers, swap_paths):
        assert (swap_handle.async_pread(buffer, path) == 0)


def swap_out_tensors(swap_handle, tensor_buffers, swap_paths):
    for buffer, path in zip(tensor_buffers, swap_paths):
        assert (swap_handle.async_pwrite(buffer, path) == 0)


def print_object(obj, name, exclude_list=[]):
    logger.info('{}:'.format(name))
    for arg in sorted(vars(obj)):
        if not arg in exclude_list:
            dots = '.' * (29 - len(arg))
            logger.info('  {} {} {}'.format(arg, dots, getattr(obj, arg)))


class SwapBuffer(object):
    def __init__(self, buffer):
        self.buffer = buffer
        self.reset()

    def reset(self):
        self.offset = 0
        self.swap_tensors = {}
        self.compute_tensors = {}
        self.swap_paths = {}
        self.num_elem = 0

    def insert_tensor(self, tensor, swap_path, aligned_numel):
        swap_tensor, compute_tensor = self.allocate_tensor(swap_path, tensor.numel(), aligned_numel)
        compute_tensor.data.copy_(tensor.data)
        return swap_tensor, compute_tensor

    def allocate_tensor(self, swap_path, numel, aligned_numel):
        assert self.has_space(aligned_numel)
        assert not self.offset in self.swap_tensors

        allocate_offset = self.offset
        swap_tensor = self.buffer.narrow(0, allocate_offset, aligned_numel)
        dest_tensor = swap_tensor.narrow(0, 0, numel)

        self.swap_tensors[allocate_offset] = swap_tensor
        self.compute_tensors[allocate_offset] = dest_tensor
        self.swap_paths[allocate_offset] = swap_path
        self.offset += aligned_numel
        self.num_elem += numel

        return self.swap_tensors[allocate_offset], self.compute_tensors[allocate_offset]

    def has_space(self, numel):
        return (self.offset + numel) <= self.buffer.numel()

    def get_swap_tensors(self):
        return [tensor for tensor in self.swap_tensors.values()]

    def get_swap_paths(self):
        return [path for path in self.swap_paths.values()]

    def get_compute_tensors(self):
        return [tensor for tensor in self.compute_tensors.values()]

    def get_num_elem(self):
        return self.num_elem

    def get_swap_tensor(self, offset):
        return self.swap_tensors.get(offset, None)

    def get_compute_tensor(self, offset):
        return self.compute_tensors.get(offset, None)

    def get_swap_path(self, offset):
        return self.swap_paths(offset, None)


class SwapBufferPool(object):
    def __init__(self, buffers):
        assert all([buf.is_pinned() for buf in buffers])
        self.buffers = [SwapBuffer(buf) for buf in buffers]
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        for buffer in self.buffers:
            buffer.reset()

    def allocate_tensor(self, numel, swap_path, aligned_numel):
        if self.has_space(aligned_numel):
            swap_tensor, compute_tensor = self._get_current_buffer().allocate_tensor(swap_path, numel, aligned_numel)
            return swap_tensor, compute_tensor

        return None, None

    def insert_tensor(self, tensor, swap_path, aligned_numel):
        if self.has_space(aligned_numel):
            swap_tensor, compute_tensor = self._get_current_buffer().insert_tensor(tensor, swap_path, aligned_numel)
            return swap_tensor, compute_tensor

        return None, None

    def get_swap_tensors(self):
        swap_tensors = []
        for buffer in self._get_used_buffers():
            swap_tensors += buffer.get_swap_tensors()

        return swap_tensors

    def get_swap_paths(self):
        swap_paths = []
        for buffer in self._get_used_buffers():
            swap_paths += buffer.get_swap_paths()

        return swap_paths

    def get_compute_tensors(self):
        compute_tensors = []
        for buffer in self._get_used_buffers():
            compute_tensors += buffer.get_compute_tensors()

        return compute_tensors

    def has_space(self, numel):
        if self._get_current_buffer().has_space(numel):
            return True

        if self.current_index == len(self.buffers) - 1:
            return False

        self.current_index += 1
        return self._get_current_buffer().has_space(numel)

    def swap_out(self, aio_handle, async_op=False):
        swap_tensors = self.get_swap_tensors()
        swap_paths = self.get_swap_paths()
        assert all([p is not None for p in swap_paths])

        swap_out_tensors(aio_handle, swap_tensors, swap_paths)

        if not async_op:
            assert len(swap_tensors) == aio_handle.wait()

    def swap_in(self, aio_handle, async_op=False):
        swap_tensors = self.get_swap_tensors()
        swap_paths = self.get_swap_paths()
        assert all([p is not None for p in swap_paths])

        swap_in_tensors(aio_handle, swap_tensors, swap_paths)

        if not async_op:
            assert len(swap_tensors) == aio_handle.wait()

    def _get_current_buffer(self):
        return self.buffers[self.current_index]

    def _get_used_buffers(self):
        return self.buffers[:self.current_index + 1]


class SwapBufferManager(object):
    def __init__(self, num_elems, count, dtype):
        self.num_elems = num_elems
        self.count = count
        self.dtype = dtype
        self.all_buffers = [
            torch.zeros(num_elems,
                        device='cpu',
                        dtype=dtype).pin_memory() for _ in range(count)
        ]
        self.free_buffer_index = [i for i in range(count)]
        self.used_buffer_index = {}
        self.gigabytes = (self.all_buffers[0].element_size() * num_elems * count) / (1024
                                                                                     **3)

        if torch.distributed.get_rank() == 0:
            exclude_list = ['all_buffers']
            print_object(obj=self, name='SwapBufferManager', exclude_list=exclude_list)

    def allocate(self, num_elems, count, dtype):
        assert dtype == self.dtype
        assert num_elems <= self.num_elems
        if count > len(self.free_buffer_index):
            return None

        used_indices = self.free_buffer_index[-count:]
        self.free_buffer_index = self.free_buffer_index[:-count]

        buffers = []
        for i in used_indices:
            tmp_buffer = self.all_buffers[i].narrow(0, 0, num_elems)
            buffers.append(tmp_buffer)
            self.used_buffer_index[id(tmp_buffer)] = i
        return buffers

    def allocate_all(self, num_elems, dtype):
        return self.allocate(num_elems=num_elems,
                             count=len(self.free_buffer_index),
                             dtype=dtype)

    def free(self, buffers):
        buffer_ids = []
        for buf in buffers:
            buffer_ids.append(id(buf))

        assert all([b_id in self.used_buffer_index for b_id in buffer_ids])

        for b_id in buffer_ids:
            self.free_buffer_index.append(self.used_buffer_index[b_id])
            del (self.used_buffer_index[b_id])


def get_sized_buffer(buffer, num_elems):
    assert num_elems <= buffer.numel(), \
        f'num_elems {num_elems} > buffer {buffer.numel()}'
    return buffer.narrow(0, 0, num_elems) if num_elems < buffer.numel() else buffer


def get_sized_buffers(buffer_list, num_elems_list):
    swap_buffers = [
        get_sized_buffer(buffer, num_elems) \
        for buffer, num_elems in zip(buffer_list, num_elems_list)
    ]
    return swap_buffers

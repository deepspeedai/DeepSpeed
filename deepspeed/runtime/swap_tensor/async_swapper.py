"""
Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.

Functionality of swapping tensors to/from (NVMe) storage devices.
"""
import torch

from deepspeed.utils.logging import logger
from deepspeed.runtime.swap_tensor.utils import swap_out_tensors, SwapBuffer

INVALID_BUFFER_INDEX = -1
ASYNC_SWAPPER_WAIT_TIMER = 'async_swap_gradient_wait'


class AsyncTensorSwapper(object):
    def __init__(self, aio_handle, numel_alignment, timers):
        self.free_buffer_index = []
        self.swapping_buffer_index = []
        self.ready_buffer_index = []
        self.current_buffer_index = INVALID_BUFFER_INDEX
        self.all_buffers = []
        self.aio_handle = aio_handle
        self.numel_alignment = numel_alignment
        self.max_numel = 0
        self.num_pending_swaps = 0
        self.timers = timers
        self.timer_names = set()
        self.num_elements_swapped = 0
        self.dtype = None

    def has_buffers(self):
        return len(self.all_buffers) > 0

    def add_buffers(self, buffer_list):
        assert len(self.all_buffers) == 0
        assert all([buffer.is_pinned() for buffer in buffer_list])
        dtype = buffer_list[0].dtype
        assert all([buffer.dtype == dtype for buffer in buffer_list])

        self.dtype = dtype
        self.all_buffers = [SwapBuffer(buffer) for buffer in buffer_list]
        self.free_buffer_index += [i for i in range(len(self.all_buffers))]
        self.max_numel = max([buffer.numel() for buffer in buffer_list])
        self.timer_names = set()

    def get_timer_names(self):
        return list(self.timer_names)

    def release_buffers(self):
        self._report_statistics('Swapped out[Before flush]')
        self._flush_buffers_until_complete()
        self._report_statistics('Swapped out[After flush]')

        pinned_buffers = [buf.buffer for buf in self.all_buffers]
        self.all_buffers = []
        self.free_buffer_index = []
        self.current_buffer_index = INVALID_BUFFER_INDEX
        self.num_elements_swapped = 0
        self.dtype = None

        return pinned_buffers

    def swap_out_tensors(self, tensor_list, path_list):
        for tensor, swap_path in zip(tensor_list, path_list):
            self._swap_out_tensor(tensor, swap_path)

    def _report_statistics(self, message):
        if torch.distributed.get_rank() == 0:
            element_size = torch.tensor([], dtype=self.dtype).element_size()
            swapped_GB = (self.num_elements_swapped * element_size) / (1024**3)
            logger.info(
                f'{message} num_elems = {self.num_elements_swapped}, {swapped_GB:5.2f} GB'
            )

    def _swap_out_tensor(self, tensor, swap_path):
        assert len(self.all_buffers) > 0

        aligned_numel = self._io_aligned_numel(tensor.numel())
        assert aligned_numel <= self.max_numel

        self._make_swap_space(aligned_numel)
        assert self.current_buffer_index != INVALID_BUFFER_INDEX

        swap_buffer = self._get_current_buffer()
        swap_buffer.insert_tensor(tensor, swap_path, aligned_numel)

    def _make_swap_space(self, numel):
        if self.current_buffer_index == INVALID_BUFFER_INDEX:
            self._allocate_buffer()
            return

        if not self._get_current_buffer().has_space(numel):
            if len(self.free_buffer_index) > 0:
                self._flush_ready_buffers()
            else:
                self._flush_buffers_until_complete()
            self._allocate_buffer()

    def _io_aligned_numel(self, numel):
        remainder = numel % self.numel_alignment
        return numel if remainder == 0 else (numel + self.numel_alignment - remainder)

    def _allocate_buffer(self):
        assert self.current_buffer_index == INVALID_BUFFER_INDEX
        assert len(self.all_buffers) > 0
        assert len(self.free_buffer_index) > 0
        self.current_buffer_index = self.free_buffer_index[-1]
        self.free_buffer_index = self.free_buffer_index[:-1]

    def _flush_ready_buffers(self):
        if self.current_buffer_index != INVALID_BUFFER_INDEX:
            self.ready_buffer_index.append(self.current_buffer_index)
            self.current_buffer_index = INVALID_BUFFER_INDEX

        self._swap_out_ready_buffers()

    def _flush_buffers_until_complete(self):
        self._flush_ready_buffers()
        assert len(self.ready_buffer_index) == 0

        self._wait_for_swap_complete()
        assert len(self.swapping_buffer_index) == 0
        assert len(self.free_buffer_index) == len(self.all_buffers)

    def _swap_out_ready_buffers(self):
        for buffer_index in self.ready_buffer_index:
            buffer = self._get_buffer(buffer_index)
            swap_tensors = buffer.get_swap_tensors()
            swap_paths = buffer.get_swap_paths()
            self.num_pending_swaps += len(swap_tensors)
            swap_out_tensors(self.aio_handle, swap_tensors, swap_paths)

        self.swapping_buffer_index += self.ready_buffer_index
        self.ready_buffer_index = []

    def _wait_for_swap_complete(self):
        assert len(self.swapping_buffer_index) > 0

        self._start_timer(ASYNC_SWAPPER_WAIT_TIMER)
        assert self.aio_handle.wait() == self.num_pending_swaps
        self._stop_timer(ASYNC_SWAPPER_WAIT_TIMER)
        self.timer_names.add(ASYNC_SWAPPER_WAIT_TIMER)

        self.num_pending_swaps = 0

        for buffer_index in self.swapping_buffer_index:
            buffer = self._get_buffer(buffer_index)
            self.num_elements_swapped += buffer.get_num_elem()
            buffer.reset()

        self.free_buffer_index += self.swapping_buffer_index
        assert len(self.free_buffer_index) <= len(self.all_buffers)
        self.swapping_buffer_index = []

    def _get_buffer(self, index):
        assert index != INVALID_BUFFER_INDEX
        return self.all_buffers[index]

    def _get_current_buffer(self):
        return self._get_buffer(self.current_buffer_index)

    def _start_timer(self, name):
        if self.timers:
            self.timers(name).start()

    def _stop_timer(self, name):
        if self.timers:
            self.timers(name).stop()

    def _log_timers(self, name_list, force=False):
        if self.timers and force:
            self.timers.log(name_list)

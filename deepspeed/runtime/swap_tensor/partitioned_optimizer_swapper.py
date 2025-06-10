# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

from deepspeed.utils.logging import logger
from deepspeed.ops.op_builder import AsyncIOBuilder
from deepspeed import comm as dist

from deepspeed.runtime.swap_tensor.constants import *
from deepspeed.runtime.swap_tensor.utils import swap_in_tensors, swap_out_tensors, print_object, \
    get_sized_buffers
from deepspeed.runtime.swap_tensor.async_swapper import AsyncTensorSwapper
from deepspeed.runtime.swap_tensor.optimizer_utils import OptimizerSwapper
from deepspeed.accelerator import get_accelerator

DEBUG_MODE = False

SWAP_IN_PARAM_TIMER = 'swap_in_param'
SWAP_OUT_PARAM_TIMER = 'swap_out_param'
SWAP_IN_GRADIENT_TIMER = 'swap_in_gradient'


class PartitionedOptimizerSwapper(OptimizerSwapper):

    def __init__(self, swap_config, aio_config, base_folder, optimizer, largest_numel, device, dtype, timers):
        super(PartitionedOptimizerSwapper, self).__init__(swap_config, aio_config, base_folder, optimizer,
                                                          largest_numel, device, dtype, timers)

        aio_op = AsyncIOBuilder().load()
        self.aio_handle = aio_op.aio_handle(block_size=aio_config[AIO_BLOCK_SIZE],
                                            queue_depth=aio_config[AIO_QUEUE_DEPTH],
                                            single_submit=aio_config[AIO_SINGLE_SUBMIT],
                                            overlap_events=aio_config[AIO_OVERLAP_EVENTS],
                                            intra_op_parallelism=aio_config[AIO_INTRA_OP_PARALLELISM])

        # Overlap swapping out
        self.gradient_swapper = AsyncTensorSwapper(aio_handle=self.aio_handle,
                                                   numel_alignment=self.numel_alignment,
                                                   timers=self.timers)

        self.print_exclude_list += ['aio_handle', 'gradient_swapper', 'print_exclude_list']

        if dist.get_rank() == 0:
            print_object(obj=self, name='PartitionedOptimizerSwapper', exclude_list=self.print_exclude_list)

    def initialize_parameters(self, parameters, src_tensors):
        self._initialize_parameters(parameters=parameters, src_tensors=src_tensors, aio_handle=self.aio_handle)

    def initialize_from_swapped_fp16_params(self, fp16_partitions_info, fp16_num_elems, fp16_pinned_buffers,
                                            fp32_parameters):
        self._initialize_from_swapped_fp16_params(aio_handle=self.aio_handle,
                                                  fp16_partitions_info=fp16_partitions_info,
                                                  fp16_num_elems=fp16_num_elems,
                                                  fp16_pinned_buffers=fp16_pinned_buffers,
                                                  fp32_parameters=fp32_parameters)

    def flush_gradients(self):
        self._flush_gradient_swapper(self.gradient_swapper)

    def release_swap_buffers(self, parameter):
        swap_info = self._get_param_swap_info(parameter)
        if swap_info is None:
            return
        swap_info.release_memory()

        self.swap_buffer_manager.free(swap_info.swap_buffers)
        swap_info.swap_buffers = []

    def swap_in_optimizer_state(self, parameter, async_parameter=None):
        swap_info = self._get_param_swap_info(parameter)
        if swap_info is None:
            return

        self._flush_gradient_swapper(self.gradient_swapper)

        required_buffer_count = swap_info.num_tensors() + (1 if swap_info.has_gradients() else 0)
        aligned_numel = self._io_aligned_numel(swap_info.numel())
        pinned_buffers = self.swap_buffer_manager.allocate(num_elems=aligned_numel,
                                                           count=required_buffer_count,
                                                           dtype=parameter.dtype)
        assert pinned_buffers is not None
        swap_info.swap_buffers = pinned_buffers.copy()

        self._start_timer(SWAP_IN_PARAM_TIMER)
        self._swap_in_parameter(aio_handle=self.aio_handle,
                                parameter=parameter,
                                dest_buffers=pinned_buffers[:swap_info.num_tensors()])
        self._stop_timer(SWAP_IN_PARAM_TIMER)
        self.timer_names.add(SWAP_IN_PARAM_TIMER)

        if swap_info.has_gradients():
            self._start_timer(SWAP_IN_GRADIENT_TIMER)
            self._swap_in_gradients(aio_handle=self.aio_handle, parameter=parameter, dest_buffer=pinned_buffers[-1])
            self._stop_timer(SWAP_IN_GRADIENT_TIMER)
            self.timer_names.add(SWAP_IN_GRADIENT_TIMER)

    def _swap_out_optimizer_state(self, swap_info):
        pinned_tensors, pinned_paths = swap_info.get_swap_buffers_and_paths(True)
        WRITE_TIMER = 'swap_submit_write'
        self._start_timer(WRITE_TIMER)

        swap_out_tensors(self.aio_handle, pinned_tensors, pinned_paths)
        assert self.aio_handle.wait() == len(pinned_tensors)

        unpinned_tensors, unpinned_paths = swap_info.get_swap_buffers_and_paths(False)
        if len(unpinned_tensors) > 0:
            pinned_buffers = self.swap_buffer_manager.allocate_all(num_elems=self.largest_numel, dtype=self.dtype)
            self._swap_out_unpinned_tensors(aio_handle=self.aio_handle,
                                            unpinned_tensors=unpinned_tensors,
                                            dest_paths=unpinned_paths,
                                            pinned_buffers=pinned_buffers)
            swap_info.swap_buffers += pinned_buffers.copy()

        self._stop_timer(WRITE_TIMER)
        self._log_timers([WRITE_TIMER])

    def writeback_optimizer_state_and_gradients(self, parameter, write_opt_state, write_gradients):
        swap_info = self._get_param_swap_info(parameter=parameter)

        if swap_info is None:
            return

        if write_opt_state:
            self._swap_out_optimizer_state(swap_info)

        if write_gradients and swap_info.has_gradients():
            param_gradients = swap_info.swapped_gradients.values()
            swap_buffers = [parameter.grad.narrow(0, grad.offset, grad.length) for grad in param_gradients]
            swap_paths = [grad.path for grad in param_gradients]
            swap_out_tensors(self.aio_handle, swap_buffers, swap_paths)
            assert len(swap_buffers) == self.aio_handle.wait()
            if swap_info.unswapped_gradients:
                swap_info.write_unswapped_gradients(src_buffer=parameter.grad)

        self.release_swap_buffers(parameter)

    def swap_out_optimizer_state(self, parameter, async_swap=False):
        swap_info = self._get_param_swap_info(parameter=parameter)

        if swap_info is None:
            return

        swap_bytes = sum(
            [self._io_aligned_numel(t.numel()) * t.element_size() for t in swap_info.get_compute_tensors()])

        self._start_timer(SWAP_OUT_PARAM_TIMER)
        self._swap_out_optimizer_state(swap_info)
        self.release_swap_buffers(parameter)
        self._stop_timer(SWAP_OUT_PARAM_TIMER)
        self.timer_names.add(SWAP_OUT_PARAM_TIMER)

        if DEBUG_MODE and dist.get_rank() == 0:
            logger.info(f'optimizer_param_swap_out: {(swap_bytes/(1024**3)):5.2f} GB')

    def swap_out_gradients(self, parameter, gradient_offsets, gradient_tensors):
        self._swap_out_gradients(parameter=parameter,
                                 gradient_offsets=gradient_offsets,
                                 gradient_tensors=gradient_tensors,
                                 gradient_swapper=self.gradient_swapper)

    def _swap_in_parameter(self, aio_handle, parameter, dest_buffers):
        swap_info = self._get_param_swap_info(parameter)
        if swap_info is None:
            return

        num_swap_tensors = swap_info.num_tensors()
        assert num_swap_tensors <= len(dest_buffers)

        swap_lengths = [self._io_aligned_numel(swap_info.numel())] * num_swap_tensors
        swap_buffers = get_sized_buffers(dest_buffers, swap_lengths)

        compute_lengths = [swap_info.numel()] * num_swap_tensors
        compute_buffers = get_sized_buffers(dest_buffers, compute_lengths)

        READ_TIMER = 'swap_submit_read_param'
        WAIT_TIMER = 'swap_wait_read_param'

        self._start_timer(READ_TIMER)
        swap_in_tensors(aio_handle, swap_buffers, swap_info.get_swap_paths())
        self._stop_timer(READ_TIMER)

        swap_bytes = sum([buffer.numel() * buffer.element_size() for buffer in swap_buffers])

        self._start_timer(WAIT_TIMER)
        aio_handle.wait()
        self._stop_timer(WAIT_TIMER)

        swap_info.set_swap_buffers(dest_buffers, self._io_aligned_numel(swap_info.numel()))

        self._log_timers([READ_TIMER, WAIT_TIMER])
        if DEBUG_MODE and dist.get_rank() == 0:
            logger.info(f'optimizer_param_swap_in: {(swap_bytes/(1024**3)):5.2f} GB')

    def _swap_in_pinned_gradients(self, aio_handle, parameter, gradient_tensor):
        swap_info = self.swap_params_info[OptimizerSwapper.parameter_id(parameter)]
        param_gradients = swap_info.swapped_gradients.values()
        swap_buffers = [gradient_tensor.narrow(0, grad.offset, grad.length) for grad in param_gradients]
        swap_paths = [grad.path for grad in param_gradients]
        SWAP_READ_GRADIENTS = 'swap_submit_read_gradient'
        SWAP_WAIT_GRADIENTS = 'swap_submit_wait_gradient'
        self._start_timer(SWAP_READ_GRADIENTS)
        swap_in_tensors(aio_handle, swap_buffers, swap_paths)
        self._stop_timer(SWAP_READ_GRADIENTS)

        self._start_timer(SWAP_WAIT_GRADIENTS)
        assert len(swap_buffers) == aio_handle.wait()
        self._stop_timer(SWAP_WAIT_GRADIENTS)

        self._log_timers([SWAP_READ_GRADIENTS, SWAP_WAIT_GRADIENTS])

    def _swap_in_gradients(self, aio_handle, parameter, dest_buffer):
        swap_info = self.swap_params_info.get(OptimizerSwapper.parameter_id(parameter), None)
        if not (swap_info and swap_info.has_gradients()):
            return

        assert get_accelerator().is_pinned(dest_buffer)
        assert parameter.numel() <= dest_buffer.numel()

        parameter.grad = dest_buffer.narrow(0, 0, parameter.numel())

        if swap_info.swapped_gradients:
            self._swap_in_pinned_gradients(aio_handle, parameter, parameter.grad)

        if swap_info.unswapped_gradients:
            self._retrieve_unswapped_grad_partitions(swap_info=swap_info, dest_buffer=parameter.grad)

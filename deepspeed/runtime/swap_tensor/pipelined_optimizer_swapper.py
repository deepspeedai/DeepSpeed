"""
Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.

Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import os
import torch

from deepspeed.utils.logging import logger
from deepspeed.ops.aio import AsyncIOBuilder

from deepspeed.runtime.zero.offload_constants import *
from deepspeed.runtime.swap_tensor.constants import *
from deepspeed.runtime.swap_tensor.utils import swap_in_tensors, swap_out_tensors, print_object, \
    MIN_AIO_BYTES, AIO_ALIGNED_BYTES
from deepspeed.runtime.swap_tensor.async_swapper import AsyncTensorSwapper
from deepspeed.runtime.swap_tensor.optimizer_utils import SwapBufferManager, get_sized_buffer
from deepspeed.runtime.swap_tensor.optimizer_utils import OptimizerSwapper


class OptimizerSwapOp(object):
    def __init__(self,
                 aio_handle,
                 read_op,
                 param_info,
                 allocated_buffers,
                 state_buffers,
                 num_ops):
        self.aio_handle = aio_handle
        self.read_op = read_op
        self.param_info = param_info
        self.allocated_buffers = allocated_buffers
        self.state_buffers = state_buffers
        self.wait_required = True
        self.num_ops = num_ops

    def is_parameter(self, parameter):
        return id(parameter) == self.param_info.param_id

    def wait(self):
        assert self.wait_required
        assert self.aio_handle.wait() == self.num_ops
        self.wait_required = False


SYNC_SWAP_IN = 'sync_swap_in'
ASYNC_SWAP_IN = 'async_swap_in'
SYNC_SWAP_OUT = 'sync_swap_out'
ASYNC_SWAP_OUT = 'async_swap_out'

SWAP_IN_STATE_TIMER = 'swap_in_state'
SWAP_OUT_STATE_TIMER = 'swap_out_state'
SWAP_OUT_GRADIENT_TIMER = 'swap_out_gradient'
ASYNC_SWAP_IN_STATE_TIMER = "async_swap_in_state"
ASYNC_SWAP_OUT_STATE_TIMER = 'async_swap_out_state'


class PipelinedOptimizerSwapper(OptimizerSwapper):
    def __init__(self,
                 swap_config,
                 aio_config,
                 base_folder,
                 optimizer,
                 largest_numel,
                 device,
                 dtype,
                 timers):
        super(PipelinedOptimizerSwapper,
              self).__init__(swap_config,
                             aio_config,
                             base_folder,
                             optimizer,
                             largest_numel,
                             device,
                             dtype,
                             timers)

        aio_op = AsyncIOBuilder().load()
        self.write_aio_handle = aio_op.aio_handle(aio_config[AIO_BLOCK_SIZE],
                                                  aio_config[AIO_QUEUE_DEPTH],
                                                  aio_config[AIO_SINGLE_SUBMIT],
                                                  aio_config[AIO_OVERLAP_EVENTS],
                                                  aio_config[AIO_THREAD_COUNT])

        self.read_aio_handle = aio_op.aio_handle(aio_config[AIO_BLOCK_SIZE],
                                                 aio_config[AIO_QUEUE_DEPTH],
                                                 aio_config[AIO_SINGLE_SUBMIT],
                                                 aio_config[AIO_OVERLAP_EVENTS],
                                                 aio_config[AIO_THREAD_COUNT])

        # Overlap gradient swap out
        self.gradient_swapper = AsyncTensorSwapper(aio_handle=self.write_aio_handle,
                                                   numel_alignment=self.numel_alignment,
                                                   timers=self.timers)

        self.async_swap_in = swap_config[OFFLOAD_OPTIMIZER_PIPELINE_READ]
        self.async_swap_out = swap_config[OFFLOAD_OPTIMIZER_PIPELINE_WRITE]

        self.swap_ops = {
            SYNC_SWAP_IN: None,
            ASYNC_SWAP_IN: None,
            SYNC_SWAP_OUT: None,
            ASYNC_SWAP_OUT: None
        }

        self.print_exclude_list += [
            'gradient_swapper',
            'read_aio_handle',
            'write_aio_handle',
            'swap_ops',
            'print_exclude_list'
        ]

        if torch.distributed.get_rank() == 0:
            print_object(obj=self,
                         name='PipelinedOptimizerSwapper',
                         exclude_list=self.print_exclude_list)

    def initialize_parameters(self, parameters, src_tensors):
        self._initialize_parameters(parameters=parameters,
                                    src_tensors=src_tensors,
                                    aio_handle=self.write_aio_handle)

    def initialize_from_swapped_fp16_params(self,
                                            fp16_partitions_info,
                                            fp16_num_elems,
                                            fp16_pinned_buffers,
                                            fp32_parameters):
        self._initialize_from_swapped_fp16_params(
            aio_handle=self.write_aio_handle,
            fp16_partitions_info=fp16_partitions_info,
            fp16_num_elems=fp16_num_elems,
            fp16_pinned_buffers=fp16_pinned_buffers,
            fp32_parameters=fp32_parameters)

    def flush_gradients(self):
        self._flush_gradient_swapper(self.gradient_swapper)

    def swap_in_optimizer_state(self, parameter, async_parameter):
        assert parameter is not None
        assert self.swap_ops[SYNC_SWAP_IN] is None

        self._flush_gradient_swapper(self.gradient_swapper)

        self._start_timer(SWAP_IN_STATE_TIMER)

        if self.swap_ops[ASYNC_SWAP_IN]:
            assert self.swap_ops[ASYNC_SWAP_IN].is_parameter(parameter)
            self.swap_ops[SYNC_SWAP_IN] = self.swap_ops[ASYNC_SWAP_IN]
            self.swap_ops[ASYNC_SWAP_IN] = None
        else:
            self.swap_ops[SYNC_SWAP_IN] = self._swap_in_optimizer_state(
                aio_handle=self.read_aio_handle,
                parameter=parameter)

        if self.swap_ops[SYNC_SWAP_IN]:
            self.swap_ops[SYNC_SWAP_IN].wait()

        if self.async_swap_in and async_parameter is not None:
            assert self.swap_ops[ASYNC_SWAP_IN] is None
            self.swap_ops[ASYNC_SWAP_IN] = self._swap_in_optimizer_state(
                aio_handle=self.read_aio_handle,
                parameter=async_parameter)

        self._stop_timer(SWAP_IN_STATE_TIMER)
        self.timer_names.add(SWAP_IN_STATE_TIMER)

    def swap_out_optimizer_state(self, parameter, async_swap):
        self._start_timer(SWAP_OUT_STATE_TIMER)

        if self.swap_ops[ASYNC_SWAP_OUT]:
            self._start_timer(ASYNC_SWAP_OUT_STATE_TIMER)
            self._complete_swap_out(ASYNC_SWAP_OUT)
            self._stop_timer(ASYNC_SWAP_OUT_STATE_TIMER)
            self.timer_names.add(ASYNC_SWAP_OUT_STATE_TIMER)

        assert self.swap_ops[SYNC_SWAP_IN] is not None
        assert not self.swap_ops[SYNC_SWAP_IN].wait_required
        swap_op = self._swap_out_optimizer_state(aio_handle=self.write_aio_handle,
                                                 parameter=parameter,
                                                 swap_in_op=self.swap_ops[SYNC_SWAP_IN])
        self.swap_ops[SYNC_SWAP_IN] = None

        if self.async_swap_out and async_swap:
            self.swap_ops[ASYNC_SWAP_OUT] = swap_op
        else:
            self.swap_ops[SYNC_SWAP_OUT] = swap_op
            self._complete_swap_out(SYNC_SWAP_OUT)

        self._stop_timer(SWAP_OUT_STATE_TIMER)
        self.timer_names.add(SWAP_OUT_STATE_TIMER)

    def swap_out_gradients(self, parameter, gradient_offsets, gradient_tensors):
        self._swap_out_gradients(parameter=parameter,
                                 gradient_offsets=gradient_offsets,
                                 gradient_tensors=gradient_tensors,
                                 gradient_swapper=self.gradient_swapper)

    def _complete_swap_out(self, swap_out_type):
        self.swap_ops[swap_out_type].wait()
        self.swap_buffer_manager.free(self.swap_ops[swap_out_type].allocated_buffers)
        self.swap_ops[swap_out_type] = None

    def _swap_out_optimizer_state(self, aio_handle, parameter, swap_in_op):
        assert swap_in_op.is_parameter(parameter)

        allocated_buffers = swap_in_op.allocated_buffers.copy()
        swap_buffers = swap_in_op.state_buffers.copy()

        param_info = swap_in_op.param_info
        self._update_param_state_info(param_info, parameter)
        unpinned_tensors = param_info.get_unpinned_state_tensors()

        if len(unpinned_tensors) > 0:
            new_alloc_buffers = self.swap_buffer_manager.allocate(
                num_elems=self._io_aligned_numel(param_info.numel()),
                count=len(unpinned_tensors),
                dtype=param_info.dtype())
            assert new_alloc_buffers is not None

            allocated_buffers += new_alloc_buffers
            swap_buffers += new_alloc_buffers

            for pinned_dst, unpinned_src in zip(new_alloc_buffers, unpinned_tensors):
                dst = get_sized_buffer(pinned_dst, unpinned_src.numel())
                dst.data.copy_(unpinned_src.data)

        swap_paths = param_info.swap_paths.copy()
        assert len(swap_paths) == len(swap_buffers)

        swap_out_tensors(aio_handle, swap_buffers, swap_paths)

        swap_out_op = OptimizerSwapOp(aio_handle=aio_handle,
                                      param_info=param_info,
                                      read_op=False,
                                      allocated_buffers=allocated_buffers,
                                      state_buffers=swap_buffers,
                                      num_ops=len(swap_buffers))

        return swap_out_op

    def _swap_in_optimizer_state(self, aio_handle, parameter):
        param_info = self._get_param_swap_info(parameter)
        if param_info is None:
            return None

        required_buffer_count = len(
            param_info.tensors) + (1 if param_info.has_gradients() else 0)
        aligned_numel = self._io_aligned_numel(param_info.numel())
        allocated_buffers = self.swap_buffer_manager.allocate(
            num_elems=aligned_numel,
            count=required_buffer_count,
            dtype=parameter.dtype)
        assert allocated_buffers is not None, \
        f"PipelinedOptimizerSwapper ran out of swap buffers, try increasing {OFFLOAD_OPTIMIZER_BUFFER_COUNT}"

        state_buffers = allocated_buffers[:len(param_info.tensors)]
        param_info.set_swap_buffers(state_buffers)

        swap_buffers = state_buffers.copy()
        swap_paths = param_info.swap_paths.copy()

        if param_info.has_gradients():
            parameter.grad = allocated_buffers[-1].narrow(0, 0, param_info.numel())
            if param_info.swapped_gradients:
                swap_buffers += param_info.get_swap_gradient_buffers(parameter.grad)
                swap_paths += param_info.get_swap_gradient_paths()

        swap_in_tensors(aio_handle, swap_buffers, swap_paths)

        if param_info.unswapped_gradients:
            self._retrieve_unswapped_grad_partitions(swap_info=param_info,
                                                     dest_buffer=parameter.grad)

        swap_in_op = OptimizerSwapOp(aio_handle=aio_handle,
                                     param_info=param_info,
                                     read_op=True,
                                     allocated_buffers=allocated_buffers,
                                     state_buffers=state_buffers,
                                     num_ops=len(swap_buffers))

        return swap_in_op

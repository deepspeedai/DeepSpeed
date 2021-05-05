"""
Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.

Functionality of swapping tensors to/from (NVMe) storage devices.
"""

import os
import torch

from deepspeed.utils.logging import logger
from deepspeed.runtime.zero.offload_constants import *
from deepspeed.runtime.swap_tensor.constants import *
from deepspeed.runtime.swap_tensor.utils import swap_in_tensors, swap_out_tensors, \
    MIN_AIO_BYTES, AIO_ALIGNED_BYTES, get_sized_buffers, get_sized_buffer
from deepspeed.runtime.swap_tensor.utils import SwapBufferManager, SwapBufferPool


class FlattenedTensorSwapInfo(object):
    def __init__(self, path, length, offset):
        self.path = path
        self.offset = offset
        self.length = length


class OptimizerStateSwapInfo(object):
    def __init__(self, parameter, numel, base_folder):
        self.tensors = []
        self.param_id = id(parameter)
        self.swap_folder = base_folder
        self.swap_paths = []
        self.swapped_gradients = {}
        self.unswapped_gradients = {}
        self.tensor_numel = numel
        self.tensor_dtype = parameter.dtype
        self.tensor_device = parameter.device
        self.has_state_tensors = False
        self._add_tensors([parameter])

    def numel(self):
        return self.tensor_numel

    def has_gradients(self):
        return self.swapped_gradients or self.unswapped_gradients

    def _add_tensors(self, tensor_list):
        for t in tensor_list:
            self.tensors.append(t)
            self.swap_paths.append(os.path.join(self.swap_folder, f'{id(t)}.tensor.swp'))

    def add_state_tensors(self, tensor_list):
        self.has_state_tensors = True
        self._add_tensors(tensor_list)

    def device(self):
        return self.tensor_device

    def dtype(self):
        return self.tensor_dtype

    def release_memory(self):
        for tensor in self.tensors:
            tensor.data = torch.Tensor()

    def get_or_create_gradient_paths(self, offsets, lengths):
        gradient_paths = []
        for offset, length in zip(offsets, lengths):
            if not offset in self.swapped_gradients.keys():
                path = os.path.join(
                    self.swap_folder,
                    f'{self.param_id}_gradient_{offset}_{length}.tensor.swp')
                self.swapped_gradients[offset] = FlattenedTensorSwapInfo(
                    path,
                    length,
                    offset)

            gradient_paths.append(self.swapped_gradients[offset].path)

        return gradient_paths

    def set_swap_buffers(self, buffers):
        compute_lengths = [self.numel()] * len(self.tensors)
        compute_buffers = get_sized_buffers(buffers, compute_lengths)
        for t, buffer in zip(self.tensors, compute_buffers):
            t.data = buffer.data

    def get_swap_gradient_buffers(self, swap_buffer):
        assert self.numel() <= swap_buffer.numel()
        return [
            swap_buffer.narrow(0,
                               grad.offset,
                               grad.length) for grad in self.swapped_gradients.values()
        ]

    def get_swap_gradient_paths(self):
        return [grad.path for grad in self.swapped_gradients.values()]

    def get_unpinned_state_tensors(self):
        return [t for t in self.tensors if not t.is_pinned()]

    def read_unswapped_gradients(self, dest_buffer):
        num_elem_count = 0
        for offset, grad_partition in self.unswapped_gradients.items():
            dst_tensor = dest_buffer.narrow(0, offset, grad_partition.numel())
            dst_tensor.data.copy_(grad_partition.data)
            num_elem_count += grad_partition.numel()

        return num_elem_count

    def release_unswapped_gradients(self):
        self.unswapped_gradients = {}


SWAPPER_DEBUG_MODE = False
SWAP_OUT_GRADIENT_TIMER = 'swap_out_gradient'


class OptimizerSwapper(object):
    def __init__(self,
                 swap_config,
                 aio_config,
                 base_folder,
                 optimizer,
                 largest_numel,
                 device,
                 dtype,
                 timers):
        self.swap_config = swap_config
        self.aio_config = aio_config

        # NVMe swap management
        self.swap_params_info = {}
        self.swap_element_size = torch.tensor([], dtype=dtype).element_size()
        self.swap_folder = os.path.join(base_folder,
                                        'optimizer',
                                        f'rank{torch.distributed.get_rank()}')
        os.makedirs(self.swap_folder, exist_ok=True)

        self.optimizer = optimizer

        # Read/Write alignment for each thread during Intra-request parallelism
        self.min_aio_bytes = max(MIN_AIO_BYTES, aio_config[AIO_BLOCK_SIZE])
        self.aligned_bytes = AIO_ALIGNED_BYTES * aio_config[AIO_THREAD_COUNT]
        self.numel_alignment = self.aligned_bytes // self.swap_element_size

        # Swap buffer management
        self.largest_numel = self._io_aligned_numel(largest_numel)
        self.dtype = dtype
        self.swap_buffer_manager = SwapBufferManager(
            num_elems=self.largest_numel,
            count=swap_config[OFFLOAD_OPTIMIZER_BUFFER_COUNT],
            dtype=dtype)

        # Timers
        self.timers = timers
        self.timer_names = set()

        # Print exclusion list
        self.print_exclude_list = [
            'optimizer',
            'swap_buffer_manager',
            'swap_params_info',
            'timers',
            'timer_names',
        ]

    def swappable_tensor(self, param=None, numel=None):
        assert param is not None or numel is not None, "Either param or numel must be provided"
        if param is not None:
            return self.min_aio_bytes <= (param.numel() * self.swap_element_size)
        return self.min_aio_bytes <= (numel * self.swap_element_size)

    def init_timers(self):
        self.timer_names = set()

    def log_timers(self):
        if self.timer_names:
            self._log_timers(list(self.timer_names), force=True)

    def pre_backward(self):
        self.init_timers()

    def post_backward(self):
        pass

    def _flush_gradient_swapper(self, gradient_swapper):
        if gradient_swapper.has_buffers():
            self._start_timer(SWAP_OUT_GRADIENT_TIMER)
            pinned_buffers = gradient_swapper.release_buffers()
            self.swap_buffer_manager.free(pinned_buffers)
            self._stop_timer(SWAP_OUT_GRADIENT_TIMER)
            self.timer_names.add(SWAP_OUT_GRADIENT_TIMER)
            self.timer_names.update(gradient_swapper.get_timer_names())

    def _swap_out_gradients(self,
                            parameter,
                            gradient_offsets,
                            gradient_tensors,
                            gradient_swapper):
        if not id(parameter) in self.swap_params_info.keys():
            return

        swap_info = self.swap_params_info[id(parameter)]

        swappable_tensors = []
        swappable_offsets = []
        swappable_lengths = []

        aligned_gradients, aligned_offsets = self._adjust_for_misaligned_lengths(
            tensors=gradient_tensors,
            offsets=gradient_offsets
        )

        self._start_timer(SWAP_OUT_GRADIENT_TIMER)
        for tensor, offset in zip(aligned_gradients, aligned_offsets):
            if not self.swappable_tensor(param=tensor):
                swap_info.unswapped_gradients[offset] = tensor
                continue

            swappable_tensors.append(tensor)
            swappable_offsets.append(offset)
            swappable_lengths.append(tensor.numel())

        if len(swappable_tensors) > 0:
            if not gradient_swapper.has_buffers():
                pinned_buffers = self.swap_buffer_manager.allocate_all(
                    num_elems=self.largest_numel,
                    dtype=self.dtype)

                gradient_swapper.add_buffers(pinned_buffers)

            swappable_paths = swap_info.get_or_create_gradient_paths(
                swappable_offsets,
                swappable_lengths)

            gradient_swapper.swap_out_tensors(tensor_list=swappable_tensors,
                                              path_list=swappable_paths)

        self._stop_timer(SWAP_OUT_GRADIENT_TIMER)
        self.timer_names.add(SWAP_OUT_GRADIENT_TIMER)

    def _initialize_from_swapped_fp16_params(self,
                                             aio_handle,
                                             fp16_partitions_info,
                                             fp16_num_elems,
                                             fp16_pinned_buffers,
                                             fp32_parameters):
        assert len(fp32_parameters) == len(fp16_partitions_info)
        assert len(fp32_parameters) == len(fp16_num_elems)
        assert all([buffer.is_pinned() for buffer in fp16_pinned_buffers])

        fp32_swap_paths = self._get_swap_paths(parameters=fp32_parameters,
                                               num_elems=fp16_num_elems)

        fp32_pinned_buffers = self.swap_buffer_manager.allocate_all(
            num_elems=self.largest_numel,
            dtype=self.dtype)

        fp16_buffer_numel = [buf.numel() for buf in fp16_pinned_buffers]
        assert all([numel >= self.largest_numel for numel in fp16_buffer_numel]), \
        f"numel of fp16 buffers {fp16_buffer_numel} is too small for initializing fp32 params {self.largest_numel}"

        fp32_swap_buffers = SwapBufferPool(fp32_pinned_buffers)
        fp16_swap_buffers = SwapBufferPool(fp16_pinned_buffers)

        curr_index = 0
        while curr_index < len(fp32_parameters):
            fp16_pinned_tensors = self._swap_in_fp16_params(
                aio_handle=aio_handle,
                fp16_num_elems=fp16_num_elems[curr_index:],
                fp16_partitions_info=fp16_partitions_info[curr_index:],
                fp16_swap_buffers=fp16_swap_buffers)

            if torch.distributed.get_rank() == 0 and SWAPPER_DEBUG_MODE:
                for i, tensor in enumerate(fp16_pinned_tensors):
                    true_index = curr_index + i
                    logger.info(
                        f'swap_in_fp16_param: fp32_id = {id(fp32_parameters[true_index])} index = {true_index} orig_num_elem = {fp16_num_elems[true_index]}, swap_num_elem = {fp16_pinned_tensors[i].numel()}'
                    )

            swap_out_count = self._swap_out_fp16_params(
                aio_handle=aio_handle,
                fp32_swap_paths=fp32_swap_paths[curr_index:],
                fp32_swap_buffers=fp32_swap_buffers,
                fp16_pinned_tensors=fp16_pinned_tensors)
            assert swap_out_count == len(fp16_pinned_tensors), \
            f"{swap_out_count} does not match {len(fp16_pinned_tensors)}"

            fp16_swap_buffers.reset()
            fp32_swap_buffers.reset()
            curr_index += swap_out_count

        self.swap_buffer_manager.free(fp32_pinned_buffers)

    def _swap_in_fp16_params(self,
                             aio_handle,
                             fp16_num_elems,
                             fp16_partitions_info,
                             fp16_swap_buffers):
        assert len(fp16_num_elems) > 0

        swapped_fp16_tensors = []
        swap_tensors = []
        swap_paths = []
        unswapped_srcs = []
        unswapped_dsts = []

        for i, numel in enumerate(fp16_num_elems):
            pinned_tensor, _ = fp16_swap_buffers.allocate_tensor(numel, None, numel)
            if pinned_tensor is None:
                break

            swapped_fp16_tensors.append(pinned_tensor)
            offset = 0
            for tensor, partition_numel, partition_path in fp16_partitions_info[i]:
                dst_tensor = pinned_tensor.narrow(0, offset, partition_numel)
                if partition_path is None:
                    unswapped_srcs.append(tensor)
                    unswapped_dsts.append(dst_tensor)
                else:
                    swap_paths.append(partition_path)
                    swap_tensors.append(dst_tensor)
                offset += partition_numel

        assert len(swapped_fp16_tensors) + len(unswapped_srcs) > 0
        ret = swap_in_tensors(aio_handle, swap_tensors, swap_paths)
        for src, dst in zip(unswapped_srcs, unswapped_dsts):
            dst.data.copy_(src.data)

        assert len(swap_tensors) == aio_handle.wait()

        return swapped_fp16_tensors

    def _swap_out_fp16_params(self,
                              aio_handle,
                              fp32_swap_paths,
                              fp32_swap_buffers,
                              fp16_pinned_tensors):

        assert len(fp16_pinned_tensors) <= len(fp32_swap_paths)
        swap_out_count = 0
        for i, fp16_tensor in enumerate(fp16_pinned_tensors):
            if not fp32_swap_buffers.has_space(fp16_tensor.numel()):
                fp32_swap_buffers.swap_out(aio_handle)
                fp32_swap_buffers.reset()

            pinned_tensor, _ = fp32_swap_buffers.insert_tensor(
                fp16_tensor,
                fp32_swap_paths[i],
                self._io_aligned_numel(fp16_tensor.numel())
                )
            assert pinned_tensor is not None
            swap_out_count += 1

        if len(fp32_swap_buffers.get_swap_tensors()) > 0:
            fp32_swap_buffers.swap_out(aio_handle)

        return swap_out_count

    def _initialize_parameters(self, parameters, src_tensors, aio_handle):
        assert len(parameters) == len(src_tensors)

        swap_paths = self._get_swap_paths(parameters=parameters,
                                          num_elems=[src.numel() for src in src_tensors])

        SWAP_INIT_TIMER = "swap_init_write"
        self._start_timer(SWAP_INIT_TIMER)

        pinned_buffers = self.swap_buffer_manager.allocate_all(
            num_elems=self.largest_numel,
            dtype=self.dtype)
        assert pinned_buffers is not None

        self._swap_out_unpinned_tensors(aio_handle=aio_handle,
                                        unpinned_tensors=src_tensors,
                                        dest_paths=swap_paths,
                                        pinned_buffers=pinned_buffers)

        if torch.distributed.get_rank() == 0 and SWAPPER_DEBUG_MODE:
            for i, tensor in enumerate(src_tensors):
                logger.info(
                    f'copy_in_fp16_param: fp32_id = {id(parameters[i])} index = {i}, swap_num_elem = {src_tensors[i].numel()}'
                )

        self.swap_buffer_manager.free(pinned_buffers)

        self._stop_timer(SWAP_INIT_TIMER)
        self._log_timers([SWAP_INIT_TIMER])

    def _get_swap_paths(self, parameters, num_elems):
        swap_info_list = [
            self._create_param_swap_info(parameter=p,
                                         numel=numel) \
            for p, numel in zip(parameters, num_elems)
        ]
        assert len(swap_info_list) == len(num_elems)

        swap_paths = [info.swap_paths[0] for info in swap_info_list]
        return swap_paths

    def _swap_out_unpinned_tensors(self,
                                   aio_handle,
                                   unpinned_tensors,
                                   dest_paths,
                                   pinned_buffers):

        swap_buffer_count = len(pinned_buffers)
        unpinned_tensor_count = len(unpinned_tensors)

        for i in range(0, unpinned_tensor_count, swap_buffer_count):
            swap_tensor_count = min((unpinned_tensor_count - i), swap_buffer_count)

            src_tensors = unpinned_tensors[i:(i + swap_tensor_count)]
            compute_lengths = [t.numel() for t in src_tensors]
            compute_buffers = get_sized_buffers(pinned_buffers, compute_lengths)

            for dst, src in zip(compute_buffers, src_tensors):
                dst.data.copy_(src.data)

            swap_lengths = [self._io_aligned_numel(t.numel()) for t in src_tensors]
            swap_buffers = get_sized_buffers(pinned_buffers, swap_lengths)

            swap_paths = dest_paths[i:(i + swap_tensor_count)]
            swap_out_tensors(aio_handle, swap_buffers, swap_paths)

            assert aio_handle.wait() == swap_tensor_count

    def _adjust_for_misaligned_lengths(self, tensors, offsets):
        new_tensors = []
        new_offsets = []

        for orig_tensor, orig_offset in zip(tensors, offsets):
            if not self.swappable_tensor(param=orig_tensor):
                new_tensors.append(orig_tensor)
                new_offsets.append(orig_offset)
                continue

            remainder = orig_tensor.numel() % self.numel_alignment
            if remainder == 0:
                new_tensors.append(orig_tensor)
                new_offsets.append(orig_offset)
                continue

            # Split into two by making remainder a tensor
            aligned_length = (orig_tensor.numel() //
                              self.numel_alignment) * self.numel_alignment
            new_tensors.append(orig_tensor.narrow(0, 0, aligned_length))
            new_offsets.append(orig_offset)

            # remainder tensor
            new_tensors.append(orig_tensor.narrow(0, aligned_length, remainder))
            new_offsets.append(orig_offset + aligned_length)

        return new_tensors, new_offsets

    def _retrieve_unswapped_grad_partitions(self, swap_info, dest_buffer):
        UNSWAPPED_READ_GRADIENTS = 'unswapped_read_gradients'
        self._start_timer(UNSWAPPED_READ_GRADIENTS)
        tensor_count = len(swap_info.unswapped_gradients)
        num_elem_count = swap_info.read_unswapped_gradients(dest_buffer)
        self._stop_timer(UNSWAPPED_READ_GRADIENTS)
        self._log_timers([UNSWAPPED_READ_GRADIENTS])

        # It shoud be safe to discard unswapped gradient partitions
        swap_info.release_unswapped_gradients()

        if SWAPPER_DEBUG_MODE:
            logger.info(
                f'optimizer_retreive_unswapped_radients: param={swap_info.param_id} tensor_count={tensor_count} elem_count={num_elem_count}'
            )

    def _get_state_tensors(self, parameter):
        if not parameter in self.optimizer.state:
            return []

        tensor_list = []
        for value in self.optimizer.state[parameter].values():
            if torch.is_tensor(value):
                tensor_list.append(value)

        return tensor_list

    def _update_param_state_info(self, swap_info, parameter):
        if not swap_info.has_state_tensors:
            state_tensors = self._get_state_tensors(parameter)
            if state_tensors:
                swap_info.add_state_tensors(state_tensors)

    def _create_param_swap_info(self, parameter, numel):
        param_id = id(parameter)
        assert not param_id in self.swap_params_info

        self.swap_params_info[param_id] = OptimizerStateSwapInfo(
            parameter=parameter,
            numel=numel,
            base_folder=self.swap_folder)
        swap_info = self.swap_params_info[param_id]

        self._update_param_state_info(swap_info, parameter)

        return swap_info

    def _get_param_swap_info(self, parameter):
        param_id = id(parameter)
        swap_info = self.swap_params_info.get(param_id, None)

        if swap_info is not None:
            self._update_param_state_info(swap_info, parameter)

        return swap_info

    def _start_timer(self, name):
        if self.timers:
            self.timers(name).start()

    def _stop_timer(self, name):
        if self.timers:
            self.timers(name).stop()

    def _log_timers(self, name_list, force=False):
        if self.timers and (SWAPPER_DEBUG_MODE or force):
            self.timers.log(name_list)

    def _io_aligned_numel(self, numel):
        remainder = numel % self.numel_alignment
        return numel if remainder == 0 else (numel + self.numel_alignment - remainder)

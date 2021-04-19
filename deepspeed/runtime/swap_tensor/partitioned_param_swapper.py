"""
Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.

Functionality of swapping tensors to/from (NVMe) storage devices.
"""

import os
from enum import Enum
import torch
import torch.distributed as dist

from deepspeed.utils.logging import logger
from deepspeed.ops.aio import AsyncIOBuilder
from .constants import *
from .utils import swap_in_tensors, swap_out_tensors, MIN_AIO_BYTES, print_object
from ..zero.offload_constants import *


def print_rank_0(message, debug=False, force=False):
    if torch.distributed.get_rank() == 0 and (debug or force):
        print(message)


class PartitionedParamStatus(Enum):
    # Partitioned parameters are present and ready for use
    AVAILABLE = 1

    # partitioned params are in some non-memory device
    NOT_AVAILABLE = 2

    # partitioned params are being read from some non-memory device.
    INFLIGHT = 3


class AsyncPartitionedParameterSwapper(object):
    def __init__(self, ds_config):

        aio_op = AsyncIOBuilder().load(verbose=False)
        self.aio_handle = aio_op.aio_handle

        #set swap buffers, create aio handles
        self._configure_aio(ds_config)

        #mapping from param id to path
        self.id_to_path = {}

        #mapping from pram_id to buffer id
        self.param_id_to_buffer_id = {}

        #number of elements in the param
        self.param_id_to_numel = {}

        self.pending_writes = 0
        self.pending_reads = 0

        #keep track of async swap in params and buffers
        self.inflight_params = []
        self.inflight_swap_in_buffers = []
        self.inflight_numel = 0

        #keep track of available params
        self.available_params = set()
        self.available_numel = 0

        self.invalid_buffer = torch.tensor(1).half()

        if dist.get_rank() == 0:
            exclude_list = ['aio_read_handle', 'aio_write_handle', 'buffers']
            print_object(obj=self,
                         name='AsyncPartitionedParameterSwapper',
                         exclude_list=exclude_list)

    def available_swap_in_buffers(self):
        return len(self.available_buffer_ids)

    def _configure_aio(self, ds_config):
        self.swap_config = ds_config.zero_config.offload_param
        self.swap_folder = os.path.join(self.swap_config[OFFLOAD_PARAM_NVME_PATH],
                                        'zero_stage_3',
                                        'fp16params',
                                        f'rank{dist.get_rank()}')
        os.makedirs(self.swap_folder, exist_ok=True)

        self.elements_per_buffer = self.swap_config[OFFLOAD_PARAM_BUFFER_SIZE]
        self.param_buffer_count = self.swap_config[OFFLOAD_PARAM_BUFFER_COUNT]

        self.available_buffer_ids = [i for i in range(self.param_buffer_count)]
        self.reserved_buffer_ids = []

        self.buffers = torch.empty(int(self.elements_per_buffer *
                                       self.param_buffer_count),
                                   dtype=torch.half,
                                   pin_memory=True,
                                   requires_grad=False)

        self.aio_config = ds_config.aio_config

        self.aio_read_handle = self.aio_handle(self.aio_config[AIO_BLOCK_SIZE],
                                               self.aio_config[AIO_QUEUE_DEPTH],
                                               self.aio_config[AIO_SINGLE_SUBMIT],
                                               self.aio_config[AIO_OVERLAP_EVENTS],
                                               self.aio_config[AIO_THREAD_COUNT])

        self.aio_write_handle = self.aio_handle(self.aio_config[AIO_BLOCK_SIZE],
                                                self.aio_config[AIO_QUEUE_DEPTH],
                                                self.aio_config[AIO_SINGLE_SUBMIT],
                                                self.aio_config[AIO_OVERLAP_EVENTS],
                                                self.aio_config[AIO_THREAD_COUNT])

        self.min_aio_bytes = max(MIN_AIO_BYTES, self.aio_config[AIO_BLOCK_SIZE])

        self.swap_element_size = torch.tensor([], dtype=torch.half).element_size()
        self.swap_out_params = []

    #Check if partiitoned param or numel in a tensor is swappable or not
    def swappable_tensor(self, param=None, numel=None):
        if param is not None:
            assert numel is None, "Both parma and numel cannot be provided"
            numel = param.ds_tensor.ds_numel
        if numel is not None:
            return self.min_aio_bytes <= numel * self.swap_element_size
        assert False, "Either param or numel must be provided"

    def get_path(self, param, must_exist=False):
        paths, _ = self._get_paths([param], must_exist=must_exist)
        return paths[0]

    def _get_paths(self, params, must_exist=False):
        paths = []
        tensors = []
        for param in params:
            param_id = param.ds_id

            if param_id in self.id_to_path.keys():
                param_path = self.id_to_path[param_id]
            else:
                assert not must_exist, f"Path for param id {param_id} does not exist"
                param_path = os.path.join(self.swap_folder,
                                          f'{param_id}_param.tensor.swp')

                self.id_to_path[param_id] = param_path
            paths.append(param_path)
            tensors.append(param.ds_tensor)
        return paths, tensors

    def _track_numel(self, params):
        for param in params:
            assert param.ds_tensor is not None, "Partitioned tensor is None"
            self.param_id_to_numel[param.ds_id] = param.ds_tensor.ds_numel

    def _allocate_and_return_buffers_for_swap_in(self, params):
        buffers = []
        for param in params:
            param_id = param.ds_id
            assert param_id in self.param_id_to_numel.keys(), f" Number of elements in param {param_id} is unknown"
            assert param_id not in self.param_id_to_buffer_id.keys(), f"param {param_id} already assigned swap buffer id {self.param_id_to_buffer_id[param_id]}"

            buffer_id = self.available_buffer_ids.pop()
            print_rank_0(
                f"param {param.ds_id} is assigned swap in buffer id {buffer_id}  ")
            self.param_id_to_buffer_id[param_id] = buffer_id
            buffer = self.buffers.narrow(0,
                                         int(buffer_id * self.elements_per_buffer),
                                         self.param_id_to_numel[param_id])
            buffers.append(buffer)

        return buffers

    #waits for inflight nvme write to complete
    def synchronize_writes(self):
        if self.pending_writes == 0:
            return
        assert self.pending_writes == self.aio_write_handle.wait()
        self.pending_writes = 0
        self.remove_partition_and_release_buffers(self.swap_out_params)
        self.swap_out_params = []

    #waits for inflight nvme reads to complete
    def synchronize_reads(self):
        if self.pending_reads == 0:
            return

        assert self.pending_reads == self.aio_read_handle.wait()

        self.pending_reads = 0

        for param, swap_in_buffer in zip(self.inflight_params, self.inflight_swap_in_buffers):
            param.ds_tensor.data = swap_in_buffer.data
            param.ds_tensor.status = PartitionedParamStatus.AVAILABLE

        self.available_params.update([param.ds_id for param in self.inflight_params])
        self.available_numel += self.inflight_numel

        self.inflight_params = []
        self.inflight_swap_in_buffers = []
        self.inflight_numel = 0

    #Removes the memory assignment and releases the buffers
    #Should only be executed after swapping out the tensors
    def remove_partition_and_release_buffers(self, params):
        for param in params:
            param_id = param.ds_id

            if param_id in self.param_id_to_buffer_id.keys():

                buffer_id = self.param_id_to_buffer_id[param_id]

                assert buffer_id is not None, "Missing buffer id for releasing"

                self.available_buffer_ids.append(buffer_id)
                del self.param_id_to_buffer_id[param_id]
                print_rank_0(f"param {param.ds_id} releases buffer id {buffer_id}  ")

                if param_id in self.available_params:
                    self.available_params.remove(param_id)
                    self.available_numel -= self.param_id_to_numel[param_id]

            param.ds_tensor.data = self.invalid_buffer.data
            param.ds_tensor.status = PartitionedParamStatus.NOT_AVAILABLE

    #writes from in memory to nvme. Does not release the buffers
    def _swap_out(self, params, async_op=True):

        swap_out_paths, swap_out_params = self._get_paths(params)

        self._track_numel(params)

        swap_out_tensors(self.aio_write_handle, swap_out_params, swap_out_paths)

        self.pending_writes += len(swap_out_params)
        self.swap_out_params += params

        if not async_op:
            self.synchronize_writes()

    #blocking swap out followed by releasing the memory buffers
    def swap_out_and_release(self, params, async_op=False, force_buffer_release=False):
        if async_op:
            assert force_buffer_release, "Should not release preallocated buffers without completing the swap out. Set force_buffer_release to True to do it anyways"
        self._swap_out(params, async_op=async_op)

    #assigns an in memory buffer and swaps in from nvme
    def swap_in(self, params, async_op=True, swap_in_buffers=None):

        assert all([param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE for param in params]), "Some params are already available or in flight"
        swap_in_paths, _ = self._get_paths(params)

        if swap_in_buffers is None:
            if len(self.available_buffer_ids) < len(swap_in_paths):
                print_rank_0(
                    f'Not enough swap in buffers {len(self.available_buffer_ids)} for params {len(swap_in_paths)}',
                    force=True)
                print_rank_0(
                    f'Num inflight: params {len(self.inflight_params)}, buffers {len(self.inflight_swap_in_buffers)}, numel = {self.inflight_numel}',
                    force=True)
                print_rank_0(
                    f'Num available: param {len(self.available_params)}, numel = {self.available_numel}',
                    force=True)

            assert len(swap_in_paths) <= len(self.available_buffer_ids), f"Not enough buffers {len(self.available_buffer_ids)} for swapping {len(swap_in_paths)}"
            swap_in_buffers = self._allocate_and_return_buffers_for_swap_in(params)

        swap_in_tensors(self.aio_read_handle, swap_in_buffers, swap_in_paths)

        self.inflight_params.extend(params)
        self.inflight_swap_in_buffers.extend(swap_in_buffers)
        self.inflight_numel += sum([t.numel() for t in swap_in_buffers])

        for param in params:
            param.ds_tensor.status = PartitionedParamStatus.INFLIGHT

        self.pending_reads += len(params)

        if not async_op:
            self.synchronize_reads()

    #assign a buffer to a param and return the buffer
    def get_buffer(self, param, numel):
        assert numel < self.elements_per_buffer, f"More elements {numel} than buffer size {self.elements_per_buffer}"
        param_id = param.ds_id
        self.param_id_to_numel[param_id] = numel
        buffer_id = self.available_buffer_ids.pop()
        self.param_id_to_buffer_id[param_id] = buffer_id

        buffer = self.buffers.narrow(0,
                                     int(buffer_id * self.elements_per_buffer),
                                     self.param_id_to_numel[param_id])
        print_rank_0(f"param {param.ds_id} is assigned swap in buffer id {buffer_id}")
        return buffer

    def reserve_available_buffers(self):
        buffers = []
        for id in self.available_buffer_ids:
            buffers.append(
                self.buffers.narrow(0,
                                    int(id * self.elements_per_buffer),
                                    int(self.elements_per_buffer)))
            self.reserved_buffer_ids.append(id)

        self.available_buffer_ids = []
        return buffers

    def release_reserved_buffers(self):
        for id in self.reserved_buffer_ids:
            self.available_buffer_ids.append(id)

        self.reserved_buffer_ids = []

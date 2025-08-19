# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass
from deepspeed.checkpoint.reshape_utils import partition_data
from deepspeed.runtime.zero.config import ZeroStageEnum
from .constants import *


@dataclass
class DataParallelWriterConfig(object):
    world_size: int
    rank: int
    global_rank: int
    local_rank: int
    pure_dp: bool


class DataParallelWriterFactory(object):

    def __init__(self, uni_parallel_info, parallel_unit):
        self._uni_parallel_info = uni_parallel_info
        self._parallel_unit = parallel_unit
        if parallel_unit == CheckpointDataParallel.SOCKET:
            self._num_resources = uni_parallel_info.num_sockets
        else:
            self._num_resources = uni_parallel_info.num_machines
        self._ranks_per_resource = max(1, self._uni_parallel_info.global_world_size // self._num_resources)

    def create_config(self, zero_stage, has_moe_layers):
        if zero_stage == ZeroStageEnum.weights:
            return self._create_config(1, 0)

        if has_moe_layers:
            writer_config = self._get_expert_data_parallel_config()
        else:
            writer_config = self._get_data_parallel_config()

        if writer_config is None and zero_stage >= ZeroStageEnum.optimizer_states:
            return self._create_config(1, 0)

        return writer_config

    def _create_config(self, world_size, rank):
        return DataParallelWriterConfig(world_size=world_size,
                                        rank=rank,
                                        global_rank=self._uni_parallel_info.global_rank,
                                        local_rank=self._uni_parallel_info.local_rank,
                                        pure_dp=self._uni_parallel_info.pure_dp)

    def _get_expert_data_parallel_config(self):
        ep_info = self._uni_parallel_info.ep_info
        if self._parallel_unit is None:
            dp_rank = ep_info.dp_rank
            return self._create_config(1, 0) if dp_rank == 0 else None

        assert self._uni_parallel_info.pure_dp, \
            '3D parallelism is not yet supported for data parallel checkpointing.'

        if self._parallel_unit == CheckpointDataParallel.REPLICA or ep_info.ep_world_size == 1:
            return self._get_parallel_write_for_ddp(ep_info.dp_world_size, ep_info.dp_rank)

        return self._get_expert_parallel_write_for_2d()

    def _get_expert_parallel_write_for_2d(self):
        ep_info = self._uni_parallel_info.ep_info

        def _get_expert_slice_resources(expert_resources, resource_name):
            ep_world_size = ep_info.ep_world_size
            slices_per_resource = min(self._ranks_per_resource, ep_world_size)
            assert slices_per_resource <= len(expert_resources)

            ep_num_resources = len(expert_resources)
            assert ep_num_resources % slices_per_resource == 0, f'{resource_name}: Expected ep_num_resources={ep_num_resources} to multiple of slices_per_resource={slices_per_resource} for ep_world_size={ep_world_size}'

            slice_partitions = partition_data(expert_resources, slices_per_resource)
            # print(
            #     f'edp_resource_partition: self._uni_parallel_info.global_rank={self._uni_parallel_info.global_rank} expert_resources={expert_resources} slices_per_resource={slices_per_resource} ep_world_size={ep_world_size} slice_partitions={slice_partitions}'
            # )
            resource_index = ep_info.ep_rank % slice_resources
            return slice_partitions[resource_index]

        dp_ranks = ep_info.dp_peer_ranks
        expert_resources = [r // self._ranks_per_resource for r in dp_ranks]
        slice_resources = _get_expert_slice_resources(expert_resources, self._parallel_unit)
        assert all([idx < self._num_resources for idx in expert_resources]), \
            f'Detected invalid resource index in expert_resources={expert_resources}, self._num_resources={self._num_resources}'
        return self._assign_resources_to_tensor_slice(slice_resources, ep_info.ep_rank, dp_ranks)

    def _get_data_parallel_config(self):
        mpu_info = self._uni_parallel_info.mpu_info
        if self._parallel_unit is None:
            dp_rank = self._uni_parallel_info.dp_rank if mpu_info is None else mpu_info.dp_rank
            return self._create_config(1, 0) if dp_rank == 0 else None

        if self._uni_parallel_info.pure_dp:
            return self._get_parallel_write_for_ddp(self._uni_parallel_info.global_world_size,
                                                    self._uni_parallel_info.global_rank)

        if self._parallel_unit == CheckpointDataParallel.REPLICA:
            return self._create_config(mpu_info.dp_world_size, mpu_info.dp_rank)

        return self._get_parallel_write_for_3d()

    def _get_parallel_write_for_3d(self):
        mpu_info = self._uni_parallel_info.mpu_info
        my_global_rank = self._uni_parallel_info.global_rank

        def _expand_resources(resource_list, new_size):
            old_size = len(resource_list)
            if old_size >= new_size:
                return resource_list

            assert new_size % old_size == 0, f'Expect new_size={new_size} to be multiple of old_size={old_size}'
            multiplier = new_size // old_size
            new_resource_list = []
            for r in resource_list:
                new_resource_list += [r] * multiplier
            # print(f'expand_resources: {my_global_rank=} {old_size=} {new_size=} {resource_list=} {new_resource_list=}')
            return new_resource_list

        # Getting resource partition for a tensor slice is a 2-step process
        # 1. Get resource partitions for all pipeline stages. A pipeline stage is a 2D grid of size TP x DP
        def _get_pipeline_stage_resources(resource_indices):
            num_resources = len(resource_indices)
            pp_world_size = mpu_info.pp_world_size
            if num_resources < pp_world_size:
                resource_indices = _expand_resources(resource_indices, pp_world_size)
                num_resources = pp_world_size
            global_resource_partitions = partition_data(resource_indices, pp_world_size)
            pp_rank = mpu_info.pp_rank
            return global_resource_partitions[pp_rank]

        # 2. Get resource partition for tensor slice. A tensor slice is a 1D vector of size DP
        def _get_tensor_slice_resources(resource_indices, resource_name):
            pipe_stage_resources = _get_pipeline_stage_resources(resource_indices)
            tp_world_size = mpu_info.tp_world_size
            if len(pipe_stage_resources) < tp_world_size:
                pipe_stage_resources = _expand_resources(pipe_stage_resources, tp_world_size)
            tp_num_resources = len(pipe_stage_resources)
            assert tp_num_resources % tp_world_size == 0, \
                f'{resource_name}: Expected tp_num_resources={tp_num_resources} to multiple of tp_world_size={tp_world_size}'

            pipe_stage_resource_partitions = partition_data(pipe_stage_resources, tp_world_size)
            tp_rank = mpu_info.tp_rank
            return pipe_stage_resource_partitions[tp_rank]

        def _get_model_parallel_slice_resources():
            # Get resources of my dp peer ranks
            resources = [(r // self._ranks_per_resource) for r in mpu_info.dp_peer_ranks]
            if len(resources) < self._ranks_per_resource:
                resources = _expand_resources(resources, self._ranks_per_resource)

            resource_partitions = partition_data(resources, self._ranks_per_resource)
            mp_rank = (mpu_info.pp_rank * mpu_info.tp_world_size) + mpu_info.tp_rank
            slice_rank = mp_rank % self._ranks_per_resource
            return resource_partitions[slice_rank]

        num_slices = mpu_info.tp_world_size * mpu_info.pp_world_size
        if num_slices > self._ranks_per_resource:
            slice_resources = _get_model_parallel_slice_resources()
        else:
            all_resources = list(range(self._num_resources))
            slice_resources = _get_tensor_slice_resources(all_resources, self._parallel_unit)

        return self._assign_resources_to_tensor_slice(slice_resources, mpu_info.tp_rank, mpu_info.dp_peer_ranks)

    def _get_slice_writers(self, slice_resources, my_dp_ranks):
        resource_map = {}
        for res in slice_resources:
            resource_map[res] = [r for r in my_dp_ranks if (r // self._ranks_per_resource) == res]

        # Only one writer per resource, and we conventionally pick the first rank as writer.
        return [ranks[0] for ranks in resource_map.values()]

    def _assign_resources_to_tensor_slice(self, slice_resources, my_slice_index, my_dp_ranks):
        my_global_rank = self._uni_parallel_info.global_rank
        slice_writer_ranks = self._get_slice_writers(slice_resources, my_dp_ranks)
        my_resource_index = my_global_rank // self._ranks_per_resource
        print(
            f'resource_assign: my_global_rank={my_global_rank} my_slice_index={my_slice_index} my_dp_ranks={my_dp_ranks} slice_resources={slice_resources} slice_writer_ranks={slice_writer_ranks}'
        )
        if my_resource_index in slice_resources and my_global_rank in slice_writer_ranks:
            my_writer_index = (my_global_rank - slice_writer_ranks[0]) // self._ranks_per_resource
            num_slice_writers = len(slice_writer_ranks)
            print(
                f'slice_writer: my_global_rank={my_global_rank} my_writer_index={my_writer_index} num_slice_writers={num_slice_writers}'
            )
            return self._create_config(num_slice_writers, my_writer_index)

        return None

    def _get_parallel_write_for_ddp(self, dp_world_size, dp_rank):
        if self._parallel_unit == CheckpointDataParallel.REPLICA:
            return self._create_config(dp_world_size, dp_rank)

        num_machines = self._uni_parallel_info.num_machines
        if self._parallel_unit == CheckpointDataParallel.SOCKET:
            if dp_world_size == num_machines:
                # There is one rank per machine
                return self._create_config(num_machines, dp_rank)

            num_sockets = self._uni_parallel_info.num_sockets
            ranks_per_socket = dp_world_size // num_sockets
            if dp_rank % ranks_per_socket == 0:
                return self._create_config(num_sockets, dp_rank // ranks_per_socket)
            else:
                return None

        ranks_per_machine = dp_world_size // num_machines
        if dp_rank % ranks_per_machine == 0:
            return self._create_config(num_machines, self._uni_parallel_info.machine_rank)

        return None

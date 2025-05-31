# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from dataclasses import dataclass
from deepspeed import comm as dist
from deepspeed.constants import CROSS_RANK, CROSS_SIZE, LOCAL_RANK
from .data_parallel_writer_factory import DataParallelWriterFactory

# TODO: parse socket number from env.
SOCKETS_PER_MACHINE = 2


@dataclass
class MPUInfo(object):
    pp_world_size: int
    pp_rank: int
    tp_world_size: int
    tp_rank: int
    dp_world_size: int
    dp_peer_ranks: list
    dp_rank: int


def _create_model_parallel_info(mpu):
    return MPUInfo(pp_world_size=mpu.get_pipeline_model_parallel_world_size(),
                   pp_rank=mpu.get_pipeline_model_parallel_rank(),
                   tp_world_size=mpu.get_tensor_model_parallel_world_size(),
                   tp_rank=mpu.get_tensor_model_parallel_rank(),
                   dp_world_size=mpu.get_data_parallel_world_size(),
                   dp_peer_ranks=mpu.get_data_parallel_group_ranks(),
                   dp_rank=mpu.get_data_parallel_rank())


@dataclass
class ExpertParallelInfo(object):
    ep_world_size: int
    ep_rank: int
    dp_world_size: int
    dp_peer_ranks: list
    dp_rank: int


def _create_expert_parallel_info(groups):
    group_name = groups._get_max_expert_size_name()
    return ExpertParallelInfo(ep_world_size=groups._get_expert_parallel_world_size(group_name),
                              ep_rank=groups._get_expert_parallel_rank(group_name),
                              dp_world_size=groups._get_expert_data_parallel_world_size(group_name),
                              dp_peer_ranks=groups._get_expert_data_parallel_group_ranks(group_name),
                              dp_rank=groups._get_expert_data_parallel_rank(group_name))


@dataclass
class UniversalParallelInfo(object):
    global_world_size: int
    global_rank: int
    local_rank: int
    mpu_info: MPUInfo
    ep_info: ExpertParallelInfo
    pure_dp: bool
    num_machines: int
    machine_rank: int
    num_sockets: int


def create_universal_parallel_info(groups, has_moe_layers):
    return UniversalParallelInfo(global_world_size=dist.get_world_size(),
                                 global_rank=dist.get_rank(),
                                 local_rank=int(os.environ[LOCAL_RANK]),
                                 mpu_info=None if groups.mpu is None else _create_model_parallel_info(groups.mpu),
                                 ep_info=_create_expert_parallel_info(groups) if has_moe_layers else None,
                                 pure_dp=groups.mpu is None
                                 or groups.mpu.get_data_parallel_world_size() == dist.get_world_size(),
                                 num_machines=int(os.environ[CROSS_SIZE]),
                                 machine_rank=int(os.environ[CROSS_RANK]),
                                 num_sockets=int(os.environ[CROSS_SIZE]) * SOCKETS_PER_MACHINE)


def create_data_parallel_writer_config(groups, parallel_unit, zero_stage, has_moe_layers):
    uni_parallel_info = create_universal_parallel_info(groups, has_moe_layers)
    writer_factory = DataParallelWriterFactory(uni_parallel_info, parallel_unit)
    return writer_factory.create_config(zero_stage, has_moe_layers)

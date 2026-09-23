# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.runtime.model_checkpointing.constants import CheckpointDataParallel
from deepspeed.runtime.model_checkpointing.data_parallel_writer_factory import DataParallelWriterFactory
from deepspeed.runtime.model_checkpointing.utils import ExpertParallelInfo, UniversalParallelInfo
from deepspeed.runtime.zero.config import ZeroStageEnum


def _expert_writer_config(rank, world_size, ep_size, parallel_unit):
    # 8 ranks on one machine with two sockets; expert DP groups are strided by ep_size.
    ep_info = ExpertParallelInfo(ep_world_size=ep_size,
                                 ep_rank=rank % ep_size,
                                 dp_world_size=world_size // ep_size,
                                 dp_peer_ranks=list(range(rank % ep_size, world_size, ep_size)),
                                 dp_rank=rank // ep_size)
    info = UniversalParallelInfo(global_world_size=world_size,
                                 global_rank=rank,
                                 local_rank=rank,
                                 mpu_info=None,
                                 ep_info=ep_info,
                                 pure_dp=True,
                                 num_machines=1,
                                 machine_rank=0,
                                 num_sockets=2)
    return DataParallelWriterFactory(info, parallel_unit).create_config(ZeroStageEnum.disabled, has_moe_layers=True)


@pytest.mark.parametrize('parallel_unit', [CheckpointDataParallel.SOCKET, CheckpointDataParallel.MACHINE])
def test_expert_writer_one_per_expert_slice(parallel_unit):
    world_size, ep_size = 8, 2
    configs = [_expert_writer_config(r, world_size, ep_size, parallel_unit) for r in range(world_size)]
    writers = [r for r, c in enumerate(configs) if c is not None]
    assert sorted(r % ep_size for r in writers) == list(range(ep_size))
    assert all((configs[r].world_size, configs[r].rank) == (1, 0) for r in writers)

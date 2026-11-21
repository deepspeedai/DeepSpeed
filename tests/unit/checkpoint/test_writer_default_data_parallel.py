# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.runtime.model_checkpointing.data_parallel_writer_factory import DataParallelWriterFactory
from deepspeed.runtime.model_checkpointing.utils import UniversalParallelInfo
from deepspeed.runtime.zero.config import ZeroStageEnum

WORLD_SIZE = 4


def _writer_config(rank, zero_stage):
    # Plain data parallelism: no mpu, no MoE, and checkpoint.writer.data_parallel left unset.
    info = UniversalParallelInfo(global_world_size=WORLD_SIZE,
                                 global_rank=rank,
                                 local_rank=rank,
                                 mpu_info=None,
                                 ep_info=None,
                                 pure_dp=True,
                                 num_machines=1,
                                 machine_rank=0,
                                 num_sockets=2)
    return DataParallelWriterFactory(info, None).create_config(zero_stage, has_moe_layers=False)


def test_default_writer_is_data_parallel_rank_zero_without_zero():
    configs = [_writer_config(rank, ZeroStageEnum.disabled) for rank in range(WORLD_SIZE)]
    assert [c is not None for c in configs] == [True, False, False, False]
    assert (configs[0].world_size, configs[0].rank) == (1, 0)


@pytest.mark.parametrize('zero_stage', [ZeroStageEnum.optimizer_states, ZeroStageEnum.gradients])
def test_default_writer_writes_every_zero_partition(zero_stage):
    configs = [_writer_config(rank, zero_stage) for rank in range(WORLD_SIZE)]
    assert all(c is not None and (c.world_size, c.rank) == (1, 0) for c in configs)

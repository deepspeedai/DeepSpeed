# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import pytest
import torch
from torch import nn

import deepspeed
import deepspeed.comm as dist
from unit.common import DistributedTest


class ScalarModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(dist.get_rank() + 1)))


class TestScalarParameterBroadcast(DistributedTest):
    world_size = 2
    backend = "gloo"
    requires_cuda_env = False

    @pytest.mark.parametrize("zero_stage", [0, 3])
    def test_initialization_broadcasts_scalar_parameter(self, zero_stage):
        engine, _, _, _ = deepspeed.initialize(
            model=ScalarModel(),
            config={
                "train_micro_batch_size_per_gpu": 1,
                "zero_optimization": {
                    "stage": zero_stage
                }
            },
        )
        parameter = engine.module.weight
        if zero_stage == 3:
            with deepspeed.zero.GatheredParameters([parameter]):
                assert parameter.shape == torch.Size([])
                assert parameter.item() == 1.0
        else:
            assert parameter.shape == torch.Size([])
            assert parameter.item() == 1.0

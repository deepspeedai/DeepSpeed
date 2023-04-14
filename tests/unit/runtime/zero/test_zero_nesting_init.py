# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from unit.common import DistributedTest

import deepspeed


class TestNestingInit(DistributedTest):
    world_size = 1

    def test_nesting_init(self):
        ds_config = dict(train_batch_size=1, zero_optimization=dict(stage=3))

        with deepspeed.zero.Init(config_dict_or_path=ds_config):
            with deepspeed.zero.Init(config_dict_or_path=ds_config):
                model = torch.nn.Linear(4, 4)

        deepspeed_engine, *_ = deepspeed.initialize(model=model, config_params=ds_config)
        # ensure that zero3 processed the parameter
        assert hasattr(deepspeed_engine.weight, "ds_id")

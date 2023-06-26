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

        # ensure that zero3 processed the parameter
        assert hasattr(model.weight, "ds_id")

        deepspeed_engine, *_ = deepspeed.initialize(model=model, config_params=ds_config)


class TestShutdownInNestingInit(DistributedTest):
    world_size = 1

    def test_shutdown_in_nesting_init(self):
        ds_config = dict(train_batch_size=1, zero_optimization=dict(stage=3))

        with deepspeed.zero.Init(config_dict_or_path=ds_config):
            with deepspeed.zero.Init(config_dict_or_path=ds_config):
                model1 = torch.nn.Linear(4, 4)

            assert hasattr(model1.weight, "ds_id")
            deepspeed_engine1, *_ = deepspeed.initialize(model=model1, config_params=ds_config)
            with deepspeed.zero.Init(config_dict_or_path=ds_config):
                model2 = torch.nn.Linear(4, 4)

        # ensure that zero3 processed the parameter
        assert hasattr(model2.weight, "ds_id")
        deepspeed_engine2, *_ = deepspeed.initialize(model=model2, config_params=ds_config)

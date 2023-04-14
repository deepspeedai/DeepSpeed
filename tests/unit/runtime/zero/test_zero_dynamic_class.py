# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from unit.common import DistributedTest

import deepspeed


class TestNewClassDeclaredInsideInit(DistributedTest):
    world_size = 1

    def test_new_class_declared_inside_init(self):
        ds_config = dict(train_batch_size=1, zero_optimization=dict(stage=3))

        with deepspeed.zero.Init(config_dict_or_path=ds_config):

            class MyModel(torch.nn.Module):

                def __init__(self):
                    super().__init__()
                    self.fc = torch.nn.Linear(4, 4)

            with deepspeed.zero.Init(config_dict_or_path=ds_config):
                model = MyModel()

        deepspeed_engine, *_ = deepspeed.initialize(model=model, config_params=ds_config)
        # ensure that zero3 processed the parameter
        assert hasattr(deepspeed_engine.fc.weight, "ds_id")


class TestNewClassDeclaredInsideInitFailure(DistributedTest):
    world_size = 1

    def test_new_class_declared_inside_init_failure(self):
        ds_config = dict(train_batch_size=1, zero_optimization=dict(stage=3))

        try:
            with deepspeed.zero.Init(config_dict_or_path=ds_config):

                class MyModel(torch.nn.Module):

                    def __init__(self):
                        super().__init__()
                        self.fc = torch.nn.Linear(1, 1)

                model = MyModel()

            assert False, "Should have failed. A subclass of torch.nn.Module must be defined before zero.Init() where an instance of the class is created."
        except RuntimeError as e:
            pass
        except:
            assert False, "Should have failed. Runtime error is expected."

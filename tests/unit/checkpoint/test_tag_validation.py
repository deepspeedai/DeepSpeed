# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed

from unit.common import DistributedTest
from unit.simple_model import *

import pytest


class TestCheckpointValidationTag(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('valid_mode', ["FAIL", "WARN", "IGNORE"])
    def test_checkpoint_unique_tag(self, tmpdir, valid_mode):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "checkpoint": {
                "tag_validation": valid_mode
            }
        }
        hidden_dim = 10
        model = SimpleModel(hidden_dim)

        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        if valid_mode == "FAIL":
            with pytest.raises(AssertionError):
                model.save_checkpoint(save_dir=tmpdir, tag=f"tag-{dist.get_rank()}")
        else:
            model.save_checkpoint(save_dir=tmpdir, tag=f"tag-{dist.get_rank()}")

    def test_checkpoint_unknown_tag_validation(self, tmpdir):

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "checkpoint": {
                "tag_validation": "foo"
            }
        }
        hidden_dim = 10
        args = args_from_dict(tmpdir, config_dict)
        model = SimpleModel(hidden_dim)

        with pytest.raises(deepspeed.DeepSpeedConfigError):
            model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

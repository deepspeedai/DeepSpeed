# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed

import pytest
from unit.common import DistributedTest
from unit.simple_model import *

from unit.checkpoint.common import checkpoint_correctness_verification
from deepspeed.ops.op_builder import FusedAdamBuilder

if not deepspeed.ops.__compatible_ops__[FusedAdamBuilder.NAME]:
    pytest.skip("This op had not been implemented on this system.", allow_module_level=True)


class TestLatestCheckpoint(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize('compile_mode', [True, False])
    def test_existing_latest(self, tmpdir, compile_mode):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            }
        }
        hidden_dim = 10
        models = [SimpleModel(hidden_dim=hidden_dim) for _ in range(2)]
        checkpoint_correctness_verification(config_dict=config_dict,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=True,
                                            load_lr_scheduler_states=False,
                                            empty_tag=True,
                                            dtype=torch.float,
                                            compile_mode=compile_mode)

    @pytest.mark.parametrize('compile_mode', [True, False])
    def test_missing_latest(self, tmpdir, compile_mode):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            }
        }
        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        if compile_mode:
            model.compile()
        # should be no-op, since latest doesn't exist
        model.load_checkpoint(tmpdir)

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

from unit.common import DistributedTest
from unit.simple_model import *
from unit.util import required_torch_version

from unit.checkpoint.common import checkpoint_correctness_verification

import pytest


class TestMoECheckpoint(DistributedTest):
    world_size = 4

    @pytest.mark.parametrize("ep_size", [4])
    def test_checkpoint_moe(self, tmpdir, ep_size):
        if not required_torch_version():
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {"train_batch_size": 8, "steps_per_print": 1, "fp16": {"enabled": True}}
        hidden_dim = 16

        models = [SimpleMoEModel(hidden_dim=hidden_dim, num_experts=ep_size, ep_size=ep_size) for _ in range(2)]
        optimizers = [torch.optim.AdamW(params=model.parameters()) for model in models]
        checkpoint_correctness_verification(config_dict,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=True,
                                            load_lr_scheduler_states=False,
                                            fp16=config_dict["fp16"]["enabled"],
                                            empty_tag=True,
                                            base_optimizers=optimizers,
                                            seq_dataloader=True)

    @pytest.mark.parametrize("ep_size, load_optim_states", [(4, True), (4, False), (2, True), (2, False)])
    def test_checkpoint_moe_and_zero(self, tmpdir, ep_size, load_optim_states):
        if not required_torch_version():
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {
            "train_batch_size": 8,
            "steps_per_print": 1,
            "optimizer": {
                "type": 'Adam',
                "params": {
                    "lr": 0.00015,
                    "betas": [0.8, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "zero_optimization": {
                "stage": 2,
            }
        }
        hidden_dim = 16

        models = [SimpleMoEModel(hidden_dim=hidden_dim, num_experts=ep_size, ep_size=ep_size) for _ in range(2)]
        # param group must have a random unique name (for now)
        # TODO: clean-up this requirement, the unique name should not be required here
        param_groups = [{'params': [p for p in model.parameters()], 'name': 'random-unique-name'} for model in models]
        params = [split_params_into_different_moe_groups_for_optimizer(group) for group in param_groups]
        optimizers = [torch.optim.AdamW(params=param) for param in params]
        checkpoint_correctness_verification(config_dict,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=load_optim_states,
                                            load_lr_scheduler_states=False,
                                            fp16=config_dict["fp16"]["enabled"],
                                            empty_tag=True,
                                            base_optimizers=optimizers,
                                            seq_dataloader=True)

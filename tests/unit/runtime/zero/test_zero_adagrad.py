# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest


class TestZeroAdagradParamGroups(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("stage", [0, 1, 2, 3])
    def test_groups_and_updates_match_torch(self, stage):
        torch.manual_seed(23)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
        model.to(get_accelerator().current_device_name())
        reference = copy.deepcopy(model)

        def groups(module):
            return [{
                "params": [module[0].weight, module[1].weight],
                "lr": 0.03,
                "eps": 1e-8,
                "initial_accumulator_value": 0.1,
                "name": "weight"
            }, {
                "params": [module[0].bias, module[1].bias],
                "lr": 0.12,
                "eps": 1e-5,
                "initial_accumulator_value": 0.2,
                "name": "bias"
            }]

        supplied = torch.optim.Adagrad(groups(model))
        # Match the unclipped Torch reference; split ZeRO-3 groups into multiple subgroups.
        config = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_clipping": 0.0,
            "zero_optimization": {
                "stage": stage,
                "sub_group_size": 1
            }
        }
        engine, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=supplied, config=config)
        reference.to(engine.device)
        oracle = torch.optim.Adagrad(groups(reference))
        keys = ("lr", "eps", "initial_accumulator_value", "name")
        assert [{
            key: group.get(key)
            for key in keys
        } for group in optimizer.param_groups] == [{
            key: group[key]
            for key in keys
        } for group in oracle.param_groups]

        for _ in range(2):
            inputs = torch.randn(2, 4, device=engine.device)
            loss = engine(inputs).square().mean()
            reference_loss = reference(inputs).square().mean()
            torch.testing.assert_close(loss, reference_loss)
            engine.backward(loss)
            engine.step()
            reference_loss.backward()
            oracle.step()
            oracle.zero_grad()
            with deepspeed.zero.GatheredParameters(list(model.parameters())):
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected)

# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import copy

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest


@pytest.mark.parametrize("zero_stage", [2, 3], ids=["zero2", "zero3"])
@pytest.mark.parametrize("multiple_groups", [False, True], ids=["single_group", "multiple_groups"])
class TestZeroAdagradParamGroups(DistributedTest):
    world_size = 2

    def test_updates_match_torch(self, zero_stage, multiple_groups):
        device = get_accelerator().current_device_name()
        torch.manual_seed(1234)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4)).to(device)
        reference = copy.deepcopy(model)

        def make_optimizer(module):
            params = list(module.parameters())
            groups = [{"params": params, "lr": 0.1, "weight_decay": 0.01}]
            if multiple_groups:
                # With sub_group_size=4, each layer spans two ZeRO-3 subgroups.
                groups[0]["params"] = list(module[0].parameters())
                groups.append({"params": list(module[1].parameters()), "lr": 0.01, "weight_decay": 0.0})
            return torch.optim.Adagrad(groups, lr=0.03, weight_decay=0.1, initial_accumulator_value=0.5)

        reference_optimizer = make_optimizer(reference)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=make_optimizer(model),
                                               config={
                                                   "train_micro_batch_size_per_gpu": 2,
                                                   "gradient_clipping": 0.0,
                                                   "zero_optimization": {
                                                       "stage": zero_stage,
                                                       "reduce_bucket_size": 32,
                                                       "allgather_bucket_size": 32,
                                                       "sub_group_size": 4,
                                                   },
                                               })

        inputs = torch.tensor([[0.2, -0.4, 0.7, 1.1], [-0.9, 0.3, 1.3, -0.1]], device=device)
        targets = torch.tensor([[0.5, -0.8, 0.1, 0.9], [-0.3, 0.4, -1.2, 0.6]], device=device)
        for _ in range(3):
            engine.backward(torch.nn.functional.mse_loss(engine(inputs), targets))
            engine.step()
            reference_optimizer.zero_grad()
            torch.nn.functional.mse_loss(reference(inputs), targets).backward()
            reference_optimizer.step()

            with deepspeed.zero.GatheredParameters(model.parameters(), enabled=zero_stage == 3):
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected)

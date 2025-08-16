# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader
import deepspeed


class BaseZenFlowTest:
    hidden_dim = 10
    batch_size = 4
    grad_acc_steps = 1

    def get_config_dict(self, stage, offload_selective_optimizer, select_strategy, select_interval, update_interval,
                        full_warm_up_rounds):
        config = {
            "train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.grad_acc_steps,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
            "zero_optimization": {
                "stage": stage,
                "offload_optimizer": {
                    "device": "cpu"
                },
                "overlap_comm": True,
                "zenflow": {
                    "topk_ratio": 0.2,
                    "select_strategy": select_strategy,
                    "select_interval": select_interval,
                    "update_interval": update_interval,
                    "overlap_step": False,
                    "offload": offload_selective_optimizer,
                    "auto_ratio": 0.99,
                    "full_warm_up_rounds": full_warm_up_rounds,
                }
            },
            "zero_allow_untested_optimizer": True,
        }

        if get_accelerator().is_bf16_supported():
            config["bf16"] = {"enabled": True}
        return config

    def run_training_distributed(self, config_dict):

        if get_accelerator().device_name() == "cpu":
            return

        model = SimpleModel(self.hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        train_dataloader = random_dataloader(model=model,
                                             total_samples=20,
                                             hidden_dim=self.hidden_dim,
                                             device=model.device)

        dist.barrier()

        for step, batch in enumerate(train_dataloader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
        model.destroy()


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("full_warm_up_rounds", [0, 3])
@pytest.mark.parametrize("offload_selective_optimizer", [True, False])
@pytest.mark.parametrize("select_strategy,select_interval,update_interval", [
    ("auto", "auto", "auto"),
    ("step", 10, 3),
    ("epoch", 1, 4),
])
class TestZenFlowSingleGPU(DistributedTest, BaseZenFlowTest):
    world_size = 1

    def test_zenflow_single_gpu(self, stage, offload_selective_optimizer, select_strategy, select_interval,
                                update_interval, full_warm_up_rounds):
        tester = BaseZenFlowTest()
        config_dict = tester.get_config_dict(stage, offload_selective_optimizer, select_strategy, select_interval,
                                             update_interval, full_warm_up_rounds)
        tester.run_training_distributed(config_dict)


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("full_warm_up_rounds", [0, 3])
@pytest.mark.parametrize("offload_selective_optimizer", [True, False])
@pytest.mark.parametrize("select_strategy,select_interval,update_interval", [
    ("auto", "auto", "auto"),
    ("step", 10, 3),
    ("epoch", 1, 4),
])
class TestZenFlowDistributed(DistributedTest, BaseZenFlowTest):
    world_size = 2

    def test_zenflow_distributed(self, stage, offload_selective_optimizer, select_strategy, select_interval,
                                 update_interval, full_warm_up_rounds):
        config_dict = self.get_config_dict(stage, offload_selective_optimizer, select_strategy, select_interval,
                                           update_interval, full_warm_up_rounds)
        self.run_training_distributed(config_dict)

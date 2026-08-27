# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import os
import deepspeed
from deepspeed.accelerator import get_accelerator
import pytest
from unit.common import DistributedTest
from unit.simple_model import Curriculum_SimpleModel, SimpleModel, random_dataloader, random_dataset
from deepspeed.runtime.data_pipeline.curriculum_scheduler import CurriculumScheduler


class MPU():

    def __init__(self, tp_world_size):
        self.rank = deepspeed.comm.get_rank()
        self.world_size = deepspeed.comm.get_world_size()
        self.tp_world_size = tp_world_size

        for i in range(0, self.world_size, tp_world_size):
            ranks = range(i, i + tp_world_size)
            group = deepspeed.comm.new_group(ranks)
            if self.rank in ranks:
                self.tp_group = group

        for i in range(0, tp_world_size):
            ranks = range(i, self.world_size, tp_world_size)
            group = deepspeed.comm.new_group(ranks)
            if self.rank in ranks:
                self.dp_group = group

    def get_model_parallel_rank(self):
        return self.rank % self.tp_world_size

    def get_model_parallel_world_size(self):
        return self.tp_world_size

    def get_data_parallel_rank(self):
        return self.rank // self.tp_world_size

    def get_data_parallel_world_size(self):
        return self.world_size // self.tp_world_size

    def get_data_parallel_group(self):
        return self.dp_group

    def get_model_parallel_group(self):
        return self.tp_group


def _curriculum_scheduler(min_difficulty, max_difficulty, difficulty_step, schedule_type="fixed_linear"):
    config = {
        "min_difficulty": min_difficulty,
        "max_difficulty": max_difficulty,
        "schedule_type": schedule_type,
        "schedule_config": {
            "total_curriculum_step": 100,
            "difficulty_step": difficulty_step,
        },
    }
    if schedule_type == "fixed_root":
        config["schedule_config"]["root_degree"] = 2
    return CurriculumScheduler(config)


@pytest.mark.parametrize("schedule_type", ["fixed_linear", "fixed_root"])
@pytest.mark.parametrize("min_difficulty, difficulty_step", [(8, 16), (1, 8), (10, 8), (64, 16), (8, 8)])
def test_curriculum_never_starts_below_min_difficulty(schedule_type, min_difficulty, difficulty_step):
    # Rounding down to a multiple of difficulty_step used to push the first steps under
    # the configured start: min_difficulty 8 with difficulty_step 16, a pair the tutorial
    # recommends for INT8 data on a million-scale model, gave a difficulty of 0, which is
    # a zero-length sequence for the seqlen metric.
    scheduler = _curriculum_scheduler(min_difficulty, 1024, difficulty_step, schedule_type)
    difficulties = [scheduler.get_difficulty(step) for step in range(20)]
    assert min(difficulties) >= min_difficulty
    assert max(difficulties) <= 1024


def test_curriculum_endpoints_are_the_configured_values():
    # Neither end is a multiple of difficulty_step here. The top has always come back as
    # max_difficulty once the schedule runs out, because the existing clamp says so; the
    # bottom now comes back as min_difficulty for the same reason.
    scheduler = _curriculum_scheduler(8, 1000, 16)
    assert scheduler.get_difficulty(0) == 8
    assert scheduler.get_difficulty(200) == 1000


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
class TestDataEfficiency(DistributedTest):
    world_size = 2

    def test_curriculum_learning(self, dtype):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU accelerator does not support this test yet")
        if not dtype in get_accelerator().supported_dtypes():
            pytest.skip(f"This test does not support {dtype=}.")

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01
                }
            },
            "gradient_clipping": 1.0,
            "data_efficiency": {
                "enabled": True,
                "seed": 1234,
                "data_sampling": {
                    "enabled": True,
                    "num_workers": 0,
                    "curriculum_learning": {
                        "enabled": True,
                        "data_cluster_path": "/tmp",
                        "curriculum_metrics": {
                            "dummy_metric": {
                                "index_to_sample_path": "dummy",
                                "index_to_metric_path": "dummy",
                                "difficulty_type": "value",
                                "clustering_type": "single_cluster",
                                "min_difficulty": 2,
                                "max_difficulty": 10,
                                "schedule_type": "fixed_root",
                                "schedule_config": {
                                    "total_curriculum_step": 8,
                                    "difficulty_step": 2,
                                    "root_degree": 1
                                }
                            }
                        }
                    }
                }
            }
        }

        if dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "loss_scale": 0, "initial_scale_power": 8}
        else:
            config_dict["bf16"] = {"enabled": True}

        def data_post_process(data, data_sampler_state_dict):
            assert 'dummy_metric' in data_sampler_state_dict['current_difficulties']
            return data

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        dataset = random_dataset(20, hidden_dim, torch.device('cpu'), dtype=dtype)
        model, _, data_loader, _ = deepspeed.initialize(config=config_dict,
                                                        model=model,
                                                        training_data=dataset,
                                                        model_parameters=model.parameters(),
                                                        mpu=MPU(1))
        if model.mpu.get_data_parallel_rank() == 0 and not os.path.exists('/tmp'):
            os.makedirs('/tmp')
        model.set_data_post_process_func(data_post_process)
        for n, batch in enumerate(data_loader):
            x = batch[0].to(get_accelerator().current_device_name())
            y = batch[1].to(get_accelerator().current_device_name())
            loss = model(x, y)
            model.backward(loss)
            model.step()
            if n >= 10:
                break


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
class TestLegacyCurriculumScheduler(DistributedTest):
    world_size = 2

    def test_fixed_discrete(self, dtype):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU accelerator does not support this test yet")
        if not dtype in get_accelerator().supported_dtypes():
            pytest.skip(f"This test does not support {dtype=}.")

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01
                }
            },
            "gradient_clipping": 1.0,
            "curriculum_learning": {
                "enabled": True,
                "curriculum_type": "seqlen",
                "min_difficulty": 1,
                "max_difficulty": 5,
                "schedule_type": "fixed_discrete",
                "schedule_config": {
                    "difficulty": [1, 2, 3, 4, 5],
                    "max_step": [2, 4, 6, 8]
                }
            }
        }
        if dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "loss_scale": 0, "initial_scale_power": 8}
        else:
            config_dict["bf16"] = {"enabled": True}
        hidden_dim = 10
        ground_truths = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}

        model = Curriculum_SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=20,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=dtype)
        for n, batch in enumerate(data_loader):
            loss, seqlen = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            true_seqlen = 5
            if n + 1 in ground_truths:
                true_seqlen = ground_truths[n + 1]
            assert seqlen == true_seqlen, f"Incorrect curriculum schedule {n=}, {seqlen=}, {true_seqlen=}"

    def test_fixed_linear(self, dtype):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU accelerator does not support this test yet")
        if not dtype in get_accelerator().supported_dtypes():
            pytest.skip(f"This test does not support {dtype=}.")

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01
                }
            },
            "gradient_clipping": 1.0,
            "curriculum_learning": {
                "enabled": True,
                "curriculum_type": "seqlen",
                "min_difficulty": 2,
                "max_difficulty": 10,
                "schedule_type": "fixed_linear",
                "schedule_config": {
                    "total_curriculum_step": 8,
                    "difficulty_step": 2
                }
            }
        }
        if dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "loss_scale": 0, "initial_scale_power": 8}
        else:
            config_dict["bf16"] = {"enabled": True}
        hidden_dim = 10
        ground_truths = {1: 2, 2: 4, 3: 4, 4: 6, 5: 6, 6: 8, 7: 8, 8: 10, 9: 10, 10: 10}

        model = Curriculum_SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=20,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=dtype)
        for n, batch in enumerate(data_loader):
            loss, seqlen = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            if n + 1 in ground_truths:
                true_seqlen = ground_truths[n + 1]
                assert seqlen == true_seqlen, "Incorrect curriculum schedule"

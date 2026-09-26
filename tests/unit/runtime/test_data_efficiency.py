# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import os
import csv
import deepspeed
from deepspeed.accelerator import get_accelerator
import pytest
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataset
from deepspeed.runtime.data_pipeline.curriculum_scheduler import CurriculumScheduler
from deepspeed.runtime.data_pipeline.data_sampling.data_analyzer import DataAnalyzer
from deepspeed.runtime.data_pipeline.data_sampling.indexed_dataset import MMapIndexedDataset


@pytest.mark.parametrize("batch_kind", ["tensor", "dict", "tuple"])
@pytest.mark.parametrize("shuffle_indices", [False, True])
def test_data_analyzer_tracks_samples_in_structured_batches(tmp_path, batch_kind, shuffle_indices):
    values = torch.arange(11, dtype=torch.long)
    if batch_kind == "dict":
        dataset = [{"value": value, "extra": value + 1} for value in values]
        metric = lambda batch: batch["value"]
    elif batch_kind == "tuple":
        dataset = torch.utils.data.TensorDataset(values, values + 1)
        metric = lambda batch: batch[0]
    else:
        dataset = values
        metric = lambda batch: batch
    sample_indices = list(reversed(range(len(dataset)))) if shuffle_indices else None

    # Two workers also exercise nonzero offsets and a partial final batch.
    observed = {}
    for worker_id in range(2):
        analyzer = DataAnalyzer(dataset,
                                num_workers=2,
                                worker_id=worker_id,
                                batch_size=3,
                                metric_names=["value"],
                                metric_functions=[metric],
                                metric_types=["single_value_per_sample"],
                                metric_dtypes=[torch.int64],
                                save_path=str(tmp_path),
                                sample_indices=sample_indices)
        analyzer.run_map()
        directory = tmp_path / "value" / f"worker{worker_id}_thread0"
        stored_values = MMapIndexedDataset(str(directory / "value_sample_to_metric"), skip_warmup=True)
        for value in stored_values:
            value = int(value[0])
            with (directory / f"value_metric_to_sample_{value}.csv").open() as handle:
                indices = [int(index) for row in csv.reader(handle) for index in row]
            assert len(indices) == 1
            observed[value] = indices[0]

    expected = {index: sample_indices[index] if shuffle_indices else index for index in range(len(dataset))}
    assert observed == expected


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
@pytest.mark.parametrize("min_difficulty, difficulty_step", [(8, 16), (1, 8), (10, 8), (64, 16), (8, 8), (100, 64)])
def test_curriculum_never_starts_below_min_difficulty(schedule_type, min_difficulty, difficulty_step):
    # Rounding down to a multiple of difficulty_step used to push the first steps under
    # the configured start: min_difficulty 8 with difficulty_step 16 gave a difficulty
    # of 0, which is a zero-length sequence for the seqlen metric. The step alignment the
    # constructor warns about has to survive the new floor, so it is asserted alongside.
    scheduler = _curriculum_scheduler(min_difficulty, 1024, difficulty_step, schedule_type)
    difficulties = [scheduler.get_difficulty(step) for step in range(100)]
    assert min(difficulties) >= min_difficulty
    assert max(difficulties) <= 1024
    assert all(difficulty % difficulty_step == 0 for difficulty in difficulties)


def test_curriculum_first_difficulty_is_the_aligned_min():
    # min_difficulty 8 is not a multiple of difficulty_step 16, so it cannot itself be a
    # difficulty. The schedule starts at the first multiple at or above it, 16, which is
    # the only value that is both no lower than asked and aligned to the step.
    scheduler = _curriculum_scheduler(8, 1024, 16)
    assert scheduler.get_difficulty(0) == 16

    # a min_difficulty that is already a multiple is used unchanged
    assert _curriculum_scheduler(64, 1024, 16).get_difficulty(0) == 64


def test_curriculum_tops_out_at_the_configured_max():
    # max_difficulty is not a multiple of difficulty_step here. The existing clamp caps
    # the ramp at the configured value, and this change does not touch that end.
    scheduler = _curriculum_scheduler(16, 1000, 16)
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

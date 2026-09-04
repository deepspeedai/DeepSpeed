# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from types import SimpleNamespace

import torch.nn as nn

import deepspeed
import deepspeed.comm as dist
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.pipe.engine import PipelineEngine
from unit.common import DistributedTest
from unit.simple_model import SimpleModel


def _bare_engine(mpu=None):
    engine = DeepSpeedEngine.__new__(DeepSpeedEngine)
    engine.mpu = mpu
    return engine


def test_pipeline_rank_defaults_without_pp():
    engine = _bare_engine(mpu=None)
    assert engine.get_pipeline_parallel_rank() == 0
    assert engine.get_pipeline_parallel_world_size() == 1


def test_pipeline_rank_uses_pipeline_model_parallel_rank():
    engine = _bare_engine(mpu=SimpleNamespace(get_pipeline_model_parallel_rank=lambda: 3))
    assert engine.get_pipeline_parallel_rank() == 3


def test_pipeline_rank_uses_pipe_parallel_rank():
    engine = _bare_engine(mpu=SimpleNamespace(get_pipe_parallel_rank=lambda: 2))
    assert engine.get_pipeline_parallel_rank() == 2


def test_pipeline_engine_rank_uses_stage_id():
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.stage_id = 4
    assert engine.get_pipeline_parallel_rank() == 4


def test_sequence_only_mpu_does_not_alias_tp_or_mp():
    mpu = SimpleNamespace(
        initialize_sequence_parallel=lambda *args, **kwargs: None,
        get_model_parallel_rank=lambda: 3,
        get_model_parallel_world_size=lambda: 4,
        get_data_parallel_rank=lambda: 1,
        get_data_parallel_world_size=lambda: None,
    )
    engine = _bare_engine(mpu=mpu)
    assert engine.get_tensor_parallel_rank() == 0
    assert engine.get_tensor_parallel_world_size() == 1
    assert engine.get_model_parallel_rank() == 0
    assert engine.get_model_parallel_world_size() == 1


def test_data_parallel_world_size_is_int_for_sequence_only_mpu(monkeypatch):
    mpu = SimpleNamespace(initialize_sequence_parallel=lambda *args, **kwargs: None)
    engine = _bare_engine(mpu=mpu)

    monkeypatch.setattr(deepspeed.utils.groups, "_get_data_parallel_world_size", lambda: None)
    monkeypatch.setattr(deepspeed.runtime.engine.dist, "get_world_size", lambda: 8)

    size = engine.get_data_parallel_world_size()
    assert size == 8
    assert isinstance(size, int)


def test_pipeline_mpu_supplies_dp_and_tp_sizes():
    mpu = SimpleNamespace(
        get_data_parallel_rank=lambda: 1,
        get_data_parallel_world_size=lambda: 2,
        get_slice_parallel_rank=lambda: 1,
        get_slice_parallel_world_size=lambda: 2,
        get_pipe_parallel_rank=lambda: 0,
        get_pipe_parallel_world_size=lambda: 2,
    )
    engine = _bare_engine(mpu=mpu)
    assert engine.get_data_parallel_rank() == 1
    assert engine.get_data_parallel_world_size() == 2
    assert engine.get_tensor_parallel_rank() == 1
    assert engine.get_tensor_parallel_world_size() == 2
    assert engine.get_pipeline_parallel_rank() == 0
    assert engine.get_pipeline_parallel_world_size() == 2


class TestEngineParallelRanks(DistributedTest):
    world_size = 1

    def test_unused_parallelism_defaults(self):
        hidden_dim = 4
        model = SimpleModel(hidden_dim)
        config_dict = {"train_batch_size": 1}
        engine, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        assert engine.get_data_parallel_rank() == dist.get_rank()
        assert engine.get_data_parallel_world_size() == dist.get_world_size()

        assert engine.get_tensor_parallel_rank() == 0
        assert engine.get_tensor_parallel_world_size() == 1
        assert engine.get_model_parallel_rank() == 0
        assert engine.get_model_parallel_world_size() == 1

        assert engine.get_pipeline_parallel_rank() == 0
        assert engine.get_pipeline_parallel_world_size() == 1

        assert engine.get_sequence_parallel_rank() == 0
        assert engine.get_sequence_parallel_world_size() == 1


class TestPipelineEngineParallelRank(DistributedTest):
    world_size = 2

    def test_pipeline_rank_matches_stage_id(self):
        config = {
            "train_batch_size": 2,
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.001
                }
            },
            "pipeline": {
                "activation_checkpoint_interval": 0
            },
        }
        layers = [nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)]
        model = PipelineModule(layers=layers, num_stages=2, loss_fn=nn.MSELoss())
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())

        assert engine.get_pipeline_parallel_rank() == engine.stage_id
        assert engine.get_pipeline_parallel_world_size() == engine.num_stages

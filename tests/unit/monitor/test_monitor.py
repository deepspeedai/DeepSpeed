# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json
import os
import sys

import deepspeed
from deepspeed.monitor.tensorboard import TensorBoardMonitor
from deepspeed.monitor.wandb import WandbMonitor
from deepspeed.monitor.csv_monitor import csvMonitor
from deepspeed.monitor.config import DeepSpeedMonitorConfig
from deepspeed.monitor.comet import CometMonitor
from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.monitor.utils import collect_monitor_config

from unit.common import DistributedTest
from unit.simple_model import SimpleModel
from unittest.mock import Mock, patch
from deepspeed.runtime.config import DeepSpeedConfig

import deepspeed.comm as dist

EXPECTED_MONITOR_CONFIG_KEYS = {
    "zero_stage",
    "offload_optimizer",
    "offload_param",
    "fp16",
    "bf16",
    "train_batch_size",
    "train_micro_batch_size_per_gpu",
    "gradient_accumulation_steps",
    "data_parallel_world_size",
    "tensor_parallel_world_size",
    "pipeline_parallel_world_size",
    "sequence_parallel_world_size",
    "data_parallel_rank",
    "tensor_parallel_rank",
    "pipeline_parallel_rank",
    "sequence_parallel_rank",
}


class TestTensorBoard(DistributedTest):
    world_size = 2

    def test_tensorboard(self):
        config_dict = {
            "train_batch_size": 2,
            "tensorboard": {
                "enabled": True,
                "output_path": "test_output/ds_logs/",
                "job_name": "test"
            }
        }
        ds_config = DeepSpeedConfig(config_dict)
        tb_monitor = TensorBoardMonitor(ds_config.monitor_config.tensorboard)
        assert tb_monitor.enabled == True
        assert tb_monitor.output_path == "test_output/ds_logs/"
        assert tb_monitor.job_name == "test"

    def test_empty_tensorboard(self):
        config_dict = {"train_batch_size": 2, "tensorboard": {}}
        ds_config = DeepSpeedConfig(config_dict)
        tb_monitor = TensorBoardMonitor(ds_config.monitor_config.tensorboard)
        defaults = DeepSpeedMonitorConfig().tensorboard
        assert tb_monitor.enabled == defaults.enabled
        assert tb_monitor.output_path == defaults.output_path
        assert tb_monitor.job_name == defaults.job_name


class TestWandB(DistributedTest):
    world_size = 2

    def test_wandb(self):
        config_dict = {
            "train_batch_size": 2,
            "wandb": {
                "enabled": False,
                "group": "my_group",
                "team": "my_team",
                "project": "my_project"
            }
        }
        ds_config = DeepSpeedConfig(config_dict)
        wandb_monitor = WandbMonitor(ds_config.monitor_config.wandb)
        assert wandb_monitor.enabled == False
        assert wandb_monitor.group == "my_group"
        assert wandb_monitor.team == "my_team"
        assert wandb_monitor.project == "my_project"

    def test_empty_wandb(self):
        config_dict = {"train_batch_size": 2, "wandb": {}}
        ds_config = DeepSpeedConfig(config_dict)
        wandb_monitor = WandbMonitor(ds_config.monitor_config.wandb)
        defaults = DeepSpeedMonitorConfig().wandb
        assert wandb_monitor.enabled == defaults.enabled
        assert wandb_monitor.group == defaults.group
        assert wandb_monitor.team == defaults.team
        assert wandb_monitor.project == defaults.project


class TestCSVMonitor(DistributedTest):
    world_size = 2

    def test_csv_monitor(self):
        config_dict = {
            "train_batch_size": 2,
            "csv_monitor": {
                "enabled": True,
                "output_path": "test_output/ds_logs/",
                "job_name": "test"
            }
        }
        ds_config = DeepSpeedConfig(config_dict)
        csv_monitor = csvMonitor(ds_config.monitor_config.csv_monitor)
        assert csv_monitor.enabled == True
        assert csv_monitor.output_path == "test_output/ds_logs/"
        assert csv_monitor.job_name == "test"

    def test_empty_csv_monitor(self):
        config_dict = {"train_batch_size": 2, "csv_monitor": {}}
        ds_config = DeepSpeedConfig(config_dict)
        csv_monitor = csvMonitor(ds_config.monitor_config.csv_monitor)
        defaults = DeepSpeedMonitorConfig().csv_monitor
        assert csv_monitor.enabled == defaults.enabled
        assert csv_monitor.output_path == defaults.output_path
        assert csv_monitor.job_name == defaults.job_name


class TestCometMonitor(DistributedTest):
    world_size = 2

    def test_comet_monitor(self):
        import comet_ml
        mock_experiment = Mock()
        mock_start = Mock(return_value=mock_experiment)

        config_dict = {
            "train_batch_size": 2,
            "comet": {
                "enabled": True,
                "samples_log_interval": 42,
                "workspace": "some-workspace",
                "project": "some-project",
                "api_key": "some-api-key",
                "experiment_name": "some-experiment-name",
                "experiment_key": "some-experiment-key",
                "mode": "get_or_create",
                "online": True
            }
        }

        ds_config = DeepSpeedConfig(config_dict)

        with patch.object(comet_ml, "start", mock_start):
            comet_monitor = CometMonitor(ds_config.monitor_config.comet)

        assert comet_monitor.enabled is True
        assert comet_monitor.samples_log_interval == 42

        # experiment should be initialized via comet_ml.start only if rank == 0
        if dist.get_rank() == 0:
            mock_start.assert_called_once_with(
                api_key="some-api-key",
                project="some-project",
                workspace="some-workspace",
                experiment_key="some-experiment-key",
                mode="get_or_create",
                online=True,
            )

            mock_experiment.set_name.assert_called_once_with("some-experiment-name")
            assert comet_monitor.experiment is mock_experiment
        else:
            mock_start.assert_not_called()

    def test_empty_comet(self):
        import comet_ml
        mock_start = Mock()

        config_dict = {"train_batch_size": 2, "comet": {}}
        ds_config = DeepSpeedConfig(config_dict)

        with patch.object(comet_ml, "start", mock_start):
            comet_monitor = CometMonitor(ds_config.monitor_config.comet)

        defaults = DeepSpeedMonitorConfig().comet
        assert comet_monitor.enabled == defaults.enabled
        assert comet_monitor.samples_log_interval == defaults.samples_log_interval
        mock_start.assert_not_called()


class TestMonitorConfigUpdate(DistributedTest):
    world_size = 2

    def test_wandb_update_config_rank0_only(self):
        config_dict = {
            "train_batch_size": 2,
            "wandb": {
                "enabled": True,
                "group": "my_group",
                "team": "my_team",
                "project": "my_project"
            }
        }
        ds_config = DeepSpeedConfig(config_dict)
        payload = {
            "zero_stage": 2,
            "data_parallel_world_size": 2,
            "fp16": False,
        }
        mock_wandb = Mock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            with patch("deepspeed.monitor.wandb.check_wandb_availability"):
                wandb_monitor = WandbMonitor(ds_config.monitor_config.wandb)
                wandb_monitor.update_config(payload)

        if dist.get_rank() == 0:
            mock_wandb.init.assert_called_once()
            mock_wandb.config.update.assert_called_once_with(payload, allow_val_change=True)
            called_cfg = mock_wandb.config.update.call_args[0][0]
            assert "zero_stage" in called_cfg
            assert "data_parallel_world_size" in called_cfg
        else:
            mock_wandb.config.update.assert_not_called()

    def test_monitor_master_update_config_rank0_only(self):
        config_dict = {"train_batch_size": 2}
        ds_config = DeepSpeedConfig(config_dict)
        master = MonitorMaster(ds_config.monitor_config)
        master.tb_monitor = Mock()
        master.wandb_monitor = Mock()
        master.csv_monitor = Mock()
        master.comet_monitor = Mock()

        payload = {"zero_stage": 1, "data_parallel_world_size": 2}
        master.update_config(payload)

        if dist.get_rank() == 0:
            master.tb_monitor.update_config.assert_called_once_with(payload)
            master.wandb_monitor.update_config.assert_called_once_with(payload)
            master.csv_monitor.update_config.assert_called_once_with(payload)
            master.comet_monitor.update_config.assert_called_once_with(payload)
        else:
            master.tb_monitor.update_config.assert_not_called()
            master.wandb_monitor.update_config.assert_not_called()
            master.csv_monitor.update_config.assert_not_called()
            master.comet_monitor.update_config.assert_not_called()

    def test_comet_update_config(self):
        import comet_ml
        mock_experiment = Mock()
        mock_start = Mock(return_value=mock_experiment)

        config_dict = {
            "train_batch_size": 2,
            "comet": {
                "enabled": True,
                "project": "some-project",
                "api_key": "some-api-key"
            }
        }
        ds_config = DeepSpeedConfig(config_dict)
        payload = {"zero_stage": 1, "bf16": True}

        with patch.object(comet_ml, "start", mock_start):
            comet_monitor = CometMonitor(ds_config.monitor_config.comet)
            comet_monitor.update_config(payload)

        if dist.get_rank() == 0:
            mock_experiment.log_parameters.assert_called_once_with(payload)
        else:
            mock_experiment.log_parameters.assert_not_called()

    def test_tensorboard_update_config(self):
        config_dict = {
            "train_batch_size": 2,
            "tensorboard": {
                "enabled": True,
                "output_path": "test_output/ds_logs/",
                "job_name": "test"
            }
        }
        ds_config = DeepSpeedConfig(config_dict)
        tb_monitor = TensorBoardMonitor(ds_config.monitor_config.tensorboard)
        writer = Mock()
        if dist.get_rank() == 0:
            tb_monitor.summary_writer = writer
        payload = {"zero_stage": 0, "fp16": True}
        tb_monitor.update_config(payload)

        if dist.get_rank() == 0:
            writer.add_text.assert_called_once()
            args, kwargs = writer.add_text.call_args
            assert args[0] == "DeepSpeed/config"
            logged = json.loads(args[1])
            assert logged["zero_stage"] == 0
            assert logged["fp16"] is True
            assert kwargs.get("global_step", args[2] if len(args) > 2 else None) == 0
        else:
            assert tb_monitor.summary_writer is None
            writer.add_text.assert_not_called()

    def test_csv_update_config(self, tmpdir):
        config_dict = {
            "train_batch_size": 2,
            "csv_monitor": {
                "enabled": True,
                "output_path": str(tmpdir),
                "job_name": "cfg_test"
            }
        }
        ds_config = DeepSpeedConfig(config_dict)
        csv_monitor = csvMonitor(ds_config.monitor_config.csv_monitor)
        payload = {"zero_stage": 2, "fp16": False, "offload_optimizer": None}
        csv_monitor.update_config(payload)

        cfg_path = os.path.join(str(tmpdir), "cfg_test", "deepspeed_config.csv")
        if dist.get_rank() == 0:
            assert os.path.isfile(cfg_path)
            with open(cfg_path, "r") as f:
                contents = f.read()
            assert "key,value" in contents.replace(" ", "")
            assert "zero_stage" in contents
            assert "2" in contents
        else:
            assert not os.path.isfile(cfg_path)


class TestCollectMonitorConfig(DistributedTest):
    world_size = 2

    def test_collect_monitor_config_from_engine_methods(self):
        engine = Mock()
        engine.zero_optimization_stage.return_value = 3
        engine.zero_offload_optimizer.return_value = Mock(device="cpu")
        engine.zero_offload_param.return_value = None
        engine.fp16_enabled.return_value = False
        engine.bfloat16_enabled.return_value = True
        engine.train_batch_size.return_value = 16
        engine.train_micro_batch_size_per_gpu.return_value = 2
        engine.gradient_accumulation_steps.return_value = 4
        engine.pipeline_parallelism = False
        engine.mpu = None

        cfg = collect_monitor_config(engine)
        assert set(cfg.keys()) == EXPECTED_MONITOR_CONFIG_KEYS
        assert cfg["zero_stage"] == 3
        assert cfg["offload_optimizer"] == "cpu"
        assert cfg["offload_param"] is None
        assert cfg["fp16"] is False
        assert cfg["bf16"] is True
        assert cfg["train_batch_size"] == 16
        assert cfg["train_micro_batch_size_per_gpu"] == 2
        assert cfg["gradient_accumulation_steps"] == 4
        assert cfg["data_parallel_world_size"] == 2
        assert cfg["tensor_parallel_world_size"] == 1
        assert cfg["pipeline_parallel_world_size"] == 1
        assert cfg["sequence_parallel_world_size"] == 1
        assert cfg["pipeline_parallel_rank"] == 0
        assert cfg["sequence_parallel_rank"] == 0
        assert cfg["data_parallel_rank"] == dist.get_rank()
        for value in cfg.values():
            assert value is None or isinstance(value, (bool, int, str))

    def test_engine_init_collects_config(self, tmpdir):
        hidden_dim = 4
        model = SimpleModel(hidden_dim)
        config_dict = {
            "train_batch_size": 2,
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.0001
                }
            },
            "zero_optimization": {
                "stage": 2
            },
            "csv_monitor": {
                "enabled": True,
                "output_path": str(tmpdir),
                "job_name": "engine_cfg"
            }
        }
        engine, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        cfg = collect_monitor_config(engine)
        assert set(cfg.keys()) == EXPECTED_MONITOR_CONFIG_KEYS
        assert cfg["zero_stage"] == 2
        assert cfg["offload_optimizer"] is None
        assert cfg["offload_param"] is None
        assert cfg["fp16"] is False
        assert cfg["bf16"] is False
        assert cfg["train_batch_size"] == 2
        assert cfg["train_micro_batch_size_per_gpu"] == 1
        assert cfg["gradient_accumulation_steps"] == 1
        assert cfg["data_parallel_world_size"] == 2
        assert cfg["tensor_parallel_world_size"] == 1
        assert cfg["pipeline_parallel_world_size"] == 1
        assert cfg["sequence_parallel_world_size"] == 1
        assert cfg["pipeline_parallel_rank"] == 0

        cfg_path = os.path.join(str(tmpdir), "engine_cfg", "deepspeed_config.csv")
        if dist.get_rank() == 0:
            assert os.path.isfile(cfg_path)
            with open(cfg_path, "r") as f:
                contents = f.read()
            assert "zero_stage" in contents
            assert "data_parallel_world_size" in contents

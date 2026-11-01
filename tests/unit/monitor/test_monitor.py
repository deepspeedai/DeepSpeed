# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.monitor.tensorboard import TensorBoardMonitor
from deepspeed.monitor.wandb import WandbMonitor
from deepspeed.monitor.csv_monitor import csvMonitor
from deepspeed.monitor.config import DeepSpeedMonitorConfig
from deepspeed.monitor.comet import CometMonitor

from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader
from unit.util import bf16_required_version_check
from unittest.mock import Mock, patch
from deepspeed.runtime.config import DeepSpeedConfig

import csv
import os
import pytest
import torch
import deepspeed
import deepspeed.comm as dist


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


class TestMonitorTrainLoss(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("managed,dtype", [(True, torch.float), (False, torch.float), (True, torch.bfloat16)])
    def test_train_loss_logged_per_step(self, tmpdir, managed, dtype):
        # Enough micro-batches per step that summing them in bf16 would visibly drift
        gas = 256
        num_steps = 2
        hidden_dim = 10
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": gas,
            "managed_gradient_accumulation": managed,
            "csv_monitor": {
                "enabled": True,
                "output_path": str(tmpdir),
                "job_name": "loss"
            },
        }
        if dtype == torch.bfloat16:
            if not bf16_required_version_check():
                pytest.skip("bf16 is not supported on this accelerator")
            config_dict["bf16"] = {"enabled": True}
            config_dict["zero_optimization"] = {"stage": 1}
            config_dict["zero_allow_untested_optimizer"] = True
        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, optimizer=optimizer)
        data_loader = random_dataloader(model=model,
                                        total_samples=gas * num_steps * model.train_micro_batch_size_per_gpu(),
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=dtype)

        # The logged value is the mean of the micro-batch losses that went into each optimizer step.
        expected = []
        micro_losses = []
        for batch in data_loader:
            loss = model(batch[0], batch[1])
            micro_losses.append(loss.float().item())
            model.backward(loss)
            if managed or len(micro_losses) == gas:
                model.step()
            if len(micro_losses) == gas:
                expected.append(sum(micro_losses) / gas)
                micro_losses = []

        csv_path = os.path.join(str(tmpdir), "loss", "Train_Samples_train_loss.csv")
        assert os.path.isfile(csv_path), "Train/Samples/train_loss was not written to the monitor"
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["step", "train_loss"]
        logged = [float(value) for _, value in rows[1:]]
        # bf16 loses precision when each micro-batch loss is divided by gas, not when the losses are summed
        rel = 1e-2 if dtype == torch.bfloat16 else 1e-6
        assert logged == pytest.approx(expected, rel=rel)


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

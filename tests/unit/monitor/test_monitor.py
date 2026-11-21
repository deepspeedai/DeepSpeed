# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys

from deepspeed.monitor.tensorboard import TensorBoardMonitor
from deepspeed.monitor.wandb import WandbMonitor
from deepspeed.monitor.csv_monitor import csvMonitor
from deepspeed.monitor.config import DeepSpeedMonitorConfig
from deepspeed.monitor.comet import CometMonitor
from deepspeed.monitor.trackio import TrackioMonitor
from deepspeed.monitor.monitor import MonitorMaster

from unit.common import DistributedTest
from unittest.mock import Mock, MagicMock, patch
from deepspeed.runtime.config import DeepSpeedConfig

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


class TestTrackio(DistributedTest):
    world_size = 2

    def test_trackio(self):
        # trackio is an optional dependency, so we stub the module rather
        # than requiring it to be installed for CI.
        mock_trackio = MagicMock()

        config_dict = {"train_batch_size": 2, "trackio": {"enabled": True, "project": "my_project"}}
        ds_config = DeepSpeedConfig(config_dict)

        with patch.dict(sys.modules, {"trackio": mock_trackio}):
            trackio_monitor = TrackioMonitor(ds_config.monitor_config.trackio)

        assert trackio_monitor.enabled == True
        assert trackio_monitor.project == "my_project"

        # trackio.init should only be called on rank 0
        if dist.get_rank() == 0:
            mock_trackio.init.assert_called_once_with(project="my_project")
        else:
            mock_trackio.init.assert_not_called()

    def test_empty_trackio(self):
        mock_trackio = MagicMock()

        config_dict = {"train_batch_size": 2, "trackio": {}}
        ds_config = DeepSpeedConfig(config_dict)

        with patch.dict(sys.modules, {"trackio": mock_trackio}):
            trackio_monitor = TrackioMonitor(ds_config.monitor_config.trackio)

        defaults = DeepSpeedMonitorConfig().trackio
        assert trackio_monitor.enabled == defaults.enabled
        assert trackio_monitor.project == defaults.project

    def test_trackio_write_events(self):
        # Verifies write_events() correctly converts 3-tuples into
        # trackio.log() calls with the right step value.
        mock_trackio = MagicMock()

        config_dict = {"train_batch_size": 2, "trackio": {"enabled": True, "project": "my_project"}}
        ds_config = DeepSpeedConfig(config_dict)

        with patch.dict(sys.modules, {"trackio": mock_trackio}):
            trackio_monitor = TrackioMonitor(ds_config.monitor_config.trackio)
            events = [("Train/Loss", 0.5, 100)]
            trackio_monitor.write_events(events)

        if dist.get_rank() == 0:
            mock_trackio.log.assert_called_once_with({"Train/Loss": 0.5}, step=100)
        else:
            mock_trackio.log.assert_not_called()


class TestMonitorMasterTrackioWiring(DistributedTest):
    world_size = 2

    def test_trackio_enabled_creates_monitor(self):
        mock_trackio = MagicMock()

        config_dict = {"train_batch_size": 2, "trackio": {"enabled": True, "project": "my_project"}}
        ds_config = DeepSpeedConfig(config_dict)

        with patch.dict(sys.modules, {"trackio": mock_trackio}):
            monitor_master = MonitorMaster(ds_config.monitor_config)

        if dist.get_rank() == 0:
            assert monitor_master.trackio_monitor is not None
            assert isinstance(monitor_master.trackio_monitor, TrackioMonitor)
        else:
            assert monitor_master.trackio_monitor is None

    def test_trackio_disabled_skips_monitor(self):
        config_dict = {"train_batch_size": 2, "trackio": {"enabled": False}}
        ds_config = DeepSpeedConfig(config_dict)
        monitor_master = MonitorMaster(ds_config.monitor_config)
        assert monitor_master.trackio_monitor is None

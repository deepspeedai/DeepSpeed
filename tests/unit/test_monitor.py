import pytest

from deepspeed.monitor.constants import *

from deepspeed.monitor.tensorboard import TensorBoardMonitor
from deepspeed.monitor.wandb import WandbMonitor
from deepspeed.monitor.csv_monitor import csvMonitor

from .simple_model import *
from .common import distributed_test
from deepspeed.runtime.config import DeepSpeedConfig

try:
    import tensorboard  # noqa: F401
    _tb_available = True
except ImportError:
    _tb_available = False
tb_available = pytest.mark.skipif(not _tb_available,
                                  reason="tensorboard is not installed")

try:
    import wandb  # noqa: F401
    _wandb_available = True
except ImportError:
    _wandb_available = False
wandb_available = pytest.mark.skipif(not _wandb_available,
                                     reason="wandb is not installed")


@tb_available
def test_tensorboard(tmpdir):
    @distributed_test(world_size=2)
    def _test_tensorboard():
        config_dict = {
            "train_batch_size": 2,
            "tensorboard": {
                "enabled": True,
                "output_path": "test_output/ds_logs/",
                "job_name": "test"
            }
        }
        ds_config = DeepSpeedConfig(config_dict)
        tb_monitor = TensorBoardMonitor(ds_config.monitor_config)
        assert tb_monitor.enabled == True
        assert tb_monitor.output_path == "test_output/ds_logs/"
        assert tb_monitor.job_name == "test"

    _test_tensorboard()


@tb_available
def test_empty_tensorboard(tmpdir):
    @distributed_test(world_size=2)
    def _test_empty_tensorboard():
        config_dict = {"train_batch_size": 2, "tensorboard": {}}
        ds_config = DeepSpeedConfig(config_dict)
        tb_monitor = TensorBoardMonitor(ds_config.monitor_config)
        assert tb_monitor.enabled == TENSORBOARD_ENABLED_DEFAULT
        assert tb_monitor.output_path == TENSORBOARD_OUTPUT_PATH_DEFAULT
        assert tb_monitor.job_name == TENSORBOARD_JOB_NAME_DEFAULT

    _test_empty_tensorboard()


@wandb_available
def test_wandb(tmpdir):
    @distributed_test(world_size=2)
    def _test_wandb():
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
        wandb_monitor = WandbMonitor(ds_config.monitor_config)
        assert wandb_monitor.enabled == False
        assert wandb_monitor.group == "my_group"
        assert wandb_monitor.team == "my_team"
        assert wandb_monitor.project == "my_project"

    _test_wandb()


@wandb_available
def test_empty_wandb(tmpdir):
    @distributed_test(world_size=2)
    def _test_empty_wandb():
        config_dict = {"train_batch_size": 2, "wandb": {}}
        ds_config = DeepSpeedConfig(config_dict)
        wandb_monitor = WandbMonitor(ds_config.monitor_config)
        assert wandb_monitor.enabled == WANDB_ENABLED_DEFAULT
        assert wandb_monitor.group == WANDB_GROUP_NAME_DEFAULT
        assert wandb_monitor.team == WANDB_TEAM_NAME_DEFAULT
        assert wandb_monitor.project == WANDB_PROJECT_NAME_DEFAULT

    _test_empty_wandb()


def test_csv_monitor(tmpdir):
    @distributed_test(world_size=2)
    def _test_csv_monitor():
        config_dict = {
            "train_batch_size": 2,
            "csv_monitor": {
                "enabled": True,
                "output_path": "test_output/ds_logs/",
                "job_name": "test"
            }
        }
        ds_config = DeepSpeedConfig(config_dict)
        csv_monitor = csvMonitor(ds_config.monitor_config)
        assert csv_monitor.enabled == True
        assert csv_monitor.output_path == "test_output/ds_logs/"
        assert csv_monitor.job_name == "test"

    _test_csv_monitor()


def test_empty_csv_monitor(tmpdir):
    @distributed_test(world_size=2)
    def _test_empty_csv_monitor():
        config_dict = {"train_batch_size": 2, "csv_monitor": {}}
        ds_config = DeepSpeedConfig(config_dict)
        csv_monitor = csvMonitor(ds_config.monitor_config)
        assert csv_monitor.enabled == CSV_MONITOR_ENABLED_DEFAULT
        assert csv_monitor.output_path == CSV_MONITOR_OUTPUT_PATH_DEFAULT
        assert csv_monitor.job_name == CSV_MONITOR_JOB_NAME_DEFAULT

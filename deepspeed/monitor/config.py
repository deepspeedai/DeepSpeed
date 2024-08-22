# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

from pydantic import model_validator
from deepspeed.runtime.config_utils import DeepSpeedConfigModel


def get_monitor_config(param_dict):
    monitor_dict = {key: param_dict.get(key, {}) for key in ("tensorboard", "wandb", "csv_monitor", "comet")}
    return DeepSpeedMonitorConfig(**monitor_dict)


class TensorBoardConfig(DeepSpeedConfigModel):
    """Sets parameters for TensorBoard monitor."""

    enabled: bool = False
    """ Whether logging to Tensorboard is enabled. Requires `tensorboard` package is installed. """

    output_path: str = ""
    """
    Path to where the Tensorboard logs will be written. If not provided, the
    output path is set under the training script’s launching path.
    """

    job_name: str = "DeepSpeedJobName"
    """ Name for the current job. This will become a new directory inside `output_path`. """


class WandbConfig(DeepSpeedConfigModel):
    """Sets parameters for WandB monitor."""

    enabled: bool = False
    """ Whether logging to WandB is enabled. Requires `wandb` package is installed. """

    group: Optional[str] = None
    """ Name for the WandB group. This can be used to group together runs. """

    team: Optional[str] = None
    """ Name for the WandB team. """

    project: str = "deepspeed"
    """ Name for the WandB project. """


class CSVConfig(DeepSpeedConfigModel):
    """Sets parameters for CSV monitor."""

    enabled: bool = False
    """ Whether logging to local CSV files is enabled. """

    output_path: str = ""
    """
    Path to where the csv files will be written. If not provided, the output
    path is set under the training script’s launching path.
    """

    job_name: str = "DeepSpeedJobName"
    """ Name for the current job. This will become a new directory inside `output_path`. """


class CometConfig(DeepSpeedConfigModel):
    """
    Sets parameters for Comet monitor. For logging data Comet uses
    experiment object.
    https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/
    """

    enabled: bool = False
    """ Whether logging to Comet is enabled. Requires `comet_ml` package is installed. """

    samples_log_interval: int = 100
    """ Metrics will be submitted to Comet after processing every `samples_log_intervas` samples"""

    project: Optional[str] = None
    """
    Comet project name. Can be set through .comet.config file or environment variable COMET_PROJECT_NAME
    https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/#explore-comet-configuration-options
    """

    workspace: Optional[str] = None
    """
    Comet workspace name. Can be set through .comet.config file or environment variable COMET_WORKSPACE
    https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/#explore-comet-configuration-options
    """

    api_key: Optional[str] = None
    """
    Comet API key. Can be set through .comet.config file or environment variable COMET_API_KEY
    https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/#explore-comet-configuration-options
    """

    experiment_name: Optional[str] = None
    """
    The name for comet experiment to be used for logging.
    Can be set through .comet.config file or environment variable COMET_EXPERIMENT_NAME
    https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/#explore-comet-configuration-options
    """

    experiment_key: Optional[str] = None
    """
    The key for comet experiment to be used for logging. Must be an alphanumeric string whose length is between 32 and 50 characters.
    Can be set through .comet.config  or environment variable COMET_EXPERIMENT_KEY
    https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/#explore-comet-configuration-options
    """

    online: Optional[bool] = None
    """
    If True, the data will be logged to Comet server, otherwise it will be stored locally in offline experiment
    Defaults to True.
    """

    mode: Optional[str] = None
    """
    Control how the Comet experiment is started, 3 options are possible.:
        - "get": Continue logging to an existing experiment identified by the `experiment_key` value.
        - "create": Always creates of a new experiment, useful for HPO sweeps.
        - "get_or_create" (default): Starts a fresh experiment if required, or persists logging to an existing one.
    """


class DeepSpeedMonitorConfig(DeepSpeedConfigModel):
    """Sets parameters for various monitoring methods."""

    tensorboard: TensorBoardConfig = {}
    """ TensorBoard monitor, requires `tensorboard` package is installed. """

    comet: CometConfig = {}
    """ Comet monitor, requires `comet_ml` package is installed """

    wandb: WandbConfig = {}
    """ WandB monitor, requires `wandb` package is installed. """

    csv_monitor: CSVConfig = {}
    """ Local CSV output of monitoring data. """

    @model_validator(mode="after")
    def check_enabled(self):
        enabled = self.tensorboard.enabled or self.wandb.enabled or self.csv_monitor.enabled or self.comet.enabled
        self.__dict__["enabled"] = enabled
        return self

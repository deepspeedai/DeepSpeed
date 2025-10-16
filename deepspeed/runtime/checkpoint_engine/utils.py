# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.model_checkpointing.constants import *
from deepspeed.runtime.model_checkpointing.utils import create_data_parallel_writer_config
from deepspeed.utils import logger
from deepspeed import comm as dist
from .decoupled_checkpoint_engine import DecoupledCheckpointEngine
from .fast_checkpoint_engine import FastCheckpointEngine
from .torch_checkpoint_engine import TorchCheckpointEngine


def create_checkpoint_engine(config_params, groups, zero_stage, has_moe_layers, optimize_dp_state):
    if config_params is not None:
        if config_params.checkpoint_config[CHECKPOINT_WRITER] is not None:
            writer_config = config_params.checkpoint_config[CHECKPOINT_WRITER]
            dp_writer_config = create_data_parallel_writer_config(
                groups=groups,
                parallel_unit=writer_config[CHECKPOINT_DATA_PARALLEL],
                zero_stage=zero_stage,
                has_moe_layers=has_moe_layers)
            if writer_config[CHECKPOINT_WRITER_DECOUPLED]:
                return DecoupledCheckpointEngine(config_params, dp_writer_config, optimize_dp_state)
            else:
                return FastCheckpointEngine(config_params, dp_writer_config, optimize_dp_state)

        if config_params is not None and config_params.nebula_config.enabled:
            try:
                from .nebula_checkpoint_engine import NebulaCheckpointEngine
            except ImportError as err:
                logger.error(f"No torch_nebula was found! Will fall back to torch.save. Details: {err}")
                return TorchCheckpointEngine(config_params)
            else:
                return NebulaCheckpointEngine(config_params=config_params.nebula_config)

        if config_params.datastates_config.enabled:
            try:
                from .datastates_checkpoint_engine import DataStatesCheckpointEngine
                return DataStatesCheckpointEngine(deepspeed_config=config_params, rank=dist.get_rank())
            except ImportError as err:
                logger.error(
                    f"No datastates engine found! Install from https://github.com/DataStates/datastates-llm. Will fall back to torch.save. Details: {err}"
                )
                return TorchCheckpointEngine(config_params)

    return TorchCheckpointEngine(config_params)

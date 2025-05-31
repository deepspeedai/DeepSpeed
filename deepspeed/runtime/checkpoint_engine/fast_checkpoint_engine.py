# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine, CheckpointCommitInfo
from deepspeed.runtime.model_checkpointing import (
    CHECKPOINT_WRITER,
    CHECKPOINT_SERIALIZATION,
    CheckpointWriterFactory,
)


class FastCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params, dp_writer_config, optimize_dp_state):
        super().__init__(config_params)
        self.name = 'FastCheckpointEngine'
        self.serialization_enabled = config_params.checkpoint_config[CHECKPOINT_SERIALIZATION]
        self.optimize_dp_state = optimize_dp_state
        if dp_writer_config is None:
            self._writer = None
        else:
            self._writer = CheckpointWriterFactory(writer_config=config_params.checkpoint_config[CHECKPOINT_WRITER],
                                                   aio_config=config_params.aio_config,
                                                   dp_writer_config=dp_writer_config)

    def create(self, info: CheckpointCommitInfo):
        pass

    def save(self, state_dict, path: str):
        if self._writer is None:
            return

        torch.save(obj=state_dict,
                   f=self._writer.create_writer(path, self.optimize_dp_state),
                   _use_new_zipfile_serialization=self.serialization_enabled)
        self._writer.release_writer()

    def load(self, path: str, map_location=None):
        sd = torch.load(path, map_location=map_location)
        return sd

    def commit(self, info: CheckpointCommitInfo):
        return True

    def is_data_parallel_writer(self, dp_rank):
        return self._writer is not None

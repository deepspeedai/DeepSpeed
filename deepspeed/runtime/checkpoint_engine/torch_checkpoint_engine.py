# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.checkpoint.utils import load_mmap
from deepspeed.utils import log_dist
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine, CheckpointCommitInfo
from deepspeed.runtime.model_checkpointing import CHECKPOINT_SERIALIZATION

ENGINE_NAME = "TorchCheckpointEngine"


class TorchCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params=None, mmap=False):
        super().__init__(config_params)
        self.name = ENGINE_NAME
        if config_params is None:
            self.zipfile_serialization = False
        else:
            self.zipfile_serialization = config_params.checkpoint_config[CHECKPOINT_SERIALIZATION]
        # Tensors from a memory-mapped load stay backed by the file, so only callers that do not keep them
        # while the file may be rewritten should ask for it.
        self.mmap = mmap
        log_dist(f'[{ENGINE_NAME}] Initialized with serialization = {self.zipfile_serialization}', ranks=[0])

    def create(self, info: CheckpointCommitInfo):
        log_dist(f"[Torch] Checkpoint {info.tag} is about to be saved!", ranks=[0])
        pass

    def save(self, state_dict, path: str):
        torch.save(state_dict, path, _use_new_zipfile_serialization=self.zipfile_serialization)

    def load(self, path: str, map_location=None):
        log_dist(f"[Torch] Begin Load checkpoint from {path}...", ranks=[0])
        if self.mmap:
            partition = load_mmap(path, map_location=map_location, weights_only=False)
        else:
            partition = torch.load(path, map_location=map_location, weights_only=False)
        log_dist(f"[Torch] End Load checkpoint from {path}...", ranks=[0])
        return partition

    def commit(self, info: CheckpointCommitInfo):
        return True

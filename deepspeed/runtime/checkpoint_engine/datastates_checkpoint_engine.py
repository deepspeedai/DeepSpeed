# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# Apache-2.0 License Copyright (c) UChicago Argonne LLC, operator of Argonne National Laboratory.

# DeepSpeed Team

from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine, CheckpointCommitInfo

ENGINE_NAME = "DataStatesCheckpointEngine"


class DataStatesCheckpointEngine(CheckpointEngine):

    def __init__(self, deepspeed_config, rank):
        super().__init__(deepspeed_config)
        self.commit_info = None
        self.ckpt_engine = None
        try:
            from datastates import CheckpointEngine as DataStatesEngine
            self.ckpt_engine = DataStatesEngine(deepspeed_config, rank)
        except ImportError:
            raise RuntimeError("Please install DataStates from https://github.com/DataStates/datastates-llm.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while initializing DataStates Checkpoint Engine: {e}")

    def __del__(self):
        self.cleanup()

    def create(self, info: CheckpointCommitInfo):
        self.commit_info = info
        return None

    def save(self, state_dict, path: str):
        return self.ckpt_engine.save(state_dict, path)

    def load(self, path: str, map_location=None):
        return self.ckpt_engine.load(path, map_location)

    def commit(self, info: CheckpointCommitInfo):
        if info is None:
            return
        assert info == self.commit_info
        self.ckpt_engine.wait(persist=True)
        self.commit_info = None
        return True

    def cleanup(self):
        self.commit(self.commit_info)
        if self.ckpt_engine:
            self.ckpt_engine.wait(persist=True)
            del self.ckpt_engine

    def is_decoupled(self):
        return True

    def preserves_storage_sharing(self):
        return False


# Copyright by DataStates Team (Argonne National Laboratory): https://github.com/DataStates/
# Maintained by DataStates Team (Argonne National Laboratory): https://github.com/DataStates/

from deepspeed.utils import logger, log_dist
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine
from datastates.llm import Checkpointing


class DataStatesCheckpointEngine(CheckpointEngine):

    def __init__(self, deepspeed_config, rank):
        super().__init__(deepspeed_config)
        self.ckpt_engine = Checkpointing(deepspeed_config, rank)
    
    def create(self, tag):
        log_dist(f"[DataStates] Checkpoint {tag} is about to be saved!", ranks=[0])
        return None

    def save(self, state_dict, path: str):
        return self.ckpt_engine.save(state_dict, path)
    
    def load(self, path: str, map_location=None):
        return self.ckpt_engine.load(path, map_location)
    
    def commit(self, tag):
        return self.ckpt_engine.commit(tag)

    def wait(self):
        return self.ckpt_engine.wait()

   

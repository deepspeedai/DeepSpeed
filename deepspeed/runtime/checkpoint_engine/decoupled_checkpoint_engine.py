# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.multiprocessing as mp
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine, CheckpointCommitInfo
from deepspeed.runtime.checkpoint_engine.fast_checkpoint_engine import FastCheckpointEngine
from deepspeed import comm as dist
from deepspeed.runtime.utils import get_checkpoint_folder_size

from enum import Enum


class DecoupledEvent(Enum):
    SAVE_EVENT = 1
    COMMIT_EVENT = 2
    EXIT_EVENT = 3


class CheckpointSize(object):

    def __init__(self):
        self._pre = None
        self._post = None
        self._gigabytes = None

    def gb_size(self):
        return self._gigabytes

    def set_pre_size(self, size):
        self._pre = size

    def set_post_size(self, size):
        self._post = size
        self._gigabytes = (self._post - self._pre) / (1024**3)


def init_decoupled_checkpoint(config_params, dp_writer_config, save_event, save_queue, optimize_dp_state):
    checkpoint_engine = FastCheckpointEngine(config_params, dp_writer_config, optimize_dp_state)
    print(f'Created FastCheckpointEngine for Decoupled Checkpointing')
    save_path_list = []
    while True:
        (save_info, event_type) = save_queue.get()
        if event_type == DecoupledEvent.SAVE_EVENT and save_info is not None:
            state_dict, save_path = save_info
            # print(f'Received decoupled checkpoint request for {save_path=}')
            save_path_list.append(save_path)
            checkpoint_engine.save(state_dict, save_path)
            del state_dict
            # print(f'Completed decoupled checkpoint request for {save_path=}')

        if event_type == DecoupledEvent.COMMIT_EVENT:
            # print(f'Recieved commit request for {save_path_list=}')
            save_path_list = []
            save_event.set()

        if event_type == DecoupledEvent.EXIT_EVENT:
            # print(f'Received decoupled exit request')
            break


ENGINE_NAME = "DecoupledCheckpointEngine"


class DecoupledCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params, dp_writer_config, optimize_dp_state):
        if mp.get_start_methods(allow_None=False) is None:
            mp.set_start_method('spawn')
        super().__init__(config_params)
        self.name = ENGINE_NAME
        self.dp_writer_config = dp_writer_config
        self.commit_info = None
        self.checkpoint_size = CheckpointSize()
        self.global_rank = dist.get_rank()
        self.optimize_dp_state = optimize_dp_state
        if dp_writer_config is None:
            self.save_event = None
            self.save_queue = None
            self.ckpt_process = None
            self.local_rank = None
            print(
                f'[{ENGINE_NAME}]: No checkpoint process self.global_rank={self.global_rank}  self.dp_writer_config={self.dp_writer_config}'
            )
        else:
            self.save_event = mp.Event()
            self.save_queue = mp.SimpleQueue()
            engine_args = (config_params, dp_writer_config, self.save_event, self.save_queue, self.optimize_dp_state)
            self.ckpt_process = mp.Process(target=init_decoupled_checkpoint, args=engine_args)
            self.ckpt_process.start()
            self.local_rank = dp_writer_config.local_rank
            print(
                f'[{ENGINE_NAME}]: Create checkpoint process self.global_rank={self.global_rank}  self.ckpt_process.pid={self.ckpt_process.pid} self.dp_writer_config={self.dp_writer_config}'
            )

    def __del__(self):
        self.cleanup()

    def create(self, info: CheckpointCommitInfo):
        self.commit_info = info
        if self.checkpoint_size.gb_size() is None:
            pre_size = get_checkpoint_folder_size(info.save_dir, info.tag, self.local_rank)
            self.checkpoint_size.set_pre_size(pre_size)

    def load(self, path: str, map_location=None):
        sd = torch.load(path, map_location=map_location)
        return sd

    def save(self, state_dict, path: str):
        if self.ckpt_process is None:
            return
        save_info = (state_dict, path)
        self.save_queue.put((save_info, DecoupledEvent.SAVE_EVENT))

    def commit(self, info: CheckpointCommitInfo):
        assert info == self.commit_info
        if self.ckpt_process is not None:
            self.save_queue.put((None, DecoupledEvent.COMMIT_EVENT))
            # print(f'[begin] wait for decoupled complete for {info.tag}')
            self.save_event.wait()
            # print(f'[end] wait for decoupled complete for {info.tag}')
            self.save_event.clear()
            self.commit_info = None

        if self.checkpoint_size.gb_size() is None:
            dist.barrier()
            post_size = get_checkpoint_folder_size(info.save_dir, info.tag, self.local_rank)
            self.checkpoint_size.set_post_size(post_size)

        if self.global_rank == 0:
            print(
                f'{self.name} self.global_rank={self.global_rank} created checkpoint of {round(self.checkpoint_size.gb_size(), 2)} GB'
            )

        return True

    def get_commit_info(self):
        # print(f'getting commit info {self.commit_info=}')
        return self.commit_info

    def is_decoupled(self):
        return True

    def cleanup(self):
        # print(f'Inside {self.name} cleanup')

        if self.get_commit_info() is not None:
            self.commit(self.commit_info)

        if self.ckpt_process is not None:
            self.save_queue.put((None, DecoupledEvent.EXIT_EVENT))
            self.ckpt_process.join()
            self.ckpt_process = None
            self.save_queue = None

    def is_data_parallel_writer(self, dp_rank):
        return self.ckpt_process is not None

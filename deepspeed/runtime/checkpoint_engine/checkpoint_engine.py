# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

import abc
from abc import ABC

from dataclasses import dataclass


@dataclass
class CheckpointCommitInfo(object):
    tag: str
    save_dir: str
    save_latest: bool


class CheckpointEngine(ABC):
    # init checkpoint engine for save/load
    def __init__(self, config_params=None):
        self.name = None

    @abc.abstractmethod
    def create(self, info: CheckpointCommitInfo):
        # create checkpoint on give tag for save/load.
        ...

    @abc.abstractmethod
    def save(self, state_dict, path: str):
        ...

    def makedirs(self, path, exist_ok=False):
        os.makedirs(path, exist_ok=exist_ok)

    @abc.abstractmethod
    def load(self, path: str, map_location=None):
        ...

    @abc.abstractmethod
    def commit(self, info: CheckpointCommitInfo):
        # to tell checkpoint services if all files are ready.
        ...

    def is_data_parallel_writer(self, dp_rank):
        return dp_rank == 0

    def is_decoupled(self):
        return False

    def set_commit_info(self, info: CheckpointCommitInfo):
        pass

    def get_commit_info(self):
        return None

    def cleanup(self):
        pass

    def preserves_storage_sharing(self):
        return True

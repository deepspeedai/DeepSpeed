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
from deepspeed.utils import logger

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
    try:
        checkpoint_engine = FastCheckpointEngine(config_params, dp_writer_config, optimize_dp_state)
        print('Created FastCheckpointEngine for Decoupled Checkpointing')
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
    except Exception as e:
        print(f'[{ENGINE_NAME}] Checkpoint subprocess crashed with error: {e}')
        raise


ENGINE_NAME = "DecoupledCheckpointEngine"

# Default timeout for checkpoint operations (5 minutes)
DEFAULT_CHECKPOINT_TIMEOUT_SECONDS = 300
# Interval for checking process health while waiting
PROCESS_HEALTH_CHECK_INTERVAL_SECONDS = 10


class DecoupledCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params, dp_writer_config, optimize_dp_state):
        # Set spawn method if not already set (needed for CUDA tensor sharing)
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass  # Already set, ignore
        super().__init__(config_params)
        self.name = ENGINE_NAME
        self.dp_writer_config = dp_writer_config
        self.commit_info = None
        self.checkpoint_size = CheckpointSize()
        self.global_rank = dist.get_rank()
        self.optimize_dp_state = optimize_dp_state
        self._cleanup_called = False
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
        try:
            self.cleanup()
        except Exception:
            # Suppress exceptions in destructor to avoid crashes during shutdown
            pass

    def _check_process_alive(self):
        """Check if the checkpoint process is still alive.

        Note: Only call this when self.ckpt_process is not None.
        Some ranks don't have a checkpoint process by design (see Figure 6 in paper).
        """
        return self.ckpt_process.is_alive()

    def _wait_for_event_with_timeout(self, timeout_seconds=DEFAULT_CHECKPOINT_TIMEOUT_SECONDS):
        """Wait for save_event with timeout and process health checks.

        Returns True if event was set, raises RuntimeError if process died or timeout occurred.
        """
        elapsed = 0
        while elapsed < timeout_seconds:
            if self.save_event.wait(timeout=PROCESS_HEALTH_CHECK_INTERVAL_SECONDS):
                return True
            elapsed += PROCESS_HEALTH_CHECK_INTERVAL_SECONDS

            # Check if process is still alive
            if not self._check_process_alive():
                raise RuntimeError(f"[{ENGINE_NAME}] Checkpoint process died unexpectedly. "
                                   f"Check logs for OOM or other errors in the checkpoint subprocess.")

        raise RuntimeError(f"[{ENGINE_NAME}] Checkpoint commit timed out after {timeout_seconds} seconds. "
                           f"Process alive: {self._check_process_alive()}")

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

        # Check process health before attempting to save
        if not self._check_process_alive():
            return

        save_info = (state_dict, path)
        self.save_queue.put((save_info, DecoupledEvent.SAVE_EVENT))

    def commit(self, info: CheckpointCommitInfo):
        # Use proper validation instead of assert (assert is disabled with python -O)
        if info != self.commit_info:
            raise ValueError(f"[{ENGINE_NAME}] Checkpoint commit info mismatch: "
                             f"expected {self.commit_info}, got {info}")

        if self.ckpt_process is not None:
            # Check process health before waiting
            if not self._check_process_alive():
                raise RuntimeError(f"[{ENGINE_NAME}] Cannot commit checkpoint: checkpoint process is not running.")

            self.save_queue.put((None, DecoupledEvent.COMMIT_EVENT))
            # Wait with timeout and health checks instead of blocking forever
            self._wait_for_event_with_timeout()
            self.save_event.clear()

        self.commit_info = None

        if self.checkpoint_size.gb_size() is None:
            dist.barrier()
            post_size = get_checkpoint_folder_size(info.save_dir, info.tag, self.local_rank)
            self.checkpoint_size.set_post_size(post_size)

        assert self.checkpoint_size.gb_size() is not None, "Checkpoint size should be set after commit"

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
        # Prevent multiple cleanup calls (especially from __del__)
        if self._cleanup_called:
            return
        self._cleanup_called = True

        try:
            if self.get_commit_info() is not None:
                self.commit(self.commit_info)
        except Exception as e:
            logger.warning(f"[{ENGINE_NAME}] Error during commit in cleanup: {e}")

        if self.ckpt_process is not None:
            try:
                self.save_queue.put((None, DecoupledEvent.EXIT_EVENT))
            except Exception:
                pass  # Queue may be broken if process died

            # Join with timeout to avoid hanging forever
            self.ckpt_process.join(timeout=DEFAULT_CHECKPOINT_TIMEOUT_SECONDS)

            # If process didn't exit, terminate it forcefully
            if self.ckpt_process.is_alive():
                logger.warning(
                    f"[{ENGINE_NAME}] Checkpoint process did not exit within timeout, terminating forcefully.")
                self.ckpt_process.terminate()
                self.ckpt_process.join(timeout=5)  # Brief wait after terminate

                # Last resort: kill
                if self.ckpt_process.is_alive():
                    self.ckpt_process.kill()

            self.ckpt_process = None
            self.save_queue = None

    def is_data_parallel_writer(self, dp_rank):
        return self.ckpt_process is not None

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import threading
import time

import deepspeed
import deepspeed.comm as dist

import pytest
from unit.common import DistributedTest
from unit.simple_model import *

from unit.checkpoint.common import checkpoint_correctness_verification
from deepspeed.ops.op_builder import FusedAdamBuilder

if not deepspeed.ops.__compatible_ops__[FusedAdamBuilder.NAME]:
    pytest.skip("This op had not been implemented on this system.", allow_module_level=True)


class TestLatestCheckpoint(DistributedTest):
    world_size = 1

    def test_existing_latest(self, tmpdir):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            }
        }
        hidden_dim = 10
        models = [SimpleModel(hidden_dim=hidden_dim) for _ in range(2)]
        checkpoint_correctness_verification(config_dict=config_dict,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=True,
                                            load_lr_scheduler_states=False,
                                            empty_tag=True,
                                            dtype=torch.float)

    def test_missing_latest(self, tmpdir):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            }
        }
        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        # should be no-op, since latest doesn't exist
        model.load_checkpoint(tmpdir)


class TestLatestWrittenAfterAllRanks(DistributedTest):
    world_size = 2

    def test_latest_waits_for_slow_rank(self, tmpdir):
        # A crash while another rank is still writing its shard must not leave `latest`
        # pointing at an incomplete tag, so `latest` may only name a tag whose files
        # from every rank are already on disk.
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015,
                    "torch_adam": True
                }
            },
            "zero_optimization": {
                "stage": 1
            }
        }
        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        save_dir = os.path.join(str(tmpdir), "ckpt")
        tag = "tag1"
        slow_rank_shard = os.path.join(save_dir, tag, "zero_pp_rank_1_mp_rank_00_optim_states.pt")
        latest_path = os.path.join(save_dir, "latest")

        if dist.get_rank() == 1:
            # Slow this rank down at the checkpoint writer's `save(state_dict, path)` contract,
            # so the test holds for any engine implementation that routes writes through it.
            checkpoint_engine = model.checkpoint_engine
            engine_save = checkpoint_engine.save

            def slow_save(state_dict, path, *args, **kwargs):
                if os.path.basename(path) == os.path.basename(slow_rank_shard):
                    time.sleep(2)
                return engine_save(state_dict, path, *args, **kwargs)

            checkpoint_engine.save = slow_save

        shard_present_when_latest_appeared = []

        def watch_latest():
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if os.path.isfile(latest_path):
                    with open(latest_path) as fd:
                        if fd.read() == tag:
                            shard_present_when_latest_appeared.append(os.path.isfile(slow_rank_shard))
                            return
                time.sleep(0.005)

        watcher = None
        if dist.get_rank() == 0:
            watcher = threading.Thread(target=watch_latest)
            watcher.start()

        model.save_checkpoint(save_dir, tag=tag)

        if watcher is not None:
            watcher.join()
            assert shard_present_when_latest_appeared == [True]
            # `latest` is replaced atomically, so no temporary file is left behind.
            leftovers = [name for name in os.listdir(save_dir) if name.startswith("latest.")]
            assert leftovers == []

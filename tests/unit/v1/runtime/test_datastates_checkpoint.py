# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys
import time
from types import SimpleNamespace

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from unit.common import DistributedTest


class TestDataStatesCheckpointDestroy(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.parametrize(("engine_type", "tag", "fail_wait"), [
        ("state_engine", "final", False),
        ("state_engine", "final", True),
        ("state_aggregated_engine", "global_step7", False),
        ("state_aggregated_engine", "custom", False),
    ])
    def test_final_checkpoint(self, tmp_path, engine_type, tag, fail_wait):
        latest = tmp_path / "latest"

        class DataStatesWriter:

            def __init__(self, config, rank):
                self.rank = rank
                self.staged = []
                self.pending = []

            def save(self, state_dict, path):
                files = self.staged if engine_type == "state_aggregated_engine" else self.pending
                files.append((state_dict, path))

            def commit(self, submitted_tag):
                assert submitted_tag == tag
                if engine_type == "state_aggregated_engine":
                    assert self.staged
                    self.pending, self.staged = self.staged, []

            def wait(self, persist):
                assert persist
                if self.rank == 1:
                    if fail_wait:
                        sys.exit(-1)
                    time.sleep(0.5)
                assert latest.read_text() == "previous"
                for state_dict, path in self.pending:
                    torch.save(state_dict, path)
                self.pending.clear()
                (tmp_path / f"persisted-{self.rank}").touch()

        config = {
            "train_micro_batch_size_per_gpu": 1,
            "datastates_ckpt": {
                "engine_type": engine_type
            },
        }
        if tag == "custom":
            config.update(zero_optimization={"stage": 1}, zero_allow_untested_optimizer=True)
        model = torch.nn.Linear(2, 1)
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "datastates", SimpleNamespace(CheckpointEngine=DataStatesWriter))
            engine, _, _, _ = deepspeed.initialize(model=model,
                                                   optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                                                   config=config)

        if engine.global_rank == 0:
            latest.write_text("previous")
        engine.save_checkpoint(str(tmp_path), tag=tag)
        assert latest.read_text() == "previous"

        if fail_wait or tag == "custom":
            failed_rank = 1 if fail_wait else 0
            with pytest.raises(RuntimeError, match=f"rank {failed_rank}"):
                engine.destroy()
            dist.barrier()
            assert latest.read_text() == "previous"
        else:
            engine.destroy()
            assert latest.read_text() == tag
            assert (tmp_path / tag / "mp_rank_00_model_states.pt").is_file()
            assert all((tmp_path / f"persisted-{rank}").exists() for rank in range(2))

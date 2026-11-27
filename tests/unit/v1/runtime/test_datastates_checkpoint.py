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

    @pytest.mark.parametrize("fail_wait", [False, True])
    def test_final_checkpoint(self, tmp_path, fail_wait):
        latest = tmp_path / "latest"

        class DataStatesWriter:

            def __init__(self, config, rank):
                self.rank = rank
                self.pending = []

            def save(self, state_dict, path):
                self.pending.append((state_dict, path))

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

        config = {"train_micro_batch_size_per_gpu": 1, "datastates_ckpt": {}}
        model = torch.nn.Linear(2, 1)
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "datastates", SimpleNamespace(CheckpointEngine=DataStatesWriter))
            engine, _, _, _ = deepspeed.initialize(model=model,
                                                   optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                                                   config=config)

        if engine.global_rank == 0:
            latest.write_text("previous")
        engine.save_checkpoint(str(tmp_path), tag="final")
        assert latest.read_text() == "previous"

        if fail_wait:
            with pytest.raises(RuntimeError, match="rank 1"):
                engine.destroy()
            dist.barrier()
            assert latest.read_text() == "previous"
        else:
            engine.destroy()
            assert latest.read_text() == "final"
            assert (tmp_path / "final" / "mp_rank_00_model_states.pt").is_file()
            assert all((tmp_path / f"persisted-{rank}").exists() for rank in range(2))

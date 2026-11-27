# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from unit.common import DistributedTest


class TestDecoupledCheckpointDestroy(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    def _create_engine(self):
        os.environ.update(CROSS_RANK="0", CROSS_SIZE="1")
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "checkpoint": {
                "writer": {
                    "type": "python",
                    "decoupled": True,
                    "data_parallel": "machine"
                }
            }
        }
        model = torch.nn.Linear(2, 1)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                                               config=config)
        return engine

    def _engine_with_previous_checkpoint(self, tmp_path):
        engine = self._create_engine()
        engine.save_checkpoint(str(tmp_path), tag="previous")
        engine.backward(engine(torch.ones(1, 2, device=engine.device)).sum())
        engine.step()
        return engine

    @pytest.mark.parametrize("save_latest", [True, False])
    def test_final_checkpoint(self, tmp_path, save_latest):
        engine = self._engine_with_previous_checkpoint(tmp_path)
        engine.save_checkpoint(str(tmp_path), tag="final", save_latest=save_latest)
        engine.destroy()

        assert (tmp_path / "final" / "mp_rank_00_model_states.pt").is_file()
        assert (tmp_path / "latest").read_text() == ("final" if save_latest else "previous")

    @pytest.mark.parametrize("previous_checkpoint", [True, False])
    def test_dead_writer(self, tmp_path, previous_checkpoint):
        engine = self._engine_with_previous_checkpoint(tmp_path) if previous_checkpoint else self._create_engine()
        if dist.get_rank() == 0:
            writer = engine.checkpoint_engine.ckpt_process
            writer.terminate()
            writer.join(timeout=10)
            assert not writer.is_alive()

        engine.save_checkpoint(str(tmp_path), tag="failed")
        with pytest.raises(RuntimeError, match="(?i)checkpoint"):
            engine.destroy()

        # A transport timeout on either rank would leave the group unusable.
        dist.barrier()
        latest = tmp_path / "latest"
        if previous_checkpoint:
            assert latest.read_bytes() == b"previous"
        else:
            assert not latest.exists()

    @pytest.mark.world_size(1)
    @pytest.mark.parametrize("retry_pending_tag", [False, True])
    def test_failed_save(self, tmp_path, retry_pending_tag):

        class Unpicklable:

            def __reduce__(self):
                raise ValueError("cannot serialize client state")

        engine = self._engine_with_previous_checkpoint(tmp_path)
        try:
            if retry_pending_tag:
                engine.save_checkpoint(str(tmp_path), tag="final")
            with pytest.raises(ValueError, match="cannot serialize client state"):
                engine.save_checkpoint(str(tmp_path), tag="final", client_state={"value": Unpicklable()})
        finally:
            engine.destroy()

        assert (tmp_path / "latest").read_text() == "previous"

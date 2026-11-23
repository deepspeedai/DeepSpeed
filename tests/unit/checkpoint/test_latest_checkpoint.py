# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

import pytest
import torch

import deepspeed
from unit.common import DistributedTest
from unit.simple_model import SimpleModel

from unit.checkpoint.common import checkpoint_correctness_verification
from deepspeed.ops.op_builder import FusedAdamBuilder


@pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedAdamBuilder.NAME],
                    reason="This op had not been implemented on this system.")
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


class TestDecoupledCheckpointDestroy(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    def _engine_with_previous_checkpoint(self, tmp_path):
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

# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Universal checkpoints carry whatever state the optimizer keeps, not Adam's state by name.

Muon keeps `momentum_buffer` for the matrices it orthogonalizes and Adam's `exp_avg` /
`exp_avg_sq` for the rest, in two param groups of the same run, so a checkpoint holds both.
"""

from types import SimpleNamespace

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.checkpoint import UNIVERSAL_CHECKPOINT_INFO
from deepspeed.checkpoint.ds_to_universal import main as convert_to_universal

from unit.common import DistributedTest

HIDDEN = 16
TAG = "state_keys"


class LinearAndNorm(torch.nn.Module):
    """A 2-D weight for the Muon half, and a bias and a norm for the Adam half."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(HIDDEN, HIDDEN)
        self.norm = torch.nn.LayerNorm(HIDDEN)

    def forward(self, x):
        return self.norm(self.linear(x))


def _config(zero_stage):
    return {
        "train_micro_batch_size_per_gpu": 2,
        "optimizer": {
            "type": "muon",
            "params": {
                "lr": 0.02,
                "momentum": 0.95
            }
        },
        "zero_optimization": {
            "stage": zero_stage,
            # ZeRO-3 refuses Muon with reduce scatter.
            "reduce_scatter": False,
        },
    }


def _engine(config):
    model = LinearAndNorm()
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    return engine


def _train(engine, steps=3):
    for _ in range(steps):
        batch = torch.randn(2, HIDDEN, device=engine.device)
        engine.backward(torch.nn.functional.mse_loss(engine(batch), torch.zeros_like(batch)))
        engine.step()


def _gathered_optimizer_state(engine, zero_stage):
    """Every rank's partition of every state, keyed by (param index, state name).

    The states are walked in sorted order because each rank's `optimizer.state` is its own
    dict: after a universal load the insertion order can differ between ranks, and an
    all-gather issued in that order would pair different states across ranks.
    """
    if zero_stage == 3:
        engine.optimizer._set_fp32_optimizer_param_groups()
    state = engine.optimizer.optimizer.state_dict()["state"]

    gathered = {}
    for param_index in sorted(state):
        for name in sorted(state[param_index]):
            value = state[param_index][name]
            if not torch.is_tensor(value) or value.numel() <= 1:
                continue
            partitions = [torch.zeros_like(value.flatten()) for _ in range(dist.get_world_size())]
            dist.all_gather(partitions, value.flatten().contiguous())
            gathered[(param_index, name)] = torch.cat(partitions).cpu()

    if zero_stage == 3:
        engine.optimizer._clear_fp32_optimizer_param_groups()
    return gathered


def _convert(tmpdir):
    convert_to_universal(
        SimpleNamespace(input_folder=f"{tmpdir}/{TAG}",
                        output_folder=f"{tmpdir}/{TAG}_universal",
                        num_extract_workers=1,
                        num_merge_workers=1,
                        keep_temp_folder=False,
                        strict=True,
                        inject_missing_state=False))


class TestUniversalCheckpointOptimizerStates(DistributedTest):
    world_size = 2

    def test_muon_state_survives_a_universal_round_trip(self, tmpdir):
        zero_stage = 3
        config = _config(zero_stage)
        engine = _engine(config)
        _train(engine)

        before = _gathered_optimizer_state(engine, zero_stage)
        state_names = {name for _, name in before}
        assert "momentum_buffer" in state_names, "the Muon half should carry a momentum buffer"
        assert "exp_avg" in state_names, "the Adam half should carry Adam's moments"

        engine.save_checkpoint(tmpdir, tag=TAG, client_state={UNIVERSAL_CHECKPOINT_INFO: {}})
        dist.barrier()
        if dist.get_rank() == 0:
            _convert(tmpdir)
        dist.barrier()
        engine.destroy()

        config["checkpoint"] = {"load_universal": True}
        resumed = _engine(config)
        resumed.load_checkpoint(tmpdir, tag=f"{TAG}_universal", load_optimizer_states=True)

        after = _gathered_optimizer_state(resumed, zero_stage)
        assert set(after) == set(before)
        for key, value in before.items():
            assert torch.equal(value, after[key]), f"{key[1]} of param {key[0]} did not survive the round trip"


class TestReplicatedOptimizerStateIsRefused(DistributedTest):
    """ZeRO-1/2 replicate Muon's momentum buffer at the size of the whole group.

    The universal format stores optimizer state in the partition's layout, so that buffer has
    no place in it yet. What matters is that the conversion says so rather than writing
    fragments that are the right size and the wrong rows.
    """

    world_size = 2

    def test_a_state_that_is_not_partition_shaped_stops_the_conversion(self, tmpdir):
        engine = _engine(_config(zero_stage=1))
        _train(engine)
        engine.save_checkpoint(tmpdir, tag=TAG, client_state={UNIVERSAL_CHECKPOINT_INFO: {}})
        dist.barrier()

        if dist.get_rank() == 0:
            with pytest.raises(ValueError, match="momentum_buffer"):
                _convert(tmpdir)
        dist.barrier()

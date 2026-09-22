# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Universal checkpoints carry whatever state the optimizer keeps, laid out however it keeps it.

Muon keeps `momentum_buffer` for the matrices it orthogonalizes and Adam's `exp_avg` /
`exp_avg_sq` for the rest, in two param groups of the same run. ZeRO-1/2 keep that momentum
whole per parameter for every parameter a partition touches, and ZeRO-3 keeps it in the
partition's layout like any other state.
"""

import glob
import os
from types import SimpleNamespace

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.checkpoint import OPTIMIZER_STATE_DICT, UNIVERSAL_CHECKPOINT_INFO, WHOLE_PARAM_OPTIMIZER_STATES
from deepspeed.checkpoint.ds_to_universal import main as convert_to_universal

from unit.common import DistributedTest, DistributedFixture

TAG = "muon"
# Four matrices for Muon, sized so that at two ranks one of them straddles the partition boundary,
# and a bias and a norm for the Adam half.
WIDTHS = [16, 24, 20, 16, 12]


class MuonModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(i, o, bias=index == len(WIDTHS) - 2)
            for index, (i, o) in enumerate(zip(WIDTHS, WIDTHS[1:])))
        self.norm = torch.nn.LayerNorm(WIDTHS[-1])

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.norm(x)


def _engine(zero_stage, load_universal=False):
    torch.manual_seed(0)
    model = MuonModel()
    config = {
        "train_micro_batch_size_per_gpu": 4,
        "optimizer": {
            "type": "muon",
            "params": {
                "lr": 0.02,
                "momentum": 0.95
            }
        },
        # Clipping would scale each run's update by a norm taken over its own partitioning.
        "gradient_clipping": 0.0,
        "zero_optimization": {
            "stage": zero_stage,
            # ZeRO-3 refuses Muon with reduce scatter.
            "reduce_scatter": False,
        },
    }
    if load_universal:
        config["checkpoint"] = {"load_universal": True}
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    return engine


def _train(engine, first_step, steps):
    # The same batch on every rank, so every data-parallel size averages to the same gradient.
    generator = torch.Generator().manual_seed(1)
    for step in range(first_step + steps):
        x = torch.randn(4, WIDTHS[0], generator=generator)
        y = torch.randn(4, WIDTHS[-1], generator=generator)
        if step >= first_step:
            engine.backward(torch.nn.functional.mse_loss(engine(x.to(engine.device)), y.to(engine.device)))
            engine.step()


def _weights(engine, zero_stage):
    params = list(engine.module.named_parameters())
    if zero_stage == 3:
        with deepspeed.zero.GatheredParameters([p for _, p in params]):
            return {name: p.detach().float().cpu().clone() for name, p in params}
    return {name: p.detach().float().cpu().clone() for name, p in params}


def _convert(tmpdir):
    convert_to_universal(
        SimpleNamespace(input_folder=f"{tmpdir}/{TAG}",
                        output_folder=f"{tmpdir}/{TAG}_universal",
                        num_extract_workers=1,
                        num_merge_workers=1,
                        keep_temp_folder=False,
                        strict=True,
                        inject_missing_state=False))


def _convert_on_rank_0(tmpdir, device):
    """Convert on rank 0 and fail on every rank if it fails, rather than leave the others at a barrier."""
    failed = torch.zeros(1, device=device)
    error = None
    if dist.get_rank() == 0:
        try:
            _convert(tmpdir)
        except Exception as exc:
            failed.fill_(1)
            error = exc
    dist.all_reduce(failed)
    if error is not None:
        raise error
    if failed.item():
        raise RuntimeError("Converting to universal failed on rank 0")


class muon_baseline_ws2(DistributedFixture):
    """Train, save and convert on two ranks, then keep going to get the steps a resume must match."""

    world_size = 2

    def run(self, tmpdir, zero_stage):
        engine = _engine(zero_stage)
        _train(engine, first_step=0, steps=3)
        engine.save_checkpoint(tmpdir, tag=TAG, client_state={UNIVERSAL_CHECKPOINT_INFO: {}})
        dist.barrier()
        _convert_on_rank_0(tmpdir, engine.device)
        _train(engine, first_step=3, steps=2)
        weights = _weights(engine, zero_stage)
        if dist.get_rank() == 0:
            torch.save(weights, os.path.join(tmpdir, "reference.pt"))
        dist.barrier()
        engine.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestMuonUniversalResume(DistributedTest):
    """A resume from the universal checkpoint takes the same next steps as the run that never stopped.

    Only the momentum, Adam's moments and each group's own settings coming back exactly lets it
    do that. The ranks then hold different partitions from the ones that saved, so ZeRO-1/2 have
    to rebuild each rank's momentum buffer from the parameters its new partition touches.
    """

    def _resume_matches(self, tmpdir, zero_stage):
        engine = _engine(zero_stage, load_universal=True)
        engine.load_checkpoint(tmpdir, tag=f"{TAG}_universal", load_optimizer_states=True)
        _train(engine, first_step=3, steps=2)
        weights = _weights(engine, zero_stage)
        reference = torch.load(os.path.join(tmpdir, "reference.pt"), weights_only=True)
        for name, expected in reference.items():
            relative = ((weights[name] - expected).norm() / expected.norm()).item()
            assert relative < 1e-5, f"{name} is {relative:.1e} away from the uninterrupted run"

    @pytest.mark.world_size(2)
    def test_resume_on_the_same_ranks(self, muon_baseline_ws2, tmpdir, zero_stage):
        self._resume_matches(tmpdir, zero_stage)

    @pytest.mark.world_size(4)
    def test_resume_on_twice_the_ranks(self, muon_baseline_ws2, tmpdir, zero_stage):
        self._resume_matches(tmpdir, zero_stage)


class TestUnrecordedMomentumLayoutIsRefused(DistributedTest):
    """ZeRO-1/2 now record which states hold each parameter whole.

    Without that record the buffer can be the same size as the partition by coincidence, and
    reading it by partition offsets would write the wrong rows without failing.
    """

    world_size = 2

    def test_a_checkpoint_without_the_record_is_not_converted(self, tmpdir):
        engine = _engine(zero_stage=2)
        _train(engine, first_step=0, steps=2)
        engine.save_checkpoint(tmpdir, tag=TAG, client_state={UNIVERSAL_CHECKPOINT_INFO: {}})
        dist.barrier()
        if dist.get_rank() == 0:
            for path in glob.glob(os.path.join(tmpdir, TAG, "*_optim_states.pt")):
                checkpoint = torch.load(path, weights_only=False)
                assert WHOLE_PARAM_OPTIMIZER_STATES in checkpoint[OPTIMIZER_STATE_DICT]
                del checkpoint[OPTIMIZER_STATE_DICT][WHOLE_PARAM_OPTIMIZER_STATES]
                torch.save(checkpoint, path)
            with pytest.raises(ValueError, match=WHOLE_PARAM_OPTIMIZER_STATES):
                _convert(tmpdir)
        dist.barrier()

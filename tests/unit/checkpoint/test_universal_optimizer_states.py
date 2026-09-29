# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Universal checkpoints carry whatever state the optimizer keeps, laid out however it keeps it.

Muon keeps `momentum_buffer` for the matrices it orthogonalizes and Adam's `exp_avg` /
`exp_avg_sq` for the rest, in two param groups of the same run. ZeRO-1/2 keep that momentum
whole per parameter for every parameter a partition touches, unless the optimizer is offloaded.
With offload, and under ZeRO-3, it is in the partition's layout like any other state.
"""

import glob
import logging
import os
from types import SimpleNamespace

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.checkpoint import OPTIMIZER_STATE_DICT, UNIVERSAL_CHECKPOINT_INFO, WHOLE_PARAM_OPTIMIZER_STATES
from deepspeed.checkpoint.ds_to_universal import main as convert_to_universal
from deepspeed.utils import logger as ds_logger

from unit.common import DistributedTest, DistributedFixture

TAG = "muon"
# Four matrices for Muon, sized so that at two ranks one of them straddles the partition boundary,
# and a bias and a norm for the Adam half.
WIDTHS = [16, 24, 20, 16, 12]
# Matrices of 35 + 21 + 15 + 10 = 81 elements: not a multiple of two ranks, so the last one pads.
ODD_WIDTHS = [5, 7, 3, 5, 2]


class MuonModel(torch.nn.Module):

    def __init__(self, widths=WIDTHS):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(i, o, bias=index == len(widths) - 2)
            for index, (i, o) in enumerate(zip(widths, widths[1:])))
        self.norm = torch.nn.LayerNorm(widths[-1])

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.norm(x)


def _engine(zero_stage, offload=False, load_universal=False, widths=WIDTHS):
    torch.manual_seed(0)
    model = MuonModel(widths)
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
    if offload:
        config["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}
    if load_universal:
        config["checkpoint"] = {"load_universal": True}
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    return engine


def _train(engine, first_step, steps, widths=WIDTHS):
    # The same batch on every rank, so every data-parallel size averages to the same gradient.
    generator = torch.Generator().manual_seed(1)
    for step in range(first_step + steps):
        x = torch.randn(4, widths[0], generator=generator)
        y = torch.randn(4, widths[-1], generator=generator)
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

    def run(self, tmpdir, zero_stage, offload):
        engine = _engine(zero_stage, offload)
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


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestMuonUniversalResume(DistributedTest):
    """A resume from the universal checkpoint takes the same next steps as the run that never stopped.

    Only the momentum, Adam's moments and each group's own settings coming back exactly lets it
    do that. The ranks then hold different partitions from the ones that saved, so ZeRO-1/2 have
    to rebuild each rank's momentum buffer from the parameters its new partition touches, and
    switching offload on or off moves ZeRO-1/2's momentum between its two layouts.
    """

    def _resume_matches(self, tmpdir, zero_stage, offload):
        engine = _engine(zero_stage, offload, load_universal=True)
        engine.load_checkpoint(tmpdir, tag=f"{TAG}_universal", load_optimizer_states=True)
        _train(engine, first_step=3, steps=2)
        weights = _weights(engine, zero_stage)
        reference = torch.load(os.path.join(tmpdir, "reference.pt"), weights_only=True)
        for name, expected in reference.items():
            relative = ((weights[name] - expected).norm() / expected.norm()).item()
            assert relative < 1e-5, f"{name} is {relative:.1e} away from the uninterrupted run"

    @pytest.mark.world_size(2)
    def test_resume_on_the_same_ranks(self, muon_baseline_ws2, tmpdir, zero_stage, offload):
        self._resume_matches(tmpdir, zero_stage, offload)

    @pytest.mark.world_size(4)
    def test_resume_on_twice_the_ranks(self, muon_baseline_ws2, tmpdir, zero_stage, offload):
        self._resume_matches(tmpdir, zero_stage, offload)

    @pytest.mark.world_size(2)
    def test_resume_with_offload_switched(self, muon_baseline_ws2, tmpdir, zero_stage, offload):
        self._resume_matches(tmpdir, zero_stage, not offload)


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
        refusal = None
        if dist.get_rank() == 0:
            for path in glob.glob(os.path.join(tmpdir, TAG, "*_optim_states.pt")):
                checkpoint = torch.load(path, weights_only=False)
                assert WHOLE_PARAM_OPTIMIZER_STATES in checkpoint[OPTIMIZER_STATE_DICT]
                del checkpoint[OPTIMIZER_STATE_DICT][WHOLE_PARAM_OPTIMIZER_STATES]
                torch.save(checkpoint, path)
            try:
                _convert(tmpdir)
            except ValueError as exc:
                refusal = exc
        # Assert only after every rank has left the collective: a failed assertion on rank 0 alone
        # would leave the others waiting at the barrier until the distributed-test timeout.
        dist.barrier()
        if dist.get_rank() == 0:
            assert refusal is not None, "the conversion should have refused a checkpoint without the record"
            assert WHOLE_PARAM_OPTIMIZER_STATES in str(refusal)


class TestAlignmentPaddingOnTheLastRank(DistributedTest):
    """ZeRO-1/2 save the fp32 partition without alignment padding but Adam's moments with it.

    A group whose size is not a multiple of the world size leaves the last rank's moments a few
    elements longer than its partition on disk. The checkpoint reader strips that padding before
    the converter checks sizes, and this pins that the two keep agreeing.
    """

    world_size = 2

    @pytest.mark.parametrize("zero_stage", [1, 2])
    def test_a_padded_partition_converts(self, tmpdir, zero_stage):
        torch.manual_seed(0)
        # 5 * 3 + 3 = 18 elements plus a 1-element bias is odd, so the two partitions cannot be equal.
        model = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.Linear(3, 1))
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "zero_optimization": {
                "stage": zero_stage
            },
        }
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        x = torch.randn(1, 5, device=engine.device)
        engine.backward(engine(x).sum())
        engine.step()
        engine.save_checkpoint(tmpdir, tag=TAG, client_state={UNIVERSAL_CHECKPOINT_INFO: {}})
        dist.barrier()

        if dist.get_rank() == dist.get_world_size() - 1:
            assert sum(engine.optimizer.groups_padding) > 0, "the model was meant to leave the last rank padding"
        _convert_on_rank_0(tmpdir, engine.device)

        if dist.get_rank() == 0:
            zero_folder = os.path.join(tmpdir, f"{TAG}_universal", "zero")
            for name, param in model.named_parameters():
                for state in ("fp32", "exp_avg", "exp_avg_sq"):
                    saved = torch.load(os.path.join(zero_folder, name, f"{state}.pt"), weights_only=False)
                    assert saved["param"].numel() == param.numel(), (name, state)
        dist.barrier()
        engine.destroy()


class muon_odd_baseline_ws2(DistributedFixture):
    """Like `muon_baseline_ws2`, on matrices whose total size is odd, so the last rank pads."""

    world_size = 2

    def run(self, tmpdir, zero_stage):
        engine = _engine(zero_stage, widths=ODD_WIDTHS)
        _train(engine, first_step=0, steps=3, widths=ODD_WIDTHS)
        engine.save_checkpoint(tmpdir, tag=TAG, client_state={UNIVERSAL_CHECKPOINT_INFO: {}})
        dist.barrier()
        _convert_on_rank_0(tmpdir, engine.device)
        _train(engine, first_step=3, steps=2, widths=ODD_WIDTHS)
        weights = _weights(engine, zero_stage)
        if dist.get_rank() == 0:
            torch.save(weights, os.path.join(tmpdir, "reference.pt"))
        dist.barrier()
        engine.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2])
class TestMuonResumeWithAlignmentPadding(DistributedTest):
    """The last rank's partition is padded to the world size, and ZeRO-1/2's whole-parameter momentum is not.

    Stripping the padding from every tensor state cut that many elements off the end of the momentum
    buffer, so the converter could not read the last parameter's momentum out of it.
    """

    world_size = 2

    def test_resume_on_the_same_ranks(self, muon_odd_baseline_ws2, tmpdir, zero_stage):
        engine = _engine(zero_stage, load_universal=True, widths=ODD_WIDTHS)
        engine.load_checkpoint(tmpdir, tag=f"{TAG}_universal", load_optimizer_states=True)
        _train(engine, first_step=3, steps=2, widths=ODD_WIDTHS)
        weights = _weights(engine, zero_stage)
        reference = torch.load(os.path.join(tmpdir, "reference.pt"), weights_only=True)
        for name, expected in reference.items():
            relative = ((weights[name] - expected).norm() / expected.norm()).item()
            assert relative < 1e-5, f"{name} is {relative:.1e} away from the uninterrupted run"


class TestParamGroupCountMismatch(DistributedTest):
    """A checkpoint saved with one param group, resumed into an optimizer with two.

    There is no group to match each saved one to, so ZeRO-3 keeps applying the first saved group
    to every group. That is a guess, and it says so rather than making it quietly.
    """

    world_size = 2

    def test_a_mismatch_is_announced(self, tmpdir):
        config = {
            "train_micro_batch_size_per_gpu": 4,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "zero_optimization": {
                "stage": 3
            },
        }

        def build(load_universal, two_groups):
            torch.manual_seed(0)
            model = MuonModel()
            groups = [{"params": list(model.layers.parameters())}, {"params": list(model.norm.parameters())}]
            params = groups if two_groups else model.parameters()
            cfg = dict(config, checkpoint={"load_universal": True}) if load_universal else config
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=params, config=cfg)
            return engine

        engine = build(load_universal=False, two_groups=False)
        _train(engine, first_step=0, steps=1)
        engine.save_checkpoint(tmpdir, tag=TAG, client_state={UNIVERSAL_CHECKPOINT_INFO: {}})
        dist.barrier()
        _convert_on_rank_0(tmpdir, engine.device)
        engine.destroy()

        records = []

        class Collect(logging.Handler):

            def emit(self, record):
                records.append(record.getMessage())

        handler = Collect(level=logging.WARNING)
        ds_logger.addHandler(handler)
        try:
            engine = build(load_universal=True, two_groups=True)
            engine.load_checkpoint(tmpdir, tag=f"{TAG}_universal", load_optimizer_states=True)
        finally:
            ds_logger.removeHandler(handler)

        assert any("1 optimizer param group(s) but this optimizer has 2" in message for message in records), records
        engine.destroy()

# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Cross-stage universal checkpoint loads, in the v1 functional subtree the Modal GPU merge-queue suite selects.

The four ZeRO-1/2 <-> ZeRO-3 directions, resuming a pre-first-step checkpoint, a TP2 x PP2 pipeline run
resumed as ZeRO-3, and the mismatched or damaged checkpoints the loaders reject.
"""

import os
import re
import traceback
from types import SimpleNamespace

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.checkpoint import UNIVERSAL_CHECKPOINT_INFO
from deepspeed.checkpoint.affine import AFFINE_MAP_FORMAT_VERSION, contiguous_split_map, replicated_map
from deepspeed.checkpoint.constants import (AFFINE_MAP, AFFINE_MAP_PARAMS, AFFINE_MAP_VERSION,
                                            PARAMETER_WITH_ROW_PARALLELISM_PATTERNS, TP_REPLICATED_PARAMETER_PATTERNS)
from deepspeed.checkpoint.ds_to_universal import main as convert_to_universal
from deepspeed.checkpoint.universal_checkpoint import _shape_flat_universal_state
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from deepspeed.runtime.zero.utils import ZeRORuntimeException
from deepspeed.utils import groups, safe_get_full_fp32_param, safe_get_full_optimizer_state

from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader

CP_TAG = "test_tag"


def _train(engine, hidden_dim, steps):
    data_loader = random_dataloader(model=engine,
                                    total_samples=steps,
                                    hidden_dim=hidden_dim,
                                    device=engine.device,
                                    dtype=torch.float32)
    for batch in data_loader:
        loss = engine(batch[0], batch[1])
        engine.backward(loss)
        engine.step()


class _CopyToTensorParallel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


class _ReduceFromTensorParallel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, group):
        x = x.clone()
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _ColumnParallelLinear(torch.nn.Module):
    """Megatron's column-parallel linear: this rank keeps a block of output rows of weight and bias."""

    def __init__(self, full, tp_rank, tp_size):
        super().__init__()
        self.weight = torch.nn.Parameter(full.weight.detach().chunk(tp_size, dim=0)[tp_rank].clone())
        self.bias = torch.nn.Parameter(full.bias.detach().chunk(tp_size, dim=0)[tp_rank].clone())
        self.group = None

    def forward(self, x):
        return torch.nn.functional.gelu(
            torch.nn.functional.linear(_CopyToTensorParallel.apply(x, self.group), self.weight, self.bias))


class _RowParallelLinear(torch.nn.Module):
    """Megatron's row-parallel linear: a block of weight columns, the bias replicated and added after the sum."""

    def __init__(self, full, tp_rank, tp_size):
        super().__init__()
        self.weight = torch.nn.Parameter(full.weight.detach().chunk(tp_size, dim=1)[tp_rank].clone())
        self.bias = torch.nn.Parameter(full.bias.detach().clone())
        self.group = None

    def forward(self, x):
        return _ReduceFromTensorParallel.apply(torch.nn.functional.linear(x, self.weight), self.group) + self.bias


class _DenseStack(torch.nn.Module):
    """The same layers unsharded, named as PipelineModule names them, for the ZeRO-3 target."""

    def __init__(self, layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)

    def forward(self, x, y):
        for i, layer in enumerate(self.children()):
            x = layer(x)
            if i % 2 == 0:
                x = torch.nn.functional.gelu(x)
        return torch.nn.functional.mse_loss(x, y)


class TestUniversalCheckpointPipelineToZeRO3(DistributedTest):
    """A TP2 x PP2 run under ZeRO-1, the most a pipeline engine allows, resumed as ZeRO-3 over 4 data-parallel ranks."""
    world_size = 4

    # The source describes its TP shards either with the Megatron name patterns or with affine maps, as AutoTP does.
    @pytest.mark.parametrize("shard_info", ["patterns", "affine_maps"])
    def test_load(self, tmpdir, shard_info):
        hidden, tp, steps_before_save, steps_after_load = 8, 2, 2, 3
        dims = [(hidden, 2 * hidden), (2 * hidden, hidden)] * 2
        row_weight, row_bias = r"^[13]\.weight$", r"^[13]\.bias$"

        def full_layers():
            torch.manual_seed(0)
            return [torch.nn.Linear(i, o) for i, o in dims]

        # Two micro-batches per step: the source accumulates both, the target's 4 ranks see each twice.
        generator = torch.Generator().manual_seed(1)
        batches = [(torch.randn(1, hidden, generator=generator), torch.randn(1, hidden, generator=generator))
                   for _ in range(2 * (steps_before_save + steps_after_load))]

        topology = PipeModelDataParallelTopology(num_pp=2, num_mp=tp, num_dp=1)
        tp_rank = topology.get_coord(dist.get_rank()).model
        layers = [(_ColumnParallelLinear if i % 2 == 0 else _RowParallelLinear)(full, tp_rank, tp)
                  for i, full in enumerate(full_layers())]
        model = PipelineModule(layers=layers,
                               topology=topology,
                               loss_fn=torch.nn.MSELoss(),
                               partition_method="uniform")
        for layer in layers:
            layer.group = model._grid.get_slice_parallel_group()
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 2,
            "gradient_clipping": 0.0,
            "zero_optimization": {
                "stage": 1
            },
            "pipeline": {
                "activation_checkpoint_interval": 0
            },
        }
        source, _, _, _ = deepspeed.initialize(config=config,
                                               model=model,
                                               optimizer=torch.optim.Adam(model.parameters(), lr=0.01))
        device = source.device
        loader = iter([(x.to(device), y.to(device)) for x, y in batches])
        for _ in range(steps_before_save):
            source.train_batch(data_iter=loader)

        def whole_source_states():
            # Each rank holds one TP block of its stage's parameters; put the whole tensors together.
            local = {name: [s.cpu() for s in states] for name, states in _full_optimizer_states(source).items()}
            everyone = [None] * self.world_size
            dist.all_gather_object(everyone, (tp_rank, local))
            blocks = {}
            for rank_tp, rank_states in everyone:
                for name, states in rank_states.items():
                    blocks.setdefault(name, {})[rank_tp] = states
            whole = {}
            for name, by_tp in blocks.items():
                per_state = list(zip(*(by_tp[r] for r in range(tp))))
                if re.match(row_bias, name):
                    assert all(torch.equal(s[0], s[1]) for s in per_state), name
                    whole[name] = [s[0] for s in per_state]
                else:
                    whole[name] = [torch.cat(s, dim=1 if re.match(row_weight, name) else 0) for s in per_state]
            assert len(whole) == 2 * len(dims)
            return whole

        expected = whole_source_states()
        if shard_info == "patterns":
            uc_info = {
                PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [row_weight],
                TP_REPLICATED_PARAMETER_PATTERNS: [row_bias],
            }
        else:
            block = [2 * hidden // tp] * tp
            column_maps = {
                "weight": contiguous_split_map((2 * hidden, hidden), block, 0),
                "bias": contiguous_split_map((2 * hidden, ), block, 0),
            }
            row_maps = {
                "weight": contiguous_split_map((hidden, 2 * hidden), block, 1),
                "bias": replicated_map((hidden, ), tp),
            }
            params = {}
            for layer in range(len(dims)):
                maps = column_maps if layer % 2 == 0 else row_maps
                for name, shard_map in maps.items():
                    params[rf"^{layer}\.{name}$"] = shard_map.to_dict()
            uc_info = {AFFINE_MAP: {AFFINE_MAP_VERSION: AFFINE_MAP_FORMAT_VERSION, AFFINE_MAP_PARAMS: params}}
        source.save_checkpoint(tmpdir, tag=CP_TAG, client_state={UNIVERSAL_CHECKPOINT_INFO: uc_info})

        # The run that never stopped, for the target to follow after the load.
        uninterrupted = []
        for _ in range(steps_after_load):
            loss = source.train_batch(data_iter=loader)
            uninterrupted.append((loss.item(), whole_source_states()))
        source.destroy()
        # A real resume starts new processes; here the pipeline grid would stay registered as the global mpu.
        groups.mpu = None

        cp_dir = os.path.join(tmpdir, CP_TAG)
        dist.barrier()
        # Rank 0 converts alone; share its outcome so a failure fails every rank instead of leaving them waiting.
        error = [None]
        if dist.get_rank() == 0:
            try:
                convert_to_universal(
                    SimpleNamespace(input_folder=cp_dir,
                                    output_folder=f"{cp_dir}_universal",
                                    num_extract_workers=1,
                                    num_merge_workers=1,
                                    keep_temp_folder=False,
                                    strict=True,
                                    inject_missing_state=False))
            except Exception:
                error = [traceback.format_exc()]
        dist.broadcast_object_list(error, src=0)
        assert error[0] is None, f"ds_to_universal failed on rank 0:\n{error[0]}"

        dense = _DenseStack(full_layers())
        target, _, _, _ = deepspeed.initialize(config={
            "train_batch_size": self.world_size,
            "gradient_clipping": 0.0,
            "zero_optimization": {
                "stage": 3
            },
            "checkpoint": {
                "load_universal": True
            },
        },
                                               model=dense,
                                               optimizer=torch.optim.Adam(dense.parameters(), lr=1.0))
        target.load_checkpoint(tmpdir, tag=f"{CP_TAG}_universal")

        loaded = _full_optimizer_states(target)
        assert loaded.keys() == expected.keys()
        for name, states in expected.items():
            assert len(loaded[name]) == len(states) == 3, name
            for expected_state, loaded_state in zip(states, loaded[name]):
                assert torch.equal(expected_state, loaded_state), name
        assert target.optimizer.param_groups[0]["lr"] == 0.01
        for state in target.optimizer.optimizer.state.values():
            assert int(state["step"]) == steps_before_save

        # Rank r takes micro-batch r % 2 of the source's pair, so the data-parallel mean of the gradients is the
        # source's accumulated gradient. The TP all-reduces sum in another order, so the match is close, not exact.
        for step, (source_loss, source_states) in enumerate(uninterrupted):
            x, y = batches[2 * (steps_before_save + step) + dist.get_rank() % 2]
            loss = target(x.to(target.device), y.to(target.device))
            target.backward(loss)
            target.step()
            mean_loss = loss.detach().clone()
            dist.all_reduce(mean_loss)
            torch.testing.assert_close(mean_loss.item() / self.world_size, source_loss, rtol=1e-5, atol=1e-6)
            for name, states in _full_optimizer_states(target).items():
                for kind, source_state, target_state, atol in zip(("fp32", "exp_avg", "exp_avg_sq"),
                                                                  source_states[name], states, (1e-6, 1e-7, 1e-9)):
                    torch.testing.assert_close(target_state,
                                               source_state,
                                               rtol=1e-5,
                                               atol=atol,
                                               msg=lambda m: f"step {step + 1}, {name} {kind}: {m}")
        for state in target.optimizer.optimizer.state.values():
            assert int(state["step"]) == steps_before_save + steps_after_load
        target.destroy()


def _full_optimizer_states(engine):
    """Every parameter's whole fp32 master and Adam moments, gathered across the ZeRO partition."""
    states = {}
    for name, param in engine.module.named_parameters():
        states[name] = [
            safe_get_full_fp32_param(param).cpu(),
            safe_get_full_optimizer_state(param, "exp_avg").cpu(),
            safe_get_full_optimizer_state(param, "exp_avg_sq").cpu(),
        ]
    return states


class _ScaledModel(SimpleModel):
    """SimpleModel behind a trainable 0-D scale, a parameter with no dimension to partition along."""

    def __init__(self, hidden_dim, nlayers):
        super().__init__(hidden_dim, nlayers=nlayers)
        self.scale = torch.nn.Parameter(torch.tensor(0.9))

    def forward(self, x, y):
        return super().forward(x * self.scale, y)


class _ScalarFirstModel(_ScaledModel):
    """Like _ScaledModel, but the caller keeps the scalar alone in its own group and places that group first. A lone
    1-element parameter padded by the ZeRO-1/2 sub-group split leaves at least one rank without a fragment of that
    group (hence no local optimizer state for it there). Putting it in group 0 makes such a rank's torch state dict be
    keyed {1, 2, ...} with group 0 absent, which is the layout that previously tripped
    ``ds_to_universal.extract_zero_shards`` (indexed by ``range(len(state_groups))``)."""


def _save_and_convert(engine, tmpdir, tag):
    """Save ``engine`` under ``tag`` and convert the checkpoint, returning the universal checkpoint's tag."""
    engine.save_checkpoint(tmpdir, tag=tag, client_state={UNIVERSAL_CHECKPOINT_INFO: {}})
    engine.destroy()

    cp_dir = os.path.join(tmpdir, tag)
    dist.barrier()
    if dist.get_rank() == 0:
        convert_to_universal(
            SimpleNamespace(input_folder=cp_dir,
                            output_folder=f"{cp_dir}_universal",
                            num_extract_workers=1,
                            num_merge_workers=1,
                            keep_temp_folder=False,
                            strict=True,
                            inject_missing_state=False))
    dist.barrier()
    return f"{tag}_universal"


def _save_as_universal(engine, tmpdir):
    """Save ``engine`` and convert the checkpoint, returning the universal checkpoint's tag."""
    return _save_and_convert(engine, tmpdir, CP_TAG)


@pytest.mark.parametrize("source_stage,target_stage", [(1, 3), (2, 3), (3, 1), (3, 2)])
class TestUniversalCheckpointAcrossZeROStages(DistributedTest):
    world_size = 2

    def test_load(self, tmpdir, source_stage, target_stage):
        hidden_dim = 10
        steps_before_save = 2

        group_lrs = [0.1, 0.05]

        def build(stage, load_universal):
            # A small sub_group_size splits ZeRO-3's groups into several sub-groups.
            ds_config = {"train_batch_size": 2, "zero_optimization": {"stage": stage, "sub_group_size": 100}}
            if load_universal:
                ds_config["checkpoint"] = {"load_universal": True}
            model = _ScaledModel(hidden_dim, nlayers=2)
            # Two groups with their own settings. The target starts from other settings, so the check below
            # sees what the load restored.
            group_params = [list(layer.parameters()) for layer in model.linears]
            group_params[0].append(model.scale)
            param_groups = [{
                "params": params,
                "lr": 1.0 if load_universal else lr,
                "weight_decay": 0.0 if load_universal else 0.01 * i
            } for i, (params, lr) in enumerate(zip(group_params, group_lrs))]
            engine, _, _, _ = deepspeed.initialize(config=ds_config,
                                                   model=model,
                                                   optimizer=torch.optim.Adam(param_groups))
            return engine

        source = build(source_stage, load_universal=False)
        _train(source, hidden_dim, steps_before_save)
        if source_stage < 3:
            # ZeRO-1/2 links optimizer state after the first group's step only, so later groups' moments are not
            # visible to safe_get_full_optimizer_state until #8956. Link them again now that all groups have state.
            source.optimizer._hp_optimizer_states_linked = False
            source.optimizer._lazy_init_hp_params_optimizer_state()
        expected = _full_optimizer_states(source)
        universal_tag = _save_as_universal(source, tmpdir)

        target = build(target_stage, load_universal=True)
        target.load_checkpoint(tmpdir, tag=universal_tag)

        loaded = _full_optimizer_states(target)
        assert loaded.keys() == expected.keys()
        for name, states in expected.items():
            for expected_state, loaded_state in zip(states, loaded[name]):
                assert torch.equal(expected_state, loaded_state), name
        optimizer = target.optimizer.optimizer
        owned = {buf for buf, st in optimizer.state.items() if "exp_avg" in st}
        assert owned
        # The restored step lands only on the groups this rank owns moments for. A rank that holds no
        # fragment for a group (e.g. a scalar's own group, whose moments live on a wider group's buffer)
        # must not get a bare step: it would be an incomplete state entry the optimizer cannot use next step.
        for buf, st in optimizer.state.items():
            if buf in owned:
                assert int(st["step"]) == steps_before_save, buf
            else:
                assert "step" not in st, buf

        for i, (param_group, lr) in enumerate(zip(target.optimizer.param_groups, group_lrs)):
            assert param_group["lr"] == lr and param_group["weight_decay"] == 0.01 * i, param_group

        # The restored optimizer must also be able to keep training, advancing every owned step once.
        _train(target, hidden_dim, 1)
        for buf in owned:
            assert int(optimizer.state[buf]["step"]) == steps_before_save + 1, buf
        target.destroy()


@pytest.mark.parametrize("source_stage", [1, 2])
class TestUniversalCheckpointBeforeFirstStepToZeRO3(DistributedTest):
    world_size = 2

    def test_load(self, tmpdir, source_stage):
        """A ZeRO-1/2 checkpoint saved before the first step has no Adam state; ZeRO-3 starts it fresh."""
        hidden_dim = 10

        def build(stage, load_universal):
            ds_config = {"train_batch_size": 2, "zero_optimization": {"stage": stage, "sub_group_size": 100}}
            if load_universal:
                ds_config["checkpoint"] = {"load_universal": True}
            model = SimpleModel(hidden_dim, nlayers=2)
            engine, _, _, _ = deepspeed.initialize(config=ds_config,
                                                   model=model,
                                                   optimizer=torch.optim.Adam(model.parameters(), lr=0.1))
            return engine

        source = build(source_stage, load_universal=False)
        expected = {name: safe_get_full_fp32_param(param).cpu() for name, param in source.module.named_parameters()}
        universal_tag = _save_as_universal(source, tmpdir)

        target = build(3, load_universal=True)
        target.load_checkpoint(tmpdir, tag=universal_tag)
        for name, param in target.module.named_parameters():
            assert torch.equal(safe_get_full_fp32_param(param).cpu(), expected[name]), name
        assert not any(target.optimizer.optimizer.state.values())

        _train(target, hidden_dim, 1)
        optimizer_states = list(target.optimizer.optimizer.state.values())
        assert optimizer_states
        for state in optimizer_states:
            assert int(state["step"]) == 1
        target.destroy()


@pytest.mark.parametrize("source_stage,target_stage", [(2, 3), (3, 2)])
class TestUniversalCheckpointAcrossZeROStagesRejects(DistributedTest):
    world_size = 2

    def _build(self, stage, hidden_dim, one_group, load_universal=False):
        ds_config = {"train_batch_size": 2, "zero_optimization": {"stage": stage, "sub_group_size": 100}}
        if load_universal:
            ds_config["checkpoint"] = {"load_universal": True}
        model = SimpleModel(hidden_dim, nlayers=2)
        if one_group:
            params = model.parameters()
        else:
            params = [{"params": list(layer.parameters())} for layer in model.linears]
        engine, _, _, _ = deepspeed.initialize(config=ds_config,
                                               model=model,
                                               optimizer=torch.optim.Adam(params, lr=0.1))
        return engine

    def test_param_group_count_mismatch(self, tmpdir, source_stage, target_stage):
        """Pairing groups by position would leave the layers of the groups one side lacks unrestored."""
        hidden_dim = 10
        source = self._build(source_stage, hidden_dim, one_group=True)
        _train(source, hidden_dim, 2)
        universal_tag = _save_as_universal(source, tmpdir)

        target = self._build(target_stage, hidden_dim, one_group=False, load_universal=True)
        with pytest.raises(ValueError, match="optimizer parameter groups"):
            target.load_checkpoint(tmpdir, tag=universal_tag)

    def test_incomplete_adam_moment_pair(self, tmpdir, source_stage, target_stage):
        """Without exp_avg_sq.pt, ZeRO-3 would restart the moments at step 1 and ZeRO-1/2 would fail its next step."""
        hidden_dim = 10
        source = self._build(source_stage, hidden_dim, one_group=True)
        _train(source, hidden_dim, 2)
        universal_tag = _save_as_universal(source, tmpdir)
        if dist.get_rank() == 0:
            zero_dir = os.path.join(tmpdir, universal_tag, "zero")
            for name in os.listdir(zero_dir):
                exp_avg_sq = os.path.join(zero_dir, name, "exp_avg_sq.pt")
                if os.path.isfile(exp_avg_sq):
                    os.remove(exp_avg_sq)
        dist.barrier()

        target = self._build(target_stage, hidden_dim, one_group=True, load_universal=True)
        with pytest.raises(ZeRORuntimeException, match="only one of the Adam moments"):
            target.load_checkpoint(tmpdir, tag=universal_tag)

    def test_tp_sharded_target_rejects_bare_source(self, source_stage, target_stage):
        """A ds_to_universal conversion from ZeRO-3 writes one bare flat tensor per parameter, without a
        tensor-parallel layout. A tensor-parallel target shard is smaller than it, so the load must
        reject the mismatch rather than slice a full tensor into a shard. Only the (3, 2) direction
        can reach the guard, since a ZeRO-1/2 source already records per-rank shards."""
        if source_stage != 3:
            return
        full = torch.arange(8)
        shard = torch.nn.Parameter(torch.zeros(4))
        with pytest.raises(ValueError, match=re.escape("only without tensor parallelism")):
            _shape_flat_universal_state(shard, full, "moment.pt")
        # A non-parallel target takes the whole tensor.
        whole = torch.nn.Parameter(torch.zeros(8))
        assert torch.equal(_shape_flat_universal_state(whole, full, "moment.pt"), full)


@pytest.mark.parametrize("source_stage,target_stage", [(3, 1), (3, 2), (1, 3), (2, 3)])
class TestUniversalCheckpointCrossStageScalarOwnGroup(DistributedTest):
    """A scalar parameter in its own optimizer group: under ZeRO-1/2 it leaves some rank without a
    fragment for that group, so the loader must not require a per-parameter step there."""

    world_size = 2

    def test_load(self, tmpdir, source_stage, target_stage):
        hidden_dim = 10
        steps_before_save = 2

        def build(stage, load_universal):
            ds_config = {"train_batch_size": 2, "zero_optimization": {"stage": stage, "sub_group_size": 100}}
            if load_universal:
                ds_config["checkpoint"] = {"load_universal": True}
            model = _ScaledModel(hidden_dim, nlayers=2)
            # The scalar keeps its own group: no other parameter in the group can carry its step on a rank
            # that holds only the scale's fragment.
            param_groups = [{"params": list(layer.parameters())} for layer in model.linears]
            param_groups.append({"params": [model.scale]})
            engine, _, _, _ = deepspeed.initialize(config=ds_config,
                                                   model=model,
                                                   optimizer=torch.optim.Adam(param_groups))
            return engine

        source = build(source_stage, load_universal=False)
        _train(source, hidden_dim, steps_before_save)
        universal_tag = _save_as_universal(source, tmpdir)

        target = build(target_stage, load_universal=True)
        target.load_checkpoint(tmpdir, tag=universal_tag)

        # Every restored parameter must carry the source's global step on the ranks that own its group's moments, so the very
        # next step continues from it. A rank that holds no fragment for the scalar's group has no local state there at all,
        # so its (empty) state must not carry a restored step: a bare step with no moments is broken state.
        optimizer = target.optimizer.optimizer
        owned = {buf for buf, state in optimizer.state.items() if "exp_avg" in state}
        assert owned
        for buf, state in optimizer.state.items():
            if buf in owned:
                assert int(state["step"]) == steps_before_save, buf
            else:
                assert "step" not in state, buf

        _train(target, hidden_dim, 1)
        for buf in owned:
            assert int(optimizer.state[buf]["step"]) == steps_before_save + 1, buf
        target.destroy()


@pytest.mark.parametrize("target_stage", [1, 2])
class TestUniversalCheckpointCrossStageScalarReConvert(DistributedTest):
    """A checkpoint saved right after a cross-stage load into ZeRO-1/2 must still convert. With the scalar alone in
    group 0, a rank that holds no fragment of that group saves no optimizer state for it, so its state dict omits
    group 0's id; ``extract_zero_shards`` must iterate the ids that were actually saved, not a positional range that
    KeyError's on the absent id."""

    world_size = 2

    def test_convert_after_cross_stage_load(self, tmpdir, target_stage):
        hidden_dim = 10
        steps_before_save = 2

        def build(stage, load_universal):
            ds_config = {"train_batch_size": 2, "zero_optimization": {"stage": stage, "sub_group_size": 100}}
            if load_universal:
                ds_config["checkpoint"] = {"load_universal": True}
            model = _ScalarFirstModel(hidden_dim, nlayers=2)
            # The scalar keeps its own group and leads the list: a rank without its fragment has group 0 absent from
            # its saved state.
            param_groups = [{
                "params": [model.scale]
            }] + [{
                "params": list(layer.parameters())
            } for layer in model.linears]
            engine, _, _, _ = deepspeed.initialize(config=ds_config,
                                                   model=model,
                                                   optimizer=torch.optim.Adam(param_groups))
            return engine

        source = build(3, load_universal=False)
        _train(source, hidden_dim, steps_before_save)
        universal_tag = _save_as_universal(source, tmpdir)

        target = build(target_stage, load_universal=True)
        target.load_checkpoint(tmpdir, tag=universal_tag)

        # Save immediately after the load, before any optimizer step, and convert. This is the moment that used to
        # raise KeyError on the group whose fragment some rank does not hold.
        reconverted_tag = _save_and_convert(target, tmpdir, "afterload")

        # The re-converted checkpoint must load and carry the source step on every rank-owned group.
        checker = build(target_stage, load_universal=True)
        checker.load_checkpoint(tmpdir, tag=reconverted_tag)
        optimizer = checker.optimizer.optimizer
        owned = {buf for buf, state in optimizer.state.items() if "exp_avg" in state}
        assert owned
        for buf, state in optimizer.state.items():
            if buf in owned:
                assert int(state["step"]) == steps_before_save, buf
        _train(checker, hidden_dim, 1)
        for buf in owned:
            assert int(optimizer.state[buf]["step"]) == steps_before_save + 1, buf
        checker.destroy()

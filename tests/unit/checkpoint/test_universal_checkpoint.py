# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
import glob
import math
import traceback

import deepspeed
from types import SimpleNamespace
from torch.utils._pytree import tree_map

from deepspeed.utils.torch import required_torch_version
from deepspeed.checkpoint import UNIVERSAL_CHECKPOINT_INFO
from deepspeed.checkpoint.constants import (OPTIMIZER_STATE_DICT, PARAMETER_WITH_ROW_PARALLELISM_PATTERNS,
                                            TP_REPLICATED_PARAMETER_PATTERNS)
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from deepspeed.checkpoint.ds_to_universal import main as convert_to_universal
from deepspeed.utils import groups, safe_get_full_fp32_param, safe_get_full_optimizer_state

from unit.common import DistributedTest, DistributedFixture
from unit.simple_model import *
from unit.util import bf16_required_version_check

from unit.checkpoint.common import compare_opt_state_dicts, compare_state_dicts

import pytest
import deepspeed.comm as dist


def get_expected_mismatch_keys():
    # torch 1.2.* stores raw tensor id numbers in checkpoint state which leads to
    # false positive mismatches in checkpoint state comparisons.
    # Newer torch versions store tensor ids as 0, 1, 2, ...
    return [] if required_torch_version(min_version=1.4) else ['params']


def maybe_step(t):
    return not torch.is_tensor(t) or (t.device.type == 'cpu' and t.numel() == 1)


def gather_opt_state(optimizer_state):

    def gather_tensor(t):

        if maybe_step(t):
            return t
        else:
            buffer = [torch.zeros_like(t.flatten()) for _ in range(dist.get_world_size())]
            dist.all_gather(buffer, t.flatten())
            return torch.cat(buffer)

    return tree_map(gather_tensor, optimizer_state)


def remove_pad_in_opt_state(optimizer_state, num_params):

    def remove_pad(t):
        if maybe_step(t):
            return t
        else:
            return t[:num_params]

    return tree_map(remove_pad, optimizer_state)


CP_TAG = "test_tag"


def init_ds_engine(model, ds_config, use_torch_adam):

    if use_torch_adam:
        ds_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        del ds_config["optimizer"]
        model, _, _, _ = deepspeed.initialize(config=ds_config, model=model, optimizer=ds_optimizer)
    else:
        model, _, _, _ = deepspeed.initialize(config=ds_config, model=model, model_parameters=model.parameters())

    return model


def train_save_convert(ds_config, hidden_dim, load_optim, use_torch_adam, dtype, tmpdir, world_size):
    if dtype == torch.bfloat16 and not bf16_required_version_check():
        return

    test_step = 8

    model = SimpleModel(hidden_dim, nlayers=2)
    model = init_ds_engine(model, ds_config, use_torch_adam)
    data_loader = random_dataloader(model=model,
                                    total_samples=test_step,
                                    hidden_dim=hidden_dim,
                                    device=model.device,
                                    dtype=dtype)
    for batch in data_loader:
        loss = model(batch[0], batch[1])
        model.backward(loss)
        model.step()

    if ds_config["zero_optimization"]["stage"] == 3:
        model.optimizer._set_fp32_optimizer_param_groups()
        sd = model.optimizer.optimizer.state_dict() if load_optim else None
        model.optimizer._clear_fp32_optimizer_param_groups()
    else:
        sd = model.optimizer.optimizer.state_dict() if load_optim else None

    client_state = {}
    client_state[UNIVERSAL_CHECKPOINT_INFO] = {}
    client_state['iteration'] = test_step
    model.save_checkpoint(tmpdir, tag=CP_TAG, client_state=client_state)

    cp_dir = os.path.join(tmpdir, CP_TAG)
    univ_cp_dir = f"{cp_dir}_universal"

    args = SimpleNamespace(input_folder=cp_dir,
                           output_folder=univ_cp_dir,
                           num_extract_workers=1,
                           num_merge_workers=1,
                           keep_temp_folder=False,
                           strict=True,
                           inject_missing_state=False)

    dist.barrier()
    if dist.get_rank() == 0:
        convert_to_universal(args)

    model_state = model.state_dict()
    optimizer_state = None
    if load_optim:
        if ds_config["zero_optimization"]["stage"] == 3:
            model.optimizer._set_fp32_optimizer_param_groups()
            optimizer_state = gather_opt_state(model.optimizer.optimizer.state_dict())
            model.optimizer._clear_fp32_optimizer_param_groups()
            update_gathered_stage3_optimizer(optimizer_state, model._get_zero_param_shapes(), world_size)
        else:
            optimizer_state = gather_opt_state(model.optimizer.optimizer.state_dict())

    if dist.get_rank() == 0:
        torch.save((model_state, optimizer_state), os.path.join(tmpdir, "baseline_state.pt"))

    dist.barrier()
    model.destroy()


@pytest.fixture
def ds_config(zero_stage, dtype, sub_group_size):
    ds_config = {
        "train_batch_size": 8,
        "optimizer": {
            "type": 'Adam'
        },
        "zero_optimization": {
            "stage": zero_stage,
        }
    }
    if dtype == torch.float16:
        ds_config["fp16"] = {"enabled": True, "initial_scale_power": 8}
    elif dtype == torch.bfloat16:
        ds_config["bf16"] = {"enabled": True}
    if sub_group_size > 0:
        ds_config["zero_optimization"]["sub_group_size"] = sub_group_size
    return ds_config


class _baseline(DistributedFixture):
    world_size = None

    def run(self, tmpdir, ds_config, zero_stage, dtype, load_optim, use_torch_adam):
        # fp16 configs crash deepspeed.initialize's sanity check on accelerators
        # without fp16 support, surfacing as a setup error for every dependent
        # test instead of a skip.
        if dtype == torch.float16 and not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported on this accelerator")
        hidden_dim = 10
        train_save_convert(ds_config, hidden_dim, load_optim, use_torch_adam, dtype, tmpdir, self.world_size)


class baseline_ws2(_baseline):
    world_size = 2


class baseline_ws4(_baseline):
    world_size = 4


# Stage3 use shard parameter, need to reorganize the optimizer parameters.
def update_gathered_stage3_optimizer(optimizer_state, param_shapes, world_size):
    for sub_group_id, group in enumerate(optimizer_state["param_groups"]):
        group["params"] = None

    new_state = {}
    for sub_group_id, sub_group_param_shape in enumerate(param_shapes):
        total_numel = optimizer_state['state'][sub_group_id]['exp_avg'].numel()
        assert total_numel % world_size == 0
        numel_per_rank = total_numel // world_size
        param_offset_in_current_rank = 0
        for param_name, param_shape in sub_group_param_shape.items():
            param_numel = param_shape.numel()
            param_partition_numel = math.ceil(param_numel / world_size)
            param_optimizer_tensor = {
                "exp_avg": torch.zeros(param_numel),
                "exp_avg_sq": torch.zeros(param_numel),
                "step": optimizer_state['state'][sub_group_id]['step'],
            }
            for key in ["exp_avg", "exp_avg_sq"]:
                write_offset = 0
                for rank in range(world_size):
                    offset = param_offset_in_current_rank + rank * numel_per_rank
                    length = min(param_partition_numel, param_numel - rank * param_partition_numel)
                    tmp = optimizer_state['state'][sub_group_id][key].narrow(0, offset, length)
                    param_optimizer_tensor[key].narrow(0, write_offset, length).copy_(tmp)
                    write_offset += length
            param_offset_in_current_rank += param_partition_numel
            new_state[param_name] = param_optimizer_tensor
    optimizer_state["state"] = new_state


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("zero_stage", [1, 3])
@pytest.mark.parametrize("use_torch_adam", [False, True])
@pytest.mark.parametrize("load_optim", [False, True])
@pytest.mark.parametrize("sub_group_size", [-1, 100])
class TestZeROUniversalCheckpointDP(DistributedTest):

    def _run_test(self, tmpdir, dtype, ds_config, load_optim, use_torch_adam, world_size):
        if dtype == torch.bfloat16 and not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        hidden_dim = 10
        loaded_model_state, loaded_optimizer_state = torch.load(f"{tmpdir}/baseline_state.pt", weights_only=False)

        ds_config["checkpoint"] = {"load_universal": True}
        univ_model = SimpleModel(hidden_dim, nlayers=2)
        univ_model = init_ds_engine(univ_model, ds_config, use_torch_adam)
        univ_model.load_checkpoint(tmpdir, tag=f"{CP_TAG}_universal", load_optimizer_states=load_optim)

        model_state = univ_model.state_dict()
        compare_state_dicts(model_state, loaded_model_state)

        if load_optim:
            if ds_config["zero_optimization"]["stage"] == 3:
                univ_model.optimizer._set_fp32_optimizer_param_groups()
                optimizer_state = gather_opt_state(univ_model.optimizer.optimizer.state_dict())
                univ_model.optimizer._clear_fp32_optimizer_param_groups()
                update_gathered_stage3_optimizer(optimizer_state, univ_model._get_zero_param_shapes(), world_size)
            else:
                optimizer_state = gather_opt_state(univ_model.optimizer.optimizer.state_dict())
            # padding sizes may differ when dp sizes are different
            param_count = sum(p.numel() for p in univ_model.parameters())
            optimizer_state = remove_pad_in_opt_state(optimizer_state, param_count)
            loaded_optimizer_state = remove_pad_in_opt_state(loaded_optimizer_state, param_count)

            compare_opt_state_dicts(optimizer_state, loaded_optimizer_state, get_expected_mismatch_keys())

        # Run training again to verify that the optimizer has necessary states
        test_step = 8
        data_loader = random_dataloader(model=univ_model,
                                        total_samples=test_step,
                                        hidden_dim=hidden_dim,
                                        device=univ_model.device,
                                        dtype=dtype)
        for batch in data_loader:
            loss = univ_model(batch[0], batch[1])
            univ_model.backward(loss)
            univ_model.step()

        univ_model.destroy()

    @pytest.mark.world_size(2)
    def test_dp_world_size_2to2(self, baseline_ws2, tmpdir, dtype, ds_config, load_optim, use_torch_adam):
        self._run_test(tmpdir, dtype, ds_config, load_optim, use_torch_adam, 2)

    @pytest.mark.world_size(2)
    def test_dp_world_size_4to2(self, baseline_ws4, tmpdir, dtype, ds_config, load_optim, use_torch_adam):
        self._run_test(tmpdir, dtype, ds_config, load_optim, use_torch_adam, 2)

    @pytest.mark.world_size(4)
    def test_dp_world_size_2to4(self, baseline_ws2, tmpdir, dtype, ds_config, load_optim, use_torch_adam):
        self._run_test(tmpdir, dtype, ds_config, load_optim, use_torch_adam, 4)


def _fused_torch_adamw(params):
    try:
        return torch.optim.AdamW(params, lr=0.1, fused=True)
    except RuntimeError as e:
        pytest.skip(f"fused torch AdamW is not available on this accelerator: {e}")


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


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("int_step", [False, True])
class TestUniversalCheckpointFusedTorchAdamW(DistributedTest):
    world_size = 1

    def test_resume(self, tmpdir, zero_stage, int_step):
        hidden_dim = 10
        steps_before_save = 2
        ds_config = {"train_batch_size": 1, "zero_optimization": {"stage": zero_stage}}

        model = SimpleModel(hidden_dim, nlayers=2)
        engine, _, _, _ = deepspeed.initialize(config=ds_config,
                                               model=model,
                                               optimizer=_fused_torch_adamw(model.parameters()))
        _train(engine, hidden_dim, steps_before_save)
        engine.save_checkpoint(tmpdir, tag=CP_TAG, client_state={UNIVERSAL_CHECKPOINT_INFO: {}})
        engine.destroy()

        cp_dir = os.path.join(tmpdir, CP_TAG)
        univ_cp_dir = f"{cp_dir}_universal"
        convert_to_universal(
            SimpleNamespace(input_folder=cp_dir,
                            output_folder=univ_cp_dir,
                            num_extract_workers=1,
                            num_merge_workers=1,
                            keep_temp_folder=False,
                            strict=True,
                            inject_missing_state=False))
        if int_step:
            # DeepSpeed's FusedAdam and CPUAdam keep a Python int step, so universal checkpoints
            # converted from them carry one, while fused torch AdamW needs a tensor.
            for step_file in glob.glob(os.path.join(univ_cp_dir, "zero", "*", "step.pt")):
                torch.save(steps_before_save, step_file)
            # ZeRO-3 restores step from the common optimizer state instead of the per-parameter files.
            if zero_stage == 3:
                optim_state_file = os.path.join(univ_cp_dir, "zero", "optimizer_state.pt")
                optim_sd = torch.load(optim_state_file, weights_only=False)
                for state in optim_sd[OPTIMIZER_STATE_DICT]["state"].values():
                    state["step"] = steps_before_save
                torch.save(optim_sd, optim_state_file)

        ds_config["checkpoint"] = {"load_universal": True}
        model = SimpleModel(hidden_dim, nlayers=2)
        engine, _, _, _ = deepspeed.initialize(config=ds_config,
                                               model=model,
                                               optimizer=_fused_torch_adamw(model.parameters()))
        engine.load_checkpoint(tmpdir, tag=f"{CP_TAG}_universal")

        # A step left as loaded (CPU tensor on an accelerator, or an int) fails the first fused step.
        steps_after_load = 3
        _train(engine, hidden_dim, steps_after_load)

        optimizer_states = list(engine.optimizer.optimizer.state.values())
        assert optimizer_states
        for state in optimizer_states:
            assert state["step"].item() == steps_before_save + steps_after_load
        engine.destroy()


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

    def test_load(self, tmpdir):
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
        uc_info = {
            PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: [row_weight],
            TP_REPLICATED_PARAMETER_PATTERNS: [row_bias],
        }
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
            model = SimpleModel(hidden_dim, nlayers=2)
            # Two groups with their own settings. The target starts from other settings, so the check below
            # sees what the load restored.
            param_groups = [{
                "params": list(layer.parameters()),
                "lr": 1.0 if load_universal else lr,
                "weight_decay": 0.0 if load_universal else 0.01 * i
            } for i, (layer, lr) in enumerate(zip(model.linears, group_lrs))]
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
        source.save_checkpoint(tmpdir, tag=CP_TAG, client_state={UNIVERSAL_CHECKPOINT_INFO: {}})
        source.destroy()

        cp_dir = os.path.join(tmpdir, CP_TAG)
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

        target = build(target_stage, load_universal=True)
        target.load_checkpoint(tmpdir, tag=f"{CP_TAG}_universal")

        loaded = _full_optimizer_states(target)
        assert loaded.keys() == expected.keys()
        for name, states in expected.items():
            for expected_state, loaded_state in zip(states, loaded[name]):
                assert torch.equal(expected_state, loaded_state), name
        optimizer_states = list(target.optimizer.optimizer.state.values())
        assert optimizer_states
        for state in optimizer_states:
            assert int(state["step"]) == steps_before_save

        for i, (param_group, lr) in enumerate(zip(target.optimizer.param_groups, group_lrs)):
            assert param_group["lr"] == lr and param_group["weight_decay"] == 0.01 * i, param_group

        # The restored optimizer must also be able to keep training, advancing every state's step once.
        _train(target, hidden_dim, 1)
        for state in target.optimizer.optimizer.state.values():
            assert int(state["step"]) == steps_before_save + 1
        target.destroy()

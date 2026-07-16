# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Compact AutoEP checkpoint tests."""

import os
from types import SimpleNamespace

import deepspeed
import pytest
import torch
import torch.nn as nn

from deepspeed import comm as dist
from deepspeed.checkpoint.ds_to_universal import main as convert_to_universal
from deepspeed.runtime.config import DeepSpeedConfig
from unit.common import DistributedFixture, DistributedTest
from unit.v1.moe.autoep_test_utils import (
    MockMoETransformer,
    UNSUPPORTED_LOAD_BALANCE_VALUES,
    assert_load_balance_coeff_rejection_message,
    engine_input_dtype,
    init_autoep_engine,
    make_autoep_config,
    make_autoep_integration_config,
    run_training_steps,
    seed_everything,
)

TOPOLOGY_TAG = "autoep-zero3-topology"
EXPERT_WEIGHT_NAMES = ("w1", "w2", "w3")
UNIVERSAL_STATE_KEYS = ("fp32", "exp_avg", "exp_avg_sq")


def _convert_checkpoint_to_universal(save_dir, tag):
    checkpoint_dir = os.path.join(save_dir, tag)
    universal_dir = os.path.join(save_dir, f"{tag}_universal")
    args = SimpleNamespace(input_folder=checkpoint_dir,
                           output_folder=universal_dir,
                           num_extract_workers=1,
                           num_merge_workers=1,
                           keep_temp_folder=False,
                           strict=True,
                           inject_missing_state=False)

    dist.barrier()
    if dist.get_rank() == 0:
        convert_to_universal(args)
    dist.barrier()
    return universal_dir


def _load_universal_file(universal_dir, param_name, key):
    return torch.load(os.path.join(universal_dir, "zero", param_name, f"{key}.pt"),
                      map_location="cpu",
                      weights_only=False)


def _load_universal_dense_state(universal_dir, param_name, key):
    state = _load_universal_file(universal_dir, param_name, key)
    assert torch.is_tensor(state), f"expected raw tensor state for dense ZeRO-3 parameter {param_name}/{key}"
    return state


def _load_universal_expert_state(universal_dir, param_name, key):
    from deepspeed.checkpoint.constants import PARAM

    state = _load_universal_file(universal_dir, param_name, key)
    assert isinstance(state, dict), f"expected metadata dict for AutoEP expert parameter {param_name}/{key}"
    return state[PARAM]


def _load_universal_optimizer_step(universal_dir):
    from deepspeed.checkpoint.constants import OPTIMIZER_STATE_DICT

    state = torch.load(os.path.join(universal_dir, "zero", "optimizer_state.pt"),
                       map_location="cpu",
                       weights_only=False)
    step = state[OPTIMIZER_STATE_DICT]["state"][0]["step"]
    return int(step.item() if torch.is_tensor(step) else step)


def _assert_universal_expert_metadata(universal_dir, num_experts):
    from deepspeed.checkpoint.constants import EP_IS_EXPERT_PARAM, EP_NUM_EXPERTS, PARAM

    found = 0
    nonzero_moments = {"exp_avg": False, "exp_avg_sq": False}
    zero_dir = os.path.join(universal_dir, "zero")
    for root, _, files in os.walk(zero_dir):
        for key in UNIVERSAL_STATE_KEYS:
            filename = f"{key}.pt"
            if filename not in files:
                continue
            state = torch.load(os.path.join(root, filename), map_location="cpu", weights_only=False)
            if not isinstance(state, dict) or not state.get(EP_IS_EXPERT_PARAM, False):
                continue
            found += 1
            assert state[EP_NUM_EXPERTS] == num_experts
            assert state[PARAM].shape[0] == num_experts
            if key in nonzero_moments and torch.count_nonzero(state[PARAM]).item() > 0:
                nonzero_moments[key] = True
    assert found > 0
    assert all(nonzero_moments.values())


def _train_save_convert_autoep_zero3(tmpdir, *, tag, ep_size, num_experts=4):
    seed_everything(8642 + ep_size + num_experts)
    config = make_autoep_integration_config(zero_stage=3, ep_size=ep_size)
    engine, _, _, _ = deepspeed.initialize(model=MockMoETransformer(num_experts=num_experts), config=config)
    run_training_steps(engine, num_steps=3)

    save_dir = str(tmpdir)
    engine.save_checkpoint(save_dir, tag=tag)
    universal_dir = _convert_checkpoint_to_universal(save_dir, tag)
    if dist.get_rank() == 0:
        _assert_universal_expert_metadata(universal_dir, num_experts)
    dist.barrier()
    engine.destroy()


def _autoep_modules(engine):
    from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer

    return [(name, module) for name, module in engine.module.named_modules() if isinstance(module, AutoEPMoELayer)]


def _expert_params(engine):
    for module_name, module in _autoep_modules(engine):
        module_prefix = f"{module_name}." if module_name else ""
        for wname in EXPERT_WEIGHT_NAMES:
            yield f"{module_prefix}experts.{wname}", module, getattr(module.experts, wname)


def _router_params(engine):
    for module_name, module in _autoep_modules(engine):
        module_prefix = f"{module_name}." if module_name else ""
        for router_name, param in module.router.named_parameters():
            yield f"{module_prefix}router.{router_name}", param


def _shared_params(engine):
    routed_expert_names = {param_name for param_name, _, _ in _expert_params(engine)}
    router_names = {param_name for param_name, _ in _router_params(engine)}
    for param_name, param in engine.module.named_parameters():
        if param_name not in routed_expert_names and param_name not in router_names:
            yield param_name, param


def _gather_zero_param(param):
    with deepspeed.zero.GatheredParameters([param]):
        return param.detach().clone()


def _collect_by_ep_rank(local_tensor, ep_rank, ep_size, device):
    local_tensor = local_tensor.contiguous()
    gathered = [torch.zeros_like(local_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_tensor)

    ep_rank_tensor = torch.tensor([ep_rank], dtype=torch.long, device=device)
    ep_rank_tensors = [torch.zeros_like(ep_rank_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(ep_rank_tensors, ep_rank_tensor)
    ep_ranks = [int(t.item()) for t in ep_rank_tensors]

    if dist.get_rank() != 0:
        return None

    representatives = {}
    for global_rank, gathered_ep_rank in enumerate(ep_ranks):
        if gathered_ep_rank in representatives:
            torch.testing.assert_close(gathered[global_rank],
                                       gathered[representatives[gathered_ep_rank]],
                                       rtol=0,
                                       atol=0)
        else:
            representatives[gathered_ep_rank] = global_rank
    assert sorted(representatives) == list(range(ep_size))
    return torch.cat([gathered[representatives[rank]] for rank in range(ep_size)], dim=0).cpu()


def _zero_optimizer_param_state(engine, param, key):
    zero_optimizer = engine.optimizer
    for sub_group_id, fp16_group in enumerate(zero_optimizer.fp16_groups):
        offset = 0
        for group_param in fp16_group:
            partition_numel = group_param.partition_numel()
            if group_param is param:
                if key == "fp32":
                    flat_state = zero_optimizer.fp32_partitioned_groups_flat[sub_group_id]
                else:
                    fp32_param = zero_optimizer.fp32_partitioned_groups_flat[sub_group_id]
                    flat_state = zero_optimizer.optimizer.state[fp32_param][key]
                return flat_state.narrow(0, offset, partition_numel).detach().clone()
            offset += partition_numel
    param_name = engine.optimizer.param_names.get(param, "<unknown>")
    raise AssertionError(f"parameter {param_name} was not found in ZeRO fp16 groups")


def _gather_optimizer_state_for_param(engine, param, key):
    local_partition = _zero_optimizer_param_state(engine, param, key).contiguous()
    partition_group = getattr(param, "ds_process_group", dist.get_world_group())
    partition_world = dist.get_world_size(group=partition_group)
    gathered = [torch.zeros_like(local_partition) for _ in range(partition_world)]
    dist.all_gather(gathered, local_partition, group=partition_group)
    full_flat = torch.cat(gathered, dim=0)[:param.ds_numel]
    return full_flat.view(param.ds_shape).contiguous()


def _assert_router_params_match_universal(engine, universal_dir):
    for param_name, param in _router_params(engine):
        restored = _gather_zero_param(param).cpu()
        expected = _load_universal_dense_state(universal_dir, param_name, "fp32").view_as(restored)
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def _assert_shared_params_match_universal(engine, universal_dir):
    for param_name, param in _shared_params(engine):
        restored = _gather_zero_param(param).cpu()
        expected = _load_universal_dense_state(universal_dir, param_name, "fp32").view_as(restored)
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def _assert_expert_params_match_universal(engine, universal_dir):
    for param_name, module, param in _expert_params(engine):
        local_experts = _gather_zero_param(param)
        restored = _collect_by_ep_rank(local_experts, module.ep_rank, module.ep_size, engine.device)
        if dist.get_rank() == 0:
            expected = _load_universal_expert_state(universal_dir, param_name, "fp32")
            torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def _assert_expert_optimizer_states_match_universal(engine, universal_dir):
    nonzero_moments = {"exp_avg": False, "exp_avg_sq": False}
    for param_name, module, param in _expert_params(engine):
        for key in UNIVERSAL_STATE_KEYS:
            local_state = _gather_optimizer_state_for_param(engine, param, key)
            restored = _collect_by_ep_rank(local_state, module.ep_rank, module.ep_size, engine.device)
            if dist.get_rank() == 0:
                expected = _load_universal_expert_state(universal_dir, param_name, key)
                torch.testing.assert_close(restored, expected, rtol=0, atol=0)
                if key in nonzero_moments and torch.count_nonzero(expected).item() > 0:
                    nonzero_moments[key] = True
    if dist.get_rank() == 0:
        assert all(nonzero_moments.values())
    dist.barrier()


def _assert_expert_fp32_master_params_match_universal(engine, universal_dir):
    for param_name, module, param in _expert_params(engine):
        local_state = _gather_optimizer_state_for_param(engine, param, "fp32")
        restored = _collect_by_ep_rank(local_state, module.ep_rank, module.ep_size, engine.device)
        if dist.get_rank() == 0:
            expected = _load_universal_expert_state(universal_dir, param_name, "fp32")
            torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    dist.barrier()


def _assert_dense_fp32_master_params_match_universal(engine, universal_dir, param_iter):
    for param_name, param in param_iter:
        restored = _gather_optimizer_state_for_param(engine, param, "fp32").cpu()
        expected = _load_universal_dense_state(universal_dir, param_name, "fp32").view_as(restored)
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def _assert_fp32_master_params_match_universal(engine, universal_dir):
    _assert_expert_fp32_master_params_match_universal(engine, universal_dir)
    _assert_dense_fp32_master_params_match_universal(engine, universal_dir, _router_params(engine))
    _assert_dense_fp32_master_params_match_universal(engine, universal_dir, _shared_params(engine))


def _assert_module_params_match_universal(engine, universal_dir):
    _assert_expert_params_match_universal(engine, universal_dir)
    _assert_router_params_match_universal(engine, universal_dir)
    _assert_shared_params_match_universal(engine, universal_dir)


def _assert_optimizer_step_restored(engine, universal_dir):
    expected_step = _load_universal_optimizer_step(universal_dir)
    steps = []
    for fp32_param in engine.optimizer.fp32_partitioned_groups_flat:
        step = engine.optimizer.optimizer.state[fp32_param]["step"]
        steps.append(int(step.item() if torch.is_tensor(step) else step))
    assert steps
    assert expected_step > 0
    assert len(set(steps)) == 1
    assert steps[0] == expected_step


def _assert_forward_runs(engine):
    with torch.no_grad():
        output = engine(torch.randn(1, 8, 64, device=engine.device, dtype=engine_input_dtype(engine)))
    assert torch.isfinite(output.float()).all()


def _run_training_steps_with_engine_input_dtype(engine, num_steps=2, seq_len=8, hidden_dim=64):
    losses = []
    for _ in range(num_steps):
        x = torch.randn(1, seq_len, hidden_dim, device=engine.device, dtype=engine_input_dtype(engine))
        loss = engine(x).mean()
        engine.backward(loss)
        engine.step()
        losses.append(loss.item())
    return losses


def _assert_topology_load_matches_universal(tmpdir,
                                            *,
                                            target_ep_size,
                                            num_experts=4,
                                            tag=TOPOLOGY_TAG,
                                            load_kwargs=None,
                                            check_optimizer_states=True):
    save_dir = str(tmpdir)
    universal_dir = os.path.join(save_dir, f"{tag}_universal")
    config = make_autoep_integration_config(zero_stage=3, ep_size=target_ep_size)
    config["checkpoint"] = {"load_universal": True}
    engine, _, _, _ = deepspeed.initialize(model=MockMoETransformer(num_experts=num_experts), config=config)
    engine.load_checkpoint(save_dir, tag=f"{tag}_universal", **(load_kwargs or {}))

    _assert_module_params_match_universal(engine, universal_dir)
    if check_optimizer_states:
        _assert_expert_optimizer_states_match_universal(engine, universal_dir)
        _assert_optimizer_step_restored(engine, universal_dir)

    _assert_forward_runs(engine)
    losses, _ = run_training_steps(engine, num_steps=1)
    assert torch.isfinite(torch.tensor(losses[0]))
    engine.destroy()


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("include_key", [False, True])
def test_load_balance_coeff_disabled_values_accepted_by_deepspeed_config(enabled, include_key):
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "expert_parallel": {
            "enabled": enabled,
            "autoep_size": 1,
            "preset_model": "mixtral",
        },
    }
    if include_key:
        config["expert_parallel"]["load_balance_coeff"] = None

    ds_config = DeepSpeedConfig(config)

    assert ds_config.expert_parallel_config.load_balance_coeff is None
    assert ds_config.expert_parallel_config._load_balance_coeff_explicit is include_key


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("value", UNSUPPORTED_LOAD_BALANCE_VALUES)
def test_load_balance_coeff_rejected_by_deepspeed_config(enabled, value):
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "expert_parallel": {
            "enabled": enabled,
            "autoep_size": 1,
            "preset_model": "mixtral",
            "load_balance_coeff": value,
        },
    }

    with pytest.raises(ValueError) as exc_info:
        DeepSpeedConfig(config)
    assert_load_balance_coeff_rejection_message(exc_info.value, value)


class TestAutoEPCheckpointSaveLoad(DistributedTest):
    world_size = 1

    def test_save_load_same_ep_and_metadata(self, tmpdir):
        engine = init_autoep_engine(ep_size=1)
        params_before = {name: param.detach().clone() for name, param in engine.module.named_parameters()}
        save_dir = str(tmpdir)
        tag = "autoep"

        engine.save_checkpoint(save_dir, tag=tag)

        checkpoint = torch.load(os.path.join(save_dir, tag, "mp_rank_00_model_states.pt"),
                                map_location="cpu",
                                weights_only=False)
        metadata = checkpoint["ds_autoep_layers"]
        assert len(metadata) == 2
        for entry in metadata:
            assert {"moe_layer_id", "module_path", "num_experts", "num_local_experts", "ep_size"} <= entry.keys()
            assert entry["num_experts"] == entry["num_local_experts"] * entry["ep_size"]

        reloaded = init_autoep_engine(ep_size=1)
        reloaded.load_checkpoint(save_dir, tag=tag)
        for name, param in reloaded.module.named_parameters():
            assert torch.equal(param, params_before[name]), f"{name} changed after same-EP reload"

    def test_autoep_metadata_schema_validation(self):
        from deepspeed.runtime.engine import DeepSpeedEngine

        with pytest.raises(RuntimeError, match="malformed"):
            DeepSpeedEngine.load_moe_state_dict(checkpoint_path="/fake",
                                                tag="fake",
                                                state_dict={},
                                                old_moe_load=False,
                                                model=nn.Linear(1, 1),
                                                autoep_layers="not_a_list")

        with pytest.raises(RuntimeError, match="missing fields"):
            DeepSpeedEngine.load_moe_state_dict(checkpoint_path="/fake",
                                                tag="fake",
                                                state_dict={},
                                                old_moe_load=False,
                                                model=nn.Linear(1, 1),
                                                autoep_layers=[{
                                                    "moe_layer_id": 0
                                                }])


class TestAutoEPZero3UniversalCheckpoint(DistributedTest):
    world_size = 2

    def test_zero3_partition_native_universal_round_trip_same_topology(self, tmpdir):
        seed_everything(2468)

        config = make_autoep_integration_config(zero_stage=3, ep_size=2)
        engine, _, _, _ = deepspeed.initialize(model=MockMoETransformer(), config=config)
        run_training_steps(engine, num_steps=1)

        save_dir = str(tmpdir)
        tag = "autoep-zero3"
        engine.save_checkpoint(save_dir, tag=tag)

        checkpoint_dir = os.path.join(save_dir, tag)
        universal_dir = os.path.join(save_dir, f"{tag}_universal")
        args = SimpleNamespace(input_folder=checkpoint_dir,
                               output_folder=universal_dir,
                               num_extract_workers=1,
                               num_merge_workers=1,
                               keep_temp_folder=False,
                               strict=True,
                               inject_missing_state=False)

        dist.barrier()
        if dist.get_rank() == 0:
            convert_to_universal(args)
        dist.barrier()

        from deepspeed.checkpoint.constants import PARAM
        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
        for module_name, module in engine.module.named_modules():
            if not isinstance(module, AutoEPMoELayer):
                continue
            module_prefix = f"{module_name}." if module_name else ""
            for wname in ("w1", "w2", "w3"):
                param = getattr(module.experts, wname)
                with deepspeed.zero.GatheredParameters([param]):
                    local_experts = param.detach().clone()
                gathered = [torch.zeros_like(local_experts) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered, local_experts)
                if dist.get_rank() == 0:
                    expected = torch.cat(gathered, dim=0).cpu()
                    universal = torch.load(
                        os.path.join(universal_dir, "zero", f"{module_prefix}experts.{wname}", "fp32.pt"),
                        map_location="cpu",
                        weights_only=False,
                    )[PARAM]
                    torch.testing.assert_close(universal, expected)

        universal_config = make_autoep_integration_config(zero_stage=3, ep_size=2)
        universal_config["checkpoint"] = {"load_universal": True}
        reloaded_engine, _, _, _ = deepspeed.initialize(model=MockMoETransformer(), config=universal_config)
        reloaded_engine.load_checkpoint(save_dir, tag=f"{tag}_universal")

        for expected, restored in zip(engine.optimizer.fp16_partitioned_groups_flat,
                                      reloaded_engine.optimizer.fp16_partitioned_groups_flat):
            torch.testing.assert_close(restored, expected)

        losses, _ = run_training_steps(reloaded_engine, num_steps=1)
        assert torch.isfinite(torch.tensor(losses[0]))

    def _assert_zero3_universal_weights_only_load(self, tmpdir, load_kwargs):
        seed_everything(6420)

        config = make_autoep_integration_config(zero_stage=3, ep_size=2)
        engine, _, _, _ = deepspeed.initialize(model=MockMoETransformer(), config=config)
        run_training_steps(engine, num_steps=2)

        save_dir = str(tmpdir)
        tag = "autoep-zero3-universal-flags"
        engine.save_checkpoint(save_dir, tag=tag)
        universal_dir = _convert_checkpoint_to_universal(save_dir, tag)

        universal_config = make_autoep_integration_config(zero_stage=3, ep_size=2)
        universal_config["checkpoint"] = {"load_universal": True}
        reloaded_engine, _, _, _ = deepspeed.initialize(model=MockMoETransformer(), config=universal_config)
        reloaded_engine.load_checkpoint(save_dir, tag=f"{tag}_universal", **load_kwargs)

        _assert_module_params_match_universal(reloaded_engine, universal_dir)
        _assert_forward_runs(reloaded_engine)
        losses, _ = run_training_steps(reloaded_engine, num_steps=1)
        assert torch.isfinite(torch.tensor(losses[0]))

        reloaded_engine.destroy()
        engine.destroy()

    def test_zero3_universal_load_optimizer_states_false_same_topology(self, tmpdir):
        self._assert_zero3_universal_weights_only_load(tmpdir, {"load_optimizer_states": False})

    def test_zero3_universal_module_only_same_topology(self, tmpdir):
        self._assert_zero3_universal_weights_only_load(tmpdir, {"load_module_only": True})

    @pytest.mark.parametrize("load_kwargs", [{"load_optimizer_states": False}, {"load_module_only": True}])
    def test_zero3_universal_weights_only_preserves_fp32_master_weights(self, tmpdir, load_kwargs):
        seed_everything(6421)

        config = make_autoep_config(zero_stage=3, ep_size=2)
        engine, _, _, _ = deepspeed.initialize(model=MockMoETransformer(), config=config)
        _run_training_steps_with_engine_input_dtype(engine, num_steps=2)

        save_dir = str(tmpdir)
        tag = "autoep-zero3-universal-fp32-master"
        engine.save_checkpoint(save_dir, tag=tag)
        universal_dir = _convert_checkpoint_to_universal(save_dir, tag)

        universal_config = make_autoep_config(zero_stage=3, ep_size=2)
        universal_config["checkpoint"] = {"load_universal": True}
        reloaded_engine, _, _, _ = deepspeed.initialize(model=MockMoETransformer(), config=universal_config)
        reloaded_engine.load_checkpoint(save_dir, tag=f"{tag}_universal", **load_kwargs)

        _assert_fp32_master_params_match_universal(reloaded_engine, universal_dir)
        _assert_forward_runs(reloaded_engine)

        reloaded_engine.destroy()
        engine.destroy()


class TestAutoEPZero3UniversalCheckpoint4GPU(DistributedTest):
    world_size = 4

    def test_zero3_partition_native_universal_round_trip_replica_groups_4gpu(self, tmpdir):
        """Same round trip as the 2-GPU test, but with expert-DP world size 2 so
        the converter consolidates multiple partition fragments per expert
        parameter and the universal/module-only loads slice real shard offsets
        instead of the degenerate world_size=1 case."""
        seed_everything(1357)

        config = make_autoep_integration_config(zero_stage=3, ep_size=2)
        engine, _, _, _ = deepspeed.initialize(model=MockMoETransformer(), config=config)
        run_training_steps(engine, num_steps=1)

        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
        autoep_modules = [(name, module) for name, module in engine.module.named_modules()
                          if isinstance(module, AutoEPMoELayer)]
        assert autoep_modules
        for _, module in autoep_modules:
            for param in module.experts.parameters():
                assert param.ds_zero_partition_world_size == 2

        save_dir = str(tmpdir)
        tag = "autoep-zero3-4gpu"
        engine.save_checkpoint(save_dir, tag=tag)

        # Module-only restore must reassemble expert weights from two real
        # partition shards per replica group.
        module_only_engine, _, _, _ = deepspeed.initialize(model=MockMoETransformer(),
                                                           config=make_autoep_integration_config(zero_stage=3,
                                                                                                 ep_size=2))
        module_only_engine.load_checkpoint(save_dir, tag=tag, load_optimizer_states=False)
        for expected, restored in zip(engine.optimizer.fp16_partitioned_groups_flat,
                                      module_only_engine.optimizer.fp16_partitioned_groups_flat):
            torch.testing.assert_close(restored, expected)

        checkpoint_dir = os.path.join(save_dir, tag)
        universal_dir = os.path.join(save_dir, f"{tag}_universal")
        args = SimpleNamespace(input_folder=checkpoint_dir,
                               output_folder=universal_dir,
                               num_extract_workers=1,
                               num_merge_workers=1,
                               keep_temp_folder=False,
                               strict=True,
                               inject_missing_state=False)

        dist.barrier()
        if dist.get_rank() == 0:
            convert_to_universal(args)
        dist.barrier()

        from deepspeed.checkpoint.constants import PARAM
        world_size = dist.get_world_size()
        for module_name, module in autoep_modules:
            module_prefix = f"{module_name}." if module_name else ""
            ep_rank_tensor = torch.tensor([module.ep_rank], dtype=torch.long, device=engine.device)
            ep_ranks = [torch.zeros_like(ep_rank_tensor) for _ in range(world_size)]
            dist.all_gather(ep_ranks, ep_rank_tensor)
            ep_ranks = [int(t.item()) for t in ep_ranks]
            for wname in ("w1", "w2", "w3"):
                param = getattr(module.experts, wname)
                with deepspeed.zero.GatheredParameters([param]):
                    local_experts = param.detach().clone()
                gathered = [torch.zeros_like(local_experts) for _ in range(world_size)]
                dist.all_gather(gathered, local_experts)
                if dist.get_rank() == 0:
                    # Replicas within an EP rank must agree; keep one
                    # representative per EP rank in EP-rank order.
                    representative = {}
                    for global_rank, ep_rank in enumerate(ep_ranks):
                        if ep_rank in representative:
                            torch.testing.assert_close(gathered[global_rank], gathered[representative[ep_rank]])
                        else:
                            representative[ep_rank] = global_rank
                    assert sorted(representative) == list(range(module.ep_size))
                    expected = torch.cat([gathered[representative[ep_rank]] for ep_rank in range(module.ep_size)],
                                         dim=0).cpu()
                    universal = torch.load(
                        os.path.join(universal_dir, "zero", f"{module_prefix}experts.{wname}", "fp32.pt"),
                        map_location="cpu",
                        weights_only=False,
                    )[PARAM]
                    torch.testing.assert_close(universal, expected)

        universal_config = make_autoep_integration_config(zero_stage=3, ep_size=2)
        universal_config["checkpoint"] = {"load_universal": True}
        reloaded_engine, _, _, _ = deepspeed.initialize(model=MockMoETransformer(), config=universal_config)
        reloaded_engine.load_checkpoint(save_dir, tag=f"{tag}_universal")

        for expected, restored in zip(engine.optimizer.fp16_partitioned_groups_flat,
                                      reloaded_engine.optimizer.fp16_partitioned_groups_flat):
            torch.testing.assert_close(restored, expected)

        losses, _ = run_training_steps(reloaded_engine, num_steps=1)
        assert torch.isfinite(torch.tensor(losses[0]))


class _AutoEPTopologyBaselineWs4Ep2(DistributedFixture):
    world_size = 4

    def run(self, tmpdir):
        _train_save_convert_autoep_zero3(tmpdir, tag=TOPOLOGY_TAG, ep_size=2)


@pytest.fixture
def autoep_topology_baseline_ws4_ep2(request):
    _AutoEPTopologyBaselineWs4Ep2()(request)


class TestAutoEPZero3UniversalTopologyChange(DistributedTest):
    world_size = 4

    @pytest.mark.world_size(2)
    def test_dp_world_size_4to2_fixed_ep_size(self, autoep_topology_baseline_ws4_ep2, tmpdir):
        _assert_topology_load_matches_universal(tmpdir, target_ep_size=2)

    @pytest.mark.world_size(8)
    def test_dp_world_size_4to8_fixed_ep_size(self, autoep_topology_baseline_ws4_ep2, tmpdir):
        _assert_topology_load_matches_universal(tmpdir, target_ep_size=2)

    @pytest.mark.world_size(4)
    def test_autoep_size_2to4_fixed_world_size(self, autoep_topology_baseline_ws4_ep2, tmpdir):
        _assert_topology_load_matches_universal(tmpdir, target_ep_size=4)

    @pytest.mark.world_size(4)
    def test_autoep_size_2to1_fixed_world_size(self, autoep_topology_baseline_ws4_ep2, tmpdir):
        _assert_topology_load_matches_universal(tmpdir, target_ep_size=1)

    @pytest.mark.world_size(8)
    def test_dp_world_size_4to8_and_autoep_size_2to4(self, autoep_topology_baseline_ws4_ep2, tmpdir):
        _assert_topology_load_matches_universal(tmpdir, target_ep_size=4)

    @pytest.mark.world_size(2)
    def test_module_only_dp_world_size_4to2_fixed_ep_size(self, autoep_topology_baseline_ws4_ep2, tmpdir):
        _assert_topology_load_matches_universal(tmpdir,
                                                target_ep_size=2,
                                                load_kwargs={"load_module_only": True},
                                                check_optimizer_states=False)

    @pytest.mark.world_size(4)
    def test_load_optimizer_states_false_autoep_size_2to4(self, autoep_topology_baseline_ws4_ep2, tmpdir):
        _assert_topology_load_matches_universal(tmpdir,
                                                target_ep_size=4,
                                                load_kwargs={"load_optimizer_states": False},
                                                check_optimizer_states=False)

    @pytest.mark.world_size(4)
    def test_universal_load_rejects_mismatched_target_expert_shape(self, autoep_topology_baseline_ws4_ep2, tmpdir):
        save_dir = str(tmpdir)
        config = make_autoep_integration_config(zero_stage=3, ep_size=2)
        config["checkpoint"] = {"load_universal": True}
        engine, _, _, _ = deepspeed.initialize(model=MockMoETransformer(num_experts=8), config=config)
        with pytest.raises(ValueError, match="target_local_experts=4, checkpoint_local_experts=2"):
            engine.load_checkpoint(save_dir, tag=f"{TOPOLOGY_TAG}_universal")
        engine.destroy()

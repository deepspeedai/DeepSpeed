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
from unit.common import DistributedTest
from unit.v1.moe.autoep_test_utils import (
    MockMoETransformer,
    UNSUPPORTED_LOAD_BALANCE_VALUES,
    assert_load_balance_coeff_rejection_message,
    init_autoep_engine,
    make_autoep_integration_config,
    run_training_steps,
    seed_everything,
)


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

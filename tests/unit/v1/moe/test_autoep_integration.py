# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Integration tests for AutoEP (multi-GPU, requires distributed backend)."""

import os

import pytest
import torch
import torch.nn as nn
import deepspeed
from deepspeed import comm as dist
from deepspeed.moe.layer import MoE
from unit.v1.moe.autoep_test_utils import (
    MockMoETransformer,
    make_autoep_integration_config as _make_autoep_config,
    run_training_steps as _run_training_steps,
    seed_everything as _seed_everything,
)
from unit.common import DistributedTest


def _assert_global_grad_norm_consistent(engine):
    norm_groups = engine.optimizer._get_norm_groups()
    local_norm = torch.linalg.vector_norm(torch.stack(norm_groups)).detach().reshape(1)
    gathered = [torch.zeros_like(local_norm) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_norm)
    for norm in gathered[1:]:
        assert torch.allclose(norm, gathered[0], rtol=1e-4, atol=1e-4), [float(item.item()) for item in gathered]


# ---------------------------------------------------------------------------
# Test class: AutoEP integration (world_size=2)
# ---------------------------------------------------------------------------


class TestAutoEPOnly(DistributedTest):
    world_size = 2

    def test_zero2_ep_2gpu(self):
        """EP with ZeRO-2 training.

        Verifies EP and ZeRO Stage 2 work together: finite losses
        and parameters actually update across training steps.
        Note: ZeRO-2 partitions gradients, so p.grad may be None on some ranks.
        """
        _seed_everything(1234)

        model = MockMoETransformer()
        config = _make_autoep_config(zero_stage=2, ep_size=2)
        engine, _, _, _ = deepspeed.initialize(model=model, config=config)

        # Verify replacement
        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
        replaced_count = sum(1 for _, m in engine.module.named_modules() if isinstance(m, AutoEPMoELayer))
        assert replaced_count == 2, (f"Expected 2 MoE layers replaced with ZeRO-2, found {replaced_count}")

        # Snapshot parameter values before training
        params_before = {n: p.data.clone().float() for n, p in engine.module.named_parameters() if p.requires_grad}

        # Run training steps (ignore grad norms since ZeRO-2 partitions them)
        losses, _ = _run_training_steps(engine, num_steps=3)

        for i, loss_val in enumerate(losses):
            assert torch.isfinite(torch.tensor(loss_val)), (f"Loss at step {i} is not finite: {loss_val}")

        # Verify at least some parameters changed (optimizer step took effect)
        params_changed = 0
        for n, p in engine.module.named_parameters():
            if n in params_before and not torch.equal(p.data.float(), params_before[n]):
                params_changed += 1
        assert params_changed > 0, "No parameters changed after 3 training steps with ZeRO-2"

    def test_zero3_ep_train_step_and_placement_2gpu(self):
        """EP with ZeRO-3 trains when AutoEP owns the MoE layers."""
        _seed_everything(1234)

        model = MockMoETransformer()
        config = _make_autoep_config(zero_stage=3, ep_size=2)
        engine, _, _, _ = deepspeed.initialize(model=model, config=config)

        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
        autoep_layers = [m for _, m in engine.module.named_modules() if isinstance(m, AutoEPMoELayer)]
        assert len(autoep_layers) == 2

        for layer in autoep_layers:
            for param in layer.experts.parameters():
                assert param.ds_zero_placement_family == "autoep_expert"
                assert param.ds_zero_partition_group_name == layer.ep_group_name
                assert param.ds_zero_partition_world_size == 1
            for param in layer.router.parameters():
                assert param.ds_zero_placement_family == "replicated"
                assert param.ds_zero_partition_world_size == 2

        losses, _ = _run_training_steps(engine, num_steps=1)
        assert torch.isfinite(torch.tensor(losses[0]))

    def test_zero3_native_moe_rejected_2gpu(self):

        class NativeMoEModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.moe = MoE(hidden_size=64, expert=nn.Linear(64, 64), num_experts=2, ep_size=2)

            def forward(self, x):
                output, _, _ = self.moe(x)
                return output

        config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                },
            },
            "zero_optimization": {
                "stage": 3,
            },
        }

        with pytest.raises(AssertionError, match="Native DeepSpeed MoE"):
            deepspeed.initialize(model=NativeMoEModel(), config=config)

    def test_zero3_ep_save_load_same_topology_2gpu(self, tmpdir):
        _seed_everything(5678)

        model = MockMoETransformer()
        config = _make_autoep_config(zero_stage=3, ep_size=2)
        engine, _, _, _ = deepspeed.initialize(model=model, config=config)
        _run_training_steps(engine, num_steps=1)

        save_dir = str(tmpdir)
        engine.save_checkpoint(save_dir, tag="autoep-zero3")
        checkpoint_dir = os.path.join(save_dir, "autoep-zero3")
        checkpoint_files = os.listdir(checkpoint_dir)
        assert not any(name.startswith("layer_") and "_expert_" in name for name in checkpoint_files)

        model_state = torch.load(os.path.join(checkpoint_dir, "zero_pp_rank_0_mp_rank_00_model_states.pt"),
                                 map_location="cpu",
                                 weights_only=False)
        from deepspeed.checkpoint.constants import (
            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION,
            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION_KEY,
            AUTOEP_ZERO3_EXPERT_STATE_FORMAT_KEY,
            AUTOEP_ZERO3_PARTITIONED_EXPERT_STATE_FORMAT,
            PARAM_SHAPES,
        )
        assert all(entry[AUTOEP_ZERO3_EXPERT_STATE_FORMAT_KEY] == AUTOEP_ZERO3_PARTITIONED_EXPERT_STATE_FORMAT
                   for entry in model_state["ds_autoep_layers"])
        assert all(entry[AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION_KEY] == AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION
                   for entry in model_state["ds_autoep_layers"])
        param_names = {name for group_shapes in model_state[PARAM_SHAPES] for name in group_shapes}
        assert any(name.endswith("experts.w1") for name in param_names)

        reloaded = MockMoETransformer()
        reloaded_engine, _, _, _ = deepspeed.initialize(model=reloaded, config=config)
        _, client_state = reloaded_engine.load_checkpoint(save_dir, tag="autoep-zero3")
        assert client_state is not None

        module_only = MockMoETransformer()
        module_only_engine, _, _, _ = deepspeed.initialize(model=module_only, config=config)
        module_only_engine.load_checkpoint(save_dir, tag="autoep-zero3", load_optimizer_states=False)

        module_only_flag = MockMoETransformer()
        module_only_flag_engine, _, _, _ = deepspeed.initialize(model=module_only_flag, config=config)
        module_only_flag_engine.load_checkpoint(save_dir, tag="autoep-zero3", load_module_only=True)

        for expected, restored in zip(engine.optimizer.fp16_partitioned_groups_flat,
                                      module_only_engine.optimizer.fp16_partitioned_groups_flat):
            torch.testing.assert_close(restored, expected)
        for expected, restored in zip(engine.optimizer.fp16_partitioned_groups_flat,
                                      module_only_flag_engine.optimizer.fp16_partitioned_groups_flat):
            torch.testing.assert_close(restored, expected)

        losses, _ = _run_training_steps(reloaded_engine, num_steps=1)
        assert torch.isfinite(torch.tensor(losses[0]))


class TestAutoEPZero3ReplicaGroups(DistributedTest):
    world_size = 4

    def test_zero3_ep_source_zero_init_expert_replica_placement_4gpu(self):
        _seed_everything(3456)

        config = _make_autoep_config(zero_stage=3, ep_size=2)
        with deepspeed.zero.Init(config_dict_or_path=config):
            model = MockMoETransformer()
        assert any(hasattr(param, "ds_id") for param in model.parameters())

        engine, _, _, _ = deepspeed.initialize(model=model, config=config)

        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
        autoep_layers = [m for _, m in engine.module.named_modules() if isinstance(m, AutoEPMoELayer)]
        assert len(autoep_layers) == 2

        for layer in autoep_layers:
            for param in layer.experts.parameters():
                assert param.ds_zero_placement_family == "autoep_expert"
                assert param.ds_zero_partition_group_name == layer.ep_group_name
                assert param.ds_zero_partition_world_size == 2
            for param in layer.router.parameters():
                assert param.ds_zero_placement_family == "replicated"
                assert param.ds_zero_partition_world_size == 4

        losses, _ = _run_training_steps(engine, num_steps=1)
        assert torch.isfinite(torch.tensor(losses[0]))

    def test_zero3_ep_expert_replica_group_train_save_load_4gpu(self, tmpdir):
        _seed_everything(9012)

        model = MockMoETransformer()
        config = _make_autoep_config(zero_stage=3, ep_size=2)
        config["gradient_clipping"] = 1.0
        engine, _, _, _ = deepspeed.initialize(model=model, config=config)

        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
        autoep_layers = [m for _, m in engine.module.named_modules() if isinstance(m, AutoEPMoELayer)]
        assert len(autoep_layers) == 2

        for layer in autoep_layers:
            for param in layer.experts.parameters():
                assert param.ds_zero_placement_family == "autoep_expert"
                assert param.ds_zero_partition_group_name == layer.ep_group_name
                assert param.ds_zero_partition_world_size == 2
            for param in layer.router.parameters():
                assert param.ds_zero_placement_family == "replicated"
                assert param.ds_zero_partition_world_size == 4

        x = torch.randn(1, 8, 64, device=engine.device)
        loss = engine(x).mean()
        engine.backward(loss)
        _assert_global_grad_norm_consistent(engine)
        engine.step()
        assert torch.isfinite(engine.optimizer._global_grad_norm)

        save_dir = str(tmpdir)
        engine.save_checkpoint(save_dir, tag="autoep-zero3")

        reloaded = MockMoETransformer()
        reloaded_engine, _, _, _ = deepspeed.initialize(model=reloaded, config=config)
        _, client_state = reloaded_engine.load_checkpoint(save_dir, tag="autoep-zero3")
        assert client_state is not None

        losses, _ = _run_training_steps(reloaded_engine, num_steps=1)
        assert torch.isfinite(torch.tensor(losses[0]))


class TestAutoEPZero3ReplicaGroups8GPU(DistributedTest):
    world_size = 8

    def test_zero3_ep_source_zero_init_expert_replica_placement_8gpu(self):
        _seed_everything(4567)

        config = _make_autoep_config(zero_stage=3, ep_size=4)
        with deepspeed.zero.Init(config_dict_or_path=config):
            model = MockMoETransformer()
        assert any(hasattr(param, "ds_id") for param in model.parameters())

        engine, _, _, _ = deepspeed.initialize(model=model, config=config)

        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
        autoep_layers = [m for _, m in engine.module.named_modules() if isinstance(m, AutoEPMoELayer)]
        assert len(autoep_layers) == 2

        for layer in autoep_layers:
            for param in layer.experts.parameters():
                assert param.ds_zero_placement_family == "autoep_expert"
                assert param.ds_zero_partition_group_name == layer.ep_group_name
                assert param.ds_zero_partition_world_size == 2
            for param in layer.router.parameters():
                assert param.ds_zero_placement_family == "replicated"
                assert param.ds_zero_partition_world_size == 8

        losses, _ = _run_training_steps(engine, num_steps=1)
        assert torch.isfinite(torch.tensor(losses[0]))

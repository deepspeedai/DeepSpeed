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
    engine_input_dtype as _engine_input_dtype,
    make_autoep_client_optimizer_config as _make_client_optimizer_config,
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


# ---------------------------------------------------------------------------
# Test class: AutoEP with a caller-supplied optimizer (world_size=2)
# ---------------------------------------------------------------------------


def _local_shard(param):
    """Local ZeRO-3 shard when partitioned, else the tensor itself."""
    tensor = getattr(param, "ds_tensor", None)
    return (tensor if tensor is not None else param.data).detach().float().clone()


class TestAutoEPClientOptimizer(DistributedTest):
    """AutoEP replaces every MoE module in ``_configure_expert_parallel`` (engine.py), which runs
    before ``_configure_optimizer``. ``torch.optim.Optimizer.__init__`` materialises its argument
    eagerly (``param_groups = list(params)``), so an optimizer the caller built from
    ``model.parameters()`` -- what HF Trainer and Accelerate do when the DeepSpeed config declares
    no ``optimizer`` block -- holds the discarded expert tensors while the live ``GroupedExperts``
    weights belong to no param group.

    Without the remap the symptoms are:
      * with ``zero.Init``: silent -- every expert and router is frozen, loss still falls because
        attention, shared experts and norms train normally;
      * without ``zero.Init``: ``AttributeError: 'Parameter' object has no attribute
        'partition_numel'`` from ``_create_fp16_sub_groups``.

    Every other AutoEP test supplies the optimizer through ds_config, so this path is otherwise
    uncovered.

    Scope: ZeRO-3 only. A caller-supplied optimizer on ZeRO-1/2 with MoE layers additionally
    requires param groups marked ``{"moe": True}`` (``stage_1_and_2.py``, ``bf16_optimizer.py``);
    that is a separate pre-existing requirement and is not addressed here.
    """

    world_size = 2

    def test_client_optimizer_covers_replacement_parameters(self):
        """Every trainable parameter of the replaced module must be optimized."""
        _seed_everything(1234)
        model = MockMoETransformer()
        client_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=client_optimizer,
                                               config=_make_client_optimizer_config())

        owned = {id(p) for group in engine.optimizer.fp16_groups for p in group}
        trainable = [(name, param) for name, param in engine.module.named_parameters() if param.requires_grad]

        orphans = [name for name, param in trainable if id(param) not in owned]
        assert not orphans, f"parameters in the model but in no optimizer param group: {orphans}"

        # The replacement really did happen, so the assertion above is meaningful.
        expert_names = [name for name, _ in engine.module.named_parameters() if ".experts." in name]
        assert expert_names, "AutoEP did not replace any MoE layer"
        assert all(name.endswith((".w1", ".w2", ".w3")) for name in expert_names), expert_names

        # And nothing detached from the module is being optimized in their place.
        live = {id(p) for _, p in engine.module.named_parameters()}
        ghosts = [p for group in engine.optimizer.fp16_groups for p in group if id(p) not in live]
        assert not ghosts, f"{len(ghosts)} optimized parameters are not part of the model"

    def test_client_optimizer_updates_expert_weights(self):
        """A step must move the expert weights, not leave them frozen.

        ``weight_decay`` is 0.1 rather than AdamW's 1e-2 default so that the check does not depend
        on which experts the router happened to pick. An expert that receives no token still has a
        gradient under ZeRO-3, but it is zero, so decoupled weight decay is the only thing acting
        on it: the parameter is scaled by ``1 - lr*weight_decay``. At the default that is 0.999, a
        0.100% move, and bf16's half-spacing is 0.195%-0.391%, so the value rounds straight back
        and ``torch.equal`` would report a correctly stepped parameter as frozen. At 0.1 the move
        is 1%, above that bound at every magnitude.
        """
        _seed_everything(1234)
        model = MockMoETransformer()
        client_optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)

        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=client_optimizer,
                                               config=_make_client_optimizer_config())

        before = {name: _local_shard(param) for name, param in engine.module.named_parameters()}

        hidden = engine.module.config.hidden_size
        inputs = torch.randn(1, 8, hidden, device=engine.device, dtype=_engine_input_dtype(engine))
        loss = engine(inputs).float().pow(2).mean()
        engine.backward(loss)
        engine.step()

        frozen = [
            name for name, param in engine.module.named_parameters()
            if param.requires_grad and torch.equal(_local_shard(param), before[name])
        ]
        assert not frozen, f"parameters unchanged after an optimizer step: {frozen}"

    def test_client_optimizer_preserves_param_group_hyperparameters(self):
        """Both param groups must survive initialize with their hyper-parameters intact.

        The expert weights live in the decayed group, which is listed second here so the remap has
        to target a group other than 0. Which group they actually land in is asserted by
        ``TestClientOptimizerRemap`` in test_autoep_unit.py -- ZeRO-3 empties the client
        optimizer's param groups during init, so it cannot be checked from here.
        """
        _seed_everything(1234)
        model = MockMoETransformer()
        decayed = [p for n, p in model.named_parameters() if not n.endswith("bias")]
        no_decay = [p for n, p in model.named_parameters() if n.endswith("bias")]
        client_optimizer = torch.optim.AdamW(
            [
                {
                    "params": no_decay,
                    "weight_decay": 0.0
                },
                {
                    "params": decayed,
                    "weight_decay": 0.1
                },
            ],
            lr=1e-3,
        )

        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=client_optimizer,
                                               config=_make_client_optimizer_config())

        owned = {id(p) for group in engine.optimizer.fp16_groups for p in group}
        orphans = [n for n, p in engine.module.named_parameters() if p.requires_grad and id(p) not in owned]
        assert not orphans, f"parameters in the model but in no optimizer param group: {orphans}"

        weight_decays = {group.get("weight_decay") for group in engine.optimizer.param_groups}
        assert weight_decays == {0.1, 0.0}, f"param-group hyper-parameters were not preserved: {weight_decays}"

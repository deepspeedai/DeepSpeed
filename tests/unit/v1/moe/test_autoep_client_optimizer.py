# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP with a caller-supplied optimizer.

AutoEP replaces every MoE module in ``_configure_expert_parallel`` (engine.py), which runs before
``_configure_optimizer``. ``torch.optim.Optimizer.__init__`` materialises its argument eagerly
(``param_groups = list(params)``), so an optimizer the caller built from ``model.parameters()`` --
what HF Trainer and Accelerate do when the DeepSpeed config declares no ``optimizer`` block -- holds
the discarded expert tensors while the live ``GroupedExperts`` weights belong to no param group.

Without the remap in ``_remap_client_optimizer_after_module_replacement`` the symptoms are:
  * with ``zero.Init``: silent -- every expert and router is frozen, loss still falls because
    attention, shared experts and norms train normally;
  * without ``zero.Init``: ``AttributeError: 'Parameter' object has no attribute 'partition_numel'``
    from ``_create_fp16_sub_groups``, because the stale params were never ZeRO-converted.

Every other AutoEP test supplies the optimizer through ``ds_config``, so this path is otherwise
uncovered.

Scope: ZeRO-3 only. A caller-supplied optimizer on ZeRO-1/2 with MoE layers additionally requires
param groups marked ``{"moe": True}`` (``stage_1_and_2.py:780``, ``bf16_optimizer.py:128``); that is
a separate pre-existing requirement and is not addressed here.
"""

import deepspeed
import torch

from unit.common import DistributedTest
from unit.v1.moe.autoep_test_utils import (
    MockMoETransformer,
    engine_input_dtype as _engine_input_dtype,
    seed_everything as _seed_everything,
)

HIDDEN = 128
INTERMEDIATE = 256
NUM_EXPERTS = 4
EP_SIZE = 2


def _make_model():
    # num_experts * hidden = 512, far below stage3_param_persistence_threshold (100_000), so the
    # separate router-release bug cannot confound this test.
    return MockMoETransformer(num_layers=2,
                              num_experts=NUM_EXPERTS,
                              hidden_size=HIDDEN,
                              intermediate_size=INTERMEDIATE)


def _config(zero_stage):
    # bf16 deliberately, not the shared fp16 helper: fp16 carries a loss scaler that skips the first
    # optimizer step(s) on overflow, so no parameter would move and the update test below could not
    # distinguish "frozen because of the bug" from "frozen because the step was skipped".
    return {
        "bf16": {
            "enabled": True
        },
        "train_micro_batch_size_per_gpu": 1,
        "zero_optimization": {
            "stage": zero_stage
        },
        "expert_parallel": {
            "enabled": True,
            "autoep_size": EP_SIZE,
            "preset_model": "mixtral",
            "use_grouped_mm": False,
        },
    }


def _local_shard(param):
    """Local ZeRO-3 shard when partitioned, else the tensor itself."""
    tensor = getattr(param, "ds_tensor", None)
    return (tensor if tensor is not None else param.data).detach().float().clone()


def _expert_param_names(module):
    return [name for name, _ in module.named_parameters() if ".experts." in name]


class TestAutoEPClientOptimizer(DistributedTest):
    world_size = 2

    def test_client_optimizer_covers_replacement_parameters(self):
        """Every trainable parameter of the replaced module must be optimized."""
        _seed_everything(1234)
        model = _make_model()
        client_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=client_optimizer,
                                               config=_config(zero_stage=3))

        owned = {id(p) for group in engine.optimizer.fp16_groups for p in group}
        trainable = [(name, param) for name, param in engine.module.named_parameters() if param.requires_grad]

        orphans = [name for name, param in trainable if id(param) not in owned]
        assert not orphans, f"parameters in the model but in no optimizer param group: {orphans}"

        # The replacement really did happen, so the assertion above is meaningful.
        expert_names = _expert_param_names(engine.module)
        assert expert_names, "AutoEP did not replace any MoE layer"
        assert all(name.endswith((".w1", ".w2", ".w3")) for name in expert_names), expert_names

        # And nothing detached from the module is being optimized in their place.
        live = {id(p) for _, p in engine.module.named_parameters()}
        ghosts = [p for group in engine.optimizer.fp16_groups for p in group if id(p) not in live]
        assert not ghosts, f"{len(ghosts)} optimized parameters are not part of the model"

    def test_client_optimizer_updates_expert_weights(self):
        """A step must move the expert weights, not leave them frozen."""
        _seed_everything(1234)
        model = _make_model()
        client_optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=client_optimizer,
                                               config=_config(zero_stage=3))

        before = {name: _local_shard(param) for name, param in engine.module.named_parameters()}

        inputs = torch.randn(1, 8, HIDDEN, device=engine.device, dtype=_engine_input_dtype(engine))
        loss = engine(inputs).float().pow(2).mean()
        engine.backward(loss)
        engine.step()

        frozen = [
            name for name, param in engine.module.named_parameters()
            if param.requires_grad and torch.equal(_local_shard(param), before[name])
        ]
        assert not frozen, f"parameters unchanged after an optimizer step: {frozen}"

    def test_client_optimizer_preserves_param_group_hyperparameters(self):
        """Replacement parameters must inherit the group they came from, not group 0 by default."""
        _seed_everything(1234)
        model = _make_model()
        decayed = [p for n, p in model.named_parameters() if not n.endswith("bias")]
        no_decay = [p for n, p in model.named_parameters() if n.endswith("bias")]
        client_optimizer = torch.optim.AdamW(
            [
                {
                    "params": decayed,
                    "weight_decay": 0.1
                },
                {
                    "params": no_decay,
                    "weight_decay": 0.0
                },
            ],
            lr=1e-3,
        )

        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=client_optimizer,
                                               config=_config(zero_stage=3))

        owned = {id(p) for group in engine.optimizer.fp16_groups for p in group}
        orphans = [n for n, p in engine.module.named_parameters() if p.requires_grad and id(p) not in owned]
        assert not orphans, f"parameters in the model but in no optimizer param group: {orphans}"

        weight_decays = {group.get("weight_decay") for group in engine.optimizer.param_groups}
        assert weight_decays == {0.1, 0.0}, f"param-group hyper-parameters were not preserved: {weight_decays}"

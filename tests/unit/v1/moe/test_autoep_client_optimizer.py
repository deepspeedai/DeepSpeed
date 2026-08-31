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
import pytest
import torch

from deepspeed.runtime.engine import DeepSpeedEngine
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

        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=client_optimizer, config=_config(zero_stage=3))

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

        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=client_optimizer, config=_config(zero_stage=3))

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
        """Both param groups must survive initialize with their hyper-parameters intact.

        The expert weights live in the decayed group, which is listed second here so the remap
        has to target a group other than 0. Which group they actually land in is asserted by
        ``test_replacement_params_join_the_group_their_source_was_in`` below -- ZeRO-3 empties
        the client optimizer's param groups during init, so it cannot be checked from here.
        """
        _seed_everything(1234)
        model = _make_model()
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

        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=client_optimizer, config=_config(zero_stage=3))

        owned = {id(p) for group in engine.optimizer.fp16_groups for p in group}
        orphans = [n for n, p in engine.module.named_parameters() if p.requires_grad and id(p) not in owned]
        assert not orphans, f"parameters in the model but in no optimizer param group: {orphans}"

        weight_decays = {group.get("weight_decay") for group in engine.optimizer.param_groups}
        assert weight_decays == {0.1, 0.0}, f"param-group hyper-parameters were not preserved: {weight_decays}"


class _StubEngine:
    """The remap only reads ``self.client_optimizer``, so the logic can be exercised without
    paying for a distributed AutoEP run."""

    def __init__(self, optimizer):
        self.client_optimizer = optimizer


def _detach_moe_blocks(model):
    """Do to the module tree what AutoEP does: every MoE block's parameters become new objects
    (a fresh router gate plus ``GroupedExperts`` w1/w2/w3), so the originals leave the model.

    Returns the same source mapping AutoEP hands the engine. The fused ``gate_up_proj`` feeds
    both w1 and w3, matching ``repack_expert_source_params``.
    """
    replacement_sources = {}
    for layer in model.model.layers:
        source_gate = layer.mlp.gate
        source_experts = layer.mlp.experts
        replacement = torch.nn.Module()
        replacement.router = torch.nn.Module()
        replacement.router.gate = torch.nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        replacement.experts = torch.nn.Module()
        for name in ("w1", "w2", "w3"):
            shard = torch.nn.Parameter(torch.empty(NUM_EXPERTS // EP_SIZE, INTERMEDIATE, HIDDEN))
            setattr(replacement.experts, name, shard)
        replacement_sources[id(replacement.router.gate.weight)] = [source_gate.weight]
        replacement_sources[id(replacement.experts.w1)] = [source_experts.gate_up_proj]
        replacement_sources[id(replacement.experts.w3)] = [source_experts.gate_up_proj]
        replacement_sources[id(replacement.experts.w2)] = [source_experts.down_proj]
        layer.mlp = replacement
    return replacement_sources


def _remap(optimizer, model, replacement_sources=None):
    DeepSpeedEngine._remap_client_optimizer_after_module_replacement(_StubEngine(optimizer), model,
                                                                     replacement_sources)


def _owned_ids(optimizer):
    return {id(p) for group in optimizer.param_groups for p in group["params"]}


def test_replacement_params_join_the_group_their_source_was_in():
    """The replacements must follow their source group, not fall back to group 0."""
    model = _make_model()
    named = list(model.named_parameters())
    no_decay = [p for n, p in named if n.endswith("bias")]
    decayed = [p for n, p in named if not n.endswith("bias")]
    # no_decay first: the expert weights belong to group 1, so a fallback to group 0 would fail
    optimizer = torch.optim.AdamW([{
        "params": no_decay,
        "weight_decay": 0.0
    }, {
        "params": decayed,
        "weight_decay": 0.1
    }],
                                  lr=1e-3)

    replacement_sources = _detach_moe_blocks(model)
    _remap(optimizer, model, replacement_sources)

    replaced = {id(p) for name, p in model.named_parameters() if ".experts." in name or ".router." in name}
    holders = [gi for gi, group in enumerate(optimizer.param_groups) if replaced & {id(p) for p in group["params"]}]
    assert holders == [1], f"replacement parameters landed in group(s) {holders}, expected [1]"
    assert optimizer.param_groups[1]["weight_decay"] == 0.1


def test_frozen_params_are_not_mistaken_for_replaced_ones():
    """A frozen parameter is still in the model, so it must not look like a replaced one.

    Counting frozen parameters as stale puts every group that holds one into ``stale_by_group``,
    which makes an unambiguous remap raise "invalidated optimizer parameters in more than one
    param group" even though only the MoE block was replaced.
    """
    model = _make_model()
    for name, param in model.named_parameters():
        if ".mlp." not in name:
            param.requires_grad_(False)
    named = list(model.named_parameters())
    decayed = [p for n, p in named if not n.endswith("bias")]
    no_decay = [p for n, p in named if n.endswith("bias")]
    optimizer = torch.optim.AdamW([{
        "params": decayed,
        "weight_decay": 0.1
    }, {
        "params": no_decay,
        "weight_decay": 0.0
    }],
                                  lr=1e-3)

    replacement_sources = _detach_moe_blocks(model)
    _remap(optimizer, model, replacement_sources)

    owned = _owned_ids(optimizer)
    orphans = [name for name, p in model.named_parameters() if p.requires_grad and id(p) not in owned]
    assert not orphans, f"trainable parameters left out of the optimizer: {orphans}"

    dropped = [name for name, p in model.named_parameters() if not p.requires_grad and id(p) not in owned]
    assert not dropped, f"frozen parameters were removed from the client optimizer: {dropped}"


def test_remap_is_a_noop_when_nothing_was_replaced():
    """Without a module replacement the caller's optimizer must be left exactly as it was."""
    model = _make_model()
    for name, param in model.named_parameters():
        if ".mlp." not in name:
            param.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [list(group["params"]) for group in optimizer.param_groups]

    _remap(optimizer, model)

    after = [list(group["params"]) for group in optimizer.param_groups]
    assert after == before, "the optimizer was modified even though no module was replaced"


def test_per_layer_param_groups_survive_replacement():
    """Layer-wise learning-rate decay gives each layer its own group, so the replacement spans
    several groups at once. Each layer's replacements must rejoin that layer's group."""
    model = _make_model()
    groups = []
    for layer_index in range(len(model.model.layers)):
        prefix = f"model.layers.{layer_index}."
        groups.append({
            "params": [p for name, p in model.named_parameters() if name.startswith(prefix)],
            "lr": 1e-4 * (0.9**layer_index)
        })
    groups.append({
        "params": [p for name, p in model.named_parameters() if not name.startswith("model.layers.")],
        "lr": 1e-4
    })
    optimizer = torch.optim.AdamW(groups)

    replacement_sources = _detach_moe_blocks(model)
    _remap(optimizer, model, replacement_sources)

    owned = _owned_ids(optimizer)
    orphans = [name for name, p in model.named_parameters() if p.requires_grad and id(p) not in owned]
    assert not orphans, f"trainable parameters left out of the optimizer: {orphans}"

    for layer_index, layer in enumerate(model.model.layers):
        in_group = {id(p) for p in optimizer.param_groups[layer_index]["params"]}
        stranded = [name for name, p in layer.mlp.named_parameters() if id(p) not in in_group]
        assert not stranded, f"layer {layer_index} replacements missed its own param group: {stranded}"


def test_sources_split_across_groups_raise():
    """``module_list`` storage packs one grouped tensor from one weight per expert. If the caller
    put those weights in different param groups the replacement has no unambiguous home."""
    model = _make_model()
    first_layer = model.model.layers[0].mlp
    source_gate_up = first_layer.experts.gate_up_proj
    source_down = first_layer.experts.down_proj
    others = [p for _, p in model.named_parameters() if p is not source_down]
    optimizer = torch.optim.AdamW([{
        "params": others,
        "weight_decay": 0.1
    }, {
        "params": [source_down],
        "weight_decay": 0.0
    }],
                                  lr=1e-3)

    replacement_sources = _detach_moe_blocks(model)
    w1 = model.model.layers[0].mlp.experts.w1
    replacement_sources[id(w1)] = [source_gate_up, source_down]

    with pytest.raises(RuntimeError, match="split across param groups"):
        _remap(optimizer, model, replacement_sources)

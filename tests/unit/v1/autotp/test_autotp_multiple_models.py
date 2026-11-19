# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Regression tests for multiple AutoTP models living in the same process.

The kv-head / grain sharding state used to be process-wide globals in tp_shard, so a second
AutoTP model overwrote the first one's metadata (#8231). AutoTPMeta is per-model now; these
tests hold that line.
"""

import copy

import pytest
import torch
from types import SimpleNamespace

from unit.common import DistributedTest
from unit.simple_model import SimpleModel, SimpleMoEModel
from unit.v1.moe.autoep_test_utils import MockMoEOnlyTransformer, engine_input_dtype, make_autoep_config

import deepspeed
import deepspeed.comm as dist
from deepspeed.constants import CROSS_RANK, CROSS_SIZE
from deepspeed.utils import groups
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from deepspeed.module_inject.auto_tp import AutoTP
from deepspeed.module_inject.autotp_config import AutoTPConfig
from deepspeed.module_inject.tp_shard import get_shard_size_list

# Only q_proj is partitioned; matches the naming the kv-head split logic looks for.
PARTITION_CONFIG = {
    "use_default_specs": False,
    "layer_specs": [{
        "patterns": [".*q_proj\\.weight$"],
        "partition_type": "column",
    }],
}


def apply_autotp(model, tp_size, partition_config):
    groups._init_tp_mesh_device(tensor_model_parallel_size=tp_size)
    autotp = AutoTP(module=model,
                    all_reduce_linears=[],
                    prefix="",
                    state_dict=None,
                    linear_layer_setting=None,
                    orig_layer_impl=None,
                    keep_module_on_host=False,
                    partition_config=AutoTPConfig.from_dict(partition_config),
                    model_config=getattr(model, "config", None))
    autotp.set_tensor_parallel_config(tp_size, groups.get_tensor_model_parallel_group())
    autotp.update_linear_policies()
    autotp._replace_module(model)
    return model


class AttentionOnlyModel(torch.nn.Module):
    """Minimal stand-in for a decoder layer, named so AutoTP treats it as attention."""

    class _Attention(torch.nn.Module):

        def __init__(self, hidden_dim, proj_dim):
            super().__init__()
            self.q_proj = torch.nn.Linear(hidden_dim, proj_dim, bias=False)

    def __init__(self, hidden_dim, proj_dim, num_kv_heads):
        super().__init__()
        self.self_attn = AttentionOnlyModel._Attention(hidden_dim, proj_dim)
        # AutoTPMeta probes attributes by name, so any config-like object will do.
        self.config = SimpleNamespace(num_key_value_heads=num_kv_heads,
                                      num_attention_heads=num_kv_heads,
                                      hidden_size=hidden_dim)


class TestAutoTPMultipleModels(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    def test_a_second_model_does_not_reshard_the_first(self):
        # 3 kv heads over 2 ranks is uneven ([128, 64] of 192), while 2 kv heads divides evenly
        # ([96, 96]). The second model's split is what a clobbered kv-head count would give the
        # first one, so the two are distinguishable.
        teacher = AttentionOnlyModel(hidden_dim=64, proj_dim=192, num_kv_heads=3)
        teacher = apply_autotp(teacher, tp_size=2, partition_config=PARTITION_CONFIG)
        teacher_layer = teacher.self_attn.q_proj
        teacher_split = list(teacher_layer._partition_sizes)
        assert teacher_split == [128, 64]

        # A second model with a different kv-head count is built in the same process.
        student = AttentionOnlyModel(hidden_dim=64, proj_dim=192, num_kv_heads=2)
        student = apply_autotp(student, tp_size=2, partition_config=PARTITION_CONFIG)
        assert list(student.self_attn.q_proj._partition_sizes) == [96, 96]

        # The teacher still describes, and re-derives, its own split.
        assert teacher_layer.tp_meta.num_kv_heads == 3
        assert list(teacher_layer._partition_sizes) == teacher_split
        assert get_shard_size_list(192, 2, teacher_layer.tp_meta, teacher_layer.name) == teacher_split
        assert teacher_layer.weight.shape[0] == teacher_split[dist.get_rank()]

    def test_multimodal_outer_config_keeps_kv_head_split(self):
        # Multimodal configs keep the head counts only under text_config; passing the outer
        # config to AutoTP would lose num_kv_heads and fall back to an even-grain split that
        # cuts through KV heads.
        model = AttentionOnlyModel(hidden_dim=64, proj_dim=192, num_kv_heads=3)
        text_config = model.config
        model.config = SimpleNamespace(text_config=text_config)

        model = apply_autotp(model, tp_size=2, partition_config=PARTITION_CONFIG)
        layer = model.self_attn.q_proj
        assert layer.tp_meta.num_kv_heads == 3
        assert list(layer._partition_sizes) == [128, 64]
        assert layer.weight.shape[0] == [128, 64][dist.get_rank()]


class WorldMPU:
    """An mpu with no tensor parallelism: data parallel over every rank."""

    def __init__(self):
        # new_group is collective, so every rank creates every single-rank group.
        singles = [dist.new_group(ranks=[rank]) for rank in range(dist.get_world_size())]
        self.model_parallel_group = singles[dist.get_rank()]

    def get_model_parallel_rank(self):
        return 0

    def get_model_parallel_world_size(self):
        return 1

    def get_model_parallel_group(self):
        return self.model_parallel_group

    def get_data_parallel_rank(self):
        return dist.get_rank()

    def get_data_parallel_world_size(self):
        return dist.get_world_size()

    def get_data_parallel_group(self):
        return dist.get_world_group()


class HalvesMPU(WorldMPU):
    """An mpu with no tensor parallelism and two data parallel groups, each over half of the ranks."""

    def __init__(self):
        super().__init__()
        half = dist.get_world_size() // 2
        halves = [dist.new_group(ranks=list(range(start, start + half))) for start in (0, half)]
        self.data_parallel_group = halves[dist.get_rank() // half]

    def get_data_parallel_rank(self):
        return dist.get_rank(group=self.data_parallel_group)

    def get_data_parallel_world_size(self):
        return dist.get_world_size(group=self.data_parallel_group)

    def get_data_parallel_group(self):
        return self.data_parallel_group


def build_engine(tp_size, mpu=None, mesh_param=None, model=None, optimizer=None, extra_config=None):
    config = {"train_micro_batch_size_per_gpu": 1, "gradient_clipping": 0.0, **(extra_config or {})}
    if tp_size > 1:
        # Nothing is sharded; the test is about which groups the engine runs on.
        config["tensor_parallel"] = {
            "autotp_size": tp_size,
            "partition_config": {
                "use_default_specs": False,
                "layer_specs": [{
                    "patterns": [".*\\.weight$"],
                    "partition_type": "skip",
                }],
            },
        }
    if model is None:
        model = SimpleModel(hidden_dim=8)
    engine, _, _, _ = deepspeed.initialize(model=model,
                                           model_parameters=model.parameters(),
                                           optimizer=optimizer,
                                           config=config,
                                           mpu=mpu,
                                           mesh_param=mesh_param)
    return engine


class TestAutoTPEnginesWithDifferentLayouts(DistributedTest):
    """The process-wide TP/DP groups belong to the first engine; an engine with another layout gets its own (#9081)."""
    world_size = 4
    reuse_dist_env = False

    def test_same_tp_size_reuses_the_groups(self):
        first = build_engine(tp_size=2)
        second = build_engine(tp_size=2)
        # Each engine has its own groups object; with the same layout, both hold the process-wide groups.
        assert first.mpu is not second.mpu
        for engine in (first, second):
            assert isinstance(engine.mpu, groups.TensorParallelGroups)
            assert engine.mpu.get_tensor_model_parallel_group() is groups.get_tensor_model_parallel_group()
            assert engine.mpu.get_data_parallel_group() is groups.get_data_parallel_group()
            assert engine.dp_world_size == 2
        assert groups.mpu is groups

    @pytest.mark.parametrize("second_tp_size", [1, 4])
    def test_different_tp_size_gets_its_own_groups(self, second_tp_size):
        # Before #9081, a TP4 engine silently ran on the first engine's TP2 groups, and a TP1 engine took their
        # data parallel group and then failed deep inside initialize.
        first = build_engine(tp_size=2)
        second = build_engine(tp_size=second_tp_size)
        world_size = dist.get_world_size()
        assert not second.mpu.process_wide
        assert second.mpu.get_tensor_model_parallel_world_size() == second_tp_size
        assert second.dp_world_size == world_size // second_tp_size
        assert dist.get_world_size(second.data_parallel_group) == world_size // second_tp_size
        assert second.get_tensor_parallel_rank() == dist.get_rank() % second_tp_size

        # The process-wide groups, and the first engine, keep the TP2 layout.
        assert groups.mpu is groups
        assert groups.get_tensor_model_parallel_world_size() == 2
        assert first.dp_world_size == 2
        assert first.get_tensor_parallel_rank() == dist.get_rank() % 2

    @pytest.mark.parametrize("second_tp_size", [1, 2, 4])
    def test_mesh_engine_after_autotp_raises(self, second_tp_size):
        # A rejected engine must not replace the mesh used by the first engine.
        first = build_engine(tp_size=2, optimizer=lambda params: torch.optim.SGD(params, lr=0.1))
        mesh_device = groups.mesh_device
        reference = SimpleModel(hidden_dim=8).to(first.device)
        reference.load_state_dict(first.module.state_dict())
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)

        with pytest.raises(RuntimeError, match="9081"):
            build_engine(tp_size=second_tp_size, mesh_param=(dist.get_world_size(), 1))

        assert groups.mesh_device is mesh_device
        inputs = torch.ones(1, 8, device=first.device)
        labels = torch.zeros(1, dtype=torch.long, device=first.device)
        expected_loss = reference(inputs, labels)
        expected_loss.backward()
        reference_optimizer.step()
        loss = first(inputs, labels)
        torch.testing.assert_close(loss, expected_loss)
        first.backward(loss)
        first.step()
        for actual, expected in zip(first.module.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_autotp_engine_after_mesh_raises(self):
        # The data parallel getters read groups.mesh_device first, so this engine would run on the mesh's
        # data parallel group, which spans ranks that hold different TP shards.
        build_engine(tp_size=1, mesh_param=(dist.get_world_size(), 1))
        with pytest.raises(RuntimeError, match="mesh.*9081"):
            build_engine(tp_size=2)

    def test_moe_engine_with_its_own_groups_raises(self):
        # Expert groups are still built from the process-wide groups, which describe the TP2 engine.
        build_engine(tp_size=2)
        model = SimpleMoEModel(hidden_dim=8)
        with pytest.raises(ValueError, match="MoE.*9081"):
            build_engine(tp_size=4, model=model)
        # It raises before initialize changes the model or creates any groups.
        assert all(param.device.type == "cpu" for param in model.parameters())
        assert groups.get_tensor_model_parallel_world_size() == 2

    @pytest.mark.parametrize("second_tp_size", [1, 4])
    def test_checkpoint_writer_uses_the_engines_own_groups(self, monkeypatch, second_tp_size):
        # The writer read the layout from groups.mpu, which describes the TP2 engine.
        monkeypatch.setenv(CROSS_RANK, "0")
        monkeypatch.setenv(CROSS_SIZE, "1")
        build_engine(tp_size=2)
        writer = {"checkpoint": {"writer": {"type": "mock", "data_parallel": "replica"}}}
        second = build_engine(tp_size=second_tp_size, extra_config=writer)
        dp_writer = second.checkpoint_engine._writer._data_parallel_writer
        assert dp_writer.world_size == dist.get_world_size() // second_tp_size
        assert dp_writer.rank == dist.get_rank() // second_tp_size

    @pytest.mark.parametrize("second_tp_size", [1, 2])
    def test_mesh_engine_after_an_explicit_mpu_raises(self, second_tp_size):
        # The mesh would replace the data parallel group the module getters return for the explicit mpu.
        build_engine(tp_size=1, mpu=HalvesMPU())
        with pytest.raises(RuntimeError, match="9081"):
            build_engine(tp_size=second_tp_size, mesh_param=(dist.get_world_size(), 1))
        assert groups.mesh_device is None
        assert groups._get_data_parallel_world_size() == dist.get_world_size() // 2

    def test_engine_without_mpu_does_not_inherit_an_explicit_mpu(self):
        # An explicit mpu replaces groups.mpu. A later engine without one used to inherit it, and with it a data
        # parallel group over part of the ranks (#9081).
        explicit = build_engine(tp_size=1, mpu=HalvesMPU())
        assert explicit.dp_world_size == dist.get_world_size() // 2
        second = build_engine(tp_size=1)
        assert second.dp_world_size == dist.get_world_size()
        assert second.mpu.get_tensor_model_parallel_world_size() == 1
        assert dist.get_world_size(second.seq_data_parallel_group) == dist.get_world_size()
        # The process-wide getters keep describing the engine that passed the mpu.
        assert groups._get_data_parallel_world_size() == dist.get_world_size() // 2

    def test_engine_without_mpu_after_autotp_and_explicit_mpu(self):
        build_engine(tp_size=2)
        build_engine(tp_size=1, mpu=WorldMPU())
        third = build_engine(tp_size=1)
        assert third.mpu.get_tensor_model_parallel_world_size() == 1
        assert third.dp_world_size == dist.get_world_size()

    def test_autotp_engine_after_an_explicit_mpu_keeps_it(self):
        # The AutoTP engine used to publish its groups over groups.mpu, so the earlier engine's code that still
        # queries groups (gradient clipping among it) ran on the AutoTP engine's data parallel group.
        explicit_mpu = HalvesMPU()
        build_engine(tp_size=1, mpu=explicit_mpu)
        autotp = build_engine(tp_size=2, optimizer=lambda params: torch.optim.SGD(params, lr=0.1))
        assert groups.mpu is explicit_mpu
        assert groups._get_data_parallel_group() is explicit_mpu.get_data_parallel_group()
        assert not autotp.mpu.process_wide
        rank = dist.get_rank()
        assert autotp.mpu.get_data_parallel_group_ranks() == [rank % 2, rank % 2 + 2]
        # No process-wide TP groups were created for it, and it trains on its own.
        inputs = torch.ones(1, 8, device=autotp.device)
        labels = torch.zeros(1, dtype=torch.long, device=autotp.device)
        loss = autotp(inputs, labels)
        autotp.backward(loss)
        autotp.step()

    def test_autotp_engine_after_autotp_and_explicit_mpu(self):
        first = build_engine(tp_size=2)
        explicit_mpu = WorldMPU()
        build_engine(tp_size=1, mpu=explicit_mpu)
        third = build_engine(tp_size=2)
        assert groups.mpu is explicit_mpu
        assert not third.mpu.process_wide
        # It has the first engine's layout, so it holds the same groups instead of new ones.
        assert third.mpu.get_tensor_model_parallel_group() is first.mpu.get_tensor_model_parallel_group()

    def test_bf16_muon_engine_after_an_autotp_engine(self):
        # The engine's own TP1 groups are dense data parallelism, which BF16 Muon supports.
        build_engine(tp_size=2)
        muon = {
            "bf16": {
                "enabled": True
            },
            "data_types": {
                "grad_accum_dtype": "fp32"
            },
            "zero_optimization": {
                "stage": 1,
                "reduce_scatter": False
            },
            "optimizer": {
                "type": "Muon",
                "params": {
                    "lr": 1e-3,
                    "momentum": 0.95
                }
            },
        }
        engine = build_engine(tp_size=1, extra_config=muon)
        assert isinstance(engine.mpu, groups.TensorParallelGroups)
        assert engine.optimizer._uses_muon

    def test_hybrid_engine_after_an_autotp_engine_builds_inference_groups(self):
        # The engine's own TP1 groups must not stand in for the inference tensor parallel groups.
        transformers = pytest.importorskip("transformers")
        build_engine(tp_size=2)
        config = transformers.OPTConfig(vocab_size=64,
                                        hidden_size=32,
                                        ffn_dim=64,
                                        num_hidden_layers=2,
                                        num_attention_heads=4,
                                        max_position_embeddings=128)
        model = transformers.OPTForCausalLM(config)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
                                               config={
                                                   "train_micro_batch_size_per_gpu": 1,
                                                   "hybrid_engine": {
                                                       "enabled": True,
                                                       "inference_tp_size": 2
                                                   },
                                               })
        assert isinstance(engine.mpu, groups.TensorParallelGroups)
        assert dist.get_world_size(group=engine.mp_group) == 2

    def test_first_engine_without_mpu_is_unchanged(self):
        engine = build_engine(tp_size=1)
        assert engine.mpu is None
        assert groups.mpu is None

    def test_checkpoint_writer_with_moe_layers(self, monkeypatch):
        # The writer also queries the expert groups, which only the groups module has.
        monkeypatch.setenv(CROSS_RANK, "0")
        monkeypatch.setenv(CROSS_SIZE, "1")
        writer = {"checkpoint": {"writer": {"type": "mock"}}}
        engine = build_engine(tp_size=2, model=SimpleMoEModel(hidden_dim=8), extra_config=writer)
        assert engine.has_moe_layers
        assert engine.checkpoint_engine is not None

    def test_autotp_engine_shares_another_engines_groups(self):
        build_engine(tp_size=2)
        # Passing the groups keeps the batch sizes of an engine without an mpu.
        batch = {"train_batch_size": 4, "gradient_accumulation_steps": 1}
        second = build_engine(tp_size=4, extra_config=batch)
        third = build_engine(tp_size=4, mpu=second.mpu, extra_config=batch)
        assert third.mpu is second.mpu
        assert third.data_parallel_group is second.data_parallel_group
        assert third.dp_world_size == dist.get_world_size() // 4
        assert third.train_batch_size() == second.train_batch_size()
        assert groups.get_tensor_model_parallel_world_size() == 2

    def test_autoep_engine_shares_the_process_wide_groups(self):
        first = build_engine(tp_size=2)
        autoep = make_autoep_config(ep_size=2, mixed_precision=False)
        autoep["optimizer"]["params"]["torch_adam"] = True
        second = build_engine(tp_size=2, mpu=first.mpu, model=MockMoEOnlyTransformer(), extra_config=autoep)
        assert second.mpu is first.mpu
        assert any(isinstance(module, AutoEPMoELayer) for module in second.module.modules())
        x = torch.randn(1, 4, 64, device=second.device, dtype=engine_input_dtype(second))
        dist.broadcast(x, groups.get_tensor_model_parallel_src_rank(), group=groups.get_tensor_model_parallel_group())
        loss = second(x).float().mean()
        second.backward(loss)
        second.step()
        assert torch.isfinite(loss.detach()).item()

    def test_shared_groups_must_have_the_engines_tensor_parallel_size(self):
        build_engine(tp_size=2)
        second = build_engine(tp_size=4)
        with pytest.raises(ValueError, match="tensor parallel size 4"):
            build_engine(tp_size=2, mpu=second.mpu)

    def test_engines_over_all_ranks_share_groups_around_a_subset_engine(self):
        # #9074 moves training from every rank to half of them and back: DP4 -> DP2 -> DP4.
        first = build_engine(tp_size=1)
        subset = build_engine(tp_size=1, mpu=HalvesMPU())
        world = build_engine(tp_size=1)
        again = build_engine(tp_size=1, mpu=world.mpu, optimizer=lambda params: torch.optim.SGD(params, lr=0.1))
        assert first.dp_world_size == dist.get_world_size()
        assert subset.dp_world_size == dist.get_world_size() // 2
        assert world.dp_world_size == dist.get_world_size()
        assert again.mpu is world.mpu
        assert again.dp_world_size == dist.get_world_size()
        # Like an engine with groups of its own, one sharing them cannot have what still reads groups.mpu.
        with pytest.raises(ValueError, match="MoE.*9081"):
            build_engine(tp_size=1, mpu=world.mpu, model=SimpleMoEModel(hidden_dim=8))

        inputs = torch.ones(1, 8, device=again.device)
        labels = torch.zeros(1, dtype=torch.long, device=again.device)
        loss = again(inputs, labels)
        again.backward(loss)
        again.step()


class MLP(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.up = torch.nn.Linear(hidden_dim, 4 * hidden_dim)
        self.down = torch.nn.Linear(4 * hidden_dim, hidden_dim)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        return self.loss(self.down(torch.relu(self.up(x))), y)


MLP_PARTITION_CONFIG = {
    "use_default_specs":
    False,
    "layer_specs": [
        {
            "patterns": [".*up\\.weight$"],
            "partition_type": "column",
        },
        {
            "patterns": [".*down\\.weight$"],
            "partition_type": "row",
        },
    ],
}


class TestAutoTPEnginesWithDifferentTPSizesTrain(DistributedTest):
    world_size = 4
    reuse_dist_env = False

    @pytest.mark.parametrize("zero_stage", [0, 1, 2, 3])
    @pytest.mark.parametrize("tp_sizes", [(2, 4), (4, 2)])
    def test_each_engine_matches_an_unsharded_model(self, tp_sizes, zero_stage):
        hidden_dim, lr, clip = 8, 1e-2, 0.1
        global_batch = dist.get_world_size()
        torch.manual_seed(0)
        reference = MLP(hidden_dim)
        engines = []
        for tp_size in tp_sizes:
            config = {
                "train_micro_batch_size_per_gpu": global_batch * tp_size // dist.get_world_size(),
                "gradient_clipping": clip,
                "optimizer": {
                    "type": "Adam",
                    "params": {
                        "lr": lr,
                        "torch_adam": True
                    }
                },
                "zero_optimization": {
                    "stage": zero_stage
                },
                "tensor_parallel": {
                    "autotp_size": tp_size,
                    "partition_config": MLP_PARTITION_CONFIG
                },
            }
            model = copy.deepcopy(reference)
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            engines.append(engine)

        for engine, tp_size in zip(engines, tp_sizes):
            assert engine.mpu.get_tensor_model_parallel_world_size() == tp_size
            # ZeRO-3 partitions the TP shard further; ds_shape is the shard it partitions.
            up_shape = getattr(engine.module.up.weight, "ds_shape", engine.module.up.weight.shape)
            down_shape = getattr(engine.module.down.weight, "ds_shape", engine.module.down.weight.shape)
            assert up_shape[0] == 4 * hidden_dim // tp_size
            assert down_shape[1] == 4 * hidden_dim // tp_size

        device = engines[0].device
        reference = reference.to(device)
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=lr)
        generator = torch.Generator().manual_seed(1)
        for _ in range(4):
            # Each data parallel rank gets its own part of the batch, so a wrong data parallel group shows up as
            # a different update. The loss compares the parameters left by the previous step.
            x = torch.randn(global_batch, hidden_dim, generator=generator).to(device)
            y = torch.randint(0, hidden_dim, (global_batch, ), generator=generator).to(device)
            for engine in engines:
                dp_size, dp_rank = engine.dp_world_size, engine.get_data_parallel_rank()
                part = slice(dp_rank * global_batch // dp_size, (dp_rank + 1) * global_batch // dp_size)
                with torch.no_grad():
                    expected_loss = reference(x[part], y[part])
                loss = engine(x[part], y[part])
                torch.testing.assert_close(loss, expected_loss, rtol=1e-4, atol=1e-5)
                engine.backward(loss)
                engine.step()
            reference_optimizer.zero_grad()
            reference(x, y).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), clip)
            reference_optimizer.step()

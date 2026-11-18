# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Regression tests for multiple AutoTP models living in the same process.

The kv-head / grain sharding state used to be process-wide globals in tp_shard, so a second
AutoTP model overwrote the first one's metadata (#8231). AutoTPMeta is per-model now; these
tests hold that line.
"""

import pytest
import torch
from types import SimpleNamespace

from unit.common import DistributedTest
from unit.simple_model import SimpleModel

import deepspeed
import deepspeed.comm as dist
from deepspeed.utils import groups
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


def build_engine(tp_size, mpu=None, mesh_param=None, optimizer=None):
    config = {"train_micro_batch_size_per_gpu": 1, "gradient_clipping": 0.0}
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
    model = SimpleModel(hidden_dim=8)
    engine, _, _, _ = deepspeed.initialize(model=model,
                                           model_parameters=model.parameters(),
                                           optimizer=optimizer,
                                           config=config,
                                           mpu=mpu,
                                           mesh_param=mesh_param)
    return engine


class TestAutoTPEnginesShareProcessGroups(DistributedTest):
    """The TP/DP groups are process-wide (#9081): a second engine must use the first engine's layout or fail."""
    world_size = 4
    reuse_dist_env = False

    def test_same_tp_size_reuses_the_groups(self):
        build_engine(tp_size=2)
        second = build_engine(tp_size=2)
        assert second.mpu.get_tensor_model_parallel_world_size() == 2
        assert second.dp_world_size == 2

    @pytest.mark.parametrize("second_tp_size", [1, 4])
    def test_different_tp_size_raises(self, second_tp_size):
        # Before the check, a TP4 engine silently ran on the first engine's TP2 groups, and a TP1 engine took
        # their data parallel group and then failed deep inside initialize.
        build_engine(tp_size=2)
        with pytest.raises(RuntimeError, match="9081"):
            build_engine(tp_size=second_tp_size)

    @pytest.mark.parametrize("second_tp_size", [1, 4])
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

    def test_engine_without_mpu_inherits_a_later_explicit_mpu(self):
        # An explicit mpu replaces groups.mpu, so a third engine without one inherits that layout, not the
        # cached AutoTP groups of the first engine.
        build_engine(tp_size=2)
        build_engine(tp_size=1, mpu=WorldMPU())
        third = build_engine(tp_size=1)
        assert third.dp_world_size == dist.get_world_size()

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Fused AutoTP layers must keep the shard widths that were frozen when they were built.

``tp_shard`` keeps the grain size and the kv head count in process-wide globals that every
AutoTP model overwrites while it is being replaced.  A layer that re-derives its split at
partition time therefore cuts the weight differently once a second model has been loaded,
which silently disagrees with the gather and with the universal checkpoint metadata.
"""

import pytest
import torch

from deepspeed.module_inject import tp_shard
from deepspeed.module_inject.layers import GateUpPack_LinearLayer, fused_LinearLayer


@pytest.fixture
def clean_tp_shard_globals():
    yield
    tp_shard.set_tp_grain_size(1)
    tp_shard.set_num_kv_heads(None)


def _build_gate_up_layer(out_features, tp_world_size, tp_index):
    layer = GateUpPack_LinearLayer(torch.nn.Linear(3, out_features, bias=False), mp_group=None, name="dense_h_to_4h")
    layer.tp_world_size = tp_world_size
    layer.tp_index = tp_index
    layer._freeze_partition_sizes(out_features)
    return layer


def test_gate_up_partition_ignores_later_grain_size_changes(clean_tp_shard_globals):
    tp_shard.set_tp_grain_size(1)
    layer = _build_gate_up_layer(out_features=10, tp_world_size=2, tp_index=0)
    assert layer._subparam_shard_widths == [[3, 2], [3, 2]]

    full_weight = torch.arange(30, dtype=torch.float32).view(10, 3)

    first = torch.nn.Parameter(full_weight.clone())
    layer._tp_partition([first, None])

    # A second AutoTP model would install its own grain size before this layer is gathered.
    tp_shard.set_tp_grain_size(4)
    second = torch.nn.Parameter(full_weight.clone())
    layer._tp_partition([second, None])

    assert tuple(second.shape) == (6, 3)
    assert torch.equal(first.data, second.data)


def test_gate_up_partition_covers_the_whole_weight(clean_tp_shard_globals):
    tp_shard.set_tp_grain_size(1)
    full_weight = torch.arange(30, dtype=torch.float32).view(10, 3)

    shards = []
    for tp_index in range(2):
        layer = _build_gate_up_layer(out_features=10, tp_world_size=2, tp_index=tp_index)
        param = torch.nn.Parameter(full_weight.clone())
        layer._tp_partition([param, None])
        shards.append(param.data)

    gate = torch.cat([shards[0][:3], shards[1][:2]], dim=0)
    up = torch.cat([shards[0][3:], shards[1][2:]], dim=0)
    assert torch.equal(torch.cat([gate, up], dim=0), full_weight)


class _QWenAttention(torch.nn.Module):

    def __init__(self, split_size):
        super().__init__()
        self.split_size = split_size


class _QWenBlock(torch.nn.Module):

    def __init__(self, split_size):
        super().__init__()
        self.attn = _QWenAttention(split_size)


def _build_qwen_attn_layer(block, hidden, tp_world_size, tp_index):
    layer = fused_LinearLayer(torch.nn.Linear(hidden, 3 * hidden, bias=False),
                              mp_group=None,
                              skip_partition=True,
                              name="c_attn",
                              fused_module=block)
    layer.tp_world_size = tp_world_size
    layer.tp_index = tp_index
    layer._freeze_partition_sizes(3 * hidden)
    return layer


def test_qwen_split_size_follows_the_frozen_shard_width(clean_tp_shard_globals):
    tp_shard.set_tp_grain_size(1)
    tp_shard.set_num_kv_heads(4)
    tp_shard.set_n_embd(12)

    block = _QWenBlock(split_size=12)
    layer = _build_qwen_attn_layer(block, hidden=12, tp_world_size=4, tp_index=3)

    weight = torch.nn.Parameter(torch.zeros(36, 12))
    layer._tp_partition([weight, None])

    # QWen unpacks query, key and value out of the fused output using this width.
    assert block.attn.split_size == 3
    assert weight.shape[0] == 3 * block.attn.split_size


def test_qwen_rejects_a_tensor_parallel_size_that_empties_a_rank(clean_tp_shard_globals):
    tp_shard.set_tp_grain_size(1)
    tp_shard.set_num_kv_heads(4)
    tp_shard.set_n_embd(12)

    with pytest.raises(RuntimeError, match="empty query/key/value shard"):
        _build_qwen_attn_layer(_QWenBlock(split_size=12), hidden=12, tp_world_size=16, tp_index=15)


class _CodeGenBlock(torch.nn.Module):
    pass


def test_interleaved_fused_layout_refuses_to_gather(clean_tp_shard_globals):
    tp_shard.set_tp_grain_size(1)
    tp_shard.set_num_kv_heads(8)
    tp_shard.set_n_embd(4)

    layer = fused_LinearLayer(torch.nn.Linear(4, 24, bias=False),
                              mp_group=None,
                              skip_partition=True,
                              name="qkv_proj",
                              fused_module=_CodeGenBlock())
    layer.tp_world_size = 2
    layer.tp_index = 0
    layer._freeze_partition_sizes(24)
    assert layer._subparam_sizes is None

    # Concatenating this layout's shards in rank order would consolidate a wrong weight.
    with pytest.raises(RuntimeError, match="interleaves or replicates blocks"):
        layer.gather_params([torch.nn.Parameter(torch.zeros(12, 4)), None])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

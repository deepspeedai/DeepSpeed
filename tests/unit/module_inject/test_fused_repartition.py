# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Fused AutoTP layers keep the shard widths frozen when they were built.

Each AutoTP model carries its own :class:`AutoTPMeta` (kv-head count, grain size, ...), so a
layer's split stays aligned with its gather and its universal checkpoint metadata regardless
of what other models do.  These tests build layers with an explicit ``tp_meta`` and check the
split is stable.
"""

import pytest
import torch

from deepspeed.module_inject.layers import GateUpPack_LinearLayer, fused_LinearLayer
from deepspeed.module_inject.tp_shard import AutoTPMeta


def _build_gate_up_layer(out_features, tp_world_size, tp_index, meta):
    layer = GateUpPack_LinearLayer(torch.nn.Linear(3, out_features, bias=False),
                                   mp_group=None,
                                   name="dense_h_to_4h",
                                   tp_meta=meta)
    layer.tp_world_size = tp_world_size
    layer.tp_index = tp_index
    layer._freeze_partition_sizes(out_features)
    return layer


def test_gate_up_partition_covers_the_whole_weight():
    meta = AutoTPMeta(tp_grain_size=1)
    full_weight = torch.arange(30, dtype=torch.float32).view(10, 3)

    shards = []
    for tp_index in range(2):
        layer = _build_gate_up_layer(out_features=10, tp_world_size=2, tp_index=tp_index, meta=meta)
        # The gate and up halves (5 each) split over 2 ranks as [3, 2]; the layer freezes these
        # widths from its own tp_meta so the partition is deterministic.
        assert layer._subparam_shard_widths == [[3, 2], [3, 2]]
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


def _build_qwen_attn_layer(block, hidden, tp_world_size, tp_index, meta):
    layer = fused_LinearLayer(torch.nn.Linear(hidden, 3 * hidden, bias=False),
                              mp_group=None,
                              skip_partition=True,
                              name="c_attn",
                              fused_module=block,
                              tp_meta=meta)
    layer.tp_world_size = tp_world_size
    layer.tp_index = tp_index
    layer._freeze_partition_sizes(3 * hidden)
    return layer


def test_qwen_split_size_follows_the_frozen_shard_width():
    meta = AutoTPMeta(num_kv_heads=4, n_embd=12, tp_grain_size=1)
    block = _QWenBlock(split_size=12)
    layer = _build_qwen_attn_layer(block, hidden=12, tp_world_size=4, tp_index=3, meta=meta)

    weight = torch.nn.Parameter(torch.zeros(36, 12))
    layer._tp_partition([weight, None])

    # QWen unpacks query, key and value out of the fused output using this width.
    assert block.attn.split_size == 3
    assert weight.shape[0] == 3 * block.attn.split_size


def test_qwen_rejects_a_tensor_parallel_size_that_empties_a_rank():
    meta = AutoTPMeta(num_kv_heads=4, n_embd=12, tp_grain_size=1)

    with pytest.raises(RuntimeError, match="empty query/key/value shard"):
        _build_qwen_attn_layer(_QWenBlock(split_size=12), hidden=12, tp_world_size=16, tp_index=15, meta=meta)


class _CodeGenBlock(torch.nn.Module):
    pass


def test_interleaved_fused_layout_refuses_to_gather():
    meta = AutoTPMeta(num_kv_heads=8, n_embd=4, tp_grain_size=1)

    layer = fused_LinearLayer(torch.nn.Linear(4, 24, bias=False),
                              mp_group=None,
                              skip_partition=True,
                              name="qkv_proj",
                              fused_module=_CodeGenBlock(),
                              tp_meta=meta)
    layer.tp_world_size = 2
    layer.tp_index = 0
    layer._freeze_partition_sizes(24)
    assert layer._subparam_sizes is None

    # Concatenating this layout's shards in rank order would consolidate a wrong weight.
    with pytest.raises(RuntimeError, match="interleaves or replicates blocks"):
        layer.gather_params([torch.nn.Parameter(torch.zeros(12, 4)), None])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

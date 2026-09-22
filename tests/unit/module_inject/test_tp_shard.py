# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
from types import SimpleNamespace

from deepspeed.module_inject import tp_shard
from deepspeed.module_inject.tp_shard import AutoTPMeta, get_shard_size, get_shard_size_list


@pytest.mark.parametrize("total_size,tp_size", [(50257, 2), (50257, 8), (151936, 8), (32000, 4)])
def test_grain_quantized_shards_tile_the_dimension(total_size, tp_size):
    # A vocabulary that is not a multiple of tp_grain_size used to lose its tail to the grain
    # quantization, so the shards no longer reconstructed the embedding table.
    meta = AutoTPMeta(tp_grain_size=64)

    shard_sizes = get_shard_size_list(total_size, tp_size, meta, "lm_head")

    assert sum(shard_sizes) == total_size
    # Only the rank that absorbs the sub-grain tail gives up its alignment.
    assert sum(1 for size in shard_sizes if size % 64) <= 1, shard_sizes


def test_uneven_shards_without_grain_quantization():
    assert get_shard_size_list(101, 2, AutoTPMeta(), "lm_head") == [51, 50]


def test_meta_descends_into_multimodal_text_config():
    # Vision-language outer configs keep the head counts only under text_config; reading the
    # outer config directly would lose them and fall back to an even-grain split.
    outer = SimpleNamespace(text_config=SimpleNamespace(num_key_value_heads=4, num_attention_heads=8, hidden_size=64))

    meta = AutoTPMeta.from_model_config(outer)

    assert meta == AutoTPMeta(num_kv_heads=4, num_attention_heads=8, n_embd=64)


def test_meta_descends_only_when_text_config_is_set():
    plain = SimpleNamespace(num_key_value_heads=3, num_attention_heads=6, hidden_size=32)

    meta = AutoTPMeta.from_model_config(plain)

    assert meta == AutoTPMeta(num_kv_heads=3, num_attention_heads=6, n_embd=32)


# 6 kv heads over 4 ranks gives 2/2/1/1 heads, so an attention projection splits 384 as
# 128/128/64/64. Every other layer kind keeps the near-even 96/96/96/96 split, either because
# the dimension has no head structure or because even shards serve it better.
@pytest.mark.parametrize("name,expected", [
    ("layers.0.self_attn.q_proj", [128, 128, 64, 64]),
    ("layers.0.mlp.dense_h_to_4h", [96, 96, 96, 96]),
    ("lm_head", [96, 96, 96, 96]),
    ("embed_out", [96, 96, 96, 96]),
    ("layers.0.experts.w1", [96, 96, 96, 96]),
])
def test_kv_head_split_applies_only_to_attention(name, expected):
    meta = AutoTPMeta(num_kv_heads=6)

    assert get_shard_size_list(384, 4, meta, name) == expected


def test_kv_head_split_needs_a_divisible_dimension():
    # 385 is not a multiple of the 6 kv heads, so there is no head-aligned split to make.
    meta = AutoTPMeta(num_kv_heads=6)

    assert get_shard_size_list(385, 4, meta, "layers.0.self_attn.q_proj") == [97, 96, 96, 96]


def test_process_group_resolves_noncontiguous_group_rank(monkeypatch):
    meta = AutoTPMeta(tp_grain_size=64)
    tp_group = object()
    monkeypatch.setattr(tp_shard.dist, "get_rank", lambda group=None: 1 if group is tp_group else 2)

    shard_sizes = get_shard_size_list(50257, 2, meta, "lm_head")
    assert get_shard_size(50257, 2, meta, "lm_head", mp_group=tp_group) == shard_sizes[1]


def test_shard_size_refuses_to_guess_subgroup_rank(monkeypatch):
    monkeypatch.setattr(tp_shard.dist, "get_world_size", lambda: 4)

    with pytest.raises(ValueError, match="group-local rank or process group"):
        get_shard_size(12, 2, AutoTPMeta())

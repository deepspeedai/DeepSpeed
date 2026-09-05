# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.utils.static_cache import DeepSpeedStaticCache, DeepSpeedStaticLayer


def test_static_layer_supports_per_row_decode_positions():
    layer = DeepSpeedStaticLayer(max_cache_len=4)
    keys = torch.zeros((2, 1, 1, 2))
    values = torch.zeros_like(keys)
    layer.lazy_initialization(keys, values)
    layer.set_write_position(torch.tensor([1, 3], dtype=torch.long))

    keys[:, :, 0, :] = torch.tensor([[[2.0, 2.0]], [[4.0, 4.0]]])
    values.copy_(keys * 10)
    layer.update(keys, values)

    assert layer.keys[0, 0, 1].tolist() == [2.0, 2.0]
    assert layer.keys[1, 0, 3].tolist() == [4.0, 4.0]
    assert torch.equal(layer.get_seq_length(), torch.tensor([2, 4]))


def test_static_cache_compact_preserves_rows_and_positions():
    config = type("Config", (), {"num_hidden_layers": 1, "num_attention_heads": 1, "hidden_size": 2})()
    cache = DeepSpeedStaticCache(config=config,
                                 batch_size=3,
                                 max_cache_len=4,
                                 device=torch.device("cpu"),
                                 dtype=torch.float32)
    cache.set_write_position(torch.tensor([1, 2, 3], dtype=torch.long))
    layer = cache.layers[0]
    layer.keys[:, 0, 0, :] = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    layer.values.copy_(layer.keys)

    cache.compact(torch.tensor([2, 0], dtype=torch.long))

    assert layer.keys[:2, 0, 0, 0].tolist() == [30.0, 10.0]
    assert cache.get_seq_length().item() == 4
    assert layer.keys[2].abs().sum().item() == 0


def test_static_layer_rejects_mismatched_per_row_positions():
    layer = DeepSpeedStaticLayer(max_cache_len=4)
    keys = torch.zeros((2, 1, 1, 2))
    layer.lazy_initialization(keys, keys)
    layer.set_write_position(torch.tensor([1], dtype=torch.long))
    with pytest.raises(ValueError, match="cover the active batch size"):
        layer.update(keys, keys)


def test_static_layer_accepts_active_prefix_of_cache_rows():
    layer = DeepSpeedStaticLayer(max_cache_len=4)
    keys = torch.zeros((3, 1, 1, 2))
    layer.lazy_initialization(keys, keys)
    layer.set_write_position(torch.tensor([1, 2, -1], dtype=torch.long))
    active_keys = torch.ones((2, 1, 1, 2))

    layer.update(active_keys, active_keys)

    assert layer.keys[:2, 0, 1:3].sum().item() == 4

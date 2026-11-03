# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.utils.static_cache import DeepSpeedStaticCache, DeepSpeedStaticLayer


@pytest.mark.parametrize("position, expected", [(None, 0), (0, 0), (3, 3), ([1, 3, -1], 3)])
def test_static_cache_query_offset_precedes_current_write(position, expected):
    config = type("Config", (), {"num_hidden_layers": 2, "num_attention_heads": 1, "hidden_size": 2})()
    cache = DeepSpeedStaticCache(config, batch_size=3, max_cache_len=6, device="cpu", dtype=torch.float32)
    if position is not None:
        cache.set_write_position(torch.tensor(position, dtype=torch.long))

    for layer_idx in range(2):
        assert cache.get_query_offset(layer_idx) == expected
        if position is not None:
            assert cache.get_seq_length(layer_idx) == expected + 1
    assert cache.get_query_offset(2) == 0


@pytest.mark.parametrize("position, allowed", [
    (None, [[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]]),
    (2, [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]]),
    ([1, 3], [[1, 1, 1, 1, 0, 0]]),
])
def test_static_cache_query_offset_builds_native_causal_mask(position, allowed):
    from transformers import LlamaConfig
    from transformers.cache_utils import Cache
    from transformers.masking_utils import create_causal_mask

    if not hasattr(Cache, "get_query_offset"):
        pytest.skip("requires Transformers cache query-offset API")

    config = LlamaConfig(num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1, hidden_size=2)
    config._attn_implementation = "eager"
    cache = DeepSpeedStaticCache(config, batch_size=2, max_cache_len=6, device="cpu", dtype=torch.float32)
    if position is not None:
        cache.set_write_position(torch.tensor(position, dtype=torch.long))

    mask = create_causal_mask(config, torch.zeros((2, len(allowed), 2)), attention_mask=None, past_key_values=cache)

    expected = torch.tensor(allowed, dtype=torch.bool).expand(2, 1, -1, -1)
    assert torch.equal(mask == 0, expected)


def test_static_cache_query_offset_supports_graph_replay():
    accelerator = get_accelerator()
    graph = accelerator.create_graph()
    if graph is None:
        pytest.skip("requires graph capture")

    config = type("Config", (), {"num_hidden_layers": 1})()
    cache = DeepSpeedStaticCache(config, max_cache_len=6)
    positions = torch.tensor([1, 2], device=accelerator.device_name())
    cache.set_write_position(positions)
    with accelerator.capture_to_graph(graph):
        allowed = torch.arange(6, device=positions.device) <= cache.get_query_offset()

    positions.copy_(torch.tensor([3, 4], device=positions.device))
    accelerator.replay_graph(graph)

    assert allowed.tolist() == [True, True, True, True, True, False]


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


def test_static_layer_per_row_positions_support_graph_replay():
    accelerator = get_accelerator()
    graph = accelerator.create_graph()
    if graph is None:
        pytest.skip("requires graph capture")

    device = accelerator.device_name()
    layer = DeepSpeedStaticLayer(max_cache_len=4)
    keys = torch.ones((2, 1, 1, 2), device=device)
    values = keys * 10
    positions = torch.tensor([0, 1], dtype=torch.long, device=device)
    layer.set_write_position(positions)
    stream = accelerator.Stream()
    stream.wait_stream(accelerator.current_stream())
    with accelerator.stream(stream):
        layer.update(keys, values)
    accelerator.current_stream().wait_stream(stream)

    with accelerator.capture_to_graph(graph):
        layer.update(keys, values)

    positions.copy_(torch.tensor([2, 3], dtype=torch.long, device=device))
    keys.fill_(2)
    values.fill_(20)
    accelerator.replay_graph(graph)

    assert layer.keys[0, 0, 2].tolist() == [2.0, 2.0]
    assert layer.keys[1, 0, 3].tolist() == [2.0, 2.0]
    assert layer.values[0, 0, 2].tolist() == [20.0, 20.0]
    assert layer.values[1, 0, 3].tolist() == [20.0, 20.0]


@pytest.mark.parametrize("position", [-1, 4])
def test_static_layer_rejects_out_of_bounds_per_row_positions(position):
    layer = DeepSpeedStaticLayer(max_cache_len=4)
    keys = torch.zeros((1, 1, 1, 2))
    layer.set_write_position(torch.tensor([position], dtype=torch.long))
    with pytest.raises(ValueError, match="within the cache bounds"):
        layer.update(keys, keys)


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


def test_static_cache_compact_identity_keeps_active_rows_and_clears_tail():
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

    cache.compact(torch.tensor([0, 1], dtype=torch.long))

    assert layer.keys[:2, 0, 0, 0].tolist() == [10.0, 20.0]
    assert layer.get_seq_length().tolist() == [2, 3, 0]
    assert layer.keys[2].abs().sum().item() == 0


def test_static_cache_trim_left_shifts_values_and_clears_tail():
    config = type("Config", (), {"num_hidden_layers": 1, "num_attention_heads": 1, "hidden_size": 1})()
    cache = DeepSpeedStaticCache(config=config,
                                 batch_size=1,
                                 max_cache_len=4,
                                 device=torch.device("cpu"),
                                 dtype=torch.float32)
    layer = cache.layers[0]
    layer.keys[0, 0, :, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    layer.values.copy_(layer.keys)

    cache.trim_left(1)

    assert layer.keys[0, 0, :, 0].tolist() == [1.0, 2.0, 3.0, 0.0]
    assert layer.values[0, 0, :, 0].tolist() == [1.0, 2.0, 3.0, 0.0]


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

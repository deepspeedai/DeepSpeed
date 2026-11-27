# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from pydantic import ValidationError

from deepspeed.inference.v2.config_v2 import RaggedInferenceEngineConfig
from deepspeed.inference.v2.ragged import DSStateManagerConfig
from deepspeed.inference.v2.ragged.kv_cache import BlockedKVCache
from deepspeed.inference.v2.ragged.manager_configs import MemoryConfig, AllocationMode, KVCacheConfig
from deepspeed.inference.v2.ragged.ragged_manager import DSStateManager


@pytest.mark.inference_v2
def test_negative_max_tracked_sequences() -> None:
    with pytest.raises(ValidationError):
        DSStateManagerConfig(max_tracked_sequences=-1)


@pytest.mark.inference_v2
def test_zero_max_tracked_sequences() -> None:
    with pytest.raises(ValidationError):
        DSStateManagerConfig(max_tracked_sequences=0)


@pytest.mark.inference_v2
def test_negative_max_ragged_batch_size() -> None:
    with pytest.raises(ValidationError):
        DSStateManagerConfig(max_ragged_batch_size=-1)


@pytest.mark.inference_v2
def test_zero_max_ragged_batch_size() -> None:
    with pytest.raises(ValidationError):
        DSStateManagerConfig(max_ragged_batch_size=0)


@pytest.mark.inference_v2
def test_negative_max_ragged_sequence_count() -> None:
    with pytest.raises(ValidationError):
        DSStateManagerConfig(max_ragged_sequence_count=-1)


@pytest.mark.inference_v2
def test_zero_max_ragged_sequence_count() -> None:
    with pytest.raises(ValidationError):
        DSStateManagerConfig(max_ragged_sequence_count=0)


@pytest.mark.inference_v2
def test_too_small_max_ragged_batch_size() -> None:
    with pytest.raises(ValidationError):
        DSStateManagerConfig(max_ragged_batch_size=512, max_ragged_sequence_count=1024)


@pytest.mark.inference_v2
def test_too_small_max_tracked_sequences() -> None:
    with pytest.raises(ValidationError):
        DSStateManagerConfig(max_tracked_sequences=512, max_ragged_sequence_count=1024)


@pytest.mark.inference_v2
def test_memory_config_defaults() -> None:
    cfg = MemoryConfig()
    assert AllocationMode(cfg.mode) == AllocationMode.RESERVE
    assert cfg.size == 1_000_000_000
    assert cfg.max_blocks is None


@pytest.mark.inference_v2
def test_memory_config_valid_max_blocks() -> None:
    cfg = MemoryConfig(max_blocks=512)
    assert cfg.max_blocks == 512


@pytest.mark.inference_v2
def test_memory_config_negative_max_blocks() -> None:
    with pytest.raises(ValidationError):
        MemoryConfig(max_blocks=-1)


@pytest.mark.inference_v2
def test_memory_config_zero_max_blocks() -> None:
    with pytest.raises(ValidationError):
        MemoryConfig(max_blocks=0)


@pytest.mark.inference_v2
def test_blocked_kv_cache_max_blocks_allocate() -> None:
    cache_config = KVCacheConfig(
        cache_shape=(1, 4, 64),
        block_size=16,
    )
    # When max_blocks is set in ALLOCATE mode
    mem_cfg = MemoryConfig(mode=AllocationMode.ALLOCATE, size=100, max_blocks=32)
    cache = BlockedKVCache((cache_config,), memory_config=mem_cfg)
    assert cache.num_blocks == 32


@pytest.mark.inference_v2
def test_blocked_kv_cache_max_blocks_reserve() -> None:
    cache_config = KVCacheConfig(
        cache_shape=(1, 4, 64),
        block_size=16,
    )
    # When max_blocks is set in RESERVE mode, allocate only up to max_blocks
    mem_cfg = MemoryConfig(mode=AllocationMode.RESERVE, size=1_000_000_000, max_blocks=16)
    cache = BlockedKVCache((cache_config,), memory_config=mem_cfg)
    assert cache.num_blocks == 16


@pytest.mark.inference_v2
def test_engine_config_and_state_manager_integration_max_blocks() -> None:
    """Exercise max_blocks supplied via RaggedInferenceEngineConfig and verify
    that DSStateManager forwards and bounds the allocated KV cache blocks."""
    config_dict = {
        "manager": {
            "max_tracked_sequences": 64,
            "max_ragged_batch_size": 64,
            "max_ragged_sequence_count": 32,
            "memory_config": {
                "mode": "reserve",
                "size": 1_000_000_000,
                "max_blocks": 24,
            },
        }
    }
    engine_cfg = RaggedInferenceEngineConfig(**config_dict)
    assert engine_cfg.state_manager.memory_config.max_blocks == 24

    kv_config = KVCacheConfig(cache_shape=(1, 4, 64), block_size=16)
    state_mgr = DSStateManager(engine_cfg.state_manager, (kv_config,))
    assert state_mgr._kv_cache.num_blocks == 24

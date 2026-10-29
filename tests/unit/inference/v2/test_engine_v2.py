# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

import pytest

from deepspeed.inference.v2.engine_v2 import InferenceEngineV2
from deepspeed.inference.v2.scheduling_utils import SchedulingResult


def _engine(free_blocks, get_kv_requirements):
    engine = InferenceEngineV2.__new__(InferenceEngineV2)
    engine._config = SimpleNamespace(state_manager=SimpleNamespace(
        max_tracked_sequences=4,
        max_ragged_sequence_count=4,
        max_ragged_batch_size=16,
    ))
    engine._state_manager = SimpleNamespace(
        free_blocks=free_blocks,
        n_tracked_sequences=0,
        get_sequence=lambda uid: None,
    )
    engine._model = SimpleNamespace(get_kv_requirements=get_kv_requirements)
    return engine


@pytest.mark.inference_v2
def test_can_schedule_tracks_scalar_free_block_count():
    observed_free_blocks = []

    def get_kv_requirements(sequence, max_new_tokens, max_new_blocks):
        assert isinstance(max_new_blocks, int)
        observed_free_blocks.append(max_new_blocks)
        return max_new_tokens, 1

    engine = _engine([2], get_kv_requirements)

    result = engine.can_schedule([1, 2], [1, 1])

    assert result == SchedulingResult.Success
    assert observed_free_blocks == [2, 1]


@pytest.mark.inference_v2
def test_can_schedule_reports_insufficient_kv_cache():

    def get_kv_requirements(sequence, max_new_tokens, max_new_blocks):
        assert isinstance(max_new_blocks, int)
        return 0, 0

    engine = _engine([0], get_kv_requirements)

    result = engine.can_schedule([1], [1])

    assert result == SchedulingResult.KVCacheLimitExceeded

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""CPU-only unit tests for HybridEngineRollout (no GPU needed).

Tests cover configuration, profiling, generation dispatch, and the pure-tensor sampling helper.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import (
    HybridEngineRollout,
    HybridEngineRolloutConfig,
)


def _make_engine():
    engine = MagicMock()
    engine.module = MagicMock()
    engine.module.parameters.return_value = iter([])
    return engine


def _make_tokenizer():
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 2
    return tok


def _make_request():
    return RolloutRequest(
        prompt_ids=torch.tensor([[0, 1, 2], [0, 3, 4]]),
        prompt_attention_mask=torch.tensor([[0, 1, 1], [0, 1, 1]]),
    )


def _make_sampling(n_samples_per_prompt=1):
    return SamplingConfig(max_new_tokens=2, temperature=0, n_samples_per_prompt=n_samples_per_prompt)


# -- config defaults ----------------------------------------------------


def test_config_defaults():
    cfg = HybridEngineRolloutConfig()
    assert cfg.use_graph_capture is False
    assert cfg.enable_profiling is False


# -- constructor --------------------------------------------------------


def test_constructor_stores_config():
    engine = _make_engine()
    tok = _make_tokenizer()
    cfg = HybridEngineRolloutConfig(use_graph_capture=True, enable_profiling=True)
    rollout = HybridEngineRollout(engine, tok, cfg=cfg)
    assert rollout.use_graph_capture is True
    assert rollout.enable_profiling is True
    assert rollout.engine is engine
    assert rollout.tokenizer is tok


def test_constructor_defaults_without_cfg():
    rollout = HybridEngineRollout(_make_engine(), _make_tokenizer())
    assert rollout.use_graph_capture is False
    assert rollout.enable_profiling is False


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.time.perf_counter")
@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_generate_records_profile_when_enabled(mock_get_accelerator, mock_perf_counter):
    engine = _make_engine()
    tok = _make_tokenizer()
    cfg = HybridEngineRolloutConfig(enable_profiling=True)
    rollout = HybridEngineRollout(engine, tok, cfg=cfg)
    rollout.engine.module.generate.return_value = torch.tensor([
        [0, 1, 2, 5, 6],
        [0, 1, 2, 7, 8],
        [0, 3, 4, 9, 10],
        [0, 3, 4, 11, 12],
    ])
    mock_perf_counter.side_effect = [1.0, 1.001, 1.011, 1.013]

    output = rollout.generate(_make_request(), _make_sampling(n_samples_per_prompt=2))

    profile = rollout.get_last_profile()
    assert profile["prompt_expansion_ms"] == pytest.approx(1.0)
    assert profile["generation_ms"] == pytest.approx(10.0)
    assert profile["post_processing_ms"] == pytest.approx(2.0)
    assert profile["total_ms"] == pytest.approx(13.0)
    assert profile["num_generated_tokens"] == 8
    assert profile["tokens_per_second"] == pytest.approx(8 / 0.013)
    assert profile["batch_size"] == 2
    assert profile["num_samples_per_prompt"] == 2
    assert profile["prompt_length"] == 3
    assert profile["response_length"] == 2
    expected_prompt_masks = [[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]]
    assert output.attention_mask[:, :3].tolist() == expected_prompt_masks
    assert mock_get_accelerator.return_value.synchronize.call_count == 4


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_generate_does_not_profile_when_disabled(mock_get_accelerator):
    engine = _make_engine()
    rollout = HybridEngineRollout(engine, _make_tokenizer())
    engine.module.generate.return_value = torch.tensor([[0, 1, 2, 5, 6], [0, 3, 4, 7, 8]])

    rollout.generate(_make_request(), _make_sampling())

    assert rollout.get_last_profile() is None
    mock_get_accelerator.assert_not_called()


def test_profiling_does_not_change_rollout_output():
    generated = torch.tensor([[0, 1, 2, 5, 6], [0, 3, 4, 7, 8]])
    engine_without_profiling = _make_engine()
    engine_without_profiling.module.generate.return_value = generated
    rollout_without_profiling = HybridEngineRollout(engine_without_profiling, _make_tokenizer())
    engine_with_profiling = _make_engine()
    engine_with_profiling.module.generate.return_value = generated
    rollout_with_profiling = HybridEngineRollout(
        engine_with_profiling,
        _make_tokenizer(),
        cfg=HybridEngineRolloutConfig(enable_profiling=True),
    )

    output_without_profiling = rollout_without_profiling.generate(_make_request(), _make_sampling())
    with patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator"):
        output_with_profiling = rollout_with_profiling.generate(_make_request(), _make_sampling())

    assert torch.equal(output_with_profiling.input_ids, output_without_profiling.input_ids)
    assert torch.equal(output_with_profiling.attention_mask, output_without_profiling.attention_mask)
    assert torch.equal(output_with_profiling.response_start_idx, output_without_profiling.response_start_idx)


def test_generate_preserves_zero_pad_token_id():
    engine = _make_engine()
    engine.module.generate.return_value = torch.tensor([[0, 1, 2, 0], [0, 3, 4, 0]])
    rollout = HybridEngineRollout(engine, _make_tokenizer())

    output = rollout.generate(_make_request(), _make_sampling())

    assert output.attention_mask[:, -1].tolist() == [0, 0]


# -- _sample_top_p ------------------------------------------------------


def test_sample_top_p_returns_correct_shape():
    logits = torch.randn(4, 100)
    tokens = HybridEngineRollout._sample_top_p(logits, temperature=1.0, top_p=1.0)
    assert tokens.shape == (4, 1)


def test_sample_top_p_deterministic_with_low_temp():
    logits = torch.tensor([[1.0, 10.0, 2.0]])
    tok = HybridEngineRollout._sample_top_p(logits, temperature=1e-10, top_p=1.0)
    assert tok.item() == 1


def test_sample_top_p_top_p_filters():
    logits = torch.tensor([[0.0, 0.0, 100.0]])
    tok = HybridEngineRollout._sample_top_p(logits, temperature=1.0, top_p=0.5)
    assert tok.item() == 2


def test_sample_top_p_batch():
    logits = torch.randn(8, 50)
    tokens = HybridEngineRollout._sample_top_p(logits, temperature=0.8, top_p=0.9)
    assert tokens.shape == (8, 1)
    assert (tokens >= 0).all() and (tokens < 50).all()


# -- sync_weights is no-op ---------------------------------------------


def test_sync_weights_is_noop():
    rollout = HybridEngineRollout(_make_engine(), _make_tokenizer())
    assert rollout.sync_weights(step=0) is None


# -- generate dispatches correctly -------------------------------------


def test_generate_calls_graph_capture_when_enabled():
    engine = _make_engine()
    tok = _make_tokenizer()
    cfg = HybridEngineRolloutConfig(use_graph_capture=True)
    rollout = HybridEngineRollout(engine, tok, cfg=cfg)
    rollout._generate_graph = MagicMock(return_value=torch.zeros(1, 5, dtype=torch.long))

    req = MagicMock()
    req.prompt_ids = torch.tensor([[1, 2]])
    req.prompt_attention_mask = torch.ones(1, 2, dtype=torch.long)
    sampling = MagicMock()
    sampling.temperature = 0
    sampling.n_samples_per_prompt = 1
    sampling.max_new_tokens = 3

    rollout.generate(req, sampling)
    rollout._generate_graph.assert_called_once()

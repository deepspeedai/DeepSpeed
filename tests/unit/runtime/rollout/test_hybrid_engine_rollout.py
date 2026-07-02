# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""CPU-only unit tests for HybridEngineRollout (no GPU needed).

Tests cover configuration defaults and the pure-tensor sampling helper.
"""

from unittest.mock import MagicMock

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


# -- config defaults ----------------------------------------------------


def test_config_defaults():
    cfg = HybridEngineRolloutConfig()
    assert cfg.use_graph_capture is False


# -- constructor --------------------------------------------------------


def test_constructor_stores_config():
    engine = _make_engine()
    tok = _make_tokenizer()
    cfg = HybridEngineRolloutConfig(use_graph_capture=True)
    rollout = HybridEngineRollout(engine, tok, cfg=cfg)
    assert rollout.use_graph_capture is True
    assert rollout.engine is engine
    assert rollout.tokenizer is tok


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


def _make_request():
    return RolloutRequest(prompt_ids=torch.tensor([[1, 2]]), prompt_attention_mask=torch.ones(1, 2, dtype=torch.long))


def test_generate_uses_module_generate_by_default():
    engine = _make_engine()
    tok = _make_tokenizer()
    rollout = HybridEngineRollout(engine, tok)
    engine.module.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 2]]))

    # Sampling (temperature > 0) routes through the engine's generate path.
    rollout.generate(_make_request(), SamplingConfig(max_new_tokens=2, temperature=1.0))
    engine.module.generate.assert_called_once()


def test_generate_calls_graph_capture_when_enabled():
    engine = _make_engine()
    tok = _make_tokenizer()
    cfg = HybridEngineRolloutConfig(use_graph_capture=True)
    rollout = HybridEngineRollout(engine, tok, cfg=cfg)
    rollout._generate_graph = MagicMock(return_value=torch.tensor([[1, 2, 3, 2]]))

    # Graph capture is used for greedy decoding (temperature <= 0).
    rollout.generate(_make_request(), SamplingConfig(max_new_tokens=2, temperature=0.0))
    rollout._generate_graph.assert_called_once()

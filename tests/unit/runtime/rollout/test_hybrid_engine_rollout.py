# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for HybridEngineRollout.

Most tests are CPU-only; the native shared-prefill cache test runs only when CUDA and
the transformer inference extension are available.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from deepspeed.ops.transformer.inference.op_binding.workspace import WorkspaceOp
from deepspeed.runtime.hybrid_engine import DeepSpeedHybridEngine
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import (
    HybridEngineRollout,
    HybridEngineRolloutConfig,
)
from deepspeed.utils.static_cache import DeepSpeedStaticCache


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
    assert cfg.use_shared_prefill is False


# -- constructor --------------------------------------------------------


def test_constructor_stores_config():
    engine = _make_engine()
    tok = _make_tokenizer()
    cfg = HybridEngineRolloutConfig(use_graph_capture=True, enable_profiling=True)
    rollout = HybridEngineRollout(engine, tok, cfg=cfg)
    assert rollout.use_graph_capture is True
    assert rollout.enable_profiling is True
    assert rollout.use_shared_prefill is False
    assert rollout.engine is engine
    assert rollout.tokenizer is tok


def test_constructor_defaults_without_cfg():
    rollout = HybridEngineRollout(_make_engine(), _make_tokenizer())
    assert rollout.use_graph_capture is False
    assert rollout.enable_profiling is False
    assert rollout.use_shared_prefill is False


def test_continuous_generation_rejects_unsupported_inputs():
    rollout = HybridEngineRollout(_make_engine(), _make_tokenizer())
    request = RolloutRequest(
        prompt_ids=torch.tensor([[0, 1, 2]]),
        prompt_attention_mask=torch.tensor([[0, 1, 1]]),
    )

    with pytest.raises(ValueError, match="at least one request"):
        rollout.generate_continuous([], [], max_batch_size=1)
    with pytest.raises(ValueError, match="same length"):
        rollout.generate_continuous([request], [], max_batch_size=1)
    with pytest.raises(ValueError, match="greedy"):
        rollout.generate_continuous(
            [request],
            [SamplingConfig(max_new_tokens=2, temperature=0.5)],
            max_batch_size=1,
        )


def test_static_cache_constructor_supports_legacy_batch_keyword():

    class LegacyStaticCache:

        def __init__(self, config, max_batch_size, max_cache_len, device, dtype):
            self.config = config
            self.max_batch_size = max_batch_size

    config = SimpleNamespace(num_attention_heads=4)
    cache = HybridEngineRollout._create_static_cache(LegacyStaticCache, config, 2, 8, "cpu", torch.float32)

    assert cache.max_batch_size == 2
    assert cache.config.num_key_value_heads == 4


def test_continuous_cache_span_does_not_sum_independent_requests():
    cache_len = HybridEngineRollout._estimate_continuous_cache_len(64, [64] * 100, 100)

    assert cache_len == 128
    assert cache_len < 64 + 64 * 100


def test_select_legacy_cache_rows_expands_flattened_heads():
    keys = torch.arange(2 * 4 * 3).reshape(2 * 4, 1, 3)
    values = keys + 100
    rows = torch.tensor([1])

    selected = HybridEngineRollout._select_legacy_cache_rows(((keys, values), ), rows, logical_batch_size=2)

    assert torch.equal(selected[0][0], keys[4:8])
    assert torch.equal(selected[0][1], values[4:8])


def test_continuous_generation_validates_each_request_length():

    class LimitedModel(torch.nn.Module):

        _supports_cache_class = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=4)

    model = LimitedModel()
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(eos_token_id=None))
    request = RolloutRequest(torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long))

    with pytest.raises(ValueError, match="request exceeds"):
        rollout.generate_continuous([request], [SamplingConfig(max_new_tokens=2, temperature=0)], 1)


def test_continuous_generation_refills_legacy_cache_batch():

    class LegacyCacheModel(torch.nn.Module):
        _supports_cache_class = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.calls = []
            self.config = SimpleNamespace(max_position_embeddings=32)

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True):
            past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
            total_length = past_length + input_ids.shape[1]
            assert attention_mask.shape == (input_ids.shape[0], total_length)
            self.calls.append((input_ids.shape[0], input_ids.shape[1], past_length))
            keys = torch.zeros((input_ids.shape[0], 1, total_length, 1))
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16))
            logits[..., 7] = 1
            return SimpleNamespace(logits=logits, past_key_values=((keys, keys.clone()), ))

    model = LegacyCacheModel()
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=None))
    requests = [
        RolloutRequest(torch.tensor([[1, 2, token]]), torch.ones((1, 3), dtype=torch.long)) for token in (3, 4, 5)
    ]
    configs = [
        SamplingConfig(max_new_tokens=1, temperature=0),
        SamplingConfig(max_new_tokens=3, temperature=0),
        SamplingConfig(max_new_tokens=2, temperature=0),
    ]

    outputs = rollout.generate_continuous(requests, configs, max_batch_size=2)

    assert [output.input_ids.shape[1] - 3 for output in outputs] == [1, 3, 2]
    assert model.calls[:3] == [(2, 3, 0), (1, 1, 3), (1, 3, 0)]


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
    assert profile["prefill_forward_ms"] is None
    assert profile["decode_forward_ms"] is None
    assert profile["generation_overhead_ms"] == pytest.approx(10.0)
    assert profile["num_decode_forwards"] == 0
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


def test_native_repeat_kv_cache_fp16_reverse_copy():
    """Exercise the native reverse copy with multiple source cache rows."""
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("CUDA is required for the native inference kernel")

    from deepspeed.ops.op_builder import InferenceBuilder

    builder = InferenceBuilder()
    try:
        is_compatible = builder.is_compatible()
    except Exception as exc:
        pytest.skip(f"Unable to inspect native transformer inference compatibility: {exc}")
    if not is_compatible:
        pytest.skip("The native transformer inference extension is not compatible")
    try:
        inference_op = builder.load()
    except Exception as exc:
        pytest.skip(f"Unable to load the native transformer inference extension: {exc}")

    repeat_kv_cache = getattr(inference_op, "repeat_kv_cache_fp16", None)
    if repeat_kv_cache is None:
        pytest.skip("The native transformer inference extension lacks repeat_kv_cache_fp16")

    device = torch.device("cuda")
    source_batch_size = 2
    repeats = 2
    target_batch_size = source_batch_size * repeats
    prompt_length = 2
    hidden_dim = 8  # The FP16 transform kernel processes eight values per thread.
    num_heads = 1

    inference_op.allocate_workspace_fp16(
        hidden_dim,
        num_heads,
        prompt_length,
        target_batch_size,
        1,
        1,
        False,
        0,
        4,
        1,
    )
    try:
        query_key_value = torch.zeros((source_batch_size, prompt_length, hidden_dim * 3),
                                      dtype=torch.float16,
                                      device=device)
        query_key_value = query_key_value.view(source_batch_size, prompt_length, 3, num_heads, hidden_dim)
        query_key_value[0, :, 1, :, :] = 1
        query_key_value[0, :, 2, :, :] = 10
        query_key_value[1, :, 1, :, :] = 3
        query_key_value[1, :, 2, :, :] = 30
        query_key_value = query_key_value.reshape(source_batch_size, prompt_length, hidden_dim * 3)

        empty_mask = torch.empty(1, dtype=torch.float16, device=device)
        inference_op.softmax_context_fp16(
            query_key_value,
            empty_mask,
            0,
            False,
            False,
            num_heads,
            0,
            1.0,
            False,
            False,
            1,
            True,
            0,
            1,
            empty_mask,
            1.0,
            True,
            None,
            None,
        )
        torch.cuda.synchronize()  #ignore-cuda

        repeated_cache = repeat_kv_cache(source_batch_size, repeats)
        torch.cuda.synchronize()  #ignore-cuda

        assert len(repeated_cache) == 2
        expected_key = torch.empty((target_batch_size, 1, prompt_length, hidden_dim),
                                   dtype=torch.float16,
                                   device=device)
        expected_key[:source_batch_size] = 1
        expected_key[source_batch_size:] = 3
        expected_value = expected_key * 10
        assert torch.equal(repeated_cache[0], expected_key)
        assert torch.equal(repeated_cache[1], expected_value)
    finally:
        inference_op.release_workspace()


def test_shared_prefill_hooks_reduce_prompt_and_expand_output():

    class PromptModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.forward_batch_sizes = []

        def forward(self, input_ids, attention_mask=None):
            self.forward_batch_sizes.append(input_ids.shape[0])
            values = input_ids[:, None, :, None].float()
            return SimpleNamespace(logits=input_ids[:, :, None].float(), past_key_values=((values, values), ))

    engine = _make_engine()
    engine.repeat_shared_prefill_cache.return_value = ((torch.zeros(4, 1, 2, 1), torch.zeros(4, 1, 2, 1)), )
    module = PromptModule()
    rollout = HybridEngineRollout(engine, _make_tokenizer())
    handles = rollout._register_shared_prefill_hooks(module, batch_size=2, repeats=2)
    prompt_ids = torch.tensor([[1, 2], [1, 2], [3, 4], [3, 4]])

    output = module(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids))
    decode_output = module(input_ids=torch.ones(4, 1, dtype=torch.long))
    for handle in handles:
        handle.remove()

    assert module.forward_batch_sizes == [2, 4]
    assert output.logits.shape[0] == 4
    assert output.past_key_values[0][0].shape[0] == 4
    assert decode_output.logits.shape[0] == 4
    engine.repeat_shared_prefill_cache.assert_called_once_with(2, 2)


def test_generate_uses_shared_prefill_for_multiple_samples():

    class GenerateModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.forward_batch_sizes = []

        def forward(self, input_ids, attention_mask=None):
            self.forward_batch_sizes.append(input_ids.shape[0])
            values = input_ids[:, None, :, None].float()
            return SimpleNamespace(logits=input_ids[:, :, None].float(), past_key_values=((values, values), ))

        def generate(self, input_ids, attention_mask=None, max_new_tokens=1, **_kwargs):
            self(input_ids=input_ids, attention_mask=attention_mask)
            response = torch.ones(input_ids.shape[0], max_new_tokens, dtype=input_ids.dtype)
            return torch.cat((input_ids, response), dim=1)

    engine = _make_engine()
    engine.module = GenerateModule()
    config = HybridEngineRolloutConfig(use_shared_prefill=True)
    rollout = HybridEngineRollout(engine, _make_tokenizer(), config)

    output = rollout.generate(_make_request(), _make_sampling(n_samples_per_prompt=2))

    assert engine.module.forward_batch_sizes == [2]
    assert output.input_ids[:, 3:].shape == (4, 2)
    engine.prepare_shared_prefill.assert_called_once_with(2, 2, 3)
    engine.repeat_shared_prefill_cache.assert_called_once_with(2, 2)


def test_shared_prefill_rejects_graph_capture():
    engine = _make_engine()
    config = HybridEngineRolloutConfig(use_graph_capture=True, use_shared_prefill=True)
    rollout = HybridEngineRollout(engine, _make_tokenizer(), config)

    with pytest.raises(RuntimeError, match="does not support CUDA graph capture"):
        rollout.generate(_make_request(), _make_sampling(n_samples_per_prompt=2))


def test_shared_prefill_fallback_repeats_prompt_cache():
    key_cache = torch.zeros(4, 1, 3, 1)
    value_cache = torch.zeros_like(key_cache)
    key_cache[:2, :, :2, :] = torch.tensor([[[[1.0], [2.0]]], [[[3.0], [4.0]]]])
    value_cache[:2, :, :2, :] = key_cache[:2, :, :2, :] + 10
    workspace = WorkspaceOp.__new__(WorkspaceOp)
    workspace.inference_context = SimpleNamespace(
        kv_cache_size=key_cache.shape,
        kv_cache=[(key_cache, value_cache)],
        current_tokens=lambda: 3,
    )

    repeated_cache = workspace.repeat_kv_cache_fallback(source_batch_size=2, repeats=2)

    expected_key = torch.tensor([1.0, 1.0, 3.0, 3.0])
    assert torch.equal(key_cache[:, 0, 0, 0], expected_key)
    assert torch.equal(value_cache[:, 0, 0, 0], expected_key + 10)
    assert repeated_cache[0].shape == (4, 1, 2, 1)
    assert repeated_cache[1].shape == (4, 1, 2, 1)


def test_engine_pairs_shared_prefill_cache_tensors():
    key_cache = torch.zeros(4, 1, 2, 1)
    value_cache = torch.ones_like(key_cache)
    workspace = MagicMock()
    workspace.repeat_kv_cache.return_value = [key_cache, value_cache]
    engine = SimpleNamespace(_shared_prefill_workspace=workspace)

    repeated_cache = DeepSpeedHybridEngine.repeat_shared_prefill_cache(engine, 2, 2)

    assert repeated_cache[0][0] is key_cache
    assert repeated_cache[0][1] is value_cache
    workspace.repeat_kv_cache.assert_called_once_with(2, 2)


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


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.time.perf_counter")
@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_generate_calls_graph_capture_when_enabled(mock_get_accelerator, mock_perf_counter):
    engine = _make_engine()
    tok = _make_tokenizer()
    cfg = HybridEngineRolloutConfig(use_graph_capture=True, enable_profiling=True)
    rollout = HybridEngineRollout(engine, tok, cfg=cfg)
    rollout._generate_graph = MagicMock(return_value=torch.zeros(1, 5, dtype=torch.long))
    mock_perf_counter.side_effect = [1.0, 1.001, 1.011, 1.013]

    req = MagicMock()
    req.prompt_ids = torch.tensor([[1, 2]])
    req.prompt_attention_mask = torch.ones(1, 2, dtype=torch.long)
    sampling = MagicMock()
    sampling.temperature = 0
    sampling.n_samples_per_prompt = 1
    sampling.max_new_tokens = 3

    rollout.generate(req, sampling)
    rollout._generate_graph.assert_called_once()
    profile = rollout.get_last_profile()
    assert profile["prefill_forward_ms"] is None
    assert profile["decode_forward_ms"] is None
    assert profile["generation_overhead_ms"] == pytest.approx(10.0)
    assert profile["num_decode_forwards"] == 0


def test_generate_keeps_ranks_in_lockstep_and_pads_after_eos():
    engine = _make_engine()
    tok = _make_tokenizer()
    rollout = HybridEngineRollout(engine, tok)
    engine.module.generate.return_value = torch.tensor([[10, 11, 5, 2, 7, 8]])

    req = MagicMock()
    req.prompt_ids = torch.tensor([[10, 11]])
    req.prompt_attention_mask = torch.ones(1, 2, dtype=torch.long)
    sampling = MagicMock()
    sampling.temperature = 0
    sampling.n_samples_per_prompt = 1
    sampling.max_new_tokens = 4
    sampling.top_p = 1.0

    result = rollout.generate(req, sampling)

    assert engine.module.generate.call_args.kwargs['eos_token_id'] is None
    assert result.input_ids.tolist() == [[10, 11, 5, 2, 0, 0]]
    assert result.attention_mask.tolist() == [[1, 1, 1, 1, 0, 0]]


def test_pad_after_eos_handles_different_lengths_and_missing_eos():
    output_ids = torch.tensor([
        [10, 11, 2, 7, 8, 9],
        [10, 11, 5, 6, 2, 9],
        [10, 11, 5, 6, 7, 8],
    ])

    padded, response_attn = HybridEngineRollout._pad_after_eos(output_ids, 2, eos_token_id=2, pad_token_id=0)

    assert padded.tolist() == [
        [10, 11, 2, 0, 0, 0],
        [10, 11, 5, 6, 2, 0],
        [10, 11, 5, 6, 7, 8],
    ]
    assert response_attn.tolist() == [
        [1, 0, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]


def test_pad_after_eos_keeps_eos_attended_when_eos_is_pad():
    output_ids = torch.tensor([[10, 11, 5, 2, 7, 8]])

    padded, response_attn = HybridEngineRollout._pad_after_eos(output_ids, 2, eos_token_id=2, pad_token_id=2)

    assert padded.tolist() == [[10, 11, 5, 2, 2, 2]]
    assert response_attn.tolist() == [[1, 1, 0, 0]]


def test_pad_after_eos_supports_multiple_eos_ids():
    output_ids = torch.tensor([[10, 11, 5, 3, 7, 2]])

    padded, response_attn = HybridEngineRollout._pad_after_eos(output_ids, 2, eos_token_id=[2, 3], pad_token_id=0)

    assert padded.tolist() == [[10, 11, 5, 3, 0, 0]]
    assert response_attn.tolist() == [[1, 1, 0, 0]]


def test_generate_accepts_zero_pad_token_id():
    engine = _make_engine()
    tok = _make_tokenizer()
    rollout = HybridEngineRollout(engine, tok)
    engine.module.generate.return_value = torch.tensor([[10, 11, 5, 6]])

    req = MagicMock()
    req.prompt_ids = torch.tensor([[10, 11]])
    req.prompt_attention_mask = torch.ones(1, 2, dtype=torch.long)
    sampling = MagicMock(temperature=0, n_samples_per_prompt=1, max_new_tokens=2, top_p=1.0)

    rollout.generate(req, sampling)

    assert engine.module.generate.call_args.kwargs['pad_token_id'] == 0


def _cache_config(head_dim=None):
    config = SimpleNamespace(num_hidden_layers=2, hidden_size=64, num_attention_heads=4, num_key_value_heads=2)
    if head_dim is not None:
        config.head_dim = head_dim
    return config


def test_static_cache_preallocates_with_config_head_dim():
    cache = DeepSpeedStaticCache(_cache_config(head_dim=32),
                                 batch_size=1,
                                 max_cache_len=8,
                                 device="cpu",
                                 dtype=torch.float32)

    assert tuple(cache.layers[0].keys.shape) == (1, 2, 8, 32)
    assert tuple(cache.layers[0].values.shape) == (1, 2, 8, 32)


def test_static_cache_falls_back_to_derived_head_dim():
    cache = DeepSpeedStaticCache(_cache_config(), batch_size=1, max_cache_len=8, device="cpu", dtype=torch.float32)

    assert tuple(cache.layers[0].keys.shape) == (1, 2, 8, 16)

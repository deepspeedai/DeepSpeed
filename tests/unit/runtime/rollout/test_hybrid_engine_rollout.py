# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for HybridEngineRollout.

Most tests are CPU-only; the native shared-prefill cache test runs only when CUDA and
the transformer inference extension are available.
"""

import gc
import weakref
from types import MethodType, SimpleNamespace
from contextlib import contextmanager, nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import PretrainedConfig

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.transformer.inference.op_binding.workspace import WorkspaceOp
from deepspeed.runtime.hybrid_engine import DeepSpeedHybridEngine
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import (
    GlobalWorkRemaining,
    HybridEngineRollout,
    HybridEngineRolloutConfig,
    _GlobalEosStoppingCriteria,
)
from deepspeed.utils.static_cache import DeepSpeedStaticCache
from unit.common import DistributedTest


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


@pytest.fixture
def graph_capture_accelerator():
    # These tests exercise graph-only validation and dispatch. On a CPU
    # accelerator the opt-in is False, so generate() would take the eager
    # fallback before reaching the code under test.
    with patch.object(type(get_accelerator()), "is_graph_capture_supported", return_value=True):
        yield


# -- config defaults ----------------------------------------------------


def test_config_defaults():
    cfg = HybridEngineRolloutConfig()
    assert cfg.use_graph_capture is False
    assert cfg.enable_profiling is False
    assert cfg.use_shared_prefill is False
    assert cfg.align_decode_fronts is False
    assert cfg.adaptive_prefill is False
    assert cfg.prefill_max_tokens == 65536
    assert cfg.enable_cache_trimming is False
    assert cfg.continuous_cache_capacity is None


# -- constructor --------------------------------------------------------


def test_constructor_stores_config():
    engine = _make_engine()
    tok = _make_tokenizer()
    cfg = HybridEngineRolloutConfig(use_graph_capture=True,
                                    enable_profiling=True,
                                    align_decode_fronts=True,
                                    enable_cache_trimming=True,
                                    continuous_cache_capacity=16)
    rollout = HybridEngineRollout(engine, tok, cfg=cfg)
    assert rollout.use_graph_capture is True
    assert rollout.enable_profiling is True
    assert rollout.use_shared_prefill is False
    assert rollout.align_decode_fronts is True
    assert rollout.enable_cache_trimming is True
    assert rollout.continuous_cache_capacity == 16
    assert rollout.engine is engine
    assert rollout.tokenizer is tok


def test_constructor_defaults_without_cfg():
    rollout = HybridEngineRollout(_make_engine(), _make_tokenizer())
    assert rollout.use_graph_capture is False
    assert rollout.enable_profiling is False
    assert rollout.use_shared_prefill is False
    assert rollout.align_decode_fronts is False
    assert rollout.enable_cache_trimming is False
    assert rollout.continuous_cache_capacity is None


@pytest.mark.parametrize("adaptive_prefill", [False, True])
def test_continuous_generation_rejects_unsupported_inputs(adaptive_prefill):
    rollout = HybridEngineRollout(_make_engine(), _make_tokenizer(),
                                  HybridEngineRolloutConfig(adaptive_prefill=adaptive_prefill))
    request = RolloutRequest(
        prompt_ids=torch.tensor([[0, 1, 2]]),
        prompt_attention_mask=torch.tensor([[0, 1, 1]]),
    )

    empty_request = RolloutRequest(torch.empty((0, 3), dtype=torch.long), torch.empty((0, 3), dtype=torch.long))
    with pytest.raises(ValueError, match="at least one request"):
        rollout.generate(empty_request, SamplingConfig(max_new_tokens=2, continuous_batch_size=1))
    with pytest.raises(ValueError, match="positive"):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, continuous_batch_size=0))
    if adaptive_prefill:
        with pytest.raises(ValueError, match="adaptive prefill.*greedy"):
            rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0.7, continuous_batch_size=1))


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (HybridEngineRolloutConfig(use_graph_capture=True, align_decode_fronts=True), "aligned decode fronts"),
        (HybridEngineRolloutConfig(use_graph_capture=True, enable_cache_trimming=True), "cache trimming"),
        (HybridEngineRolloutConfig(use_graph_capture=True, continuous_cache_capacity=4), "must fit"),
    ],
)
def test_continuous_graph_rejects_incompatible_fixed_layout_options(cfg, message, graph_capture_accelerator):
    rollout = HybridEngineRollout(_make_engine(), _make_tokenizer(), cfg=cfg)
    request = RolloutRequest(
        prompt_ids=torch.tensor([[0, 1, 2]]),
        prompt_attention_mask=torch.tensor([[0, 1, 1]]),
    )

    with pytest.raises(ValueError, match=message):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=1))


def test_static_graph_rejects_flash_attention(graph_capture_accelerator):
    module = torch.nn.Linear(1, 1)
    module.config = SimpleNamespace(_attn_implementation="flash_attention_2")
    rollout = HybridEngineRollout(SimpleNamespace(module=module),
                                  _make_tokenizer(),
                                  cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    request = RolloutRequest(torch.tensor([[1, 2]]), torch.ones((1, 2), dtype=torch.long))

    with pytest.raises(ValueError, match="does not support flash_attention_2"):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0))


def test_continuous_graph_rejects_flash_attention(graph_capture_accelerator):
    engine = _make_engine()
    engine.module.config = SimpleNamespace(_attn_implementation="flash_attention_2")
    rollout = HybridEngineRollout(engine, _make_tokenizer(), cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    request = RolloutRequest(
        prompt_ids=torch.tensor([[0, 1, 2]]),
        prompt_attention_mask=torch.tensor([[0, 1, 1]]),
    )

    with pytest.raises(ValueError, match="does not support flash_attention_2"):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=1))


def test_continuous_generation_falls_back_to_eager_without_graph_capture_opt_in():
    # The eager continuous path is a full fallback, so a backend without the
    # graph capture opt-in must use it instead of raising.

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = PretrainedConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            key_states = input_ids[:, None, :, None].to(dtype=torch.float32)
            _, cache_values = past_key_values.update(key_states, key_states, layer_idx=0, cache_kwargs=kwargs)
            next_tokens = torch.where(cache_values[:, 0].sum(dim=(1, 2)) == 6, 2, 7).long()
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16))
            logits.scatter_(2, next_tokens[:, None, None].expand(-1, input_ids.shape[1], 1), 1)
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    rollout = HybridEngineRollout(
        SimpleNamespace(module=CacheClassModel()),
        SimpleNamespace(pad_token_id=0, eos_token_id=2),
        cfg=HybridEngineRolloutConfig(use_graph_capture=True),
    )
    request = RolloutRequest(
        torch.tensor([[1, 2, 3], [1, 2, 4], [1, 2, 5]]),
        torch.ones((3, 3), dtype=torch.long),
    )

    with patch.object(get_accelerator(), "is_graph_capture_supported", return_value=False), \
            patch.object(rollout, "_generate_continuous_graph") as generate_graph, \
            patch("deepspeed.runtime.rollout.hybrid_engine_rollout.logger.warning_once") as warning_once:
        output = rollout.generate(request, SamplingConfig(max_new_tokens=3, temperature=0, continuous_batch_size=2))

    generate_graph.assert_not_called()
    warning_once.assert_called_once()
    assert output.input_ids.tolist() == [[1, 2, 3, 2, 0, 0], [1, 2, 4, 7, 7, 7], [1, 2, 5, 7, 7, 7]]


def test_continuous_generation_routes_graph_capture_to_fixed_capacity_path(graph_capture_accelerator):
    rollout = HybridEngineRollout(
        _make_engine(),
        _make_tokenizer(),
        cfg=HybridEngineRolloutConfig(use_graph_capture=True),
    )
    request = _make_request()
    sampling = SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=4)
    expected = MagicMock()

    with patch.object(rollout, "_generate_continuous_graph", return_value=expected) as generate_graph:
        assert rollout.generate(request, sampling) is expected

    generate_graph.assert_called_once()
    original_request, requests, captured_sampling, max_batch_size = generate_graph.call_args.args
    assert original_request is request
    assert len(requests) == request.prompt_ids.shape[0]
    assert captured_sampling is sampling
    assert max_batch_size == 4


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_continuous_graph_capture_uses_fixed_capacity_buffers(mock_get_accelerator):

    class DecodeModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, input_ids, **_kwargs):
            self.calls.append(input_ids)
            return SimpleNamespace(logits=torch.zeros((input_ids.shape[0], 1, 8)))

    stream = MagicMock()
    accelerator = mock_get_accelerator.return_value
    accelerator.Stream.return_value = stream
    accelerator.current_stream.return_value = stream
    accelerator.stream.side_effect = lambda _stream: nullcontext()
    accelerator.capture_to_graph.side_effect = lambda _graph: nullcontext()

    module = DecodeModule()
    static_input = torch.zeros((4, 1), dtype=torch.long)
    static_attention = torch.zeros((4, 6), dtype=torch.long)
    static_write_positions = torch.zeros(4, dtype=torch.long)
    static_cache_position = torch.tensor([5], dtype=torch.long)
    static_position_ids = torch.zeros((4, 1), dtype=torch.long)

    graph, logits = HybridEngineRollout._capture_continuous_graph(
        module,
        MagicMock(),
        static_input,
        static_attention,
        static_write_positions,
        static_cache_position,
        static_position_ids,
    )

    assert graph is accelerator.create_graph.return_value
    assert logits.shape == (4, 1, 8)
    assert len(module.calls) == 4
    assert all(call is static_input for call in module.calls)
    accelerator.capture_to_graph.assert_called_once_with(graph)


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_continuous_graph_capture_preserves_attention_implementation(mock_get_accelerator):

    class DecodeModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_attn_implementation="sdpa")
            self.attention_implementations = []

        def forward(self, input_ids, **_kwargs):
            self.attention_implementations.append(self.config._attn_implementation)
            return SimpleNamespace(logits=torch.zeros((input_ids.shape[0], 1, 8)))

    stream = MagicMock()
    accelerator = mock_get_accelerator.return_value
    accelerator.Stream.return_value = stream
    accelerator.current_stream.return_value = stream
    accelerator.stream.side_effect = lambda _stream: nullcontext()
    accelerator.capture_to_graph.side_effect = lambda _graph: nullcontext()

    module = DecodeModule()
    HybridEngineRollout._capture_continuous_graph(
        module,
        MagicMock(),
        torch.zeros((4, 1), dtype=torch.long),
        torch.zeros((4, 6), dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        torch.tensor([5], dtype=torch.long),
        torch.zeros((4, 1), dtype=torch.long),
    )

    assert module.attention_implementations == ["sdpa"] * 4
    assert module.config._attn_implementation == "sdpa"


def test_continuous_graph_keeps_cache_position_alive_for_replay():
    accelerator = get_accelerator()
    if not accelerator.is_available() or not accelerator.is_graph_capture_supported():
        pytest.skip("requires graph capture")

    class DecodeModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_hidden_layers=1, num_attention_heads=1, hidden_size=1)
            self.cache_position_ref = None

        def forward(self, input_ids, cache_position, **_kwargs):
            self.cache_position_ref = weakref.ref(cache_position)
            return SimpleNamespace(logits=(input_ids + cache_position).unsqueeze(-1))

    module = DecodeModule()
    rollout = HybridEngineRollout(SimpleNamespace(module=module), _make_tokenizer())
    state = rollout._get_continuous_graph_state(module, 1, 3, 5, torch.device(accelerator.current_device_name()),
                                                torch.float32, DeepSpeedStaticCache)
    gc.collect()

    # A captured kernel retains only the device pointer, not the Python tensor.
    assert module.cache_position_ref() is not None
    state.static_input.fill_(7)
    accelerator.replay_graph(state.graph)
    assert torch.equal(state.static_logits, torch.full_like(state.static_logits, 11))


def test_continuous_graph_state_reuses_matching_capture_and_rebuilds_for_weight_updates():

    class GraphCache:

        def __init__(self, *_args, **_kwargs):
            self.reset_calls = 0

        def set_write_position(self, _write_position):
            return None

        def reset(self):
            self.reset_calls += 1

    class DecodeModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace()

    module = DecodeModule()
    rollout = HybridEngineRollout(SimpleNamespace(module=module), _make_tokenizer())
    graph = MagicMock()
    logits = torch.empty((1, 1, 8))

    with patch.object(rollout, "_capture_continuous_graph", return_value=(graph, logits)) as capture:
        first = rollout._get_continuous_graph_state(module, 1, 3, 5, torch.device("cpu"), torch.float32, GraphCache)
        second = rollout._get_continuous_graph_state(module, 1, 3, 5, torch.device("cpu"), torch.float32, GraphCache)
        with torch.no_grad():
            module.weight.add_(1)
        third = rollout._get_continuous_graph_state(module, 1, 3, 5, torch.device("cpu"), torch.float32, GraphCache)

    assert first is second
    assert third is not first
    assert capture.call_count == 2


def test_static_cache_constructor_supports_max_batch_keyword():

    class MaxBatchStaticCache:

        def __init__(self, config, max_batch_size, max_cache_len, device, dtype):
            self.config = config
            self.max_batch_size = max_batch_size

    config = SimpleNamespace(num_attention_heads=4)
    cache = HybridEngineRollout._create_static_cache(MaxBatchStaticCache, config, 2, 8, "cpu", torch.float32)

    assert cache.max_batch_size == 2
    assert cache.config.num_key_value_heads == 4


def test_continuous_cache_span_does_not_sum_independent_requests():
    cache_len = HybridEngineRollout._estimate_continuous_cache_len(64, [64] * 100, 100)

    assert cache_len == 128
    assert cache_len < 64 + 64 * 100


@pytest.mark.parametrize("adaptive_prefill", [False, True])
@pytest.mark.parametrize("length,max_positions", [(3, 4), (65536, 32768)])
def test_continuous_generation_validates_each_request_length(adaptive_prefill, length, max_positions):

    class LimitedModel(torch.nn.Module):

        _supports_cache_class = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=max_positions)

    model = LimitedModel()
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(eos_token_id=None),
                                  HybridEngineRolloutConfig(adaptive_prefill=adaptive_prefill))
    prompt = torch.ones((1, length), dtype=torch.long)
    request = RolloutRequest(prompt, torch.ones_like(prompt))

    with pytest.raises(ValueError, match="request exceeds"):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=1))


def test_continuous_generation_reports_cache_capacity_remedies():

    class CacheConfig(SimpleNamespace):

        def get_text_config(self, **_kwargs):
            return self

        @property
        def per_layer_config(self):
            # StaticCache on current Transformers main reads this before choosing layer types.
            return [self]

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = CacheConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            kwargs.pop("cache_position", None)
            kwargs.pop("position_ids", None)
            past_key_values.update(states, states, layer_idx=0, **kwargs)
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 4))
            logits[:, :, 1] = 1
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    rollout = HybridEngineRollout(
        SimpleNamespace(module=CacheClassModel()),
        SimpleNamespace(pad_token_id=0, eos_token_id=2),
        cfg=HybridEngineRolloutConfig(align_decode_fronts=True, continuous_cache_capacity=2),
    )
    request = RolloutRequest(torch.tensor([[1], [3]]), torch.ones((2, 1), dtype=torch.long))

    with pytest.raises(ValueError, match=r"capacity \(2\).*continuous_cache_capacity.*enable_cache_trimming"):
        rollout.generate(request, SamplingConfig(max_new_tokens=3, temperature=0, continuous_batch_size=1))


def test_continuous_generation_rejects_legacy_cache_model():

    class LegacyModel(torch.nn.Module):
        _supports_cache_class = False

        _supports_cache_class = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=32)

    model = LegacyModel()
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(eos_token_id=None))
    request = RolloutRequest(torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long))

    with pytest.raises(ValueError, match="cache-class support"):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=1))


def test_continuous_graph_generation_rejects_legacy_cache_model():

    class LegacyModel(torch.nn.Module):
        _supports_cache_class = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=32)

    model = LegacyModel()
    rollout = HybridEngineRollout(
        SimpleNamespace(module=model),
        SimpleNamespace(eos_token_id=None),
        cfg=HybridEngineRolloutConfig(use_graph_capture=True),
    )
    request = RolloutRequest(torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long))

    with pytest.raises(ValueError, match="cache-class support"):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=1))


def test_continuous_generation_covers_modern_static_cache_path():

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.calls = []

            self.config = PretrainedConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            key_states = input_ids[:, None, :, None].to(dtype=torch.float32)
            _, cache_values = past_key_values.update(key_states, key_states, layer_idx=0, cache_kwargs=kwargs)
            cache_sums = cache_values[:, 0].sum(dim=(1, 2))
            next_tokens = torch.where(cache_sums == 6, 2, 7).long()
            self.calls.append((input_ids.shape[0], input_ids.shape[1]))
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16))
            logits.scatter_(2, next_tokens[:, None, None].expand(-1, input_ids.shape[1], 1), 1)
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    model = CacheClassModel()
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=2))
    request = RolloutRequest(
        torch.tensor([[1, 2, 3], [1, 2, 4], [1, 2, 5]]),
        torch.ones((3, 3), dtype=torch.long),
    )
    output = rollout.generate(request, SamplingConfig(max_new_tokens=3, temperature=0, continuous_batch_size=2))

    assert output.input_ids.shape == (3, 6)
    assert output.input_ids[:, :3].tolist() == request.prompt_ids.tolist()
    assert output.input_ids[:, 3:].tolist() == [[2, 0, 0], [7, 7, 7], [7, 7, 7]]
    assert output.attention_mask[:, 3:].tolist() == [[1, 0, 0], [1, 1, 1], [1, 1, 1]]
    assert output.response_start_idx.tolist() == [3, 3, 3]
    assert model.calls[0] == (2, 3)
    assert (1, 3) in model.calls


def test_continuous_prefill_requests_last_logits_only():

    class LastLogitsModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.prefill_logits_to_keep = []

            self.config = PretrainedConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        # Mirrors the Transformers contract: 0 returns logits for every position, k for the last k.
        def forward(self,
                    input_ids,
                    attention_mask,
                    past_key_values=None,
                    use_cache=True,
                    cache_position=None,
                    position_ids=None,
                    logits_to_keep=0):
            key_states = input_ids[:, None, :, None].to(dtype=torch.float32)
            past_key_values.update(key_states,
                                   key_states,
                                   layer_idx=0,
                                   cache_kwargs={"cache_position": cache_position})
            if input_ids.shape[1] > 1:
                self.prefill_logits_to_keep.append(logits_to_keep)
            # Each position predicts its own token + 5, so reading any position but the last changes the output.
            logits = torch.nn.functional.one_hot((input_ids + 5) % 16, num_classes=16).to(torch.float32)
            if logits_to_keep:
                logits = logits[:, -logits_to_keep:]
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    model = LastLogitsModel()
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=2))
    request = RolloutRequest(torch.tensor([[1, 2, 3], [1, 2, 4]]), torch.ones((2, 3), dtype=torch.long))
    output = rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=2))

    assert output.input_ids[:, 3:].tolist() == [[8, 13], [9, 14]]
    assert model.prefill_logits_to_keep == [1]


def test_aligned_continuous_generation_supports_mixed_effective_prompt_lengths():

    class CacheConfig(SimpleNamespace):

        def get_text_config(self, **_kwargs):
            return self

        @property
        def per_layer_config(self):
            # StaticCache on current Transformers main reads this before choosing layer types.
            return [self]

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.prefill_lengths = []
            self.decode_positions = []
            self.config = CacheConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            if input_ids.shape[1] > 1:
                self.prefill_lengths.append(input_ids.shape[1])
            else:
                write_position = getattr(past_key_values, "_write_position", None)
                if write_position is not None:
                    self.decode_positions.append(int(write_position[0].item()))
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            kwargs.pop("cache_position", None)
            kwargs.pop("position_ids", None)
            _, values = past_key_values.update(states, states, layer_idx=0, **kwargs)
            cache_sums = values[:, 0].sum(dim=(1, 2))
            next_tokens = torch.where(cache_sums == 6, 2, 7).long()
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16))
            logits.scatter_(2, next_tokens[:, None, None].expand(-1, input_ids.shape[1], 1), 1)
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    model = CacheClassModel()
    rollout = HybridEngineRollout(
        SimpleNamespace(module=model),
        SimpleNamespace(pad_token_id=0, eos_token_id=2),
        cfg=HybridEngineRolloutConfig(align_decode_fronts=True, enable_cache_trimming=True),
    )
    request = RolloutRequest(
        torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4], [0, 0, 1, 2]]),
        torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]]),
    )

    output = rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=2))

    assert output.input_ids.tolist() == [
        [0, 1, 2, 3, 2, 0],
        [1, 2, 3, 4, 7, 7],
        [0, 0, 1, 2, 7, 7],
    ]
    assert output.attention_mask.tolist() == [
        [0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
    ]
    assert model.prefill_lengths == [4, 3, 2]
    assert model.decode_positions == [4, 3]
    stats = rollout.get_last_continuous_stats()
    assert stats["cache_capacity"] == 6
    assert stats["peak_cache_length"] == 5
    assert stats["trim_count"] == 1
    assert stats["trimmed_columns"] == 2
    assert stats["trim_frequency"] == pytest.approx(0.5)
    assert stats["trim_bytes_moved"] > 0
    assert stats["end_to_end_ms"] is None


def test_aligned_continuous_generation_reclaims_dead_prefix_when_cache_would_exhaust():

    class CacheConfig(SimpleNamespace):

        def get_text_config(self, **_kwargs):
            return self

        @property
        def per_layer_config(self):
            # StaticCache on current Transformers main reads this before choosing layer types.
            return [self]

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = CacheConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            kwargs.pop("cache_position", None)
            kwargs.pop("position_ids", None)
            _, values = past_key_values.update(states, states, layer_idx=0, **kwargs)
            cache_sums = values[:, 0].sum(dim=(1, 2))
            next_tokens = torch.where(cache_sums == 20, 2, 7).long()
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16))
            logits.scatter_(2, next_tokens[:, None, None].expand(-1, input_ids.shape[1], 1), 1)
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    rollout = HybridEngineRollout(
        SimpleNamespace(module=CacheClassModel()),
        SimpleNamespace(pad_token_id=0, eos_token_id=2),
        cfg=HybridEngineRolloutConfig(align_decode_fronts=True),
    )
    request = RolloutRequest(
        torch.tensor([[1, 2, 3], [1, 2, 4], [1, 2, 5]]),
        torch.ones((3, 3), dtype=torch.long),
    )

    output = rollout.generate(request, SamplingConfig(max_new_tokens=4, temperature=0, continuous_batch_size=2))

    assert output.input_ids[:, 3:].tolist() == [[7, 7, 2, 0], [7, 7, 7, 7], [7, 7, 7, 7]]
    assert output.attention_mask[:, 3:].tolist() == [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]]
    assert rollout.get_last_continuous_stats()["trim_count"] == 1


@pytest.mark.parametrize("adaptive_prefill", [False, True])
def test_continuous_generation_trims_cache_after_staggered_eos(adaptive_prefill):

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = PretrainedConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            _, values = past_key_values.update(states, states, layer_idx=0, cache_kwargs=kwargs)
            cache_sums = values[:, 0].sum(dim=(1, 2))
            eos_rows = (cache_sums == 6) | (cache_sums == 8) | (cache_sums == 10)
            next_tokens = torch.where(eos_rows, 2, 7).long()
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16))
            logits.scatter_(2, next_tokens[:, None, None].expand(-1, input_ids.shape[1], 1), 1)
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    model = CacheClassModel()
    rollout = HybridEngineRollout(
        SimpleNamespace(module=model),
        SimpleNamespace(pad_token_id=0, eos_token_id=2),
        cfg=HybridEngineRolloutConfig(enable_cache_trimming=True, adaptive_prefill=adaptive_prefill),
    )
    request = RolloutRequest(
        torch.tensor([[1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6], [1, 2, 7], [1, 2, 8]]),
        torch.ones((6, 3), dtype=torch.long),
    )

    output = rollout.generate(request, SamplingConfig(max_new_tokens=4, temperature=0, continuous_batch_size=2))

    assert output.input_ids.shape == (6, 7)
    assert output.input_ids[:, 3:].tolist() == [
        [2, 0, 0, 0],
        [7, 7, 7, 7],
        [2, 0, 0, 0],
        [7, 7, 7, 7],
        [2, 0, 0, 0],
        [7, 7, 7, 7],
    ]
    assert output.attention_mask[:, 3:].tolist() == [
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 1, 1, 1],
    ]


def test_continuous_generation_refills_padded_prompts_after_trim():

    class CacheConfig(SimpleNamespace):

        def get_text_config(self, **_kwargs):
            return self

        @property
        def per_layer_config(self):
            # StaticCache on current Transformers main reads this before choosing layer types.
            return [self]

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = CacheConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            kwargs.pop("cache_position", None)
            kwargs.pop("position_ids", None)
            _, values = past_key_values.update(states, states, layer_idx=0, **kwargs)
            cache_sums = values[:, 0].sum(dim=(1, 2))
            next_tokens = torch.where(cache_sums == 6, 2, 7).long()
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16))
            logits.scatter_(2, next_tokens[:, None, None].expand(-1, input_ids.shape[1], 1), 1)
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    model = CacheClassModel()
    rollout = HybridEngineRollout(
        SimpleNamespace(module=model),
        SimpleNamespace(pad_token_id=0, eos_token_id=2),
        cfg=HybridEngineRolloutConfig(enable_cache_trimming=True),
    )
    prompts = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2]] * 16)
    attention_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]] * 16)
    request = RolloutRequest(prompts, attention_mask)

    output = rollout.generate(request, SamplingConfig(max_new_tokens=3, temperature=0, continuous_batch_size=8))

    assert output.input_ids.shape == (32, 7)
    assert output.attention_mask[:, 4:].any(dim=1).all()
    assert rollout.get_last_continuous_stats()["trim_count"] > 0


def _argmax(logits):
    return logits.argmax(dim=-1, keepdim=True)


def test_continuous_generation_applies_repetition_penalty():
    request = RolloutRequest(torch.tensor([[0]]), torch.ones((1, 1), dtype=torch.long))
    responses = {0: [torch.tensor([[0]])]}
    module = SimpleNamespace(generation_config=SimpleNamespace(repetition_penalty=1.5))

    # Pin the real-model regression: a bare argmax would select the repeated
    # token 0, while Transformers' repetition processor must select token 1.
    next_tokens = HybridEngineRollout._continuous_next_tokens(
        torch.tensor([[6.0, 5.0]]),
        (0, ),
        {0: request},
        responses,
        module,
        _argmax,
    )

    assert next_tokens.tolist() == [[1]]


def test_continuous_generation_treats_none_repetition_penalty_as_one():
    request = RolloutRequest(torch.tensor([[0]]), torch.ones((1, 1), dtype=torch.long))
    responses = {0: [torch.tensor([[0]])]}
    module = SimpleNamespace(generation_config=SimpleNamespace(repetition_penalty=None))

    next_tokens = HybridEngineRollout._continuous_next_tokens(
        torch.tensor([[6.0, 5.0]]),
        (0, ),
        {0: request},
        responses,
        module,
        _argmax,
    )

    assert next_tokens.tolist() == [[0]]


def test_continuous_sampling_applies_repetition_penalty_before_warpers():
    request = RolloutRequest(torch.tensor([[0]]), torch.ones((1, 1), dtype=torch.long))
    responses = {0: [torch.tensor([[0]])]}
    module = SimpleNamespace(generation_config=SimpleNamespace(repetition_penalty=1.5))
    select = HybridEngineRollout._token_selector(SamplingConfig(max_new_tokens=1, temperature=1.0, top_k=1))

    # generate() penalizes first: token 0 drops from 6 to 4, so top-k 1 keeps only token 1. Warping first would
    # keep only token 0.
    next_tokens = HybridEngineRollout._continuous_next_tokens(
        torch.tensor([[6.0, 5.0]]),
        (0, ),
        {0: request},
        responses,
        module,
        select,
    )

    assert next_tokens.tolist() == [[1]]


def test_continuous_generation_applies_repetition_penalty_in_float32():
    request = RolloutRequest(torch.tensor([[0]]), torch.ones((1, 1), dtype=torch.long))
    responses = {0: [torch.tensor([[0]])]}
    module = SimpleNamespace(generation_config=SimpleNamespace(repetition_penalty=1.5))
    logits = torch.tensor([[6.03125, 4.03125]], dtype=torch.bfloat16)

    # generate() penalizes in float32: token 0 drops to 4.0208, below token 1. In bf16 it would round up to 4.03125,
    # tying with token 1, and argmax would keep token 0.
    next_tokens = HybridEngineRollout._continuous_next_tokens(
        logits,
        (0, ),
        {0: request},
        responses,
        module,
        _argmax,
    )

    assert next_tokens.tolist() == [[1]]
    assert logits.tolist() == [[6.03125, 4.03125]]


def test_continuous_prefill_applies_repetition_penalty():

    class Model(torch.nn.Module):
        _supports_cache_class = True
        config = PretrainedConfig(num_hidden_layers=1, num_attention_heads=1, hidden_size=1, head_dim=1)
        generation_config = SimpleNamespace(repetition_penalty=1.5)

        def forward(self, input_ids, **kwargs):
            # Token 0 is in the prompt and wins before the penalty; token 1 is not in the prompt.
            logits = torch.tensor([6.0, 5.0]).expand(input_ids.shape[0], 1, 2)
            return SimpleNamespace(logits=logits)

    rollout = HybridEngineRollout(SimpleNamespace(module=Model()),
                                  SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                  cfg=HybridEngineRolloutConfig())
    prompt_ids = torch.tensor([[0, 0]])

    tokens, _ = rollout._continuous_prefill_cache(rollout.engine.module, lambda *args, **kwargs: None, prompt_ids,
                                                  torch.ones_like(prompt_ids), torch.float32, torch.device("cpu"),
                                                  SamplingConfig(max_new_tokens=1, temperature=0))

    assert tokens.tolist() == [[1]]


def _reference_probs(logits, temperature, top_k, top_p):
    """The distribution ``SamplingConfig`` describes, derived from ``logits`` independently of the rollout."""
    scores = logits / temperature
    if top_k > 0:
        kth_score = scores.topk(top_k).values[-1]
        scores = scores.masked_fill(scores < kth_score, float("-inf"))
    probs = scores.softmax(dim=-1)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(descending=True)
        # A token is kept while the more likely tokens hold less than top_p of the mass.
        keep_sorted = (sorted_probs.cumsum(dim=-1) - sorted_probs) < top_p
        keep = torch.zeros_like(keep_sorted).scatter(0, order, keep_sorted)
        probs = torch.where(keep, probs, torch.zeros_like(probs))
        probs = probs / probs.sum()
    return probs


@pytest.mark.parametrize("temperature, top_k, top_p", [(1.0, -1, 1.0), (0.5, -1, 1.0), (1.0, 3, 1.0), (1.0, -1, 0.7),
                                                       (0.7, 4, 0.9)])
def test_token_selector_samples_the_configured_distribution(temperature, top_k, top_p):
    logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0])
    expected = _reference_probs(logits, temperature, top_k, top_p)
    select = HybridEngineRollout._token_selector(
        SamplingConfig(max_new_tokens=1, temperature=temperature, top_k=top_k, top_p=top_p))
    draws = 20000

    torch.manual_seed(0)
    tokens = select(logits.expand(draws, -1)).squeeze(1)

    observed = torch.bincount(tokens, minlength=logits.numel()).float() / draws
    assert (observed[expected == 0] == 0).all(), "sampled a token outside the top-k / top-p support"
    tolerance = 4 * (expected * (1 - expected) / draws).sqrt() + 1e-6
    assert ((observed - expected).abs() <= tolerance).all(), f"observed {observed}, expected {expected}"


def test_token_selector_is_argmax_without_temperature():
    logits = torch.tensor([[0.1, 3.0, 2.0], [5.0, -1.0, 4.9]])
    select = HybridEngineRollout._token_selector(SamplingConfig(max_new_tokens=1, temperature=0))

    assert select(logits).tolist() == [[1], [0]]


def test_continuous_generation_samples_when_temperature_is_set():

    class TiedLogitsModel(torch.nn.Module):
        """Tokens 7 and 8 share the top logit at every position, so greedy decoding always picks 7."""
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = PretrainedConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            past_key_values.update(states, states, layer_idx=0, cache_kwargs=kwargs)
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16))
            logits[..., 7] = 5.0
            logits[..., 8] = 5.0
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    rollout = HybridEngineRollout(SimpleNamespace(module=TiedLogitsModel()),
                                  SimpleNamespace(pad_token_id=0, eos_token_id=2))
    request = RolloutRequest(torch.tensor([[1, 2, 3]] * 8), torch.ones((8, 3), dtype=torch.long))
    sampling = SamplingConfig(max_new_tokens=4, temperature=1.0, top_k=2, continuous_batch_size=2)

    torch.manual_seed(0)
    output = rollout.generate(request, sampling)

    generated = output.input_ids[:, 3:]
    # Top-k 2 keeps only tokens 7 and 8, each with probability 1/2.
    assert set(generated.flatten().tolist()) == {7, 8}


def test_global_work_remaining_uses_requested_group(monkeypatch):
    process_group = object()
    all_reduce = MagicMock(side_effect=lambda flag, **kwargs: flag.fill_(1))
    monkeypatch.setattr("deepspeed.runtime.rollout.hybrid_engine_rollout.dist.is_initialized", lambda: True)
    monkeypatch.setattr("deepspeed.runtime.rollout.hybrid_engine_rollout.dist.get_world_size", lambda group: 2)
    monkeypatch.setattr("deepspeed.runtime.rollout.hybrid_engine_rollout.dist.all_reduce", all_reduce)

    assert GlobalWorkRemaining(process_group)(torch.tensor(0)) is True
    assert all_reduce.call_args.kwargs["group"] is process_group


def test_global_eos_stops_only_when_no_global_work_remains():
    rendezvous = MagicMock(side_effect=[True, False])
    criterion = _GlobalEosStoppingCriteria(response_start=2, eos_token_id=2, work_remaining=rendezvous)

    keep_going = criterion(torch.tensor([[10, 11, 2], [10, 11, 5]]), scores=None)
    stop = criterion(torch.tensor([[10, 11, 2], [10, 11, 2]]), scores=None)

    assert keep_going is False
    assert stop is True
    # Transformers 4.32 applies Python any() to criterion results.
    assert any([keep_going]) is False
    assert any([stop]) is True
    assert [call.args[0].item() for call in rendezvous.call_args_list] == [1, 0]


def test_zero3_rendezvous_uses_optimizer_dp_group(monkeypatch):
    engine = _make_engine()
    process_group = object()
    engine.zero_optimization_partition_weights.return_value = True
    engine.optimizer.dp_process_group = process_group
    monkeypatch.setattr("deepspeed.runtime.rollout.hybrid_engine_rollout.dist.is_initialized", lambda: True)

    rendezvous = HybridEngineRollout(engine, _make_tokenizer())._zero3_work_remaining()

    assert rendezvous.process_group is process_group


def test_non_zero3_generation_uses_native_eos():
    engine = _make_engine()
    engine.zero_optimization_partition_weights.return_value = False
    engine.module.generate.return_value = torch.tensor([[0, 1, 2, 5, 6], [0, 3, 4, 7, 8]])
    rollout = HybridEngineRollout(engine, _make_tokenizer())

    rollout.generate(_make_request(), _make_sampling())

    assert engine.module.generate.call_args.kwargs["eos_token_id"] == 2
    assert engine.module.generate.call_args.kwargs["stopping_criteria"] is None
    # Non-ZeRO-3 generation must keep the Transformers default.
    assert engine.module.generate.call_args.kwargs["synced_gpus"] is None


def test_zero3_generation_disables_native_synced_gpus(monkeypatch):
    engine = _make_engine()
    engine.zero_optimization_partition_weights.return_value = True
    engine.optimizer.dp_process_group = object()
    engine.module.generate.return_value = torch.tensor([[0, 1, 2, 5, 6], [0, 3, 4, 7, 8]])
    monkeypatch.setattr("deepspeed.runtime.rollout.hybrid_engine_rollout.dist.is_initialized", lambda: True)

    HybridEngineRollout(engine, _make_tokenizer()).generate(_make_request(), _make_sampling())

    kwargs = engine.module.generate.call_args.kwargs
    # Transformers would otherwise derive synced_gpus from the default world
    # size and issue an ungrouped world all-reduce inside the decode loop.
    assert kwargs["synced_gpus"] is False
    assert kwargs["eos_token_id"] is None
    assert kwargs["stopping_criteria"] is not None


class TestGlobalWorkRemainingSubgroups(DistributedTest):
    world_size = 4
    backend = "gloo"
    requires_cuda_env = False
    init_distributed = False

    def test(self):
        # This is a CPU/Gloo topology test. Avoid binding the Gloo default
        # group to the host's visible accelerator when running on a GPU node.
        with patch("deepspeed.comm.torch.get_accelerator") as mock_accelerator:
            mock_accelerator.return_value.device_name.return_value = "cpu"
            deepspeed.init_distributed(dist_backend="gloo")
        first_group = dist.new_group(ranks=[0, 1])
        second_group = dist.new_group(ranks=[2, 3])
        rank = dist.get_rank()
        process_group = first_group if rank < 2 else second_group

        # Only rank 0 has local work. Its DP sibling must continue, while the
        # independent DP subgroup must be able to stop.
        local_work_remaining = torch.tensor(rank == 0, dtype=torch.int32)
        global_work_remaining = GlobalWorkRemaining(process_group)(local_work_remaining)

        assert global_work_remaining is (rank < 2)


class TestZero3SubgroupGeneration(DistributedTest):
    """ZeRO-3 rollout inside one data-parallel subgroup must not touch WORLD.

    When Transformers can observe ZeRO-3 it derives ``synced_gpus`` from the
    default world size, and the decode loop then issues an ungrouped world
    all-reduce. A subgroup that enters rollout while the remaining WORLD ranks
    stay out would block forever on that collective. Here ranks 0-1 generate
    inside one subgroup and ranks 2-3 never enter rollout at all.
    """
    world_size = 4
    backend = "gloo"
    requires_cuda_env = False
    init_distributed = False

    def test(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from transformers.integrations.deepspeed import HfDeepSpeedConfig, is_deepspeed_zero3_enabled

        # This is a CPU/Gloo topology test. Avoid binding the Gloo default
        # group to the host's visible accelerator when running on a GPU node.
        with patch("deepspeed.comm.torch.get_accelerator") as mock_accelerator:
            mock_accelerator.return_value.device_name.return_value = "cpu"
            deepspeed.init_distributed(dist_backend="gloo")
        first_group = dist.new_group(ranks=[0, 1])
        dist.new_group(ranks=[2, 3])
        rank = dist.get_rank()
        dist.barrier()

        # Make ZeRO-3 visible to Transformers so the native default for
        # synced_gpus would be True. Keep the config alive for the process.
        deepspeed_config = HfDeepSpeedConfig({"zero_optimization": {"stage": 3}})
        assert is_deepspeed_zero3_enabled()

        if rank >= 2:
            # Non-participating WORLD ranks must not be required by rollout.
            del deepspeed_config
            return

        model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                max_position_embeddings=16,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            ))
        engine = SimpleNamespace(
            module=model,
            zero_optimization_partition_weights=lambda: True,
            optimizer=SimpleNamespace(dp_process_group=first_group),
        )
        rollout = HybridEngineRollout(engine, SimpleNamespace(pad_token_id=0, eos_token_id=2))
        request = RolloutRequest(
            prompt_ids=torch.tensor([[5, 6]]),
            prompt_attention_mask=torch.ones(1, 2, dtype=torch.long),
        )

        result = rollout.generate(request, SamplingConfig(max_new_tokens=3, temperature=0))

        assert result.input_ids.shape == (1, 5)
        assert result.attention_mask.shape == (1, 5)
        assert result.input_ids[0, :2].tolist() == [5, 6]
        del deepspeed_config


class TestZero3SynchronizedEarlyStop(DistributedTest):
    world_size = 2

    def test(self):
        """Different local EOS steps stop at the latest EOS, not max length."""
        from transformers import Qwen2Config, Qwen2ForCausalLM

        rank = dist.get_rank()
        device = torch.device(get_accelerator().current_device_name())
        model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                max_position_embeddings=16,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            ))
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-5,
                    "torch_adam": True
                },
            },
            "zero_optimization": {
                "stage": 3
            },
            "hybrid_engine": {
                "enabled": True,
                "inference_tp_size": 1,
                "pin_parameters": True,
            },
        }
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
        engine.eval()

        target_eos_step = 2 if rank == 0 else 5
        forward_count = 0
        original_forward = engine.module.forward

        def scripted_forward(_module, *args, **kwargs):
            nonlocal forward_count
            output = original_forward(*args, **kwargs)
            forward_count += 1
            scripted_token = 2 if forward_count >= target_eos_step else 3
            logits = torch.full_like(output.logits, torch.finfo(output.logits.dtype).min)
            logits[:, -1, scripted_token] = 0
            output.logits = logits
            return output

        engine.module.forward = MethodType(scripted_forward, engine.module)
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
        rollout = HybridEngineRollout(engine, tokenizer)
        request = RolloutRequest(
            prompt_ids=torch.tensor([[5, 6]], device=device),
            prompt_attention_mask=torch.ones(1, 2, dtype=torch.long, device=device),
        )
        result = rollout.generate(request, SamplingConfig(max_new_tokens=8, temperature=0))

        counts = [torch.zeros(1, dtype=torch.int32, device=device) for _ in range(2)]
        dist.all_gather(counts, torch.tensor([forward_count], dtype=torch.int32, device=device))
        assert [count.item() for count in counts] == [5, 5]
        assert int(result.attention_mask[:, 2:].sum()) == target_eos_step
        assert result.input_ids[0, 1 + target_eos_step].item() == 2


def test_continuous_repetition_penalty_matches_hf_in_bfloat16():
    from transformers import RepetitionPenaltyLogitsProcessor

    logits = torch.tensor([[1.0, 0.333984375]], dtype=torch.bfloat16)
    prompt = torch.tensor([[0]])
    request = RolloutRequest(prompt, torch.ones_like(prompt))
    model = SimpleNamespace(generation_config=SimpleNamespace(repetition_penalty=3.0))
    expected = RepetitionPenaltyLogitsProcessor(3.0)(prompt, logits.float()).argmax(dim=-1, keepdim=True)
    # Pin the BF16 rounding bug: dividing in BF16 ties the scores and incorrectly selects token 0.
    actual = HybridEngineRollout._continuous_next_tokens(logits, (0, ), {0: request}, {0: ()}, model, _argmax)
    assert expected.tolist() == [[1]]
    assert torch.equal(actual, expected)


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

    with pytest.raises(RuntimeError, match="does not support graph capture"):
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


# -- sync_weights is no-op ---------------------------------------------


def test_sync_weights_is_noop():
    rollout = HybridEngineRollout(_make_engine(), _make_tokenizer())
    assert rollout.sync_weights(step=0) is None


# -- generate dispatches correctly -------------------------------------


@pytest.mark.parametrize("temperature", [0, 0.7])
@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.time.perf_counter")
@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_generate_calls_graph_capture_when_enabled(mock_get_accelerator, mock_perf_counter, temperature):
    mock_get_accelerator.return_value.is_graph_capture_supported.return_value = True
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
    sampling.temperature = temperature
    sampling.n_samples_per_prompt = 1
    sampling.max_new_tokens = 3
    sampling.continuous_batch_size = None

    rollout.generate(req, sampling)
    rollout._generate_graph.assert_called_once()
    # Sampling must not fall back to Hugging Face generate().
    engine.module.generate.assert_not_called()
    profile = rollout.get_last_profile()
    assert profile["prefill_forward_ms"] is None
    assert profile["decode_forward_ms"] is None
    assert profile["generation_overhead_ms"] == pytest.approx(10.0)
    assert profile["num_decode_forwards"] == 0


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_generate_falls_back_to_eager_without_graph_capture_opt_in(mock_get_accelerator):
    # A backend that exposes a graph API but has not opted in must not have
    # its unvalidated graph path selected by use_graph_capture.
    mock_get_accelerator.return_value.is_graph_capture_supported.return_value = False
    engine = _make_engine()
    engine.module.generate.return_value = torch.tensor([[10, 11, 5, 6]])
    rollout = HybridEngineRollout(engine, _make_tokenizer(), cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    rollout._generate_graph = MagicMock()

    req = MagicMock()
    req.prompt_ids = torch.tensor([[10, 11]])
    req.prompt_attention_mask = torch.ones(1, 2, dtype=torch.long)
    sampling = MagicMock()
    sampling.temperature = 0
    sampling.n_samples_per_prompt = 1
    sampling.max_new_tokens = 2
    sampling.top_p = 1.0
    sampling.continuous_batch_size = None

    result = rollout.generate(req, sampling)

    rollout._generate_graph.assert_not_called()
    engine.module.generate.assert_called_once()
    assert result.input_ids.tolist() == [[10, 11, 5, 6]]


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
    sampling.continuous_batch_size = None

    result = rollout.generate(req, sampling)

    assert engine.module.generate.call_args.kwargs['eos_token_id'] is None
    assert result.input_ids.tolist() == [[10, 11, 5, 2, 0, 0]]
    assert result.attention_mask.tolist() == [[1, 1, 1, 1, 0, 0]]


def test_generate_forwards_top_k_for_sampling_with_multiple_samples_and_eos():
    engine = _make_engine()
    tok = _make_tokenizer()
    rollout = HybridEngineRollout(engine, tok)
    engine.module.generate.return_value = torch.tensor([
        [10, 11, 5, 2, 7, 8],
        [10, 11, 6, 7, 2, 8],
    ])
    request = RolloutRequest(torch.tensor([[10, 11]]), torch.ones((1, 2), dtype=torch.long))
    sampling = SamplingConfig(max_new_tokens=4, temperature=0.7, top_p=0.8, top_k=5, n_samples_per_prompt=2)

    result = rollout.generate(request, sampling)

    generate_call = engine.module.generate.call_args
    assert generate_call.args[0].tolist() == [[10, 11], [10, 11]]
    assert generate_call.kwargs["attention_mask"].tolist() == [[1, 1], [1, 1]]
    assert generate_call.kwargs["do_sample"] is True
    assert generate_call.kwargs["temperature"] == 0.7
    assert generate_call.kwargs["top_p"] == 0.8
    assert generate_call.kwargs["top_k"] == 5
    assert generate_call.kwargs["eos_token_id"] is None
    assert result.input_ids.tolist() == [[10, 11, 5, 2, 0, 0], [10, 11, 6, 7, 2, 0]]
    assert result.attention_mask.tolist() == [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]]


@pytest.mark.parametrize("top_k", [-1, 0])
def test_generate_disables_top_k_for_sampling(top_k):
    engine = _make_engine()
    rollout = HybridEngineRollout(engine, _make_tokenizer())
    engine.module.generate.return_value = torch.tensor([[10, 11, 5]])
    request = RolloutRequest(torch.tensor([[10, 11]]), torch.ones((1, 2), dtype=torch.long))

    rollout.generate(request, SamplingConfig(max_new_tokens=1, temperature=0.7, top_k=top_k))

    assert engine.module.generate.call_args.kwargs["top_k"] == 0


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
    sampling = MagicMock(temperature=0,
                         n_samples_per_prompt=1,
                         max_new_tokens=2,
                         top_p=1.0,
                         continuous_batch_size=None)

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


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_static_graph_capture_receives_a_prebuilt_4d_mask(mock_get_accelerator):
    """The captured decode forward must not get a 2-D attention_mask.

    Transformers builds its SDPA mask from a 2-D mask with host-side checks (``padding_mask.all()``), which
    synchronize the stream and invalidate CUDA graph capture. The captured forward therefore receives a
    pre-built 4-D mask, and every replay must see exactly the prompt's valid columns plus the decode
    positions written so far. Additive (finite min) and boolean (False) masks are both accepted.
    """

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.capturing = False
            self.captured_masks = []
            self.config = PretrainedConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            if self.capturing:
                self.captured_masks.append(attention_mask)
            key_states = input_ids[:, None, :, None].to(dtype=torch.float32)
            past_key_values.update(key_states, key_states, layer_idx=0, cache_kwargs=kwargs)
            return SimpleNamespace(logits=torch.zeros((input_ids.shape[0], input_ids.shape[1], 8)))

    def masked_columns(mask):
        if mask.dtype == torch.bool:
            return ~mask[:, 0, 0, :]
        return mask[:, 0, 0, :] < 0

    module = CacheClassModel()
    prompt_attention = torch.tensor([[0, 1, 1], [1, 1, 1]])
    prompt_len = prompt_attention.shape[1]
    max_new_tokens = 4
    replay_views = []

    @contextmanager
    def capture(_graph):
        module.capturing = True
        try:
            yield
        finally:
            module.capturing = False

    stream = MagicMock()
    accelerator = mock_get_accelerator.return_value
    accelerator.Stream.return_value = stream
    accelerator.current_stream.return_value = stream
    accelerator.stream.side_effect = lambda _stream: nullcontext()
    accelerator.capture_to_graph.side_effect = capture
    accelerator.replay_graph.side_effect = lambda _graph: replay_views.append(
        masked_columns(module.captured_masks[0]).clone())

    rollout = HybridEngineRollout(SimpleNamespace(module=module),
                                  _make_tokenizer(),
                                  cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    request = RolloutRequest(torch.tensor([[1, 3, 4], [5, 6, 7]]), prompt_attention)
    batch = rollout.generate(request, SamplingConfig(max_new_tokens=max_new_tokens, temperature=0))

    assert batch.input_ids.shape == (2, prompt_len + max_new_tokens)
    assert len(module.captured_masks) == 1
    captured = module.captured_masks[0]
    assert captured.ndim == 4 and captured.shape == (2, 1, 1, prompt_len + max_new_tokens)
    # The first token comes from prefill; every later token comes from one replay.
    assert len(replay_views) == max_new_tokens - 1
    for step, masked in enumerate(replay_views):
        position = prompt_len + step
        expected = torch.ones_like(masked)
        expected[:, :prompt_len] = prompt_attention == 0
        expected[:, prompt_len:position + 1] = False
        assert torch.equal(masked, expected), f"replay {step}"


def _make_small_qwen(max_position_embeddings=64):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from deepspeed.accelerator import get_accelerator

    torch.manual_seed(8497)
    device = get_accelerator().device_name()
    config = Qwen2Config(vocab_size=32,
                         hidden_size=32,
                         intermediate_size=64,
                         num_hidden_layers=2,
                         num_attention_heads=4,
                         num_key_value_heads=2,
                         max_position_embeddings=max_position_embeddings,
                         pad_token_id=0,
                         eos_token_id=None)
    return Qwen2ForCausalLM(config).to(device).eval()


@pytest.mark.parametrize("lengths,capacity,strategy", [
    ([4, 4, 3, 3], 4, "batched"),
    ([4, 4, 3, 3], 2, "bucketed"),
    ([16, 1, 1, 1], 4, "batched"),
    ([4, 2, 1, 1], 4, "batched"),
    ([1], 1, "batched"),
])
def test_adaptive_generation_matches_single_request_eager(lengths, capacity, strategy):
    model = _make_small_qwen()
    device = next(model.parameters()).device
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None)
    width = max(lengths)
    prompt_ids = torch.zeros((len(lengths), width), dtype=torch.long, device=device)
    mask = torch.zeros_like(prompt_ids, dtype=torch.float32)
    expected = []
    with torch.no_grad():
        for row, length in enumerate(lengths):
            prompt = torch.arange(1, length + 1, device=device).unsqueeze(0)
            prompt_ids[row, -length:] = prompt
            mask[row, -length:] = 1
            expected.append(model.generate(prompt, max_new_tokens=3, do_sample=False)[0, length:])
        rollout = HybridEngineRollout(
            SimpleNamespace(module=model), tokenizer,
            HybridEngineRolloutConfig(adaptive_prefill=True, enable_profiling=True, align_decode_fronts=True))
        sampling = SamplingConfig(max_new_tokens=3, temperature=0, continuous_batch_size=capacity)
        result = rollout.generate(RolloutRequest(prompt_ids, mask), sampling)

    assert torch.equal(result.input_ids[:, width:], torch.stack(expected))
    assert torch.equal(result.input_ids[:, :width], prompt_ids)
    assert torch.equal(result.attention_mask[:, :width], mask)
    assert result.response_start_idx.tolist() == [width] * len(lengths)
    assert rollout.get_last_profile()["generation_strategy"] == strategy
    assert sampling.continuous_batch_size == capacity
    assert rollout.align_decode_fronts is True
    if strategy == "batched":
        assert rollout.get_last_continuous_stats() is None
    else:
        assert rollout.get_last_profile()["active_batch_size"] <= capacity


@pytest.mark.parametrize("adaptive_prefill", [False, True])
def test_continuous_opt_matches_eager_with_static_cache(adaptive_prefill):
    from deepspeed.accelerator import get_accelerator
    from transformers import OPTConfig, OPTForCausalLM

    torch.manual_seed(8497)
    config = OPTConfig(vocab_size=32,
                       hidden_size=32,
                       ffn_dim=64,
                       num_hidden_layers=2,
                       num_attention_heads=4,
                       max_position_embeddings=32,
                       pad_token_id=0,
                       eos_token_id=None,
                       attn_implementation="sdpa")
    model = OPTForCausalLM(config).to(get_accelerator().device_name()).eval()
    prompt = torch.tensor([[1, 4, 5], [0, 4, 5]], device=next(model.parameters()).device)
    mask = prompt.ne(0).long()
    with torch.no_grad():
        expected = model.generate(prompt, attention_mask=mask, max_new_tokens=3, do_sample=False)
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=None),
                                  HybridEngineRolloutConfig(adaptive_prefill=adaptive_prefill))
    result = rollout.generate(RolloutRequest(prompt, mask),
                              SamplingConfig(max_new_tokens=3, temperature=0, continuous_batch_size=1))
    assert torch.equal(result.input_ids, expected)
    assert result.attention_mask[:, :3].tolist() == mask.tolist()


@pytest.mark.parametrize("name,value", [
    ("prefill_fixed_cost_ms", -1.0),
    ("prefill_token_cost_ms", float("nan")),
    ("prefill_attention_cost_ms", float("inf")),
    ("prefill_max_tokens", 0),
])
def test_adaptive_prefill_rejects_invalid_cost_settings(name, value):
    with pytest.raises(ValueError, match=name):
        HybridEngineRollout(_make_engine(), _make_tokenizer(),
                            HybridEngineRolloutConfig(adaptive_prefill=True, **{name: value}))


@patch("deepspeed.comm.get_world_size", return_value=2)
@patch("deepspeed.comm.is_initialized", return_value=True)
def test_adaptive_prefill_rejects_unsynchronized_distributed_selection(_initialized, _world_size):
    rollout = HybridEngineRollout(_make_engine(), _make_tokenizer(), HybridEngineRolloutConfig(adaptive_prefill=True))
    with pytest.raises(ValueError, match="single-process"):
        rollout.generate(_make_request(), SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=2))


def test_config_preserves_legacy_positional_arguments():
    cfg = HybridEngineRolloutConfig(False, False, False, False, True, 16)
    assert cfg.enable_cache_trimming is True
    assert cfg.continuous_cache_capacity == 16
    assert cfg.adaptive_prefill is False
    assert cfg.prefill_max_tokens == 65536


@pytest.mark.parametrize("setting,value", [
    ("no_repeat_ngram_size", 1),
    ("num_beams", 2),
    ("return_dict_in_generate", True),
])
@pytest.mark.parametrize("config_source", ["generation", "model"])
def test_adaptive_generation_rejects_unsupported_generation_settings(setting, value, config_source):
    model = _make_small_qwen()
    config = model.generation_config if config_source == "generation" else model.config
    setattr(config, setting, value)
    prompt = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], device=next(model.parameters()).device)
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=None),
                                  HybridEngineRolloutConfig(adaptive_prefill=True))
    with pytest.raises(ValueError, match=f"adaptive prefill.*{setting}"):
        rollout.generate(RolloutRequest(prompt, torch.ones_like(prompt)),
                         SamplingConfig(max_new_tokens=4, temperature=0, continuous_batch_size=2))


@pytest.mark.parametrize("value", [True, False, None])
@pytest.mark.parametrize("config_source", ["generation", "prepared"])
def test_adaptive_generation_rejects_unknown_public_settings(value, config_source):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    model = Qwen2ForCausalLM(
        Qwen2Config(vocab_size=32,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    num_key_value_heads=2)).eval()
    native_prepare = model._prepare_generation_config
    if config_source == "generation":
        model.generation_config.custom_generation_switch = value

    def prepare(*args, **kwargs):
        config, model_kwargs = native_prepare(*args, **kwargs)
        if config_source == "prepared":
            config.custom_generation_switch = value
        return config, model_kwargs

    prompt = torch.tensor([[1, 2, 3]])
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=None),
                                  HybridEngineRolloutConfig(adaptive_prefill=True))
    with patch.object(model, "_prepare_generation_config", side_effect=prepare):
        with pytest.raises(ValueError, match="custom_generation_switch"):
            rollout.generate(RolloutRequest(prompt, torch.ones_like(prompt)),
                             SamplingConfig(max_new_tokens=1, temperature=0, continuous_batch_size=1))


@pytest.mark.parametrize("capacity", [1, 2])
@pytest.mark.parametrize("stop_at_first", [True, False])
def test_adaptive_profile_counts_only_productive_tokens_after_eos(capacity, stop_at_first):
    model = _make_small_qwen()
    device = next(model.parameters()).device
    prompt_tokens = [1, 2, 3, 4] if stop_at_first else [1, 5]
    width = len(prompt_tokens)
    prompt = torch.tensor([prompt_tokens, prompt_tokens], device=device)
    with torch.no_grad():
        if stop_at_first:
            eos = model.generate(prompt[:1], max_new_tokens=1, do_sample=False)[0, -1].item()
        else:
            # A zero output head deterministically emits PAD 0 before EOS 2.
            model.lm_head.weight.zero_()
            eos = 2
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=eos)
    rollout = HybridEngineRollout(SimpleNamespace(module=model), tokenizer,
                                  HybridEngineRolloutConfig(adaptive_prefill=True, enable_profiling=True))
    result = rollout.generate(
        RolloutRequest(prompt, torch.ones_like(prompt)),
        SamplingConfig(max_new_tokens=4 if stop_at_first else 1, temperature=0, continuous_batch_size=capacity))
    profile = rollout.get_last_profile()
    assert result.attention_mask[:, width:].sum().item() == 2
    if not stop_at_first:
        assert result.input_ids[:, width:].tolist() == [[0], [0]]
    assert profile["num_generated_tokens"] == 2
    if stop_at_first:
        assert profile["num_decode_forwards"] == 0
        assert result.input_ids.shape[1] == width + 1
    assert profile["tokens_per_second"] == pytest.approx(2 * 1000 / profile["total_ms"])
    assert profile["active_batch_size"] == min(capacity, 2)
    assert profile["continuous_batch_size"] == capacity


@pytest.mark.parametrize("capacity", [1, 2])
@pytest.mark.parametrize("config_source", ["generation", "model"])
def test_adaptive_generation_applies_repetition_penalty_from_first_token(capacity, config_source):
    from transformers import GenerationConfig

    model = _make_small_qwen()
    model._supports_cache_class = True
    if config_source == "generation":
        # An explicitly changed generation config takes precedence over legacy model settings.
        model.config.num_beams = 2
        model.generation_config.repetition_penalty = 3.0
        expected_config = model.generation_config
    else:
        model.config.repetition_penalty = 3.0
        expected_config = GenerationConfig.from_model_config(model.config)
    prompt = torch.tensor([[11, 1, 2, 3, 4], [11, 1, 2, 3, 4]], device=next(model.parameters()).device)
    with torch.no_grad():
        # Passing the oracle config explicitly leaves the target's stale default untouched.
        expected = model.generate(prompt[:1], generation_config=expected_config, max_new_tokens=2, do_sample=False)[0,
                                                                                                                    5:]
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=None),
                                  HybridEngineRolloutConfig(adaptive_prefill=True))
    result = rollout.generate(RolloutRequest(prompt, torch.ones_like(prompt)),
                              SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=capacity))
    assert torch.equal(result.input_ids[:, 5:], expected.unsqueeze(0).expand(2, -1))


@pytest.mark.parametrize("cache_support,bucketed,adaptive_prefill", [(True, True, True), (True, False, True),
                                                                     (None, True, True), (None, False, True),
                                                                     (True, True, False)])
def test_adaptive_generation_preserves_padded_repetition_history(cache_support, bucketed, adaptive_prefill):
    model = _make_small_qwen()
    model._supports_cache_class = cache_support
    model.generation_config.repetition_penalty = 3.0
    device = next(model.parameters()).device
    prompt = torch.full((8, 16), 11, dtype=torch.long, device=device)
    mask = torch.zeros_like(prompt)
    prompt[0] = torch.arange(1, 17, device=device)
    mask[0] = 1
    prompt[1:, -4:] = torch.arange(1, 5, device=device)
    mask[1:, -4:] = 1
    with torch.no_grad():
        expected = torch.cat([
            model.generate(prompt[row:row + 1],
                           attention_mask=mask[row:row + 1],
                           max_new_tokens=1,
                           do_sample=False,
                           eos_token_id=None,
                           pad_token_id=11)[:, -1:] for row in range(8)
        ])
    rollout = HybridEngineRollout(
        SimpleNamespace(module=model), SimpleNamespace(pad_token_id=11, eos_token_id=11),
        HybridEngineRolloutConfig(adaptive_prefill=adaptive_prefill,
                                  align_decode_fronts=not adaptive_prefill,
                                  prefill_fixed_cost_ms=0.01 if bucketed else 1e6,
                                  prefill_token_cost_ms=1.0,
                                  prefill_attention_cost_ms=0.0,
                                  prefill_kv_cost_ms=0.0,
                                  continuous_decode_cost_ms=0.0))
    result = rollout.generate(RolloutRequest(prompt, mask),
                              SamplingConfig(max_new_tokens=1, temperature=0, continuous_batch_size=8))
    assert torch.equal(result.input_ids[:, 16:], expected)
    assert result.attention_mask[:, 16:].tolist() == [[1]] * 8


@pytest.mark.parametrize("cache_support", [True, None])
@pytest.mark.parametrize("capacity,prefill_forwards", [(4, 2), (2, 3)])
def test_dp_prefill_groups_requests_and_matches_eager(cache_support, capacity, prefill_forwards):
    model = _make_small_qwen()
    model._supports_cache_class = cache_support
    device = next(model.parameters()).device
    lengths = [1, 16, 2, 3]
    prompt = torch.zeros((4, 16), dtype=torch.long, device=device)
    mask = torch.zeros_like(prompt)
    expected = []
    with torch.no_grad():
        for row, length in enumerate(lengths):
            prompt[row, -length:] = torch.arange(1, length + 1, device=device)
            mask[row, -length:] = 1
            expected.append(
                model.generate(prompt[row:row + 1],
                               attention_mask=mask[row:row + 1],
                               max_new_tokens=3,
                               do_sample=False)[0, 16:])
    cfg = HybridEngineRolloutConfig(adaptive_prefill=True,
                                    enable_profiling=True,
                                    prefill_fixed_cost_ms=2.0,
                                    prefill_token_cost_ms=1.0,
                                    prefill_attention_cost_ms=0.0,
                                    prefill_kv_cost_ms=0.0,
                                    continuous_decode_cost_ms=0.0)
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=None),
                                  cfg)
    result = rollout.generate(RolloutRequest(prompt, mask),
                              SamplingConfig(max_new_tokens=3, temperature=0, continuous_batch_size=capacity))
    assert torch.equal(result.input_ids[:, 16:], torch.stack(expected))
    assert torch.equal(result.attention_mask[:, :16], mask)
    assert result.response_start_idx.tolist() == [16] * 4
    assert rollout.get_last_profile()["generation_strategy"] == "bucketed"
    assert rollout.get_last_profile()["num_prefill_forwards"] == prefill_forwards


def test_dp_prefill_rejects_single_prompt_above_forward_limit():
    model = _make_small_qwen()
    prompt = torch.ones((1, 17), dtype=torch.long, device=next(model.parameters()).device)
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=None),
                                  HybridEngineRolloutConfig(adaptive_prefill=True, prefill_max_tokens=16))
    with pytest.raises(ValueError, match="prefill.*limit"):
        rollout.generate(RolloutRequest(prompt, torch.ones_like(prompt)),
                         SamplingConfig(max_new_tokens=1, temperature=0, continuous_batch_size=1))


def test_adaptive_short_batch_does_not_read_one_gpu_scalar_per_request():
    model = _make_small_qwen()
    prompt = torch.ones((128, 4), dtype=torch.long, device=next(model.parameters()).device)
    request = RolloutRequest(prompt, torch.ones_like(prompt))
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None)
    scalar_reads = []
    outputs = []
    for adaptive in (False, True):
        rollout = HybridEngineRollout(SimpleNamespace(module=model), tokenizer,
                                      HybridEngineRolloutConfig(adaptive_prefill=adaptive))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            output = rollout.generate(
                request,
                SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=128 if adaptive else None))
        outputs.append(output.input_ids)
        # Pin the reported regression: per-row mask.any() caused 128 device-to-host scalar reads.
        scalar_reads.append(
            sum(event.count for event in profile.key_averages() if event.key == "aten::_local_scalar_dense"))
    assert torch.equal(*outputs)
    assert scalar_reads[1] <= scalar_reads[0] + 4


@pytest.mark.parametrize("capacity", [1, 2])
@pytest.mark.parametrize("trigger", ["budget", "eos"])
@pytest.mark.parametrize(("name", "value"), [("min_length", 5), ("do_sample", True), ("max_new_tokens", 1)])
def test_adaptive_generation_resolves_custom_generation_config(capacity, trigger, name, value):
    model = _make_small_qwen()
    native_prepare = model._prepare_generation_config

    def prepare(*args, **kwargs):
        config, model_kwargs = native_prepare(*args, **kwargs)
        if ((trigger == "budget" and kwargs.get("max_new_tokens") == 2)
                or (trigger == "eos" and kwargs.get("eos_token_id") == 2)):
            setattr(config, name, value)
        return config, model_kwargs

    prompt = torch.ones((2, 4), dtype=torch.long, device=next(model.parameters()).device)
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=2),
                                  HybridEngineRolloutConfig(adaptive_prefill=True))
    with patch.object(model, "_prepare_generation_config", side_effect=prepare):
        with pytest.raises(ValueError, match=name):
            rollout.generate(RolloutRequest(prompt, torch.ones_like(prompt)),
                             SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=capacity))


@pytest.mark.parametrize("capacity", [1, 2])
@pytest.mark.parametrize("in_place", [False, True])
def test_adaptive_generation_rejects_mask_changed_by_custom_preparer(capacity, in_place):
    model = _make_small_qwen()
    native_prepare = model._prepare_generation_config

    def prepare(*args, **kwargs):
        config, model_kwargs = native_prepare(*args, **kwargs)
        mask = model_kwargs["attention_mask"]
        if not in_place:
            mask = model_kwargs["attention_mask"] = mask.clone()
        # Hide the first prompt token from generate() only.
        mask[:, 0] = 0
        return config, model_kwargs

    prompt = torch.ones((2, 4), dtype=torch.long, device=next(model.parameters()).device)
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=2),
                                  HybridEngineRolloutConfig(adaptive_prefill=True))
    with patch.object(model, "_prepare_generation_config", side_effect=prepare):
        with pytest.raises(ValueError, match="model kwargs"):
            rollout.generate(RolloutRequest(prompt, torch.ones_like(prompt)),
                             SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=capacity))


@pytest.mark.parametrize("align_decode_fronts", [False, True])
def test_aligned_generation_counts_bfloat16_mask_lengths_exactly(align_decode_fronts):
    model = _make_small_qwen(max_position_embeddings=512)
    # A BF16 sum rounds 257 real tokens to 256, which no longer matches the selected prompt tokens.
    prompt = torch.ones((2, 257), dtype=torch.long, device=next(model.parameters()).device)
    cfg = HybridEngineRolloutConfig(adaptive_prefill=not align_decode_fronts, align_decode_fronts=align_decode_fronts)
    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=None),
                                  cfg)
    sampling = SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=1)

    expected = rollout.generate(RolloutRequest(prompt, torch.ones_like(prompt)), sampling)
    actual = rollout.generate(RolloutRequest(prompt, torch.ones_like(prompt, dtype=torch.bfloat16)), sampling)

    assert torch.equal(actual.input_ids, expected.input_ids)


@pytest.mark.parametrize("cache_support", [True, None])
@pytest.mark.parametrize("capacity", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_adaptive_generation_uses_resolved_penalty_on_both_routes(cache_support, capacity, dtype):
    from transformers import GenerationConfig

    model = _make_small_qwen().to(dtype=dtype)
    model._supports_cache_class = cache_support
    native_prepare = model._prepare_generation_config
    prompt = torch.tensor([[11, 1, 2, 3, 4], [11, 1, 2, 3, 4]], device=next(model.parameters()).device)
    expected_config = GenerationConfig.from_model_config(model.config)
    expected_config.repetition_penalty = 3.0
    with torch.no_grad():
        expected = model.generate(prompt, generation_config=expected_config, max_new_tokens=3, do_sample=False)
        unpenalized = model.generate(prompt, max_new_tokens=3, do_sample=False)
    assert not torch.equal(expected[:, 5:], unpenalized[:, 5:])

    def prepare(*args, **kwargs):
        config, model_kwargs = native_prepare(*args, **kwargs)
        config.repetition_penalty = 3.0
        return config, model_kwargs

    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=None),
                                  HybridEngineRolloutConfig(adaptive_prefill=True))
    with patch.object(model, "_prepare_generation_config", side_effect=prepare):
        result = rollout.generate(RolloutRequest(prompt, torch.ones_like(prompt)),
                                  SamplingConfig(max_new_tokens=3, temperature=0, continuous_batch_size=capacity))
    assert torch.equal(result.input_ids, expected)
    assert model.generation_config.repetition_penalty == 1.0
    assert (rollout.get_last_continuous_stats() is None) == (capacity == 2)


@pytest.mark.parametrize("align_decode_fronts", [False, True])
def test_manual_continuous_generation_omits_unsupported_output_logits(align_decode_fronts):
    model = _make_small_qwen()
    model._supports_cache_class = None
    native_generate = model.generate
    prompt = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], device=next(model.parameters()).device)
    with torch.no_grad():
        expected = native_generate(prompt, max_new_tokens=2, do_sample=False)

    def legacy_generate(*args, **kwargs):
        if "output_logits" in kwargs:
            raise ValueError("output_logits is not supported by this generation API")
        return native_generate(*args, **kwargs)

    rollout = HybridEngineRollout(SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=None),
                                  HybridEngineRolloutConfig(align_decode_fronts=align_decode_fronts))
    with patch.object(model, "generate", side_effect=legacy_generate):
        result = rollout.generate(RolloutRequest(prompt, torch.ones_like(prompt)),
                                  SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=1))
    assert torch.equal(result.input_ids, expected)


@pytest.mark.parametrize("force_cb", [False, True])
def test_adaptive_profile_includes_route_planning_time(force_cb):
    model = _make_small_qwen()
    device = next(model.parameters()).device
    prompt = torch.ones((1024, 4), dtype=torch.long, device=device)
    rollout = HybridEngineRollout(
        SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=None),
        HybridEngineRolloutConfig(adaptive_prefill=True, enable_profiling=True, enable_cache_trimming=force_cb))
    # The clock advances only during the first planning step, so the profile must report exactly that time.
    clock = [0.0]
    validate_inputs = rollout._validate_continuous_inputs

    def slow_validate_inputs(*args):
        clock[0] += 1.0
        return validate_inputs(*args)

    with patch("deepspeed.runtime.rollout.hybrid_engine_rollout.time.perf_counter", side_effect=lambda: clock[0]):
        with patch.object(rollout, "_validate_continuous_inputs", side_effect=slow_validate_inputs):
            result = rollout.generate(RolloutRequest(prompt, torch.ones_like(prompt)),
                                      SamplingConfig(max_new_tokens=1, temperature=0, continuous_batch_size=1024))
    profile = rollout.get_last_profile()
    # The former profile excluded initial DP planning and over-reported throughput.
    assert profile["generation_strategy"] == ("bucketed" if force_cb else "batched")
    assert profile["total_ms"] == pytest.approx(1000.0)
    assert profile["num_generated_tokens"] == result.attention_mask[:, 4:].sum().item()
    if force_cb:
        assert rollout.get_last_continuous_stats()["end_to_end_ms"] == pytest.approx(1000.0)

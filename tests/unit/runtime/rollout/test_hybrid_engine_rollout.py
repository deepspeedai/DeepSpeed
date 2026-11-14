# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for HybridEngineRollout.

Most tests are CPU-only; the native shared-prefill cache test runs only when CUDA and
the transformer inference extension are available, and the graph-capture tests
only when CUDA is available.
"""

import asyncio
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
from deepspeed.runtime.config import HybridEngineConfig
from deepspeed.runtime.hybrid_engine import DeepSpeedHybridEngine
from deepspeed.runtime.rollout.base import RolloutConfig, RolloutRecord, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.tool import Tool, ToolCall, ToolQueue, ToolSet, parse_qwen3_coder_tool_calls
from deepspeed.runtime.rollout.utils import get_eos_token_ids, tokenize_tool_responses, truncate_tool_output
from deepspeed.runtime.rollout.hybrid_engine_rollout import (
    GlobalWorkRemaining,
    HybridEngineRollout,
    HybridEngineRolloutConfig,
    _GlobalEosStoppingCriteria,
)
from deepspeed.runtime.rollout.hybrid_engine_rollout import _TokenSelector
from deepspeed.utils.static_cache import DeepSpeedStaticCache
from unit.common import DistributedTest


def _engine_for(module, max_out_tokens=None):
    """A stand-in for a DeepSpeed engine wrapping ``module``, without ZeRO-3 or tensor parallelism.

    ``max_out_tokens`` defaults to the model's position limit, so it caps nothing a test generates.
    """
    if max_out_tokens is None:
        max_out_tokens = getattr(getattr(module, "config", None), "max_position_embeddings", None) or 1024
    return SimpleNamespace(module=module,
                           zero_optimization_partition_weights=lambda: False,
                           autotp_size=lambda: 1,
                           _config=SimpleNamespace(hybrid_engine=HybridEngineConfig(max_out_tokens=max_out_tokens)))


def _make_engine(max_out_tokens=1024):
    engine = MagicMock()
    engine.module = MagicMock()
    engine.module.parameters.return_value = iter([])
    # Not ZeRO-3, so generation may stop at EOS.
    engine.zero_optimization_partition_weights.return_value = False
    engine.autotp_size.return_value = 1
    engine._config.hybrid_engine = HybridEngineConfig(max_out_tokens=max_out_tokens)
    return engine


def _make_engine_on_cpu():
    """An engine whose module has one CPU parameter, so rollout can resolve the model device."""
    engine = _make_engine()
    engine.module.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
    return engine


def _make_tokenizer():
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 2
    tok.decode.side_effect = lambda ids, **_kwargs: " ".join(map(str, ids))
    return tok


def _prompt(tokens, attention_mask=None):
    """A pre-tokenized prompt sample; samples with tokens skip chat-template rendering."""
    if attention_mask is None:
        attention_mask = [1] * len(tokens)
    return RolloutRecord(tokens=list(tokens), attention_mask=list(attention_mask))


def _make_request():
    return RolloutRequest(samples=[_prompt([0, 1, 2], [0, 1, 1]), _prompt([0, 3, 4], [0, 1, 1])])


def _tensor_request(prompt_ids, attention_mask):
    """A request of pre-tokenized prompts given as left-padded ``[B, T]`` tensors."""
    return RolloutRequest(
        samples=[_prompt(ids.tolist(), mask.tolist()) for ids, mask in zip(prompt_ids, attention_mask)])


def _mock_cpu_accelerator(mock_get_accelerator):
    mock_get_accelerator.return_value.current_device_name.return_value = "cpu"


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
    assert cfg.enable_cache_trimming is False
    assert cfg.continuous_cache_capacity is None
    assert cfg.continuous_batch_size is None
    assert cfg.return_log_probs is False


@pytest.mark.parametrize("kwargs, message", [
    (dict(multi_turn=True, max_turns=None), "max_turns must be a positive integer"),
    (dict(multi_turn=True, max_turns=0), "max_turns must be a positive integer"),
    (dict(multi_turn=True, max_turns=1.5), "max_turns must be a positive integer"),
    (dict(continuous_batch_size=0), "continuous_batch_size must be positive"),
    (dict(tool_call_parser="xml"), "tool_call_parser"),
])
def test_config_rejects_invalid_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HybridEngineRolloutConfig(**kwargs)


def test_rollout_rejects_a_config_of_another_backend():
    # Its settings would otherwise all fall back to their defaults.
    with pytest.raises(TypeError, match="HybridEngineRolloutConfig"):
        HybridEngineRollout(_make_engine(), _make_tokenizer(), RolloutConfig())


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


def test_continuous_generation_returns_nothing_for_an_empty_request():
    engine = _make_engine()
    engine.module = torch.nn.Linear(1, 1)
    rollout = HybridEngineRollout(engine, _make_tokenizer(), cfg=HybridEngineRolloutConfig(continuous_batch_size=1))

    empty = rollout.generate(RolloutRequest(samples=[]), SamplingConfig(max_new_tokens=2))
    assert empty.samples == []


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (HybridEngineRolloutConfig(use_graph_capture=True, continuous_batch_size=1,
                                   align_decode_fronts=True), "aligned decode fronts"),
        (HybridEngineRolloutConfig(use_graph_capture=True, continuous_batch_size=1,
                                   enable_cache_trimming=True), "cache trimming"),
        # The prompt has two real tokens and two to generate; rows drop the padding, so 3 slots are too few.
        (HybridEngineRolloutConfig(use_graph_capture=True, continuous_batch_size=1,
                                   continuous_cache_capacity=3), "must fit"),
    ],
)
def test_continuous_graph_rejects_incompatible_fixed_layout_options(cfg, message, graph_capture_accelerator):
    rollout = HybridEngineRollout(_make_engine_on_cpu(), _make_tokenizer(), cfg=cfg)
    request = _tensor_request(torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, 1]]))

    with pytest.raises(ValueError, match=message):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0))


def test_continuous_graph_rejects_flash_attention(graph_capture_accelerator):
    engine = _make_engine_on_cpu()
    engine.module.config = SimpleNamespace(_attn_implementation="flash_attention_2")
    rollout = HybridEngineRollout(engine,
                                  _make_tokenizer(),
                                  cfg=HybridEngineRolloutConfig(continuous_batch_size=1, use_graph_capture=True))
    request = _tensor_request(torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, 1]]))

    with pytest.raises(ValueError, match="does not support flash_attention_2"):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0))


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
        _engine_for(CacheClassModel()),
        SimpleNamespace(pad_token_id=0, eos_token_id=2, decode=lambda ids, **_kwargs: ""),
        cfg=HybridEngineRolloutConfig(continuous_batch_size=2, use_graph_capture=True),
    )
    request = _tensor_request(
        torch.tensor([[1, 2, 3], [1, 2, 4], [1, 2, 5]]),
        torch.ones((3, 3), dtype=torch.long),
    )

    with patch.object(get_accelerator(), "is_graph_capture_supported", return_value=False), \
            patch.object(rollout, "_generate_continuous_graph") as generate_graph, \
            patch("deepspeed.runtime.rollout.hybrid_engine_rollout.logger.warning_once") as warning_once:
        output = rollout.generate(request, SamplingConfig(max_new_tokens=3, temperature=0))

    generate_graph.assert_not_called()
    warning_once.assert_called_once()
    assert [sample.tokens for sample in output.samples] == [[1, 2, 3, 2], [1, 2, 4, 7, 7, 7], [1, 2, 5, 7, 7, 7]]


def test_continuous_generation_routes_graph_capture_to_fixed_capacity_path(graph_capture_accelerator):
    rollout = HybridEngineRollout(
        _make_engine_on_cpu(),
        _make_tokenizer(),
        cfg=HybridEngineRolloutConfig(continuous_batch_size=4, use_graph_capture=True),
    )
    request = _make_request()
    sampling = SamplingConfig(max_new_tokens=2, temperature=0)

    with patch.object(rollout, "_generate_continuous_graph") as generate_graph:
        output = rollout.generate(request, sampling)

    generate_graph.assert_called_once()
    trajectories, captured_sampling, num_prompts = generate_graph.call_args.args
    assert [sample.tokens for sample in trajectories] == [sample.tokens for sample in request.samples]
    assert output.samples == trajectories
    assert captured_sampling is sampling
    assert num_prompts == len(request.samples)


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
    rollout = HybridEngineRollout(_engine_for(module), _make_tokenizer())
    state = rollout._get_continuous_graph_state(module, 1, 5, torch.device(accelerator.current_device_name()),
                                                torch.float32, DeepSpeedStaticCache)
    gc.collect()

    # A captured kernel retains only the device pointer, not the Python tensor.
    assert module.cache_position_ref() is not None
    state.static_input.fill_(7)
    accelerator.replay_graph(state.graph)
    assert torch.equal(state.static_logits, torch.full_like(state.static_logits, 11))


def test_continuous_graph_state_survives_in_place_weight_updates():

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
            self.weight = torch.nn.Parameter(torch.zeros(2, 3))
            self.register_buffer("inv_freq", torch.ones(3), persistent=False)
            self.config = SimpleNamespace()

    module = DecodeModule()
    rollout = HybridEngineRollout(_engine_for(module), _make_tokenizer())
    graph = MagicMock()
    logits = torch.empty((1, 1, 8))
    device = torch.device("cpu")

    with patch.object(rollout, "_capture_continuous_graph", return_value=(graph, logits)) as capture:
        first = rollout._get_continuous_graph_state(module, 1, 5, device, torch.float32, GraphCache)
        # An optimizer step writes in place, so replay already reads the new values.
        with torch.no_grad():
            module.weight.add_(1)
        updated = rollout._get_continuous_graph_state(module, 1, 5, device, torch.float32, GraphCache)
        # Captured kernels keep the old strides even when the address is unchanged.
        with torch.no_grad():
            module.weight.transpose_(0, 1)
        relaid = rollout._get_continuous_graph_state(module, 1, 5, device, torch.float32, GraphCache)
        # A replaced parameter lives at an address the captured kernels never read.
        module.weight = torch.nn.Parameter(torch.zeros(2, 3))
        moved = rollout._get_continuous_graph_state(module, 1, 5, device, torch.float32, GraphCache)
        # Re-registering a buffer, e.g. a rotary inv_freq or quant scale, moves it too.
        module.register_buffer("inv_freq", torch.full((3, ), 2.0), persistent=False)
        rebuffered = rollout._get_continuous_graph_state(module, 1, 5, device, torch.float32, GraphCache)
        resized = rollout._get_continuous_graph_state(module, 1, 6, device, torch.float32, GraphCache)
        rollout.shutdown()
        released = rollout._get_continuous_graph_state(module, 1, 6, device, torch.float32, GraphCache)

    assert updated is first
    assert relaid is not first
    assert moved is not relaid
    assert rebuffered is not moved
    assert resized is not rebuffered
    assert released is not resized
    assert capture.call_count == 6


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


def test_continuous_generation_validates_each_request_length():

    class LimitedModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=4)

    model = LimitedModel()
    rollout = HybridEngineRollout(_engine_for(model),
                                  SimpleNamespace(pad_token_id=0, eos_token_id=None),
                                  cfg=HybridEngineRolloutConfig(continuous_batch_size=1))
    request = _tensor_request(torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long))

    with pytest.raises(ValueError, match="request exceeds"):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0))


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
        _engine_for(CacheClassModel()),
        _make_tokenizer(),
        cfg=HybridEngineRolloutConfig(continuous_batch_size=1, align_decode_fronts=True, continuous_cache_capacity=2),
    )
    request = _tensor_request(torch.tensor([[1], [3]]), torch.ones((2, 1), dtype=torch.long))

    with pytest.raises(ValueError, match=r"capacity \(2\).*continuous_cache_capacity.*enable_cache_trimming"):
        rollout.generate(request, SamplingConfig(max_new_tokens=3, temperature=0))


def test_continuous_generation_rejects_legacy_cache_model():

    class LegacyModel(torch.nn.Module):
        _supports_cache_class = False

        _supports_cache_class = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=32)

    model = LegacyModel()
    rollout = HybridEngineRollout(_engine_for(model),
                                  SimpleNamespace(pad_token_id=0, eos_token_id=None),
                                  cfg=HybridEngineRolloutConfig(continuous_batch_size=1))
    request = _tensor_request(torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long))

    with pytest.raises(ValueError, match="cache-class support"):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0))


def test_continuous_graph_generation_rejects_legacy_cache_model():

    class LegacyModel(torch.nn.Module):
        _supports_cache_class = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=32)

    model = LegacyModel()
    rollout = HybridEngineRollout(
        _engine_for(model),
        SimpleNamespace(pad_token_id=0, eos_token_id=None),
        cfg=HybridEngineRolloutConfig(continuous_batch_size=1, use_graph_capture=True),
    )
    request = _tensor_request(torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long))

    with pytest.raises(ValueError, match="cache-class support"):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0))


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
    rollout = HybridEngineRollout(_engine_for(model),
                                  _make_tokenizer(),
                                  cfg=HybridEngineRolloutConfig(continuous_batch_size=2))
    request = _tensor_request(
        torch.tensor([[1, 2, 3], [1, 2, 4], [1, 2, 5]]),
        torch.ones((3, 3), dtype=torch.long),
    )
    output = rollout.generate(request, SamplingConfig(max_new_tokens=3, temperature=0))

    assert [sample.tokens for sample in output.samples] == [[1, 2, 3, 2], [1, 2, 4, 7, 7, 7], [1, 2, 5, 7, 7, 7]]
    assert [sample.loss_mask for sample in output.samples] == [[1], [1, 1, 1], [1, 1, 1]]
    assert [sample.status for sample in output.samples
            ] == [RolloutRecord.Status.COMPLETED, RolloutRecord.Status.TRUNCATED, RolloutRecord.Status.TRUNCATED]
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
    rollout = HybridEngineRollout(_engine_for(model),
                                  _make_tokenizer(),
                                  cfg=HybridEngineRolloutConfig(continuous_batch_size=2))
    request = _tensor_request(torch.tensor([[1, 2, 3], [1, 2, 4]]), torch.ones((2, 3), dtype=torch.long))
    output = rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0))

    assert [sample.tokens[3:] for sample in output.samples] == [[8, 13], [9, 14]]
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
        _engine_for(model),
        _make_tokenizer(),
        cfg=HybridEngineRolloutConfig(continuous_batch_size=2, align_decode_fronts=True, enable_cache_trimming=True),
    )
    request = _tensor_request(
        torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4], [0, 0, 1, 2]]),
        torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]]),
    )

    output = rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0))

    assert [sample.tokens for sample in output.samples] == [
        [0, 1, 2, 3, 2],
        [1, 2, 3, 4, 7, 7],
        [0, 0, 1, 2, 7, 7],
    ]
    assert [sample.attention_mask for sample in output.samples] == [
        [0, 1, 1, 1, 1],
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
        _engine_for(CacheClassModel()),
        _make_tokenizer(),
        cfg=HybridEngineRolloutConfig(continuous_batch_size=2, align_decode_fronts=True),
    )
    request = _tensor_request(
        torch.tensor([[1, 2, 3], [1, 2, 4], [1, 2, 5]]),
        torch.ones((3, 3), dtype=torch.long),
    )

    output = rollout.generate(request, SamplingConfig(max_new_tokens=4, temperature=0))

    assert [sample.tokens[3:] for sample in output.samples] == [[7, 7, 2], [7, 7, 7, 7], [7, 7, 7, 7]]
    assert [sample.attention_mask[3:] for sample in output.samples] == [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    assert rollout.get_last_continuous_stats()["trim_count"] == 1


def test_continuous_generation_trims_cache_after_staggered_eos():

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
        _engine_for(model),
        _make_tokenizer(),
        cfg=HybridEngineRolloutConfig(continuous_batch_size=2, enable_cache_trimming=True),
    )
    request = _tensor_request(
        torch.tensor([[1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6], [1, 2, 7], [1, 2, 8]]),
        torch.ones((6, 3), dtype=torch.long),
    )

    output = rollout.generate(request, SamplingConfig(max_new_tokens=4, temperature=0))

    assert [sample.tokens[3:] for sample in output.samples] == [[2], [7, 7, 7, 7]] * 3
    assert [sample.status
            for sample in output.samples] == [RolloutRecord.Status.COMPLETED, RolloutRecord.Status.TRUNCATED] * 3


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
        _engine_for(model),
        _make_tokenizer(),
        cfg=HybridEngineRolloutConfig(continuous_batch_size=8, enable_cache_trimming=True),
    )
    prompts = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2]] * 16)
    attention_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]] * 16)
    request = _tensor_request(prompts, attention_mask)

    output = rollout.generate(request, SamplingConfig(max_new_tokens=3, temperature=0))

    assert len(output.samples) == 32
    assert all(sample.response_length > 0 for sample in output.samples)
    assert rollout.get_last_continuous_stats()["trim_count"] > 0


def test_token_selector_tracks_each_rows_history():
    select = _TokenSelector(SamplingConfig(max_new_tokens=1, temperature=0), repetition_penalty=4.0, num_rows=3)
    # Token 2 scores highest until it has been seen, then token 1 takes over.
    logits = torch.tensor([[0.0, 3.0, 4.0, 0.0]]).expand(3, -1)

    # The pad position of row 1 holds token 2, which the mask leaves out of its history.
    first, _ = select.first(logits[:2], torch.tensor([[3, 3], [2, 3]]), torch.tensor([[1, 1], [0, 1]]))
    assert first.tolist() == [[2], [2]]
    # Each row now has token 2 in its history, from its own pick.
    assert select(logits[:2])[0].tolist() == [[1], [1]]

    # Row 2 starts fresh in the spare row, then row 0 retires and rows 1 and 2 move to the front.
    assert select.first(logits[:1], torch.tensor([[3]]), rows=torch.tensor([2]))[0].tolist() == [[2]]
    select.keep(torch.tensor([1, 2]))
    assert select.seen[:2, 2].tolist() == [True, True]
    assert select.seen[:2, 1].tolist() == [True, False]


class _StubCacheConfig(SimpleNamespace):

    def get_text_config(self, **_kwargs):
        return self

    @property
    def per_layer_config(self):
        # StaticCache on current Transformers main reads this before choosing layer types.
        return [self]


class _LastTokenModel(torch.nn.Module):
    """Writes its inputs to the cache and scores each position's next token with ``score(input_ids)``."""

    _supports_cache_class = True

    def __init__(self, score, checkpoint_repetition_penalty=1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.score = score
        self.generation_config = SimpleNamespace(repetition_penalty=checkpoint_repetition_penalty)
        self.config = _StubCacheConfig(
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
        return SimpleNamespace(logits=self.score(input_ids), past_key_values=past_key_values)


_VOCAB = 16


def _next_token_logits(previous):
    """Logits after token ``previous``, peaked at ``previous + 1`` and sharper for larger tokens."""
    vocab = torch.arange(_VOCAB, dtype=torch.float32)
    return -((vocab - (previous + 1) % _VOCAB)**2) / (previous + 2)


def test_continuous_generation_records_greedy_log_probs():
    model = _LastTokenModel(lambda input_ids: _next_token_logits(input_ids[..., None].float()))
    rollout = HybridEngineRollout(_engine_for(model),
                                  _make_tokenizer(),
                                  cfg=HybridEngineRolloutConfig(return_log_probs=True, continuous_batch_size=1))
    request = RolloutRequest(samples=[_prompt([3, 4]), _prompt([0, 5], [0, 1]), _prompt([7, 8])])
    # One row at a time, so later prompts are admitted as earlier ones retire.
    sampling = SamplingConfig(max_new_tokens=3, temperature=0)

    samples = rollout.generate(request, sampling).samples

    assert [sample.tokens for sample in samples] == [[3, 4, 5, 6, 7], [0, 5, 6, 7, 8], [7, 8, 9, 10, 11]]
    for sample in samples:
        assert len(sample.rollout_log_probs) == sample.response_length == 3
        tokens = sample.tokens
        # The reference is each greedy token's log-prob under the logits that follow the token before it.
        expected = [
            torch.log_softmax(_next_token_logits(torch.tensor(float(previous))), dim=-1)[token].item()
            for previous, token in zip(tokens[-4:-1], tokens[-3:])
        ]
        assert sample.rollout_log_probs == pytest.approx(expected, abs=1e-6)


def _repeat_preferring_logits(input_ids):
    """Prefer repeating the last token (2.0) over moving to the next one (1.5)."""
    logits = torch.zeros((*input_ids.shape, _VOCAB))
    logits.scatter_(-1, input_ids[..., None], 2.0)
    logits.scatter_(-1, ((input_ids + 1) % _VOCAB)[..., None], 1.5)
    return logits


def test_sampling_repetition_penalty_overrides_the_checkpoint():
    request = RolloutRequest(samples=[_prompt([5, 6])])

    def generate(checkpoint_repetition_penalty, repetition_penalty):
        model = _LastTokenModel(_repeat_preferring_logits, checkpoint_repetition_penalty=checkpoint_repetition_penalty)
        rollout = HybridEngineRollout(_engine_for(model),
                                      _make_tokenizer(),
                                      cfg=HybridEngineRolloutConfig(continuous_batch_size=1))
        sampling = SamplingConfig(max_new_tokens=3, temperature=0, repetition_penalty=repetition_penalty)
        return rollout.generate(request, sampling).samples[0].tokens

    # 1.0 turns off a checkpoint's penalty, and a set value applies over a checkpoint without one.
    assert generate(100.0, 1.0) == generate(1.0, None)
    assert generate(1.0, 100.0) == generate(100.0, None)
    assert generate(1.0, 100.0) != generate(1.0, None)


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.logger")
def test_return_log_probs_warns_about_a_checkpoint_repetition_penalty(mock_logger):
    model = _LastTokenModel(_repeat_preferring_logits, checkpoint_repetition_penalty=1.5)
    rollout = HybridEngineRollout(_engine_for(model),
                                  _make_tokenizer(),
                                  cfg=HybridEngineRolloutConfig(continuous_batch_size=1, return_log_probs=True))
    sampling = SamplingConfig(max_new_tokens=2, temperature=0)

    samples = rollout.generate(RolloutRequest(samples=[_prompt([5, 6])]), sampling).samples

    assert samples[0].response_length == 2
    assert "repetition_penalty=1.5" in mock_logger.warning_once.call_args.args[0]


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


def test_global_eos_counts_a_row_past_its_budget_as_finished():
    rendezvous = MagicMock(side_effect=lambda local: bool(local.item()))
    # Row 0 may generate two tokens and row 1 four; neither emits EOS.
    criterion = _GlobalEosStoppingCriteria(response_start=2,
                                           eos_token_id=2,
                                           work_remaining=rendezvous,
                                           row_budgets=[2, 4])

    assert criterion(torch.tensor([[10, 11, 5, 6], [10, 11, 5, 6]]), scores=None) is False
    # Row 0 is at its budget; row 1 has reached EOS.
    assert criterion(torch.tensor([[10, 11, 5, 6], [10, 11, 5, 2]]), scores=None) is True
    # A placeholder row with budget 0 is finished from its first token, and needs no EOS ids.
    placeholder = _GlobalEosStoppingCriteria(response_start=1,
                                             eos_token_id=None,
                                             work_remaining=rendezvous,
                                             row_budgets=[0])
    assert placeholder(torch.tensor([[0, 9]]), scores=None) is True


def test_zero3_rank_with_no_budget_left_joins_with_a_finished_placeholder_row():
    engine = _make_engine(max_out_tokens=2)
    engine.zero_optimization_partition_weights.return_value = True
    engine.module.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
    engine.module.generate.return_value = torch.tensor([[0, 9]])
    rollout = HybridEngineRollout(engine, _make_tokenizer())

    # The prompt already fills max_out_tokens, so this rank has nothing to generate.
    result = rollout.generate(RolloutRequest(samples=[_prompt([10, 11])]),
                              SamplingConfig(max_new_tokens=4, max_new_tokens_per_turn=3, temperature=0))

    assert result.samples[0].status == RolloutRecord.Status.TRUNCATED
    kwargs = engine.module.generate.call_args.kwargs
    assert engine.module.generate.call_args.args[0].shape == (1, 1)
    # Ranks decode up to the same configured cap, whatever their own budgets.
    assert kwargs["max_new_tokens"] == 3
    assert kwargs["stopping_criteria"][0].row_budgets.tolist() == [0]


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

    assert engine.module.generate.call_args.kwargs["eos_token_id"] == [2]
    assert engine.module.generate.call_args.kwargs["stopping_criteria"] is None
    # Non-ZeRO-3 generation must keep the Transformers default.
    assert engine.module.generate.call_args.kwargs["synced_gpus"] is None


def test_zero3_generation_disables_native_synced_gpus(monkeypatch):
    engine = _make_engine()
    engine.zero_optimization_partition_weights.return_value = True
    engine.optimizer.dp_process_group = object()
    engine.module.generate.return_value = torch.tensor([[0, 1, 2, 5, 6], [0, 3, 4, 7, 8]])
    monkeypatch.setattr("deepspeed.runtime.rollout.hybrid_engine_rollout.dist.is_initialized", lambda: True)

    rollout = HybridEngineRollout(engine, _make_tokenizer())
    rollout.generate(_make_request(), _make_sampling())

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
            seq_data_parallel_group=first_group,
            autotp_size=lambda: 1,
            _config=SimpleNamespace(hybrid_engine=HybridEngineConfig(max_out_tokens=16)),
        )
        rollout = HybridEngineRollout(
            engine, SimpleNamespace(pad_token_id=0, eos_token_id=2, decode=lambda ids, **_kwargs: ""))

        result = rollout.generate(RolloutRequest(samples=[_prompt([5, 6])]),
                                  SamplingConfig(max_new_tokens=3, temperature=0))

        sample = result.samples[0]
        assert sample.tokens[:2] == [5, 6]
        assert 1 <= sample.response_length <= 3
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
                "max_out_tokens": 16,
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
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, decode=lambda ids, **_kwargs: "")
        rollout = HybridEngineRollout(engine, tokenizer)
        result = rollout.generate(RolloutRequest(samples=[_prompt([5, 6])]),
                                  SamplingConfig(max_new_tokens=8, temperature=0))

        counts = [torch.zeros(1, dtype=torch.int32, device=device) for _ in range(2)]
        dist.all_gather(counts, torch.tensor([forward_count], dtype=torch.int32, device=device))
        assert [count.item() for count in counts] == [5, 5]
        sample = result.samples[0]
        assert sample.response_length == target_eos_step
        assert sample.tokens[1 + target_eos_step] == 2


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.time.perf_counter")
@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_generate_records_profile_when_enabled(mock_get_accelerator, mock_perf_counter):
    _mock_cpu_accelerator(mock_get_accelerator)
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
    assert profile["max_prompt_length"] == 3
    assert profile["response_length"] == 2
    assert [sample.attention_mask[:3] for sample in output.samples] == [[0, 1, 1]] * 4
    assert mock_get_accelerator.return_value.synchronize.call_count == 4


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_generate_does_not_profile_when_disabled(mock_get_accelerator):
    _mock_cpu_accelerator(mock_get_accelerator)
    engine = _make_engine()
    rollout = HybridEngineRollout(engine, _make_tokenizer())
    engine.module.generate.return_value = torch.tensor([[0, 1, 2, 5, 6], [0, 3, 4, 7, 8]])

    rollout.generate(_make_request(), _make_sampling())

    assert rollout.get_last_profile() is None
    mock_get_accelerator.return_value.synchronize.assert_not_called()


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
    with patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator") as mock_get_accelerator:
        _mock_cpu_accelerator(mock_get_accelerator)
        output_with_profiling = rollout_with_profiling.generate(_make_request(), _make_sampling())

    assert output_with_profiling.samples == output_without_profiling.samples


def test_generate_pads_with_eos_when_the_tokenizer_has_no_pad_token():
    engine = _make_engine()
    engine.module.generate.side_effect = lambda prompt_ids, **_kwargs: torch.cat(
        (prompt_ids, torch.full((prompt_ids.shape[0], 1), 2, device=prompt_ids.device)), dim=1)
    tokenizer = _make_tokenizer()
    tokenizer.pad_token_id = None
    rollout = HybridEngineRollout(engine, tokenizer)

    output = rollout.generate(RolloutRequest(samples=[_prompt([5, 6, 7]), _prompt([8])]), _make_sampling())

    prompt_ids = engine.module.generate.call_args.args[0]
    assert prompt_ids[1].tolist() == [2, 2, 8]
    assert engine.module.generate.call_args.kwargs["pad_token_id"] == 2
    assert [sample.status for sample in output.samples] == [RolloutRecord.Status.COMPLETED] * 2


def test_generate_requires_a_pad_or_eos_token():
    tokenizer = _make_tokenizer()
    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = None
    rollout = HybridEngineRollout(_make_engine(), tokenizer)

    with pytest.raises(ValueError, match="must define pad_token_id or eos_token_id"):
        rollout.generate(_make_request(), _make_sampling())


def test_generate_preserves_zero_pad_token_id():
    engine = _make_engine()
    engine.module.generate.return_value = torch.tensor([[0, 1, 2, 0], [0, 3, 4, 0]])
    rollout = HybridEngineRollout(engine, _make_tokenizer())

    output = rollout.generate(_make_request(), _make_sampling())

    # A generated token equal to pad_token_id is still a response token, since only positions after EOS are padding.
    assert [sample.response_length for sample in output.samples] == [1, 1]
    assert [sample.tokens for sample in output.samples] == [[0, 1, 2, 0], [0, 3, 4, 0]]


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
            self.config = SimpleNamespace()
            self.forward_batch_sizes = []

        def forward(self, input_ids, attention_mask=None):
            self.forward_batch_sizes.append(input_ids.shape[0])
            values = input_ids[:, None, :, None].float()
            return SimpleNamespace(logits=input_ids[:, :, None].float(), past_key_values=((values, values), ))

        def generate(self, input_ids, attention_mask=None, max_new_tokens=1, **_kwargs):
            self(input_ids=input_ids, attention_mask=attention_mask)
            response = torch.ones(input_ids.shape[0], max_new_tokens, dtype=input_ids.dtype, device=input_ids.device)
            return torch.cat((input_ids, response), dim=1)

    engine = _make_engine()
    engine.module = GenerateModule()
    config = HybridEngineRolloutConfig(use_shared_prefill=True)
    rollout = HybridEngineRollout(engine, _make_tokenizer(), config)

    output = rollout.generate(_make_request(), _make_sampling(n_samples_per_prompt=2))

    assert engine.module.forward_batch_sizes == [2]
    assert [sample.tokens[3:] for sample in output.samples] == [[1, 1]] * 4
    engine.prepare_shared_prefill.assert_called_once_with(2, 2, 3)
    engine.repeat_shared_prefill_cache.assert_called_once_with(2, 2)


@pytest.mark.parametrize(("cfg", "has_inference_policy", "message"), [
    (HybridEngineRolloutConfig(use_graph_capture=True,
                               use_shared_prefill=True), True, "does not support graph capture"),
    (HybridEngineRolloutConfig(continuous_batch_size=2, use_shared_prefill=True), True,
     "continuous batching does not support shared prompt prefill"),
    (HybridEngineRolloutConfig(use_shared_prefill=True), False, "DeepSpeed inference policy"),
])
def test_shared_prefill_rejects_unsupported_setups_at_construction(cfg, has_inference_policy, message):
    engine = _make_engine()
    # A model without a DeepSpeed inference policy, such as Qwen3, gets no hybrid engine inference containers.
    engine._inference_containers = [object()] if has_inference_policy else []

    with pytest.raises(ValueError, match=message):
        HybridEngineRollout(engine, _make_tokenizer(), cfg)


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


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.time.perf_counter")
@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_generate_calls_graph_capture_when_enabled(mock_get_accelerator, mock_perf_counter):
    _mock_cpu_accelerator(mock_get_accelerator)
    engine = _make_engine()
    tok = _make_tokenizer()
    cfg = HybridEngineRolloutConfig(use_graph_capture=True, enable_profiling=True)
    rollout = HybridEngineRollout(engine, tok, cfg=cfg)
    rollout._generate_graph = MagicMock(return_value=(torch.tensor([[1, 3, 5, 2]]), torch.zeros(1, 2)))
    mock_perf_counter.side_effect = [1.0, 1.001, 1.011, 1.013]
    request = RolloutRequest(samples=[_prompt([1, 3])])

    output = rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0))

    rollout._generate_graph.assert_called_once()
    engine.module.generate.assert_not_called()
    assert output.samples[0].tokens == [1, 3, 5, 2]
    profile = rollout.get_last_profile()
    assert profile["prefill_forward_ms"] is None
    assert profile["decode_forward_ms"] is None
    assert profile["generation_overhead_ms"] == pytest.approx(10.0)
    assert profile["num_decode_forwards"] == 0


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_generate_falls_back_to_eager_without_graph_capture_opt_in(mock_get_accelerator):
    # A backend that exposes a graph API but has not opted in must not have
    # its unvalidated graph path selected by use_graph_capture.
    _mock_cpu_accelerator(mock_get_accelerator)
    mock_get_accelerator.return_value.is_graph_capture_supported.return_value = False
    engine = _make_engine_on_cpu()
    engine.module.generate.return_value = torch.tensor([[10, 11, 5, 6]])
    rollout = HybridEngineRollout(engine, _make_tokenizer(), cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    rollout._generate_graph = MagicMock()

    result = rollout.generate(_tensor_request(torch.tensor([[10, 11]]), torch.ones(1, 2, dtype=torch.long)),
                              SamplingConfig(max_new_tokens=2, temperature=0))

    rollout._generate_graph.assert_not_called()
    engine.module.generate.assert_called_once()
    assert result.samples[0].tokens == [10, 11, 5, 6]


def test_repetition_penalty_applies_before_warpers_and_in_float32():
    prompt = torch.tensor([[0]])
    select = _TokenSelector(SamplingConfig(max_new_tokens=1, temperature=1.0, top_k=1), repetition_penalty=1.5)
    # generate() penalizes before warping, which drops token 0 from 6 to 4 so top-k 1 keeps only token 1.
    tokens, _ = select.first(torch.tensor([[6.0, 5.0]]), prompt)
    assert tokens.tolist() == [[1]]

    # In bf16 token 0 would round to 4.0312 and tie with token 1, and argmax would keep token 0.
    logits = torch.tensor([[6.03125, 4.03125]], dtype=torch.bfloat16)
    greedy = _TokenSelector(SamplingConfig(max_new_tokens=1, temperature=0), repetition_penalty=1.5)
    tokens, _ = greedy.first(logits, prompt)
    assert tokens.tolist() == [[1]]
    assert logits.tolist() == [[6.03125, 4.03125]]


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


@pytest.mark.parametrize("temperature, top_k, top_p", [(1.0, -1, 1.0), (0.5, -1, 1.0), (1.0, 3, 1.0), (1.0, -1, 0.8),
                                                       (0.7, 4, 0.9)])
def test_token_selector_samples_the_configured_distribution(temperature, top_k, top_p):
    logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0])
    expected = _reference_probs(logits, temperature, top_k, top_p)
    select = _TokenSelector(SamplingConfig(max_new_tokens=1, temperature=temperature, top_k=top_k, top_p=top_p))
    draws = 20000

    torch.manual_seed(0)
    tokens, log_probs = select(logits.expand(draws, -1))
    tokens = tokens.squeeze(1)

    observed = torch.bincount(tokens, minlength=logits.numel()).float() / draws
    assert (observed[expected == 0] == 0).all(), "sampled a token outside the top-k / top-p support"
    tolerance = 4 * (expected * (1 - expected) / draws).sqrt() + 1e-6
    assert ((observed - expected).abs() <= tolerance).all(), f"observed {observed}, expected {expected}"
    torch.testing.assert_close(log_probs.squeeze(1).exp(), expected[tokens], atol=1e-5, rtol=0)


def test_token_selector_is_argmax_without_temperature():
    logits = torch.tensor([[0.1, 3.0, 2.0], [5.0, -1.0, 4.9]])
    tokens, _ = _TokenSelector(SamplingConfig(max_new_tokens=1, temperature=0))(logits)

    assert tokens.tolist() == [[1], [0]]


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_token_selector_skips_log_probs_unless_asked(temperature):
    select = _TokenSelector(SamplingConfig(max_new_tokens=1, temperature=temperature), return_log_probs=False)

    tokens, log_probs = select(torch.tensor([[0.1, 3.0, 2.0]]))

    assert tokens.shape == (1, 1)
    assert log_probs is None


class _EchoTool(Tool):
    name = "echo"
    description = "Returns ok."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, arguments):
        return "ok"


_TOOL_CALL_TOKEN = 7


def _multi_turn_rollout(responses_per_turn,
                        call_text='<tool_call>{"name": "echo", "arguments": {}}</tool_call>',
                        tools=None,
                        max_out_tokens=1024,
                        **cfg_kwargs):
    """A multi-turn rollout whose model emits ``responses_per_turn`` in order, and ``call_text`` on token 7."""
    engine = _make_engine(max_out_tokens)
    responses = iter(responses_per_turn)

    def generate(prompt_ids, **_kwargs):
        response = torch.tensor([next(responses)], device=prompt_ids.device)
        return torch.cat((prompt_ids, response), dim=1)

    engine.module.generate.side_effect = generate
    tokenizer = _make_tokenizer()
    tokenizer.decode.side_effect = lambda ids, **_kwargs: call_text if _TOOL_CALL_TOKEN in ids else ""
    cfg = HybridEngineRolloutConfig(multi_turn=True, max_turns=4, **cfg_kwargs)
    tools = tools if tools is not None else ToolSet([_EchoTool()])
    return engine, HybridEngineRollout(engine, tokenizer, cfg=cfg, tools=tools)


def test_single_turn_does_not_run_tool_calls():
    # A single-turn rollout ends after one turn, even when the response looks like a tool call.
    engine, rollout = _multi_turn_rollout([[_TOOL_CALL_TOKEN, 2]])
    rollout.multi_turn = False

    sample = rollout.generate(RolloutRequest(samples=[_prompt([10, 11])]), _make_sampling()).samples[0]

    assert engine.module.generate.call_count == 1
    assert sample.tokens == [10, 11, _TOOL_CALL_TOKEN, 2]
    assert sample.status == RolloutRecord.Status.COMPLETED


def test_single_turn_caps_the_context_at_max_out_tokens():
    engine = _make_engine(max_out_tokens=4)
    engine.module.generate.side_effect = lambda prompt_ids, **kwargs: torch.cat(
        (prompt_ids, torch.full((prompt_ids.shape[0], kwargs["max_new_tokens"]), 5, device=prompt_ids.device)), dim=1)
    rollout = HybridEngineRollout(engine, _make_tokenizer())

    sample = rollout.generate(RolloutRequest(samples=[_prompt([10, 11, 12])]),
                              SamplingConfig(max_new_tokens=8, temperature=0)).samples[0]

    assert sample.tokens == [10, 11, 12, 5]
    assert sample.status == RolloutRecord.Status.TRUNCATED


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.tokenize_tool_responses", return_value=[9] * 10)
def test_multi_turn_ends_when_tool_output_would_pass_max_out_tokens(_tokenize_tool_responses):
    engine, rollout = _multi_turn_rollout([[_TOOL_CALL_TOKEN, 2]], max_out_tokens=8)

    sample = rollout.generate(RolloutRequest(samples=[_prompt([10, 11])]),
                              SamplingConfig(max_new_tokens=6, temperature=0)).samples[0]

    # The tool output is dropped rather than cut, and the trajectory keeps its last response.
    assert engine.module.generate.call_count == 1
    assert sample.tokens == [10, 11, _TOOL_CALL_TOKEN, 2]
    assert sample.loss_mask == [1, 1]
    assert sample.status == RolloutRecord.Status.TRUNCATED


class _LongEchoTool(Tool):
    name = "echo"
    description = "Returns ten letters."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, arguments):
        return "abcdefghij"


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.tokenize_tool_responses", return_value=[9])
def test_multi_turn_cuts_and_counts_tool_outputs_over_max_tool_output_chars(tokenize_tool_responses):
    engine, rollout = _multi_turn_rollout([[_TOOL_CALL_TOKEN, 2], [_TOOL_CALL_TOKEN, 2], [5, 2]],
                                          tools=ToolSet([_LongEchoTool()]))
    sampling = SamplingConfig(max_new_tokens=8, temperature=0, max_tool_output_chars=4, tool_output_keep="start")

    sample = rollout.generate(RolloutRequest(samples=[_prompt([10, 11])]), sampling).samples[0]

    assert [call.args[1] for call in tokenize_tool_responses.call_args_list] == [["abcd...(truncated)"]] * 2
    assert sample.num_tool_outputs_truncated == 2
    assert sample.num_turns == 3
    assert sample.status == RolloutRecord.Status.COMPLETED


@pytest.mark.parametrize("max_chars, keep, expected", [
    (10, "both", "abcdefghij"),
    (4, "start", "abcd...(truncated)"),
    (4, "end", "(truncated)...ghij"),
    (5, "both", "ab...(truncated)...hij"),
    (1, "both", "...(truncated)...j"),
])
def test_truncate_tool_output_keeps_the_chosen_ends(max_chars, keep, expected):
    assert truncate_tool_output("abcdefghij", max_chars, keep) == expected


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.tokenize_tool_responses", return_value=[9])
def test_multi_turn_shares_the_prefill_of_the_first_turn_only(_tokenize_tool_responses):
    # Copies of a prompt are identical only until their first responses, so later turns prefill every row.
    engine, rollout = _multi_turn_rollout([[_TOOL_CALL_TOKEN, 2], [_TOOL_CALL_TOKEN, 2], [5, 2], [5, 2]])
    rollout.use_shared_prefill = True
    engine.module.generate.side_effect = None
    outputs = iter([[[_TOOL_CALL_TOKEN, 2]] * 2, [[5, 2]] * 2])

    def generate(prompt_ids, **_kwargs):
        return torch.cat((prompt_ids, torch.tensor(next(outputs), device=prompt_ids.device)), dim=1)

    engine.module.generate.side_effect = generate

    sampling = SamplingConfig(max_new_tokens=8, temperature=0, n_samples_per_prompt=2)
    samples = rollout.generate(RolloutRequest(samples=[_prompt([10, 11])]), sampling)

    engine.prepare_shared_prefill.assert_called_once_with(1, 2, 2)
    assert [sample.status for sample in samples.samples] == [RolloutRecord.Status.COMPLETED] * 2


@pytest.mark.parametrize(("cfg_kwargs", "message"), [
    ({
        "multi_turn": True,
        "max_turns": 2
    }, "multi-turn rollout does not support ZeRO-3"),
    ({
        "use_graph_capture": True
    }, "graph capture does not support ZeRO-3"),
    ({
        "continuous_batch_size": 2
    }, "continuous batching does not support ZeRO-3"),
])
def test_rejects_zero3_where_ranks_would_run_different_forwards(cfg_kwargs, message):
    engine = _make_engine_on_cpu()
    engine.zero_optimization_partition_weights.return_value = True
    rollout = HybridEngineRollout(engine, _make_tokenizer(), HybridEngineRolloutConfig(**cfg_kwargs))

    with pytest.raises(ValueError, match=message):
        rollout.generate(_make_request(), _make_sampling())


def test_vocabulary_parallel_head_gathers_only_while_generating():
    """A tied head that AutoTP shards by vocabulary must give full logits to rollout, and rank-local ones to training."""

    class FailingModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(2, 4)
            self.lm_head.is_vocab_parallel_lm_head = True
            self.lm_head.gather_output = False
            self.gathered_while_generating = None

        def generate(self, *_args, **_kwargs):
            self.gathered_while_generating = self.lm_head.gather_output
            raise RuntimeError("generation failed")

    module = FailingModel()
    rollout = HybridEngineRollout(_engine_for(module), _make_tokenizer())

    with pytest.raises(RuntimeError, match="generation failed"):
        rollout.generate(_make_request(), _make_sampling())

    assert module.gathered_while_generating is True
    assert module.lm_head.gather_output is False


def test_rejects_hybrid_engine_inference_tensor_parallelism():
    engine = _make_engine()
    engine._config.hybrid_engine.inference_tp_size = 2
    rollout = HybridEngineRollout(engine, _make_tokenizer(), HybridEngineRolloutConfig())

    with pytest.raises(ValueError, match="does not support hybrid engine inference tensor parallelism"):
        rollout.generate(_make_request(), _make_sampling())


class _EchoTextTool(Tool):
    name = "echo"
    description = "Echoes its text."
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    def __init__(self):
        self.calls = 0

    async def execute(self, arguments):
        self.calls += 1
        return arguments["text"]


def test_tool_queue_takes_tool_output_from_the_first_tensor_parallel_rank():
    """Only the first rank runs the tools; the others get the output it sends, so the group stays in step."""
    sent = []

    class FakeDist:
        rank = 0

        def get_rank(self):
            return self.rank

        def broadcast(self, tensor, src, group=None):
            if self.rank == src:
                sent.append(tensor.clone())
            else:
                tensor.copy_(sent.pop(0))

    tool = _EchoTextTool()
    # Non-ASCII text and a lone surrogate must survive the trip unchanged.
    calls = [ToolCall(name="echo", arguments={"text": "é \ud800"})]
    fake = FakeDist()
    ready = {}
    with patch("deepspeed.runtime.rollout.tool.dist", fake):
        for rank in (0, 1):
            fake.rank = rank
            with ToolQueue(ToolSet([tool]), group=object(), src=0) as queue:
                queue.submit(0, calls)
                assert queue.busy
                queue.wait()
                ready[rank] = queue.pop_ready()
                assert not queue.busy
                assert queue.tool_ms > 0.0

    assert tool.calls == 1
    assert not sent
    assert ready[0] == ready[1] == [(0, calls, ["é \ud800"])]


class _SlowEchoTool(_EchoTextTool):

    async def execute(self, arguments):
        await asyncio.sleep(0.05 if arguments["text"] == "slow" else 0.0)
        return await super().execute(arguments)


def test_tool_queue_drain_waits_for_every_running_tool():
    """Static batching starts the next turn only once every trajectory's tools have returned."""
    tool = _SlowEchoTool()
    with ToolQueue(ToolSet([tool])) as queue:
        queue.submit(0, [ToolCall(name="echo", arguments={"text": "fast"})])
        queue.submit(1, [ToolCall(name="echo", arguments={"text": "slow"})])
        assert sorted((i, outputs) for i, _, outputs in queue.drain()) == [(0, ["fast"]), (1, ["slow"])]
        assert not queue.busy

    assert tool.calls == 2


def test_tool_queue_sends_only_a_size_when_no_tool_finished():
    sizes = []

    class FakeDist:

        def get_rank(self):
            return 1

        def broadcast(self, tensor, src, group=None):
            sizes.append(tensor.clone())
            tensor.zero_()

    with patch("deepspeed.runtime.rollout.tool.dist", FakeDist()):
        queue = ToolQueue(ToolSet([]), group=object(), src=0)
        assert queue.pop_ready() == []
    assert len(sizes) == 1


@pytest.mark.parametrize("max_new_tokens_per_turn, turn_budgets", [(None, [3, 1]), (2, [2, 1])])
@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.tokenize_tool_responses", return_value=[9])
def test_multi_turn_caps_each_turn_and_the_whole_response(_tokenize_tool_responses, max_new_tokens_per_turn,
                                                          turn_budgets):
    engine, rollout = _multi_turn_rollout([[_TOOL_CALL_TOKEN, 2], [5, 6, 2][:turn_budgets[1]]])
    sampling = SamplingConfig(max_new_tokens=3, max_new_tokens_per_turn=max_new_tokens_per_turn, temperature=0)

    sample = rollout.generate(RolloutRequest(samples=[_prompt([10, 11])]), sampling).samples[0]

    # The second turn gets what is left of max_new_tokens (3 - 2), whatever the per-turn cap.
    assert [call.kwargs["max_new_tokens"] for call in engine.module.generate.call_args_list] == turn_budgets
    assert sample.tokens == [10, 11, _TOOL_CALL_TOKEN, 2, 9, 5]
    assert sample.loss_mask == [1, 1, 0, 1]
    assert sample.num_turns == 2
    assert sample.status == RolloutRecord.Status.TRUNCATED


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.tokenize_tool_responses", return_value=[9])
def test_multi_turn_per_turn_cap_resets_every_turn(_tokenize_tool_responses):
    # The per-turn cap applies to each turn on its own, so the whole response may grow beyond it.
    engine, rollout = _multi_turn_rollout([[_TOOL_CALL_TOKEN, 2], [_TOOL_CALL_TOKEN, 2], [5, 2]])
    sampling = SamplingConfig(max_new_tokens=100, max_new_tokens_per_turn=2, temperature=0)

    sample = rollout.generate(RolloutRequest(samples=[_prompt([10, 11])]), sampling).samples[0]

    assert [call.kwargs["max_new_tokens"] for call in engine.module.generate.call_args_list] == [2, 2, 2]
    assert sample.response_length == 8
    assert sample.status == RolloutRecord.Status.COMPLETED


class _TypedTool(Tool):
    name = "typed"
    description = "Takes one argument of each schema type."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string"
            },
            "count": {
                "type": "integer"
            },
            "scale": {
                "type": "number"
            },
            "flag": {
                "type": "boolean"
            },
            "options": {
                "type": "object"
            },
            "items": {
                "type": "array"
            },
            "limit": {
                "type": ["integer", "null"]
            },
        },
    }

    def __init__(self):
        self.calls = []

    async def execute(self, arguments):
        self.calls.append(arguments)
        return "ok"


# Tool calls as the Qwen3.5 chat template renders them, with each value on lines of its own.
_QWEN3_CODER_TYPED_CALL = ("I will run it.\n\n<tool_call>\n<function=typed>\n"
                           "<parameter=code>\nx = 42\nprint(x)\n</parameter>\n"
                           "<parameter=count>\n3\n</parameter>\n"
                           "<parameter=scale>\n0.5\n</parameter>\n"
                           "<parameter=flag>\ntrue\n</parameter>\n"
                           '<parameter=options>\n{"mode": "fast"}\n</parameter>\n'
                           "<parameter=items>\n[1, 2]\n</parameter>\n"
                           "<parameter=limit>\nnull\n</parameter>\n"
                           "</function>\n</tool_call>")


def test_qwen3_coder_parser_converts_arguments_to_their_schema_types():
    calls = parse_qwen3_coder_tool_calls(_QWEN3_CODER_TYPED_CALL, ToolSet([_TypedTool()]))

    assert calls == [
        ToolCall(name="typed",
                 arguments={
                     "code": "x = 42\nprint(x)",
                     "count": 3,
                     "scale": 0.5,
                     "flag": True,
                     "options": {
                         "mode": "fast"
                     },
                     "items": [1, 2],
                     "limit": None,
                 })
    ]


@pytest.mark.parametrize(("text", "expected"), [("true", True), ("False", False), ("yes", "yes"), ("1", "1")])
def test_qwen3_coder_parser_converts_only_true_and_false_to_booleans(text, expected):
    call = f"<tool_call>\n<function=typed>\n<parameter=flag>\n{text}\n</parameter>\n</function>\n</tool_call>"

    calls = parse_qwen3_coder_tool_calls(call, ToolSet([_TypedTool()]))

    assert calls == [ToolCall(name="typed", arguments={"flag": expected})]


def test_qwen3_coder_parser_keeps_text_where_the_schema_does_not_say_otherwise():
    text = ("<tool_call>\n<function=typed>\n<parameter=code>\n42\n</parameter>\n"
            "<parameter=count>\nthree\n</parameter>\n<parameter=extra>\n7\n</parameter>\n</function>\n</tool_call>")

    calls = parse_qwen3_coder_tool_calls(text, ToolSet([_TypedTool()]))

    # A string parameter that looks like a number stays a string; so do unconvertible and undeclared values.
    assert calls == [ToolCall(name="typed", arguments={"code": "42", "count": "three", "extra": "7"})]


def test_qwen3_coder_parser_reads_several_calls_in_one_turn():
    text = ("<tool_call>\n<function=echo>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=typed>\n<parameter=count>\n1\n</parameter>\n</function>\n</tool_call>")

    calls = parse_qwen3_coder_tool_calls(text, ToolSet([_EchoTool(), _TypedTool()]))

    assert calls == [ToolCall(name="echo"), ToolCall(name="typed", arguments={"count": 1})]


@pytest.mark.parametrize(
    "text",
    [
        "<tool_call>\nno function here\n</tool_call>",
        "<tool_call>\n<function=typed>\n<parameter=code>\nx\n</parameter>\n</tool_call>",
        "<tool_call>\n<function=typed>\n<parameter=code>\nx\n</function>\n</tool_call>",
        "<tool_call>\n<function=typed>\n<parameter=code>\nx",
    ],
    ids=["no_function", "unclosed_function", "unclosed_parameter", "cut_off"],
)
def test_qwen3_coder_parser_reports_malformed_calls(text):
    calls = parse_qwen3_coder_tool_calls(text, ToolSet([_TypedTool()]))

    # The model is told what went wrong rather than the call disappearing.
    assert len(calls) == 1
    assert calls[0].error is not None


def test_qwen3_coder_parser_reads_the_qwen3_5_chat_template():
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen/Qwen3.5-4B tokenizer is not in the local Hugging Face cache")
    arguments = {"code": "for i in range(3):\n    print(i)", "count": 2, "options": {"mode": "fast"}}
    tool_call = {"type": "function", "function": {"name": "typed", "arguments": arguments}}
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "", "tool_calls": [tool_call]}]

    rendered = tokenizer.apply_chat_template(messages, tokenize=False)

    assert parse_qwen3_coder_tool_calls(rendered, ToolSet([_TypedTool()])) == [ToolCall("typed", arguments)]


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.tokenize_tool_responses", return_value=[9])
def test_multi_turn_runs_tools_called_in_the_configured_format(tokenize_tool_responses):
    tool = _TypedTool()
    engine, rollout = _multi_turn_rollout([[_TOOL_CALL_TOKEN, 2], [5, 2]],
                                          call_text=_QWEN3_CODER_TYPED_CALL,
                                          tools=ToolSet([tool]),
                                          tool_call_parser="qwen3_coder")

    sample = rollout.generate(RolloutRequest(samples=[_prompt([10, 11])]),
                              SamplingConfig(max_new_tokens=8, temperature=0)).samples[0]

    assert [call["count"] for call in tool.calls] == [3]
    calls, outputs = tokenize_tool_responses.call_args.args[:2]
    assert [(call.name, output) for call, output in zip(calls, outputs)] == [("typed", "ok")]
    assert sample.tokens == [10, 11, _TOOL_CALL_TOKEN, 2, 9, 5, 2]
    assert sample.status == RolloutRecord.Status.COMPLETED


def test_single_turn_applies_the_per_turn_cap():
    engine = _make_engine()
    engine.module.generate.return_value = torch.tensor([[10, 11, 5, 6]])
    rollout = HybridEngineRollout(engine, _make_tokenizer())
    sampling = SamplingConfig(max_new_tokens=8, max_new_tokens_per_turn=2, temperature=0)

    rollout.generate(RolloutRequest(samples=[_prompt([10, 11])]), sampling)

    assert engine.module.generate.call_args.kwargs["max_new_tokens"] == 2


def test_generate_keeps_ranks_in_lockstep_and_pads_after_eos():
    engine = _make_engine()
    # ZeRO-3 gathers parameters in every decode step, so ranks must not stop at EOS.
    engine.zero_optimization_partition_weights.return_value = True
    rollout = HybridEngineRollout(engine, _make_tokenizer())
    engine.module.generate.return_value = torch.tensor([[10, 11, 5, 2, 7, 8]])
    request = RolloutRequest(samples=[_prompt([10, 11])])

    result = rollout.generate(request, SamplingConfig(max_new_tokens=4, temperature=0))

    kwargs = engine.module.generate.call_args.kwargs
    assert kwargs['eos_token_id'] is None
    assert kwargs['max_new_tokens'] == 4
    assert kwargs['stopping_criteria'][0].row_budgets.tolist() == [4]
    sample = result.samples[0]
    assert sample.tokens == [10, 11, 5, 2]
    assert sample.attention_mask == [1, 1, 1, 1]
    assert sample.loss_mask == [1, 1]
    assert sample.response_length == 2
    assert sample.status == RolloutRecord.Status.COMPLETED


def test_get_eos_token_ids_merges_the_tokenizer_and_generation_config():
    engine = _make_engine()
    tok = _make_tokenizer()
    rollout = HybridEngineRollout(engine, tok)
    # A mocked generation config has no real EOS ids.
    assert get_eos_token_ids(tok, engine.module) == [2]

    engine.module.generation_config = SimpleNamespace(eos_token_id=[2, 8, 9])
    assert get_eos_token_ids(tok, engine.module) == [2, 8, 9]

    tok.eos_token_id = [9, 3]
    engine.module.generation_config = SimpleNamespace(eos_token_id=8)
    assert get_eos_token_ids(tok, engine.module) == [9, 3, 8]

    tok.eos_token_id = None
    engine.module.generation_config = SimpleNamespace(eos_token_id=None)
    assert get_eos_token_ids(tok, engine.module) == []
    assert rollout._eos_tensor("cpu").tolist() == [-1]


@pytest.mark.parametrize("tokenizer_eos, config_eos", [([2, 3], None), (2, [2, 3])])
def test_static_generation_ends_turns_on_every_eos_id(tokenizer_eos, config_eos):
    engine = _make_engine()
    engine.module.generation_config = SimpleNamespace(eos_token_id=config_eos)
    tok = _make_tokenizer()
    tok.eos_token_id = tokenizer_eos
    cfg = HybridEngineRolloutConfig(multi_turn=True, max_turns=1)
    rollout = HybridEngineRollout(engine, tok, cfg=cfg, tools=ToolSet([_EchoTool()]))
    # The first row ends in the second EOS id, then runs on; the second never ends.
    engine.module.generate.return_value = torch.tensor([[10, 11, 5, 3, 7], [10, 11, 6, 7, 8]])
    request = RolloutRequest(samples=[_prompt([10, 11]), _prompt([10, 11])])

    with patch.object(rollout, "parse_tool_calls", return_value=[]) as parse_tool_calls:
        result = rollout.generate(request, SamplingConfig(max_new_tokens=3, temperature=0))

    assert engine.module.generate.call_args.kwargs["eos_token_id"] == [2, 3]
    parse_tool_calls.assert_called_once()
    assert [sample.tokens for sample in result.samples] == [[10, 11, 5, 3], [10, 11, 6, 7, 8]]
    assert [sample.status for sample in result.samples] == [
        RolloutRecord.Status.COMPLETED,
        RolloutRecord.Status.TRUNCATED,
    ]


def _tool_template_tokenizer(assistant_end):
    """A tokenizer whose template ends the user turn with <eot> and the tool-call turn with ``assistant_end``."""
    tok = MagicMock()

    def render(messages, add_generation_prompt=False, **_kwargs):
        text = f"<u>dummy<eot>\n<a>call{assistant_end}\n"
        if len(messages) > 2:
            text += "<t>out<eot>\n"
        if add_generation_prompt:
            text += "<a>"
        return text

    tok.apply_chat_template.side_effect = render
    tok.convert_ids_to_tokens.side_effect = lambda ids: [{2: "<eot>", 3: "<eom>"}[i] for i in ids]
    tok.encode.side_effect = lambda text, **_kwargs: list(text)
    return tok


def test_tokenize_tool_responses_joins_after_the_last_eos_token_of_any_kind():
    ids = tokenize_tool_responses([ToolCall(name="echo")], ["out"], _tool_template_tokenizer("<eom>"), [2, 3])

    assert "".join(ids) == "\n<t>out<eot>\n<a>"


def test_tokenize_tool_responses_requires_an_eos_token_in_the_template():
    with pytest.raises(ValueError, match="without an EOS token"):
        tokenize_tool_responses([ToolCall(name="echo")], ["out"], _tool_template_tokenizer("<end>"), [3])


def test_matches_eos_checks_membership_in_every_eos_id():
    eos_ids = torch.tensor([2, 3])

    assert HybridEngineRollout._matches_eos(torch.tensor([1, 2, 3, 4]), eos_ids).tolist() == [False, True, True, False]


def test_generate_forwards_top_k_for_sampling_with_multiple_samples_and_eos():
    engine = _make_engine()
    tok = _make_tokenizer()
    rollout = HybridEngineRollout(engine, tok)
    engine.module.generate.return_value = torch.tensor([
        [10, 11, 5, 2, 7, 8],
        [10, 11, 6, 7, 2, 8],
    ])
    request = RolloutRequest(samples=[_prompt([10, 11])])
    sampling = SamplingConfig(max_new_tokens=4, temperature=0.7, top_p=0.8, top_k=5, n_samples_per_prompt=2)

    result = rollout.generate(request, sampling)

    generate_call = engine.module.generate.call_args
    assert generate_call.args[0].tolist() == [[10, 11], [10, 11]]
    assert generate_call.kwargs["attention_mask"].tolist() == [[1, 1], [1, 1]]
    assert generate_call.kwargs["do_sample"] is True
    assert generate_call.kwargs["temperature"] == 0.7
    assert generate_call.kwargs["top_p"] == 0.8
    assert generate_call.kwargs["top_k"] == 5
    assert generate_call.kwargs["eos_token_id"] == [tok.eos_token_id]
    assert [sample.tokens for sample in result.samples] == [[10, 11, 5, 2], [10, 11, 6, 7, 2]]
    assert [sample.loss_mask for sample in result.samples] == [[1, 1], [1, 1, 1]]
    assert [sample.group_index for sample in result.samples] == [0, 0]
    assert [sample.status for sample in result.samples] == [RolloutRecord.Status.COMPLETED] * 2


@pytest.mark.parametrize("top_k", [-1, 0])
def test_generate_disables_top_k_for_sampling(top_k):
    engine = _make_engine()
    rollout = HybridEngineRollout(engine, _make_tokenizer())
    engine.module.generate.return_value = torch.tensor([[10, 11, 5]])
    request = RolloutRequest(samples=[_prompt([10, 11])])

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


def test_pad_after_eos_keeps_generated_pad_ids_before_eos():
    # A model can sample the pad id; dropping it would shift every later token and its log-prob.
    output_ids = torch.tensor([[10, 11, 0, 5, 2, 7]])

    padded, response_attn = HybridEngineRollout._pad_after_eos(output_ids, 2, eos_token_id=2, pad_token_id=0)

    assert padded.tolist() == [[10, 11, 0, 5, 2, 0]]
    assert response_attn.tolist() == [[1, 1, 1, 0]]


def test_pad_after_eos_supports_multiple_eos_ids():
    output_ids = torch.tensor([[10, 11, 5, 3, 7, 2]])

    padded, response_attn = HybridEngineRollout._pad_after_eos(output_ids, 2, eos_token_id=[2, 3], pad_token_id=0)

    assert padded.tolist() == [[10, 11, 5, 3, 0, 0]]
    assert response_attn.tolist() == [[1, 1, 0, 0]]


def test_generate_accepts_zero_pad_token_id():
    engine = _make_engine()
    rollout = HybridEngineRollout(engine, _make_tokenizer())
    engine.module.generate.return_value = torch.tensor([[10, 11, 5, 6]])
    request = RolloutRequest(samples=[_prompt([10, 11])])

    result = rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0))

    assert engine.module.generate.call_args.kwargs['pad_token_id'] == 0
    assert result.samples[0].tokens == [10, 11, 5, 6]
    # No EOS within max_new_tokens.
    assert result.samples[0].status == RolloutRecord.Status.TRUNCATED


def test_single_turn_records_log_probs_of_the_sampled_distribution():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64,
                         hidden_size=32,
                         intermediate_size=64,
                         num_hidden_layers=1,
                         num_attention_heads=2,
                         num_key_value_heads=1,
                         max_position_embeddings=64)
    # The rollout places prompts on the accelerator's current device, where the policy lives.
    device = torch.device(get_accelerator().current_device_name())
    model = LlamaForCausalLM(config).to(device).eval()
    engine = _engine_for(model)
    rollout = HybridEngineRollout(engine, _make_tokenizer(), cfg=HybridEngineRolloutConfig(return_log_probs=True))
    request = RolloutRequest(samples=[_prompt([5, 6, 7]), _prompt([0, 8, 9], [0, 1, 1])])
    sampling = SamplingConfig(max_new_tokens=4, temperature=0.7, n_samples_per_prompt=2)

    samples = rollout.generate(request, sampling).samples

    assert [sample.group_index for sample in samples] == [0, 0, 1, 1]
    for sample in samples:
        assert len(sample.rollout_log_probs) == len(sample.loss_mask) == sample.response_length > 0
        # The reference is a full forward over the sample's real tokens, scored at the sampling temperature.
        ids = torch.tensor([token for token, keep in zip(sample.tokens, sample.attention_mask) if keep], device=device)
        with torch.no_grad():
            logits = model(ids[None]).logits[0].float() / sampling.temperature
        response = ids[-sample.response_length:]
        expected = torch.log_softmax(logits[-sample.response_length - 1:-1], dim=-1).gather(1, response[:, None])
        torch.testing.assert_close(torch.tensor(sample.rollout_log_probs),
                                   expected.squeeze(1).cpu(),
                                   atol=1e-4,
                                   rtol=1e-4)


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


def test_rollout_requires_an_explicit_max_out_tokens():
    # The hybrid engine's default would silently cap every trajectory.
    engine = _make_engine()
    engine._config.hybrid_engine = HybridEngineConfig()
    with pytest.raises(ValueError, match="hybrid_engine.max_out_tokens"):
        HybridEngineRollout(engine, _make_tokenizer())


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.get_accelerator")
def test_static_graph_capture_receives_a_prebuilt_4d_mask(mock_get_accelerator):
    """The captured decode forward gets a pre-built 4-D mask that shows the prompt and the positions decoded so far.

    A 2-D mask would make Transformers run host-side checks, which invalidate graph capture.
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
    accelerator.current_device_name.return_value = "cpu"
    accelerator.Stream.return_value = stream
    accelerator.current_stream.return_value = stream
    accelerator.stream.side_effect = lambda _stream: nullcontext()
    accelerator.capture_to_graph.side_effect = capture
    accelerator.replay_graph.side_effect = lambda _graph: replay_views.append(
        masked_columns(module.captured_masks[0]).clone())

    rollout = HybridEngineRollout(_engine_for(module),
                                  _make_tokenizer(),
                                  cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    request = RolloutRequest(samples=[_prompt([1, 3, 4], [0, 1, 1]), _prompt([5, 6, 7])])
    rollout.generate(request, SamplingConfig(max_new_tokens=max_new_tokens, temperature=0))

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


def test_static_graph_rejects_flash_attention(graph_capture_accelerator):
    module = torch.nn.Linear(1, 1)
    module.config = SimpleNamespace(_attn_implementation="flash_attention_2")
    rollout = HybridEngineRollout(_engine_for(module),
                                  _make_tokenizer(),
                                  cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    request = RolloutRequest(samples=[_prompt([1, 2])])

    with pytest.raises(ValueError, match="does not support flash_attention_2"):
        rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0))

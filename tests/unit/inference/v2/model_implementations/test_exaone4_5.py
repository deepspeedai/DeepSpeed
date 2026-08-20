# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers", minversion="5.3.0")

from transformers import Exaone4Config
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from deepspeed.inference.v2.model_implementations.exaone4_5 import model as exaone4_5_model
from deepspeed.inference.v2.model_implementations.exaone4_5.model import (
    Exaone4_5InferenceModel,
    _llama3_inverse_frequencies,
    _supported_sequence_length,
)
from deepspeed.inference.v2.ragged import PlaceholderSequenceDescriptor


def _model_config() -> Exaone4Config:
    return Exaone4Config(
        hidden_size=5120,
        num_hidden_layers=4,
        num_attention_heads=40,
        num_key_value_heads=8,
        max_position_embeddings=262144,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        sliding_window=4096,
        rope_parameters={
            "factor": 16.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_theta": 1000000.0,
            "rope_type": "llama3",
        },
    )


def test_llama3_inverse_frequencies_match_transformers() -> None:
    config = _model_config()

    expected, attention_factor = ROPE_INIT_FUNCTIONS["llama3"](config, device=torch.device("cpu"))
    actual = _llama3_inverse_frequencies(config, torch.device("cpu"))

    assert attention_factor == 1.0
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_attention_uses_supplied_inverse_frequencies() -> None:
    rope_config = Exaone4_5InferenceModel.positional_embedding_config.fget(SimpleNamespace())

    assert rope_config.use_trained_freqs


def test_model_retains_inverse_frequencies_in_fp32(monkeypatch) -> None:

    def initialize_base_model(model, config, engine_config, base_mp_group) -> None:
        torch.nn.Module.__init__(model)
        model._config = config
        model._engine_config = engine_config

    monkeypatch.setattr(exaone4_5_model.Exaone4InferenceModel, "__init__", initialize_base_model)
    monkeypatch.setattr(Exaone4_5InferenceModel, "_build_global_attention", lambda model: object())
    monkeypatch.setattr(
        exaone4_5_model,
        "get_accelerator",
        lambda: SimpleNamespace(current_device=lambda: torch.device("cpu")),
    )

    model = Exaone4_5InferenceModel(_model_config(), SimpleNamespace(), None)

    assert model._rope_inv_freqs.dtype == torch.float32


def test_sequence_length_is_capped_to_sliding_window() -> None:
    assert _supported_sequence_length(_model_config()) == 4096


def test_kv_requirements_stop_at_sequence_length_cap() -> None:
    model = Exaone4_5InferenceModel.__new__(Exaone4_5InferenceModel)
    torch.nn.Module.__init__(model)
    model._config = _model_config()
    model.attn = SimpleNamespace(kv_block_size=64)
    sequence = PlaceholderSequenceDescriptor(seen_tokens=4000, cur_allocated_blocks=63)

    schedulable_tokens, required_blocks = model.get_kv_requirements(sequence, max_new_tokens=128, max_new_blocks=10)

    assert schedulable_tokens == 96
    assert required_blocks == 1


class _AttentionRecorder:

    def __init__(self, result: str) -> None:
        self.result = result
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


def test_attention_dispatches_by_layer_type() -> None:
    sliding_attn = _AttentionRecorder("sliding")
    global_attn = _AttentionRecorder("global")
    inv_freqs = object()
    model = SimpleNamespace(
        _layer_types=("sliding_attention", "full_attention"),
        attn=sliding_attn,
        _global_attn=global_attn,
        _rope_inv_freqs=inv_freqs,
    )
    qkv, kv_cache, batch = object(), object(), object()

    assert Exaone4_5InferenceModel._forward_attention(model, 0, qkv, kv_cache, batch) == "sliding"
    assert Exaone4_5InferenceModel._forward_attention(model, 1, qkv, kv_cache, batch) == "global"
    assert sliding_attn.calls == [(qkv, kv_cache, batch, inv_freqs)]
    assert global_attn.calls == [(qkv, kv_cache, batch)]

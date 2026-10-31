# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import importlib.util
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

import deepspeed.runtime.hybrid_engine as hybrid_engine
from deepspeed.runtime.hybrid_engine import DeepSpeedHybridEngine
from deepspeed.module_inject.layers import LinearLayer, OPTEmbedding


class SupportedLayer(nn.Module):
    pass


class UnsupportedLayer(nn.Module):
    pass


class SupportedPolicy:
    _orig_layer_class = SupportedLayer

    def __init__(self, client_module, inference=False):
        pass


def _make_engine(module):
    engine = DeepSpeedHybridEngine.__new__(DeepSpeedHybridEngine)
    object.__setattr__(engine, 'module', module)
    return engine


def test_hybrid_engine_import_does_not_resolve_transformers_opt(monkeypatch):
    from transformers.models import opt

    class TransformersOPTImportGuard:

        def __getattr__(self, name):
            raise AssertionError(f'Hybrid Engine resolved transformers.models.opt.modeling_opt.{name} during import')

    monkeypatch.setattr(opt, 'modeling_opt', TransformersOPTImportGuard(), raising=False)
    spec = importlib.util.spec_from_file_location('deepspeed.runtime._hybrid_engine_import_test',
                                                  hybrid_engine.__file__)
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)


def test_unsupported_model_uses_native_fallback(monkeypatch):
    monkeypatch.setattr(hybrid_engine, 'replace_policies', [SupportedPolicy])
    engine = _make_engine(nn.Sequential(UnsupportedLayer(), nn.Linear(2, 2), nn.LayerNorm(2)))

    with patch.object(hybrid_engine.logger, 'warning') as mock_warning:
        engine.populate_all_inference_policies()

    assert engine.inference_policies == {}
    mock_warning.assert_called_once()
    assert "Hybrid Engine inference acceleration is unavailable" in mock_warning.call_args.args[0]
    assert mock_warning.call_args.args[1] == "Sequential"


def test_supported_model_registers_auxiliary_policies(monkeypatch):
    from transformers.models.opt.modeling_opt import OPTLearnedPositionalEmbedding

    monkeypatch.setattr(hybrid_engine, 'replace_policies', [SupportedPolicy, hybrid_engine.HFOPTLayerPolicy])
    engine = _make_engine(nn.Sequential(SupportedLayer(), nn.Linear(2, 2)))

    engine.populate_all_inference_policies()

    assert SupportedLayer in engine.inference_policies
    assert engine.inference_policies[nn.Linear][0] is LinearLayer
    assert engine.inference_policies[OPTLearnedPositionalEmbedding] == (OPTEmbedding, )


def test_supported_model_without_transformers_registers_auxiliary_policies(monkeypatch):
    if importlib.util.find_spec('transformers') is not None:
        pytest.skip('Requires an environment without Transformers')

    monkeypatch.setattr(hybrid_engine, 'replace_policies', [SupportedPolicy, hybrid_engine.HFOPTLayerPolicy])
    engine = _make_engine(nn.Sequential(SupportedLayer(), nn.Linear(2, 2)))

    engine.populate_all_inference_policies()

    assert SupportedLayer in engine.inference_policies
    assert engine.inference_policies[nn.Linear][0] is LinearLayer
    assert all(policy.__name__ != 'OPTLearnedPositionalEmbedding' for policy in engine.inference_policies)


def test_transformers_checkpoint_roundtrip(tmp_path):
    from transformers import BertConfig, BertForSequenceClassification

    torch.manual_seed(1234)
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        ))
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = model(input_ids).logits

    checkpoint = tmp_path / 'ordinary_transformers_model.pt'
    torch.save({'model': model, 'input_ids': input_ids, 'expected': expected}, checkpoint)

    restored = torch.load(checkpoint, map_location='cpu', weights_only=False)
    restored['model'].eval()
    with torch.inference_mode():
        actual = restored['model'](restored['input_ids']).logits
    torch.testing.assert_close(actual, restored['expected'])


def test_modern_opt_uses_native_fallback():
    import inspect
    from transformers import OPTConfig, OPTForCausalLM
    from transformers.models.opt.modeling_opt import OPTDecoderLayer

    parameters = inspect.signature(OPTDecoderLayer.forward).parameters
    if not ({'cache_position', 'past_key_values'} & parameters.keys()):
        pytest.skip('Installed OPT uses the supported legacy cache contract')
    model = OPTForCausalLM(
        OPTConfig(hidden_size=16, ffn_dim=32, num_hidden_layers=1, num_attention_heads=2, vocab_size=32))
    engine = _make_engine(model)
    engine.populate_all_inference_policies()
    assert engine.inference_policies == {}

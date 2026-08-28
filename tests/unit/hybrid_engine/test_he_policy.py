# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from unittest.mock import patch

import torch.nn as nn

import deepspeed.runtime.hybrid_engine as hybrid_engine
from deepspeed.runtime.hybrid_engine import DeepSpeedHybridEngine
from deepspeed.module_inject.layers import LinearLayer


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
    monkeypatch.setattr(hybrid_engine, 'replace_policies', [SupportedPolicy])
    engine = _make_engine(nn.Sequential(SupportedLayer(), nn.Linear(2, 2)))

    engine.populate_all_inference_policies()

    assert SupportedLayer in engine.inference_policies
    assert engine.inference_policies[nn.Linear][0] is LinearLayer

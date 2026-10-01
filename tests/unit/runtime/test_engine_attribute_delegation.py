# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import pytest
import torch

from deepspeed.runtime.engine import DeepSpeedEngine


def _make_engine():
    engine = DeepSpeedEngine.__new__(DeepSpeedEngine)
    torch.nn.Module.__init__(engine)
    engine.module = torch.nn.Module()
    engine.module.model_attribute = object()
    engine.workspace = torch.nn.Module()
    return engine


def test_getattr_resolves_registered_workspace_module():
    engine = _make_engine()

    assert engine.workspace is engine._modules["workspace"]


def test_getattr_delegates_missing_attributes_to_model():
    engine = _make_engine()

    assert engine.model_attribute is engine.module.model_attribute


def test_getattr_raises_for_missing_attribute():
    engine = _make_engine()

    with pytest.raises(AttributeError) as exc_info:
        _ = engine.missing_attribute

    assert str(exc_info.value) == "'DeepSpeedEngine' object has no attribute 'missing_attribute'"

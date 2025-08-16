# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.pipe.engine import PipelineEngine


# Silence destructors because instances are created via __new__ (no init)
@pytest.fixture(autouse=True)
def _silence_engine_destructors(monkeypatch):
    monkeypatch.setattr(DeepSpeedEngine, "__del__", lambda self: None, raising=False)
    monkeypatch.setattr(PipelineEngine, "__del__", lambda self: None, raising=False)
    monkeypatch.setattr(DeepSpeedEngine, "destroy", lambda self: None, raising=False)
    monkeypatch.setattr(PipelineEngine, "destroy", lambda self: None, raising=False)


# Skip if methods are absent (e.g., running against an older DS build)
if (not hasattr(DeepSpeedEngine, "get_parallel_world_sizes")
        or not hasattr(PipelineEngine, "get_parallel_world_sizes")):
    pytest.skip("Required methods missing on this DeepSpeed build.", allow_module_level=True)


def _patch_groups(monkeypatch, dp=8, tp=4):
    """Patch deepspeed.utils.groups to avoid initializing any distributed backend."""
    import deepspeed.utils.groups as groups
    monkeypatch.setattr(groups, "get_data_parallel_world_size", lambda: dp, raising=True)
    monkeypatch.setattr(groups, "get_tensor_model_parallel_world_size", lambda: tp, raising=True)


def _make_engine():
    """Create engine without running __init__ to avoid side effects."""
    return DeepSpeedEngine.__new__(DeepSpeedEngine)


def _make_pipeline_engine(num_stages=6):
    """Create pipeline engine without init; set the minimal required attribute."""
    pe = PipelineEngine.__new__(PipelineEngine)
    pe.num_stages = num_stages
    return pe


def test_deepspeedengine_get_parallel_world_sizes(monkeypatch):
    _patch_groups(monkeypatch, dp=8, tp=4)
    eng = _make_engine()
    assert eng.get_parallel_world_sizes() == {"dp": 8, "tp": 4}


def test_pipelineengine_get_parallel_world_sizes(monkeypatch):
    _patch_groups(monkeypatch, dp=8, tp=4)
    peng = _make_pipeline_engine(num_stages=6)
    assert peng.get_parallel_world_sizes() == {"dp": 8, "tp": 4, "pp": 6}

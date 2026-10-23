# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Unit tests for the per-hardware triton_ops implementation dispatch.

``resolve_triton_ops_impl`` asks the accelerator for its triton_ops subpackage
(op_builder-style): an accelerator without one resolves the portable module, an
accelerator with one resolves its own module, and a feature the accelerator
does not override falls back to the portable module. The accelerator is a
duck-typed double, so these tests run on any host without a device.
"""

import pytest

import deepspeed.accelerator
from deepspeed.ops.triton_ops._triton import _TRITON_OPS_IMPL_CACHE, resolve_triton_ops_impl


class _FakeAccelerator:
    """Duck-typed accelerator double: only the dispatch hook is needed."""

    def __init__(self, subpackage=None):
        self._subpackage = subpackage

    def triton_ops_dir(self):
        return self._subpackage


@pytest.fixture(autouse=True)
def _clear_resolver_cache():
    # The resolver caches per feature, and each test pins a different
    # accelerator, so the cache is cleared between tests.
    _TRITON_OPS_IMPL_CACHE.clear()
    yield


def test_no_subpackage_resolves_the_portable_module(monkeypatch):
    monkeypatch.setattr(deepspeed.accelerator, "get_accelerator", lambda: _FakeAccelerator())
    impl = resolve_triton_ops_impl("swiglu_triton")
    assert impl.__name__ == "deepspeed.ops.triton_ops.swiglu_triton"


def test_platform_subpackage_resolves_the_platform_module(monkeypatch):
    monkeypatch.setattr(deepspeed.accelerator, "get_accelerator",
                        lambda: _FakeAccelerator("deepspeed.ops.triton_ops.npu"))
    impl = resolve_triton_ops_impl("swiglu_triton")
    assert impl.__name__ == "deepspeed.ops.triton_ops.npu.swiglu_triton"


def test_unoverridden_feature_falls_back_to_the_portable_module(monkeypatch):
    monkeypatch.setattr(deepspeed.accelerator, "get_accelerator",
                        lambda: _FakeAccelerator("deepspeed.ops.triton_ops.npu"))
    impl = resolve_triton_ops_impl("fused_rms_norm")
    assert impl.__name__ == "deepspeed.ops.triton_ops.fused_rms_norm"


def test_missing_feature_raises_after_the_platform_fallback(monkeypatch):
    monkeypatch.setattr(deepspeed.accelerator, "get_accelerator",
                        lambda: _FakeAccelerator("deepspeed.ops.triton_ops.npu"))
    with pytest.raises(ModuleNotFoundError):
        resolve_triton_ops_impl("no_such_feature")

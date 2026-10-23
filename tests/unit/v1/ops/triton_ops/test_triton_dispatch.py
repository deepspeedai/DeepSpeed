# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Unit tests for the per-hardware triton_ops implementation dispatch.

``resolve_triton_ops_impl`` asks the accelerator for its triton_ops subpackage
(op_builder-style): an accelerator without one resolves the portable module, an
accelerator with one resolves its own module, and a feature the accelerator
does not override falls back to the portable module. The ``dispatch`` decorator
built on top of it runs the platform's same-named function when it exists and
the portable original otherwise; a platform module that merely re-exports the
decorated portable function counts as "no override" and never recurses. The
accelerator is a duck-typed double and the decorator cases pin the resolver
cache directly, so these tests run on any host without a device.
"""

import types

import pytest

import deepspeed.accelerator
from deepspeed.ops.triton_ops._triton import _TRITON_OPS_IMPL_CACHE, dispatch, resolve_triton_ops_impl


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
    impl = resolve_triton_ops_impl("group_gemm_triton")
    assert impl.__name__ == "deepspeed.ops.triton_ops.group_gemm_triton"


def test_missing_feature_raises_after_the_platform_fallback(monkeypatch):
    monkeypatch.setattr(deepspeed.accelerator, "get_accelerator",
                        lambda: _FakeAccelerator("deepspeed.ops.triton_ops.npu"))
    with pytest.raises(ModuleNotFoundError):
        resolve_triton_ops_impl("no_such_feature")


def test_out_of_package_feature_resolves_its_portable_module(monkeypatch):
    # Triton kernels living outside triton_ops (MoE permute, chunked CE, fp8
    # GEMM) pass their portable module explicitly; the platform variant still
    # lives under the accelerator's triton_ops subpackage.
    monkeypatch.setattr(deepspeed.accelerator, "get_accelerator",
                        lambda: _FakeAccelerator("deepspeed.ops.triton_ops.npu"))
    impl = resolve_triton_ops_impl("ep_kernels", portable_module="deepspeed.moe.ep_kernels")
    assert impl.__name__ == "deepspeed.moe.ep_kernels"


# --- The dispatch decorator ---
# The resolver cache is pinned per test, so the decorator cases never import a
# real module and run on any host without a device.

_FAKE_FEATURE = "_dispatch_test_feature"


def test_dispatch_calls_the_platform_override():
    calls = []

    def my_op(value):
        calls.append("portable")
        return f"portable:{value}"

    def platform_op(value):
        calls.append("platform")
        return f"platform:{value}"

    _TRITON_OPS_IMPL_CACHE[_FAKE_FEATURE] = types.SimpleNamespace(my_op=platform_op)
    decorated = dispatch(_FAKE_FEATURE)(my_op)
    assert decorated(1) == "platform:1"
    assert calls == ["platform"]


def test_dispatch_falls_back_to_the_original_without_an_override():
    calls = []

    def my_op(value):
        calls.append("portable")
        return f"portable:{value}"

    _TRITON_OPS_IMPL_CACHE[_FAKE_FEATURE] = types.SimpleNamespace()  # no my_op
    decorated = dispatch(_FAKE_FEATURE)(my_op)
    assert decorated(1) == "portable:1"
    assert calls == ["portable"]


def test_dispatch_treats_a_reexported_wrapper_as_no_override():
    calls = []

    def my_op(value):
        calls.append("portable")
        return f"portable:{value}"

    decorated = dispatch(_FAKE_FEATURE)(my_op)
    # A platform module that re-exports the decorated portable function holds
    # this very wrapper; dispatch must recognize it by identity and run the
    # original instead of recursing.
    _TRITON_OPS_IMPL_CACHE[_FAKE_FEATURE] = types.SimpleNamespace(my_op=decorated)
    assert decorated(1) == "portable:1"
    assert calls == ["portable"]

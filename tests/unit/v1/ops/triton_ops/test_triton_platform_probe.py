# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for the platform-owned Triton probe helper.

``probe_once`` is a building block for platform ``triton_ops`` variants, not a
global gate: the variant owns the key, the probe body, and the failure policy.
These tests pin the caching and error-capture contract and run without a
device.
"""

from deepspeed.ops.triton_ops._triton import (_TRITON_KERNEL_PROBE_CACHE, _TRITON_KERNEL_PROBE_ERRORS, probe_error,
                                              probe_once)


def _clear(key):
    _TRITON_KERNEL_PROBE_CACHE.pop(key, None)
    _TRITON_KERNEL_PROBE_ERRORS.pop(key, None)


def test_probe_once_runs_once_and_caches_success():
    key = "unit:platform_probe:success"
    _clear(key)
    calls = []

    def run():
        calls.append(1)

    assert probe_once(key, run)
    assert probe_once(key, run)
    assert calls == [1]
    assert probe_error(key) is None


def test_probe_once_caches_failure_and_preserves_the_error():
    key = "unit:platform_probe:failure"
    _clear(key)
    calls = []

    class ProbeError(RuntimeError):
        pass

    def failing():
        calls.append(1)
        raise ProbeError("probe failed")

    assert not probe_once(key, failing)

    def success():
        calls.append(1)

    # A later successful callable does not replace the cached failure.
    assert not probe_once(key, success)
    assert calls == [1]
    error = probe_error(key)
    assert isinstance(error, ProbeError)
    assert str(error) == "probe failed"


def test_probe_keys_are_independent():
    ok_key = "unit:platform_probe:ok"
    bad_key = "unit:platform_probe:bad"
    _clear(ok_key)
    _clear(bad_key)

    def bad():
        raise RuntimeError("bad")

    assert probe_once(ok_key, lambda: None)
    assert not probe_once(bad_key, bad)
    assert probe_error(ok_key) is None
    assert isinstance(probe_error(bad_key), RuntimeError)

# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for the per-kernel Triton probe gate.

These tests pin the probe mechanism's contract, not its timing: the probe runs
at most once per process per key, later calls answer from the cache without
running or constructing the probe again, a failing probe closes the gate with a
cached False, a probe does not advance the caller's RNG sequence, and
DS_TRITON_KERNEL_GATE=0 bypasses probing entirely. The probe's wall-clock cost
is covered by a standalone benchmark outside the repo (numbers in the PR
description).
"""

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.triton_ops import _probes
from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, _TRITON_PROBE_CACHE, is_triton_kernel_supported

_ACCELERATOR = get_accelerator()

pytestmark = pytest.mark.skipif(not _ACCELERATOR.is_available() or not _TRITON_AVAILABLE,
                                reason="requires an accelerator with an active Triton backend")


def _dtype_and_device():
    device = torch.device(_ACCELERATOR.current_device_name())
    dtype = torch.bfloat16
    return dtype, device


def _key(name, dtype, device):
    return f"triton_kernel:{name}:{device.type}:{device.index}"


def test_probe_runs_once_and_later_calls_answer_from_the_cache():
    dtype, device = _dtype_and_device()
    calls = {"n": 0}

    def probe_fn():
        calls["n"] += 1
        return _probes.swiglu(dtype, device)

    _TRITON_PROBE_CACHE.pop(_key("unit_probe_once", dtype, device), None)
    first = is_triton_kernel_supported("unit_probe_once", dtype, device, probe_fn)
    second = is_triton_kernel_supported("unit_probe_once", dtype, device, probe_fn)
    assert first == second
    # The second call answered from the cache: the probe factory never ran again.
    assert calls["n"] == 1


def test_failing_probe_closes_the_gate_with_a_cached_false():
    dtype, device = _dtype_and_device()

    def failing_probe_fn():

        def run():
            raise RuntimeError("deliberate unit-test probe failure")

        return run

    _TRITON_PROBE_CACHE.pop(_key("unit_probe_fail", dtype, device), None)
    assert not is_triton_kernel_supported("unit_probe_fail", dtype, device, failing_probe_fn)

    calls = {"n": 0}

    def counting_probe_fn():
        calls["n"] += 1
        return _probes.swiglu(dtype, device)

    # The cached False stands and the replacement probe is never even constructed.
    assert not is_triton_kernel_supported("unit_probe_fail", dtype, device, counting_probe_fn)
    assert calls["n"] == 0


def test_probe_does_not_advance_the_caller_rng_sequence():
    dtype, device = _dtype_and_device()
    # Compare the first draw after seeding WITH the probe in between against
    # the same draw WITHOUT the probe: reseed, run the probe, draw
    # ``after_probe``, then reseed again and draw ``control``. If the probe
    # consumed any random numbers, ``after_probe`` starts further along the
    # sequence than ``control`` and the two draws differ.
    _ACCELERATOR.manual_seed(4321)
    is_triton_kernel_supported("unit_probe_rng", dtype, device, _probes.fused_rms_norm, dtype, device)
    after_probe = torch.randn(4, device=device)

    _ACCELERATOR.manual_seed(4321)
    control = torch.randn(4, device=device)

    assert torch.equal(after_probe, control)


def test_env_escape_hatch_bypasses_probing(monkeypatch):
    dtype, device = _dtype_and_device()
    monkeypatch.setenv("DS_TRITON_KERNEL_GATE", "0")
    calls = {"n": 0}

    def probe_fn():
        calls["n"] += 1
        return _probes.swiglu(dtype, device)

    # The gate answers at the backend level without ever touching the probe.
    assert is_triton_kernel_supported("unit_probe_env", dtype, device, probe_fn) == _ACCELERATOR.is_triton_supported()
    assert calls["n"] == 0


def test_static_declaration_answers_from_the_backend_gate_without_a_probe():
    dtype, device = _dtype_and_device()
    # probe_fn=None is the static-declaration form for probe-unsafe paths:
    # nothing runs, and a kernel name no backend declares answers with the
    # backend-level gate.
    _TRITON_PROBE_CACHE.pop(_key("unit_probe_static", dtype, device), None)
    assert is_triton_kernel_supported("unit_probe_static", dtype, device, None) == _ACCELERATOR.is_triton_supported()

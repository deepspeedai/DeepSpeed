# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

import psutil
import pytest

from deepspeed.ops.adam.cpu_adam import single_thread_warning


@pytest.fixture
def cores(monkeypatch):

    def set_cores(n, physical=None, affinity=None):
        """n logical CPUs; the physical core count and the affinity mask default to n."""
        physical = physical or n
        affinity = affinity or n
        monkeypatch.setattr(os, "cpu_count", lambda: n)
        monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: n if logical else physical)
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(affinity)), raising=False)

    return set_cores


def test_warns_when_torchrun_default_leaves_one_thread(monkeypatch, cores):
    cores(8)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    message = single_thread_warning()
    assert message is not None and "4 cores" in message


@pytest.mark.parametrize("omp", [None, "4", "16"])
def test_silent_when_threads_are_set(monkeypatch, cores, omp):
    cores(8)
    if omp is None:
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    else:
        monkeypatch.setenv("OMP_NUM_THREADS", omp)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    assert single_thread_warning() is None


def test_silent_when_rank_has_one_core(monkeypatch, cores):
    cores(2)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    assert single_thread_warning() is None


def test_single_process_uses_all_cores(monkeypatch, cores):
    cores(6)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    assert "6 cores" in single_thread_warning()


def test_counts_physical_cores(monkeypatch, cores):
    # 8 physical cores with two hardware threads each, all of them in the affinity mask.
    cores(16, physical=8)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    assert "4 cores" in single_thread_warning()


def test_affinity_limits_cores(monkeypatch, cores):
    cores(16, physical=8, affinity=4)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    assert "4 cores" in single_thread_warning()


def test_unknown_physical_core_count(monkeypatch, cores):
    # psutil.cpu_count(logical=False) returns None when it cannot tell.
    cores(6)
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 6 if logical else None)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    assert "6 cores" in single_thread_warning()

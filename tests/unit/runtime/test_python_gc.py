# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from deepspeed.runtime.python_gc import PythonGCManager


def test_python_gc_manager_restores_enabled_state(monkeypatch):
    calls = []
    enabled = True

    def isenabled():
        return enabled

    def collect():
        calls.append("collect")
        return 7

    def disable():
        nonlocal enabled
        enabled = False
        calls.append("disable")

    def enable():
        nonlocal enabled
        enabled = True
        calls.append("enable")

    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.isenabled", isenabled)
    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.collect", collect)
    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.disable", disable)
    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.enable", enable)

    manager = PythonGCManager()
    generation = manager.acquire()
    assert manager.acquire() == generation
    assert manager.active_engines == 2
    assert calls == ["collect", "disable"]

    manager.release(generation)
    assert manager.active_engines == 1
    assert calls == ["collect", "disable"]

    manager.release(generation)
    assert manager.active_engines == 0
    assert calls == ["collect", "disable", "enable"]


def test_python_gc_manager_preserves_disabled_state(monkeypatch):
    calls = []

    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.isenabled", lambda: False)
    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.collect", lambda: calls.append("collect"))
    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.disable", lambda: calls.append("disable"))
    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.enable", lambda: calls.append("enable"))

    manager = PythonGCManager()
    generation = manager.acquire()
    manager.release(generation)

    assert calls == []


def test_python_gc_manager_explicit_collection(monkeypatch):
    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.collect", lambda: 11)
    assert PythonGCManager().collect() == 11


def test_python_gc_manager_restores_enabled_state_after_fork(monkeypatch):
    calls = []
    enabled = False

    def isenabled():
        return enabled

    def enable():
        nonlocal enabled
        enabled = True
        calls.append("enable")

    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.isenabled", isenabled)
    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.enable", enable)

    manager = PythonGCManager()
    manager._active_engines = 1
    manager._restore_enabled = True
    manager._after_fork_child()

    assert calls == ["enable"]
    assert manager.active_engines == 0


def test_python_gc_manager_preserves_disabled_state_after_fork(monkeypatch):
    calls = []
    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.isenabled", lambda: False)
    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.enable", lambda: calls.append("enable"))

    manager = PythonGCManager()
    manager._active_engines = 1
    manager._restore_enabled = False
    manager._after_fork_child()

    assert calls == []
    assert manager.active_engines == 0


def test_python_gc_manager_ignores_pre_fork_release(monkeypatch):
    enabled = True

    def isenabled():
        return enabled

    def disable():
        nonlocal enabled
        enabled = False

    def enable():
        nonlocal enabled
        enabled = True

    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.isenabled", isenabled)
    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.collect", lambda: 0)
    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.disable", disable)
    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.enable", enable)

    manager = PythonGCManager()
    parent_generation = manager.acquire()
    manager._after_fork_child()
    child_generation = manager.acquire()

    manager.release(parent_generation)
    assert manager.active_engines == 1
    assert enabled is False

    manager.release(child_generation)
    assert manager.active_engines == 0
    assert enabled is True


def test_python_gc_manager_collection_allows_reentrant_release(monkeypatch):
    manager = PythonGCManager()
    enabled = True

    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.isenabled", lambda: enabled)

    def collect():
        manager.release(-1)
        return 0

    def disable():
        nonlocal enabled
        enabled = False

    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.collect", collect)
    monkeypatch.setattr("deepspeed.runtime.python_gc.gc.disable", disable)

    generation = manager.acquire()
    assert manager.active_engines == 1
    manager.release(generation)

# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Contract tests for the DS_MULTIRANK_ONLY collection selector in tests/conftest.py.

Each test builds a sandbox containing the real conftest.py and pytest.ini plus
representative test classes, runs pytest --collect-only with the gate off and
on, and asserts on which items survive collection -- the same observable
surface the nightly multi-rank workflow relies on.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent.parent

SANDBOX_TESTS = '''
import pytest


class DistBase:
    is_dist_test = True


class PlainBase:
    pass


class TestClassWorldSizeTwo(DistBase):
    world_size = 2

    def test_class_level_multi(self):
        pass


class TestClassWorldSizeOne(DistBase):
    world_size = 1

    def test_class_level_single(self):
        pass


class TestNoWorldSize(DistBase):
    def test_no_world_size(self):
        pass


class TestMarkRaisesToMulti(DistBase):
    world_size = 1

    @pytest.mark.world_size(2)
    def test_mark_raises(self):
        pass


class TestMarkLowersToSingle(DistBase):
    world_size = 2

    @pytest.mark.world_size(1)
    def test_mark_lowers(self):
        pass


class TestListWorldSize(DistBase):
    world_size = [1, 2]

    def test_list_has_multi(self):
        pass


class TestListAllSingle(DistBase):
    world_size = [1, 1]

    def test_list_all_single(self):
        pass


def test_plain_function():
    pass
'''


def _collect(multirank_only: bool) -> list:
    """Collect the sandbox suite with the gate off/on and return the node ids."""
    root = tempfile.mkdtemp(prefix="multirank_select_")
    try:
        for name in ("conftest.py", "pytest.ini"):
            shutil.copy(TESTS_DIR / name, Path(root) / name)
        (Path(root) / "test_sandbox.py").write_text(SANDBOX_TESTS)
        env = dict(os.environ, DS_MULTIRANK_ONLY="1" if multirank_only else "0")
        result = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "test_sandbox.py"],
                                cwd=root,
                                env=env,
                                capture_output=True,
                                text=True,
                                check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        # The conftest under test forces verbose mode, so --collect-only emits a
        # tree of <Class>/<Function> lines instead of plain node ids.
        ids = []
        current_class = None
        class_indent = None
        for line in result.stdout.splitlines():
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if stripped.startswith("<Class "):
                current_class = stripped[len("<Class "):-len(">")]
                class_indent = indent
            elif stripped.startswith("<Function "):
                function = stripped[len("<Function "):-len(">")]
                # A function at the class's own indentation level is
                # module-level, not a member of the class above it.
                if current_class is not None and indent > class_indent:
                    ids.append(f"test_sandbox.py::{current_class}::{function}")
                else:
                    current_class = None
                    ids.append(f"test_sandbox.py::{function}")
        return sorted(ids)
    finally:
        shutil.rmtree(root, ignore_errors=True)


ALL_ITEMS = [
    "test_sandbox.py::TestClassWorldSizeTwo::test_class_level_multi",
    "test_sandbox.py::TestClassWorldSizeOne::test_class_level_single",
    "test_sandbox.py::TestNoWorldSize::test_no_world_size",
    "test_sandbox.py::TestMarkRaisesToMulti::test_mark_raises",
    "test_sandbox.py::TestMarkLowersToSingle::test_mark_lowers",
    "test_sandbox.py::TestListWorldSize::test_list_has_multi",
    "test_sandbox.py::TestListAllSingle::test_list_all_single",
    "test_sandbox.py::test_plain_function",
]

MULTIRANK_ITEMS = [
    "test_sandbox.py::TestClassWorldSizeTwo::test_class_level_multi",
    "test_sandbox.py::TestMarkRaisesToMulti::test_mark_raises",
    "test_sandbox.py::TestListWorldSize::test_list_has_multi",
]


def test_gate_off_collects_everything():
    # Catches an inverted or over-eager gate: local and regular CI runs must
    # never lose tests to the nightly-only selector.
    assert _collect(multirank_only=False) == sorted(ALL_ITEMS)


def test_gate_on_keeps_only_multirank_items():
    # Catches both directions: single-rank items leaking into the nightly
    # (wasted budget, already covered by cpu-torch-latest) and multi-rank
    # items being dropped (a silently green, test-free nightly).
    assert _collect(multirank_only=True) == sorted(MULTIRANK_ITEMS)


def test_mark_overrides_class_world_size():
    # The selector must honor the same precedence as the launcher: a method
    # mark beats the class attribute in both directions.
    collected = _collect(multirank_only=True)
    assert "test_sandbox.py::TestMarkLowersToSingle::test_mark_lowers" not in collected
    assert "test_sandbox.py::TestMarkRaisesToMulti::test_mark_raises" in collected

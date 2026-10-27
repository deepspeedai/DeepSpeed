# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Contract tests for ci/classify_multirank_exit.py.

The helper maps the multi-rank run's exit code to the DS_CI_FAILURE_CLASS
sentinel that ci/nightly_classify.sh greps from the job log, so the tests
drive the real CLI via subprocess and assert on exit code and stdout only.
Pure-stdlib, no DeepSpeed/torch install needed.

Run standalone::

    python ci/test_classify_multirank_exit.py

or under pytest::

    pytest ci/test_classify_multirank_exit.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPER = REPO_ROOT / "ci" / "classify_multirank_exit.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "cpu-torch-multi-latest.yml"


def _run(raw: str) -> tuple[int, str]:
    result = subprocess.run([sys.executable, str(HELPER), raw], capture_output=True, text=True, check=False)
    return result.returncode, result.stdout


def test_timeout_exit_classifies_as_timeout() -> None:
    # Catches GNU timeout's 124 being reported as a plain test failure, which
    # would route the nightly to regression triage instead of the timeout path.
    code, stdout = _run("124")
    assert code == 1
    assert "DS_CI_FAILURE_CLASS=timeout" in stdout
    assert "DS_CI_FAILURE_CLASS=test" not in stdout


def test_nonzero_exit_classifies_as_test_and_propagates_the_code() -> None:
    # Catches a misrouted test failure (e.g. reported as timeout) and a lost
    # pytest exit code, which the triage report relies on.
    code, stdout = _run("7")
    assert code == 7
    assert "DS_CI_FAILURE_CLASS=test" in stdout
    assert "DS_CI_FAILURE_CLASS=timeout" not in stdout


def test_zero_exit_reports_success_without_any_sentinel() -> None:
    # Catches a passing nightly being classified as a failure at all.
    code, stdout = _run("0")
    assert code == 0
    assert "DS_CI_FAILURE_CLASS" not in stdout


def test_missing_exit_reports_no_sentinel() -> None:
    # "none" means the test step never ran; a sentinel here would misroute a
    # setup death to regression/timeout triage instead of the unknown path.
    code, stdout = _run("none")
    assert code == 1
    assert "DS_CI_FAILURE_CLASS" not in stdout


def test_workflow_does_not_emit_the_sentinel_inline() -> None:
    # Pins the bug this helper exists for: GitHub Actions echoes a run: script
    # into the log, so an inline `echo "DS_CI_FAILURE_CLASS=..."` is matched by
    # nightly_classify.sh's head -1 before any runtime line, routing every
    # failure as a timeout. The sentinel may only be printed from this file.
    for line in WORKFLOW.read_text().splitlines():
        assert 'echo "DS_CI_FAILURE_CLASS=' not in line, line


def _all_test_functions():
    return sorted((name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj))


def main() -> int:
    failures = 0
    for name, fn in _all_test_functions():
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    total = len(_all_test_functions())
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

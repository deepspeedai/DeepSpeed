# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Self-tests for ci/tests_fetcher.py (#9).

These build small synthetic git repos on disk and assert the selector makes the
right call (narrow vs. full vs. nothing). They are pure-stdlib (only need ``git``
on PATH), so they run anywhere -- including the ``collect-tests`` CI job that uses
the fetcher -- without a DeepSpeed/torch install.

Run standalone::

    python ci/test_tests_fetcher.py

or under pytest::

    pytest ci/test_tests_fetcher.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import the sibling module

import tests_fetcher  # noqa: E402
from tests_fetcher import Selection, TestSelector, WorkflowConfig  # noqa: E402

CONFIG = WorkflowConfig(
    name="test",
    test_scopes=("tests/unit/v1", ),
    extra_run_all_globs=(".github/workflows/modal*.yml", ),
)

# A tiny but representative repo:
#   deepspeed.leaf    -> imported only by test_leaf
#   deepspeed.shared  -> imported by test_shared + test_shared2
#   deepspeed.orphan  -> imported by nobody (for clean-delete test)
#   runtime/engine, comm/comm, accelerator/* -> core run-all globs
#   module_inject/*   -> dynamic edge -> moe tests
BASELINE = {
    "deepspeed/__init__.py": "from deepspeed import runtime  # noqa\n",
    "deepspeed/leaf.py": "VALUE = 1\n",
    "deepspeed/shared.py": "VALUE = 2\n",
    "deepspeed/orphan.py": "VALUE = 3\n",
    "deepspeed/runtime/__init__.py": "",
    "deepspeed/runtime/engine.py": "VALUE = 4\n",
    "deepspeed/comm/__init__.py": "",
    "deepspeed/comm/comm.py": "VALUE = 5\n",
    "deepspeed/module_inject/__init__.py": "",
    "deepspeed/module_inject/replace.py": "VALUE = 6\n",
    "tests/__init__.py": "",
    "tests/unit/__init__.py": "",
    "tests/unit/common.py": "import deepspeed  # noqa\n",
    "tests/unit/v1/__init__.py": "",
    "tests/unit/v1/test_leaf.py": "from deepspeed.leaf import VALUE\n\n\ndef test_x():\n    assert VALUE == 1\n",
    "tests/unit/v1/test_shared.py": "from deepspeed.shared import VALUE\n\n\ndef test_x():\n    assert VALUE == 2\n",
    "tests/unit/v1/test_shared2.py": "from deepspeed.shared import VALUE\n\n\ndef test_x():\n    assert VALUE == 2\n",
    "tests/unit/v1/test_plain.py": "import deepspeed  # noqa\n\n\ndef test_x():\n    assert True\n",
    "tests/unit/v1/moe/__init__.py": "",
    "tests/unit/v1/moe/test_moe.py": "import deepspeed  # noqa\n\n\ndef test_x():\n    assert True\n",
    "README.md": "# synthetic repo\n",
}


class TmpRepo:
    """A throwaway git repo seeded with BASELINE, committed on ``master``."""

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="ds-fetcher-test-")).resolve()
        for rel, content in BASELINE.items():
            self.write(rel, content)
        self._git("init", "-q", "-b", "master")
        self._git("config", "user.email", "ci@example.com")
        self._git("config", "user.name", "ci")
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "baseline")
        # Work on a feature branch so ``select("master")`` has a real diff.
        self._git("checkout", "-q", "-b", "feature")

    def _git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True, text=True).stdout

    def write(self, rel: str, content: str = "") -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def delete(self, rel: str) -> None:
        (self.root / rel).unlink()

    def commit(self, message: str) -> None:
        self._git("add", "-A")
        self._git("commit", "-q", "--allow-empty", "-m", message)

    def selector(self) -> TestSelector:
        return TestSelector(self.root, CONFIG)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _rel_names(repo: TmpRepo, tests) -> set[str]:
    return {p.relative_to(repo.root).as_posix() for p in tests}


# --------------------------------------------------------------------------- #
# Individual checks. Each builds a fresh repo, mutates it, and asserts.
# --------------------------------------------------------------------------- #
def test_leaf_change_narrows_to_one_test() -> None:
    repo = TmpRepo()
    try:
        repo.write("deepspeed/leaf.py", "VALUE = 11\n")
        repo.commit("touch leaf")
        sel = repo.selector().select("master")
        assert sel.mode == "subset", sel.reason
        assert _rel_names(repo, sel.tests) == {"tests/unit/v1/test_leaf.py"}
    finally:
        repo.cleanup()


def test_shared_change_selects_all_importers() -> None:
    repo = TmpRepo()
    try:
        repo.write("deepspeed/shared.py", "VALUE = 22\n")
        repo.commit("touch shared")
        sel = repo.selector().select("master")
        assert sel.mode == "subset", sel.reason
        assert _rel_names(repo, sel.tests) == {
            "tests/unit/v1/test_shared.py",
            "tests/unit/v1/test_shared2.py",
        }
    finally:
        repo.cleanup()


def test_core_runtime_file_runs_all() -> None:
    repo = TmpRepo()
    try:
        repo.write("deepspeed/runtime/engine.py", "VALUE = 44\n")
        repo.commit("touch engine")
        sel = repo.selector().select("master")
        assert sel.mode == "all", sel.reason
    finally:
        repo.cleanup()


def test_comm_change_runs_all() -> None:
    repo = TmpRepo()
    try:
        repo.write("deepspeed/comm/comm.py", "VALUE = 55\n")
        repo.commit("touch comm")
        sel = repo.selector().select("master")
        assert sel.mode == "all", sel.reason
    finally:
        repo.cleanup()


def test_shared_fixture_runs_all() -> None:
    repo = TmpRepo()
    try:
        repo.write("tests/unit/common.py", "import deepspeed  # touched\n")
        repo.commit("touch common")
        sel = repo.selector().select("master")
        assert sel.mode == "all", sel.reason
    finally:
        repo.cleanup()


def test_commit_tag_forces_all() -> None:
    repo = TmpRepo()
    try:
        repo.write("deepspeed/leaf.py", "VALUE = 11\n")
        repo.commit("touch leaf [test all]")
        sel = repo.selector().select("master")
        assert sel.mode == "all", sel.reason
    finally:
        repo.cleanup()


def test_non_python_change_selects_nothing() -> None:
    repo = TmpRepo()
    try:
        repo.write("README.md", "# changed\n")
        repo.commit("docs only")
        sel = repo.selector().select("master")
        assert sel.mode == "none", sel.reason
    finally:
        repo.cleanup()


def test_new_test_file_is_selected() -> None:
    repo = TmpRepo()
    try:
        repo.write(
            "tests/unit/v1/test_new.py",
            "import deepspeed  # noqa\n\n\ndef test_x():\n    assert True\n",
        )
        repo.commit("add new test")
        sel = repo.selector().select("master")
        assert sel.mode == "subset", sel.reason
        assert "tests/unit/v1/test_new.py" in _rel_names(repo, sel.tests)
    finally:
        repo.cleanup()


def test_clean_delete_does_not_run_all() -> None:
    # Deleting a module nobody imports must NOT trigger a full run (#5).
    repo = TmpRepo()
    try:
        repo.delete("deepspeed/orphan.py")
        repo.commit("remove orphan")
        sel = repo.selector().select("master")
        assert sel.mode == "none", sel.reason
    finally:
        repo.cleanup()


def test_dangling_delete_runs_all() -> None:
    # Deleting a module that a surviving file still imports IS unsafe -> full run (#5).
    repo = TmpRepo()
    try:
        repo.delete("deepspeed/shared.py")  # test_shared / test_shared2 still import it
        repo.commit("remove shared but leave importers")
        sel = repo.selector().select("master")
        assert sel.mode == "all", sel.reason
    finally:
        repo.cleanup()


def test_dynamic_edge_pulls_in_moe_tests() -> None:
    # module_inject is wired in at runtime; test_moe never imports it (#4).
    repo = TmpRepo()
    try:
        repo.write("deepspeed/module_inject/replace.py", "VALUE = 66\n")
        repo.commit("touch module_inject")
        sel = repo.selector().select("master")
        assert sel.mode == "subset", sel.reason
        assert "tests/unit/v1/moe/test_moe.py" in _rel_names(repo, sel.tests)
    finally:
        repo.cleanup()


def test_missing_base_runs_all() -> None:
    repo = TmpRepo()
    try:
        sel = repo.selector().select("")
        assert sel.mode == "all", sel.reason
        sel = repo.selector().select("origin/does-not-exist")
        assert sel.mode == "all", sel.reason
    finally:
        repo.cleanup()


def test_external_repo_root_keeps_output_in_trusted_checkout() -> None:
    repo = TmpRepo()
    trusted_root = Path(tempfile.mkdtemp(prefix="ds-fetcher-trusted-"))
    original_root = tests_fetcher.REPO_ROOT
    original_argv = sys.argv
    try:
        repo.write("deepspeed/leaf.py", "VALUE = 11\n")
        repo.commit("touch leaf")
        candidate_output = repo.root / "ci" / ".test_selection" / "test_list.txt"
        repo.write("ci/.test_selection/test_list.txt", "candidate-controlled\n")

        tests_fetcher.REPO_ROOT = trusted_root
        sys.argv = [
            "tests_fetcher.py",
            "--workflow",
            "modal-torch-latest",
            "--repo-root",
            str(repo.root),
            "--base",
            "master",
        ]
        tests_fetcher.main()

        trusted_output = trusted_root / "ci" / ".test_selection" / "test_list.txt"
        assert trusted_output.read_text(encoding="utf-8") == "tests/unit/v1/test_leaf.py\n"
        assert candidate_output.read_text(encoding="utf-8") == "candidate-controlled\n"
    finally:
        tests_fetcher.REPO_ROOT = original_root
        sys.argv = original_argv
        repo.cleanup()
        shutil.rmtree(trusted_root, ignore_errors=True)


def test_github_output_contains_only_validated_mode_and_count() -> None:
    root = Path(tempfile.mkdtemp(prefix="ds-fetcher-output-"))
    output = root / "github-output"
    original = os.environ.get("GITHUB_OUTPUT")
    try:
        os.environ["GITHUB_OUTPUT"] = str(output)
        tests_fetcher._emit_github_output("subset", 1)
        assert output.read_text(encoding="utf-8") == "mode=subset\ncount=1\n"
        try:
            tests_fetcher._emit_github_output("subset\npwned=1", 1)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid output mode was accepted")
    finally:
        if original is None:
            os.environ.pop("GITHUB_OUTPUT", None)
        else:
            os.environ["GITHUB_OUTPUT"] = original
        shutil.rmtree(root, ignore_errors=True)


def test_summary_escapes_candidate_controls_and_markup() -> None:
    root = Path(tempfile.mkdtemp(prefix="ds-fetcher-summary-"))
    summary = root / "summary"
    original = os.environ.get("GITHUB_STEP_SUMMARY")
    try:
        os.environ["GITHUB_STEP_SUMMARY"] = str(summary)
        selection = Selection("all", [], "changed <tag>\n::warning:: `text`")
        tests_fetcher._write_step_summary(CONFIG, selection, ["tests/unit/v1"])
        text = summary.read_text(encoding="utf-8")
        assert "changed &lt;tag&gt;\\x0a::warning:: `text`" in text
        assert "\n::warning::" not in text
    finally:
        if original is None:
            os.environ.pop("GITHUB_STEP_SUMMARY", None)
        else:
            os.environ["GITHUB_STEP_SUMMARY"] = original
        shutil.rmtree(root, ignore_errors=True)


def test_output_path_cannot_escape_or_replace_trusted_checkout() -> None:
    root = Path(tempfile.mkdtemp(prefix="ds-fetcher-path-"))
    outside = Path(tempfile.mkdtemp(prefix="ds-fetcher-outside-"))
    try:
        expected = root / "ci" / ".test_selection" / "test_list.txt"
        assert tests_fetcher._resolve_output_path(root, "ci/.test_selection/test_list.txt") == expected
        for value in ("../escape", str(outside / "absolute")):
            try:
                tests_fetcher._resolve_output_path(root, value)
            except ValueError:
                pass
            else:
                raise AssertionError(f"unsafe output path was accepted: {value}")

        expected.parent.mkdir(parents=True)
        expected.symlink_to(outside / "test_list.txt")
        try:
            tests_fetcher._resolve_output_path(root, "ci/.test_selection/test_list.txt")
        except ValueError:
            pass
        else:
            raise AssertionError("symlink output path was accepted")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(outside, ignore_errors=True)


def _all_test_functions():
    return sorted((name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj))


def main() -> int:
    failures = 0
    for name, fn in _all_test_functions():
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    total = len(_all_test_functions())
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

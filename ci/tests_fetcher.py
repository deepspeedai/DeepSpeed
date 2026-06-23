# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Diff-based test selector for DeepSpeed CI.

A small, self-contained take on HuggingFace transformers' ``utils/tests_fetcher.py``.

Given a git diff against a base ref, it figures out the *minimal* set of test
files (under a workflow's test scope, e.g. ``tests/unit/v1``) that could be
affected, so CI doesn't have to run the whole suite on every PR:

  1. Collect the Python files changed in the diff.
  2. Build an import-dependency graph over the ``deepspeed`` package and the test
     helpers (the ``unit`` package under ``tests/unit``). Module A "impacts"
     module B when B imports A, directly or transitively.
  3. Walk that graph backwards from the changed files to every impacted test in
     the workflow's scope.
  4. Write the impacted test paths to an output file (one per line), which the CI
     runner (``ci/torch_latest.py``) feeds to ``pytest``.

Design principles
-----------------
* **Fail safe, never fail closed silently.** A missing base ref, no merge-base
  (shallow clone), a touched shared fixture / build / CI file, a dangling deleted
  module, a ``[test all]`` commit tag, or *any unexpected error in the selector*
  all fall back to running the entire scope. The one thing the selector will
  never do on error is select *fewer* tests than reality.
* **Config-driven.** One ``WorkflowConfig`` per CI workflow (see ``WORKFLOWS``),
  so the same engine can drive several workflows with different scopes / extra
  run-all triggers via ``--workflow``.
* **Testable.** All repo/config state lives on ``TestSelector`` (constructed with
  an arbitrary ``repo_root``), so the logic can be unit-tested against synthetic
  repos -- see ``ci/test_tests_fetcher.py``.

Escape hatches (for humans)
---------------------------
* Put ``[test all]`` (or ``[no filter]``) anywhere in a commit message to force
  the full suite for that push.
* Changing a file matched by the run-all globs (CI config, build system, shared
  fixtures, core runtime) always runs everything.

Preview what CI would run for your branch::

    python ci/tests_fetcher.py --base origin/master
    cat ci/.test_selection/test_list.txt

Explain *why* a test was (de)selected (prints the import chains)::

    python ci/tests_fetcher.py --base origin/master --explain
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import traceback
from collections import deque
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Per-workflow configuration (#11: drive multiple workflows from one config).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorkflowConfig:
    """Knobs that differ between CI workflows.

    name              : identifier, also the ``--workflow`` value.
    test_scopes       : repo-root-relative dirs whose ``test_*.py`` files this
                        workflow runs (and the targets used for ``mode=all``).
    extra_run_all_globs: workflow-specific paths that force a full run (on top of
                         ``TestSelector.COMMON_RUN_ALL_GLOBS``).
    """

    name: str
    test_scopes: tuple[str, ...]
    extra_run_all_globs: tuple[str, ...] = field(default_factory=tuple)


WORKFLOWS: dict[str, WorkflowConfig] = {
    "modal-torch-latest":
    WorkflowConfig(
        name="modal-torch-latest",
        test_scopes=("tests/unit/v1", ),
        extra_run_all_globs=(".github/workflows/modal*.yml", ),
    ),
}

DEFAULT_WORKFLOW = "modal-torch-latest"


@dataclass
class Selection:
    """Outcome of a selection run.

    mode  : "all" | "subset" | "none".
    tests : selected test files (only meaningful for "subset").
    reason: human-readable explanation, surfaced in logs and the job summary.
    """

    mode: str
    tests: list[Path]
    reason: str


class TestSelector:
    """Diff-driven test selection engine for one repo + one workflow."""

    # Importable roots that participate in the graph, as (base_subdir, top_package)
    # *relative to the repo root*:
    #   - ``deepspeed`` lives at the repo root (base_subdir = "").
    #   - tests import shared helpers as the ``unit`` package rooted at ``tests/``
    #     (e.g. ``from unit.common import DistributedTest``).
    PACKAGE_ROOTS = (("", "deepspeed"), ("tests", "unit"))

    # "Opaque" package roots whose ``__init__.py`` imports are NOT expanded in the
    # graph. Almost every test imports ``unit.common``, which does ``import
    # deepspeed``, ``import deepspeed.comm`` and ``from deepspeed.accelerator import
    # get_accelerator``. Each of those ``__init__`` files eagerly imports a large
    # subtree, so treating them as normal nodes makes nearly any ``deepspeed/**``
    # change fan out to the entire suite. Treating them as opaque links a test only
    # to the ``__init__.py`` itself (not its whole subtree); submodule changes are
    # traced to the tests that import *that* submodule. Changes to the hubs
    # themselves are covered by the run-all globs below.
    OPAQUE_MODULES = ("deepspeed", "deepspeed.comm", "deepspeed.accelerator")

    # Commit-message tags that force the whole suite.
    RUN_ALL_COMMIT_TAGS = ("[test all]", "[no filter]")

    # Changing any of these means "we can't safely narrow the suite" -> run
    # everything. (Shell globs, matched against repo-root-relative POSIX paths.)
    COMMON_RUN_ALL_GLOBS = (
        "ci/**",  # the CI runner / this fetcher
        "csrc/**",  # C/CUDA kernel sources (compiled ops)
        "op_builder/**",  # op build system
        "accelerator/**",  # accelerator abstraction
        "requirements/**",  # dependency pins
        "setup.py",
        "tests/conftest.py",  # auto-loaded by every test
        "tests/pytest.ini",  # global pytest config (markers, addopts)
        "tests/unit/common.py",  # the distributed-test base used across the suite
        "tests/unit/util.py",
        # Core orchestration on the deepspeed.initialize() critical path. Tests
        # exercise these at *runtime* (via deepspeed.initialize) without importing
        # them directly, so the import graph can't see the dependency once
        # deepspeed/__init__ is opaque (see OPAQUE_MODULES). Changing them runs the
        # whole suite.
        "deepspeed/__init__.py",
        "deepspeed/runtime/engine.py",
        "deepspeed/runtime/hybrid_engine.py",
        "deepspeed/runtime/config.py",
        "deepspeed/runtime/config_utils.py",
        "deepspeed/comm/**",  # collectives used by every distributed test
        "deepspeed/accelerator/**",  # accelerator abstraction used by every test
    )

    # Dynamic dependency edges the *static* import graph can't see (#4).
    #
    # Some code is wired up at runtime -- monkey-patching, registry/plugin lookup,
    # JIT-loaded ops, replace_module() injection at deepspeed.initialize() time --
    # so a test can depend on a module it never ``import``s. This curated map fills
    # those gaps: it maps a *changed-file* glob to extra *test-path* globs that must
    # be pulled in whenever a matching file changes. Keep entries conservative and
    # reviewed; over-broad entries just cost CO2, under-broad ones miss coverage.
    #
    # NOTE: this is additive on top of the static graph; a file that is also a
    # run-all trigger short-circuits to the full suite before this map is consulted.
    DYNAMIC_EDGES: dict[str, tuple[str, ...]] = {
        # module_inject is applied at deepspeed.initialize() time (replace_module /
        # replace_transformer_layer); the tests it affects don't import it directly.
        "deepspeed/module_inject/**": ("tests/unit/v1/moe/**", ),
    }

    def __init__(self, repo_root: Path | str, config: WorkflowConfig):
        self.repo_root = Path(repo_root).resolve()
        self.config = config
        self.run_all_globs = self.COMMON_RUN_ALL_GLOBS + tuple(config.extra_run_all_globs)
        # repo-root-relative ``pkg`` prefixes, e.g. ("deepspeed", "tests/unit").
        self._source_prefixes = tuple((f"{base}/{pkg}" if base else pkg) for base, pkg in self.PACKAGE_ROOTS)

    # ----------------------------------------------------------------- git --- #
    def _run_git(self, args: list[str]) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def _rev_exists(self, ref: str) -> bool:
        try:
            self._run_git(["rev-parse", "--verify", f"{ref}^{{commit}}"])
            return True
        except subprocess.CalledProcessError:
            return False

    def _merge_base(self, base: str) -> str:
        """Merge-base of ``base`` and ``HEAD``, or "" if it can't be determined.

        An empty result (e.g. shallow clone that doesn't reach the fork point)
        must be treated as "can't trust a narrow diff" by the caller -> run all.
        """
        try:
            return self._run_git(["merge-base", base, "HEAD"]).strip()
        except subprocess.CalledProcessError:
            return ""

    def _diff_files(self, base_rev: str) -> tuple[list[str], list[str]]:
        """Return (changed, deleted) repo-root-relative paths for ``base_rev..HEAD``.

        'changed' = added / modified / renamed-new / copied-new; 'deleted' =
        deleted / renamed-old. ``base_rev`` should be the merge-base commit so
        two-dot == three-dot.
        """
        out = self._run_git(["diff", "--name-status", "--find-renames", f"{base_rev}", "HEAD"])
        changed: list[str] = []
        deleted: list[str] = []
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            status = parts[0]
            if status.startswith("R") and len(parts) >= 3:
                # rename: old path goes away, new path is what changed.
                deleted.append(parts[1])
                changed.append(parts[2])
            elif status.startswith("C") and len(parts) >= 3:
                # copy: source is untouched (do NOT mark it deleted); the new
                # destination path is the one that's effectively "changed".
                changed.append(parts[2])
            elif status.startswith("D"):
                deleted.append(parts[1])
            elif len(parts) >= 2:
                changed.append(parts[1])
        return changed, deleted

    def _commit_messages(self, base_rev: str) -> str:
        try:
            return self._run_git(["log", "--format=%B", f"{base_rev}..HEAD"])
        except subprocess.CalledProcessError:
            return ""

    # ------------------------------------------------- file / module maps --- #
    def _is_test_file(self, path: Path) -> bool:
        try:
            rel = path.relative_to(self.repo_root).as_posix()
        except ValueError:
            return False
        if not (path.name.startswith("test_") and path.suffix == ".py"):
            return False
        return any(rel.startswith(scope + "/") for scope in self.config.test_scopes)

    def _all_test_files(self) -> list[Path]:
        files: set[Path] = set()
        for scope in self.config.test_scopes:
            for p in (self.repo_root / scope).rglob("test_*.py"):
                if self._is_test_file(p):
                    files.add(p)
        return sorted(files)

    def _all_source_files(self) -> list[Path]:
        files: list[Path] = []
        seen: set[Path] = set()
        for base, pkg in self.PACKAGE_ROOTS:
            for p in (self.repo_root / base / pkg).rglob("*.py"):
                # Ignore build artifacts (e.g. build/lib/deepspeed/...).
                if p.relative_to(self.repo_root).parts[:1] == ("build", ):
                    continue
                if p not in seen:
                    seen.add(p)
                    files.append(p)
        return sorted(files)

    def _module_name(self, path: Path) -> str | None:
        """Dotted module name for a file under a package root, or None.

        e.g. deepspeed/runtime/engine.py -> deepspeed.runtime.engine
             tests/unit/common.py        -> unit.common
        """
        for base, pkg in self.PACKAGE_ROOTS:
            base_dir = self.repo_root / base
            pkg_dir = base_dir / pkg
            if path == pkg_dir or pkg_dir in path.parents:
                parts = list(path.relative_to(base_dir).with_suffix("").parts)
                if parts and parts[-1] == "__init__":
                    parts = parts[:-1]
                return ".".join(parts)
        return None

    def _rel_to_module(self, rel_posix: str) -> str | None:
        """Dotted module name from a repo-root-relative path string (for deleted files)."""
        if not rel_posix.endswith(".py"):
            return None
        for base, pkg in self.PACKAGE_ROOTS:
            prefix = f"{base}/{pkg}" if base else pkg
            if rel_posix == f"{prefix}.py" or rel_posix.startswith(prefix + "/"):
                sub = rel_posix[len(base) + 1:] if base else rel_posix
                parts = sub[:-3].split("/")
                if parts and parts[-1] == "__init__":
                    parts = parts[:-1]
                return ".".join(parts)
        return None

    def _under_sources(self, rel_posix: str) -> bool:
        return rel_posix.endswith(".py") and any(rel_posix == f"{p}.py" or rel_posix.startswith(p + "/")
                                                 for p in self._source_prefixes)

    # ------------------------------------------------------ import parsing --- #
    def _build_indexes(self, files: list[Path]) -> tuple[dict[str, Path], dict[Path, dict[str, Path]]]:
        module_index: dict[str, Path] = {}
        dir_local: dict[Path, dict[str, Path]] = {}
        for f in files:
            mod = self._module_name(f)
            if mod:
                module_index[mod] = f
            dir_local.setdefault(f.parent, {})[f.stem] = f
        return module_index, dir_local

    @staticmethod
    def _resolve_candidate(name: str, module_index: dict[str, Path]) -> Path | None:
        parts = name.split(".")
        while parts:
            hit = module_index.get(".".join(parts))
            if hit is not None:
                return hit
            parts = parts[:-1]
        return None

    def _file_imports(
        self,
        path: Path,
        module_index: dict[str, Path],
        dir_local: dict[Path, dict[str, Path]],
    ) -> tuple[set[Path], set[str]]:
        """Parse ``path`` once and return (intra-repo dep files, raw dotted import names).

        - dep files: package modules + sibling helpers it imports (graph edges).
        - raw dotted names: every dotted module name referenced, resolved or not.
          Used to detect dangling imports of deleted modules (#5).
        """
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError, OSError) as e:
            print(f"WARNING: could not parse {path}: {e}", file=sys.stderr)
            return set(), set()

        self_mod = self._module_name(path)
        is_init = path.name == "__init__.py"
        deps: set[Path] = set()
        raw: set[str] = set()
        siblings = dir_local.get(path.parent, {})

        # Don't expand a universal-hub package root (see OPAQUE_MODULES). We still
        # record raw names so dangling-delete detection keeps working.
        opaque = self_mod in self.OPAQUE_MODULES

        def add_dotted(name: str) -> None:
            if not name:
                return
            raw.add(name)
            if opaque:
                return
            hit = self._resolve_candidate(name, module_index)
            if hit is not None:
                deps.add(hit)

        def add_bare(top: str) -> None:
            if opaque:
                return
            hit = siblings.get(top)
            if hit is not None and hit != path:
                deps.add(hit)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_dotted(alias.name)
                    add_bare(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    module = node.module or ""
                    if module:
                        add_dotted(module)
                        add_bare(module.split(".")[0])
                        for alias in node.names:
                            add_dotted(f"{module}.{alias.name}")
                    else:
                        for alias in node.names:
                            add_bare(alias.name)
                elif self_mod is not None:
                    pkg_parts = self_mod.split(".") if is_init else self_mod.split(".")[:-1]
                    base_parts = pkg_parts[:len(pkg_parts) - (node.level - 1)]
                    base = ".".join(base_parts + ([node.module] if node.module else []))
                    if base:
                        add_dotted(base)
                        for alias in node.names:
                            add_dotted(f"{base}.{alias.name}")
        return deps, raw

    def _parse_all(
        self,
        files: list[Path],
        module_index: dict[str, Path],
        dir_local: dict[Path, dict[str, Path]],
    ) -> tuple[dict[Path, set[Path]], dict[Path, set[str]]]:
        """Parse every file once; return (deps_by_file, raw_imports_by_file)."""
        deps_by_file: dict[Path, set[Path]] = {}
        raw_by_file: dict[Path, set[str]] = {}
        for f in files:
            deps, raw = self._file_imports(f, module_index, dir_local)
            deps_by_file[f] = deps
            raw_by_file[f] = raw
        return deps_by_file, raw_by_file

    @staticmethod
    def _reverse_graph(deps_by_file: dict[Path, set[Path]]) -> dict[Path, set[Path]]:
        """Map each file -> the set of files that import it (directly)."""
        reverse: dict[Path, set[Path]] = {f: set() for f in deps_by_file}
        for f, deps in deps_by_file.items():
            for dep in deps:
                reverse.setdefault(dep, set()).add(f)
        return reverse

    @staticmethod
    def _impacted_files(seeds: set[Path], reverse: dict[Path, set[Path]]) -> set[Path]:
        seen: set[Path] = set()
        stack = list(seeds)
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(reverse.get(cur, ()))
        return seen

    @staticmethod
    def _reachable_with_parents(seed: Path, reverse: dict[Path, set[Path]]) -> dict[Path, Path | None]:
        """BFS from ``seed`` recording a predecessor for each node (for --explain).

        Uses a FIFO queue so each node's recorded parent is on a *shortest* path,
        giving the most concise import chain in ``--explain`` output.
        """
        parent: dict[Path, Path | None] = {seed: None}
        queue: deque[Path] = deque([seed])
        while queue:
            cur = queue.popleft()
            for imp in sorted(reverse.get(cur, ()), key=str):
                if imp not in parent:
                    parent[imp] = cur
                    queue.append(imp)
        return parent

    def _matches_glob(self, rel_posix: str, globs: tuple[str, ...]) -> bool:
        for g in globs:
            if fnmatch(rel_posix, g):
                return True
            # Support ``dir/**`` matching ``dir/anything/deep``.
            if g.endswith("/**") and (rel_posix == g[:-3] or rel_posix.startswith(g[:-2])):
                return True
        return False

    def _dynamic_edge_tests(self, changed: list[str], all_tests: list[Path]) -> set[Path]:
        """Extra tests pulled in by the curated dynamic-edge map (#4)."""
        wanted_globs: set[str] = set()
        for rel in changed:
            for src_glob, test_globs in self.DYNAMIC_EDGES.items():
                if self._matches_glob(rel, (src_glob, )):
                    wanted_globs.update(test_globs)
        if not wanted_globs:
            return set()
        globs = tuple(wanted_globs)
        hits: set[Path] = set()
        for t in all_tests:
            rel = t.relative_to(self.repo_root).as_posix()
            if self._matches_glob(rel, globs):
                hits.add(t)
        return hits

    # ----------------------------------------------------------- selection --- #
    def select(self, base: str | None, commit_message: str = "") -> Selection:
        all_tests = self._all_test_files()

        if not base:
            return Selection("all", all_tests, "no base ref (push/manual) -> full suite")

        if not self._rev_exists(base):
            return Selection("all", all_tests, f"base ref {base!r} not resolvable -> full suite")

        merge_base = self._merge_base(base)
        if not merge_base:
            # Shallow clone that doesn't reach the fork point, or unrelated history:
            # a diff here would be wrong, so don't risk a narrow (false-negative) run.
            return Selection("all", all_tests, f"no merge-base with {base!r} (shallow clone?) -> full suite")

        messages = commit_message + "\n" + self._commit_messages(merge_base)
        for tag in self.RUN_ALL_COMMIT_TAGS:
            if tag in messages:
                return Selection("all", all_tests, f"commit message contains {tag!r} -> full suite")

        changed, deleted = self._diff_files(merge_base)

        triggers = [p for p in changed if self._matches_glob(p, self.run_all_globs)]
        if triggers:
            shown = ", ".join(triggers[:5]) + (" ..." if len(triggers) > 5 else "")
            return Selection("all", all_tests, f"changed shared/infra file(s) [{shown}] -> full suite")

        # --- import-graph narrowing ------------------------------------------ #
        source_files = self._all_source_files()
        module_index, dir_local = self._build_indexes(source_files)
        deps_by_file, raw_by_file = self._parse_all(source_files, module_index, dir_local)
        reverse = self._reverse_graph(deps_by_file)

        # #5: a deleted .py only forces a full run if a *surviving* file still
        # imports it (a dangling import the graph can't follow). A clean deletion --
        # where the PR also removed/updated every importer -- is covered by those
        # importer edits (which are in ``changed`` and seeded below), so we don't
        # blanket-run on it any more.
        deleted_modules = {
            m
            for p in deleted if (m := self._rel_to_module(p)) and not self._is_test_file(self.repo_root / p)
        }
        if deleted_modules:
            dangling = self._dangling_importers(deleted_modules, raw_by_file)
            if dangling:
                shown = ", ".join(sorted(deleted_modules)[:5])
                return Selection(
                    "all",
                    all_tests,
                    f"deleted module(s) [{shown}] still imported by surviving file(s) -> full suite",
                )

        selected: set[Path] = set()
        for rel in changed:
            if not self._under_sources(rel):
                continue  # non-Python / out-of-scope change with no direct test impact
            path = (self.repo_root / rel).resolve()
            if path.name == "conftest.py":
                # A directory-scoped conftest affects every test under that directory.
                for t in all_tests:
                    if str(t).startswith(str(path.parent) + os.sep):
                        selected.add(t)
                continue
            impacted = self._impacted_files({path}, reverse)
            selected.update(f for f in impacted if self._is_test_file(f))
            if self._is_test_file(path):
                selected.add(path)  # a brand-new test file has no importers yet

        # #4: add dynamic-edge tests the static graph can't see.
        selected |= self._dynamic_edge_tests(changed, all_tests)

        if not selected:
            scope = ", ".join(self.config.test_scopes)
            return Selection("none", [], f"no {scope} tests impacted by this diff")
        if selected >= set(all_tests):
            return Selection("all", sorted(selected), "diff impacts the whole suite -> full suite")
        return Selection("subset", sorted(selected), f"{len(selected)} test file(s) impacted by the diff")

    @staticmethod
    def _dangling_importers(deleted_modules: set[str], raw_by_file: dict[Path, set[str]]) -> set[Path]:
        """Surviving files whose imports still reference a deleted module."""
        dangling: set[Path] = set()
        for f, raw in raw_by_file.items():
            for name in raw:
                if name in deleted_modules or any(name.startswith(m + ".") for m in deleted_modules):
                    dangling.add(f)
                    break
        return dangling

    # ------------------------------------------------------------- explain --- #
    def explain(self, base: str | None) -> str:
        """Human-readable trace of why each test was selected (#7)."""
        sel = self.select(base)
        lines = [f"# Test selection for workflow {self.config.name!r}", f"# decision: {sel.mode} -- {sel.reason}", ""]
        if sel.mode != "subset":
            scope = ", ".join(self.config.test_scopes)
            lines.append(
                f"(mode={sel.mode}: running {'the full ' + scope + ' suite' if sel.mode == 'all' else 'no tests'})")
            return "\n".join(lines)

        merge_base = self._merge_base(base) if base else ""
        changed, _ = self._diff_files(merge_base) if merge_base else ([], [])
        changed_src = [self.repo_root / r for r in changed if self._under_sources(r)]

        source_files = self._all_source_files()
        module_index, dir_local = self._build_indexes(source_files)
        deps_by_file, _ = self._parse_all(source_files, module_index, dir_local)
        reverse = self._reverse_graph(deps_by_file)

        selected = set(sel.tests)
        for seed in sorted(changed_src):
            parent = self._reachable_with_parents(seed, reverse)
            seed_tests = sorted(t for t in selected if t in parent)
            rel_seed = seed.relative_to(self.repo_root).as_posix()
            if not seed_tests:
                lines.append(f"{rel_seed}: (no impacted tests in scope)")
                continue
            lines.append(f"{rel_seed} impacts:")
            for t in seed_tests:
                chain = []
                node: Path | None = t
                while node is not None:
                    chain.append(node.relative_to(self.repo_root).as_posix())
                    node = parent.get(node)
                lines.append("    " + " <- ".join(chain))
            lines.append("")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI / CI glue.
# --------------------------------------------------------------------------- #
def _emit_github_output(mode: str, num_tests: int, reason: str) -> None:
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if not gh_out:
        return
    with open(gh_out, "a", encoding="utf-8") as fh:
        fh.write(f"mode={mode}\n")
        fh.write(f"num_tests={num_tests}\n")
        fh.write(f"reason={reason}\n")


def _write_step_summary(config: WorkflowConfig, sel: Selection, targets: list[str]) -> None:
    """Emit a GitHub job summary explaining the decision (#6)."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    # chr(96) is a backtick; kept out of f-string literals so flake8's W604
    # (Python-2 backtick) check -- which doesn't mute f-string contents -- won't fire.
    bt = chr(96)

    def code(text: object) -> str:
        return f"{bt}{text}{bt}"

    lines = [
        f"### Test selection: {code(config.name)}",
        "",
        f"- **decision:** {code(sel.mode)}",
        f"- **reason:** {sel.reason}",
        f"- **pytest targets:** {len(targets)}",
        "",
    ]
    if sel.mode == "subset":
        lines.append("<details><summary>selected test files</summary>")
        lines.append("")
        for t in targets[:200]:
            lines.append(f"- {code(t)}")
        if len(targets) > 200:
            lines.append(f"- ... and {len(targets) - 200} more")
        lines.append("")
        lines.append("</details>")
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError as e:
        print(f"WARNING: could not write job summary: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--workflow",
        default=DEFAULT_WORKFLOW,
        choices=sorted(WORKFLOWS),
        help="Which CI workflow's scope/config to select for. Default: %(default)s.",
    )
    parser.add_argument(
        "--base",
        default="origin/master",
        help="Git ref to diff against (merge-base is used). Empty string -> run all tests. "
        "Default: origin/master.",
    )
    parser.add_argument(
        "--output-file",
        default="ci/.test_selection/test_list.txt",
        help="Where to write the test targets to run (one per line). "
        "mode=all writes the scope dir(s); mode=subset writes individual files; mode=none writes nothing.",
    )
    parser.add_argument(
        "--commit-message",
        default="",
        help="Commit message to scan for [test all] / [no filter] override tags.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print the import chains from changed files to selected tests, then exit.",
    )
    args = parser.parse_args()

    config = WORKFLOWS[args.workflow]
    selector = TestSelector(REPO_ROOT, config)

    if args.explain:
        try:
            print(selector.explain(args.base))
        except Exception:  # noqa: BLE001 -- explain is a diagnostic, never fail CI on it
            traceback.print_exc()
        return

    # #3: any unexpected failure in the selector degrades to running EVERYTHING,
    # never to running nothing. A bug here must not silently skip tests.
    try:
        sel = selector.select(args.base, args.commit_message)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sel = Selection("all", selector._all_test_files(),
                        "selector raised an unexpected error -> full suite (fail-safe)")

    if sel.mode == "all":
        targets = list(config.test_scopes)
    else:
        targets = [p.relative_to(REPO_ROOT).as_posix() for p in sel.tests]

    out_path = (REPO_ROOT / args.output_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(targets) + ("\n" if targets else ""), encoding="utf-8")

    print(f"\n### MODE: {sel.mode} ({sel.reason}) ###")
    print(f"### {len(targets)} pytest target(s) -> {out_path.relative_to(REPO_ROOT)} ###")
    for t in targets:
        print(f"  {t}")

    _emit_github_output(sel.mode, len(sel.tests), sel.reason)
    _write_step_summary(config, sel, targets)


if __name__ == "__main__":
    main()

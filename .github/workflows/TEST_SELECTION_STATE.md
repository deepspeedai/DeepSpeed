# Test-selection — implementation state / handoff

Working notes for the diff-based CI test-selection system, so the work can be
resumed without re-deriving context. For *how the system works / how to use it*,
see [`TEST_SELECTION.md`](TEST_SELECTION.md); this file is only about the **state
of the implementation**.

_Last updated: 2026-06-18._


## Status: complete & locally verified; in review on PR #8077

The system is implemented end-to-end and passes local checks. Opened as
[deepspeedai/DeepSpeed#8077](https://github.com/deepspeedai/DeepSpeed/pull/8077)
(scoped to `modal-torch-latest` first; replicate to other workflows if accepted).
It has **not** yet run a live `subset`/`deploy` on modal, so the first real runs
should be watched.

### Review fixes applied (Codex bot, PR #8077)

- **P1 — run the selector from trusted code.** `collect-tests` now restores `ci/`
  from the base branch before running the selector + self-tests, so a PR can't
  rewrite `tests_fetcher.py` to emit `mode=none` and skip the Required check. (A
  PR's `ci/` changes still show in the diff and force a full run via `ci/**`.)
  Includes a bootstrap fallback: if base has no `tests_fetcher.py` yet, run all.
- **P1 — hidden-file artifact.** `actions/upload-artifact@v4` skips dot-prefixed
  paths by default; `ci/.test_selection/` is one. Added `include-hidden-files:
  true` so `deploy`'s download doesn't fail.

> Reminder: under `pull_request_target`, `deploy` restores `ci/` from the base
> branch, so changes under `ci/*` (the fetcher included) only take effect once
> **merged**. To validate the live behavior before merge, trigger via a
> `pull_request`-based run or the `modal` CLI.


## What was built

A diff-driven selector that, on a PR, runs only the `tests/unit/v1` tests impacted
by the changed files, with a fail-safe fallback to the full suite. Driven by the
`modal-torch-latest` workflow.

### Files

| File | State | Notes |
| --- | --- | --- |
| `ci/tests_fetcher.py` | new/rewritten | Config-driven `TestSelector` class; import-graph selector. |
| `ci/test_tests_fetcher.py` | new | 12 stdlib self-tests over synthetic git repos. |
| `ci/torch_latest.py` | modified | Reads `DS_TEST_LIST_FILE`, runs pytest on the selected targets (falls back to full `tests/unit/v1`). |
| `.github/workflows/modal-torch-latest.yml` | rewritten | `collect-tests` (fetcher) gates `deploy` (modal). |
| `.github/workflows/TEST_SELECTION.md` | new | Full design + usage doc. |
| `.github/workflows/TEST_SELECTION_STATE.md` | new | This file. |
| `CONTRIBUTING.md` | modified | "Diff-based CI test selection" section + link. |
| `.gitignore` | modified | Ignores `ci/.test_selection/`. |


## Key design decisions (and why)

- **Always-runs Required job, selects instead of skips.** `modal-torch-latest` is a
  Required check, so path filters can't skip it. `collect-tests` decides
  `all|subset|none`; `none` skips `deploy` while still satisfying the Required
  status.
- **`deploy` tests PR code but restores `ci/` from base.** Selection must reflect
  PR code, but the orchestration that holds modal/HF secrets must not be
  PR-controlled. (Security trade-off accepted earlier in the work.)
- **Opaque hub modules + curated run-all globs.** `deepspeed`, `deepspeed.comm`,
  `deepspeed.accelerator` are opaque in the graph to stop universal fan-out; core
  hubs (`deepspeed/__init__.py`, `runtime/engine.py`, `comm/**`, `accelerator/**`,
  …) instead live in the run-all globs. (Coarseness fix accepted earlier.)
- **Fail-safe to `all`, never `none`.** Every uncertain case (no base, no
  merge-base, parse error, unexpected exception) widens to the full suite.

### Improvements implemented in the latest pass (items #3–#11)

- **#3** Degrade to `all` on any selector error (top-level guard in `main()`).
- **#4** `DYNAMIC_EDGES` map for runtime/dynamic deps the AST can't see
  (seeded: `deepspeed/module_inject/** → tests/unit/v1/moe/**`).
- **#5** Deleted `.py` only forces `all` when a *surviving* file still imports it
  (dangling import); clean deletions no longer over-trigger.
- **#6** GitHub job summary (`$GITHUB_STEP_SUMMARY`) with mode/reason/file list;
  `reason` also on `$GITHUB_OUTPUT`.
- **#7** `--explain` prints import chains from changed files to selected tests.
- **#8** Escape hatches documented in `CONTRIBUTING.md` + `TEST_SELECTION.md`.
- **#9** Self-tests in `ci/test_tests_fetcher.py`, also run in `collect-tests`.
- **#10** Robust merge-base: explicit check, `all` if missing; workflow keeps
  `fetch-depth: 0` with rationale.
- **#11** Multi-workflow config: `WORKFLOWS` registry + `WorkflowConfig` +
  `--workflow` flag.


## Verification done locally

- `python ci/tests_fetcher.py` self-tests: **12/12 pass**
  (narrowing, shared importers, core/comm/fixture run-all, commit tag, docs-only →
  none, new test file, clean vs. dangling delete, dynamic edge, missing base).
- Real-repo smoke: `--base`, `--explain`, `--workflow` validation all behave.
- Confirmed `$GITHUB_OUTPUT` + `$GITHUB_STEP_SUMMARY` emission and the #3
  error-fallback path directly.
- No linter errors.

Re-run the core check any time:

```bash
python ci/test_tests_fetcher.py
python ci/tests_fetcher.py --base origin/master --explain
```


## Open items / possible follow-ups

- **Live validation:** watch the first real PR; confirm `collect-tests` artifact
  hand-off and `deploy` gating behave on Actions + modal.
- **`DYNAMIC_EDGES` is minimal.** Only one seeded edge. Expand as blind spots are
  found (tests that fail on `master`'s full run but weren't selected on their PR
  are the signal).
- **Single workflow wired.** The engine is config-driven (#11) but only
  `modal-torch-latest` is registered/wired. Other GPU workflows could adopt it by
  adding a `WorkflowConfig` and mirroring the `collect-tests` job.
- **Run-all glob tuning.** The core-hub list is a conservative first cut; revisit
  if it proves too coarse (full runs on minor edits) or too narrow (missed impact).
- **No telemetry.** Consider logging selection stats over time to tune
  globs/opaque-modules against real false-positive/negative rates.

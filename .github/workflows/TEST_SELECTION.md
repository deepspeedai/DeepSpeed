# Diff-based CI test selection

This document explains the diff-driven test-selection system that lets some GPU CI
workflows run only the tests a PR could affect, instead of the whole suite, while
still satisfying Required status checks.

It currently drives **`modal-torch-latest`** (which runs `tests/unit/v1/` on
[modal.com](https://modal.com) GPUs), but is built to drive more workflows from
one config — see [Adding a workflow](#adding-a-new-workflow).

- [TL;DR](#tldr)
- [Why](#why)
- [Moving parts](#moving-parts)
- [How a decision is made](#how-a-decision-is-made)
- [Driving it (as a contributor)](#driving-it-as-a-contributor)
- [Changing it (as a maintainer)](#changing-it-as-a-maintainer)
- [Security model](#security-model)
- [Failure modes & guarantees](#failure-modes--guarantees)
- [FAQ / troubleshooting](#faq--troubleshooting)


## TL;DR

- On a PR, `ci/tests_fetcher.py` diffs your branch against the base branch, traces
  the import graph from your changed files to the impacted tests, and writes the
  list to `ci/.test_selection/test_list.txt`.
- The workflow runs pytest on just that list. `push` to `master` and manual runs
  always run everything.
- It is **fail-safe**: anything it can't reason about safely → run the *full* suite.
  It never silently runs *fewer* tests than reality.
- Preview locally:
  ```bash
  python ci/tests_fetcher.py --base origin/master            # what would CI run?
  python ci/tests_fetcher.py --base origin/master --explain  # ...and why?
  ```
- Force a full run: put `[test all]` (or `[no filter]`) in a commit message.


## Why

`modal-torch-latest` is a **Required** check, so it must report a status on every
PR — which means we can't use GitHub's path filters (`on.<event>.paths`) to skip
it, because a skipped Required job blocks merges. Instead, the job always runs but
*selects* what to test: a docs-only PR resolves to "no impacted tests" and exits
fast (still green), while a PR touching a leaf module runs only the tests that
import it.

The design is a small, self-contained take on HuggingFace `transformers`'
`utils/tests_fetcher.py`.


## Moving parts

| File | Role |
| --- | --- |
| `.github/workflows/modal-torch-latest.yml` | The workflow: a `collect-tests` job (runs the fetcher) gating a `deploy` job (runs pytest on modal). |
| `ci/tests_fetcher.py` | The selector. AST-parses the repo, builds an import graph, decides `all` / `subset` / `none`, writes the test-list file, emits a job summary. |
| `ci/test_tests_fetcher.py` | Self-tests for the selector (pure stdlib; run in `collect-tests`). |
| `ci/torch_latest.py` | The modal runner. Reads the test-list file and feeds it to pytest. |
| `ci/.test_selection/test_list.txt` | The hand-off artifact (one pytest target per line). Git-ignored. |

### Job flow

```
                pull_request_target / push / workflow_dispatch
                                  │
                    ┌─────────────┴─────────────┐
                    │        collect-tests       │   (no secrets; AST-only)
                    │  checkout PR head          │
                    │  restore ci/ from base     │
                    │  self-test the fetcher     │
                    │  run tests_fetcher.py      │
                    │   → mode (all|subset|none) │
                    │   → upload test_list.txt   │
                    └─────────────┬──────────────┘
                                  │ needs:
                                  ▼
                    ┌────────────────────────────┐
                    │           deploy            │   (has modal/HF secrets)
                    │  if mode != none (or manual │
                    │     / collector failed)     │
                    │  checkout PR head           │
                    │  restore ci/ from base      │
                    │  download test_list.txt     │
                    │  modal run ci.torch_latest  │
                    └────────────────────────────┘
```

`mode` controls `deploy`:

- **`none`** → `deploy` is skipped. The Required status is still satisfied (a
  skipped dependent job counts as success here).
- **`subset`** → `deploy` runs pytest on exactly the impacted files.
- **`all`** → `deploy` runs the whole scope (`tests/unit/v1`).


## How a decision is made

`TestSelector.select()` in `ci/tests_fetcher.py` runs these checks in order; the
first that matches wins:

1. **No base ref** (push / manual) → `all`.
2. **Base ref unresolvable** → `all`.
3. **No merge-base** with the base (e.g. shallow clone, unrelated history) → `all`.
   A diff here would be wrong, so we never narrow on it.
4. **Commit message tag** `[test all]` / `[no filter]` anywhere on the branch → `all`.
5. **A changed file matches a run-all glob** (`COMMON_RUN_ALL_GLOBS` +
   the workflow's `extra_run_all_globs`) → `all`. These are files too central or
   too dynamic to narrow safely: CI scripts, build system, `csrc/`, `op_builder/`,
   `accelerator/`, shared fixtures (`tests/unit/common.py`, `tests/conftest.py`,
   `pytest.ini`), and core runtime hubs (`deepspeed/__init__.py`,
   `deepspeed/runtime/engine.py`, `deepspeed/comm/**`, `deepspeed/accelerator/**`, …).
6. **A deleted module is still imported** by a surviving file (a dangling import
   the graph can't follow) → `all`. A *clean* deletion (importers removed/updated
   in the same PR) does **not** trigger this.
7. Otherwise, **narrow via the import graph** (below). If nothing is impacted →
   `none`; if everything is → `all`; else → `subset`.

### The import graph

- Nodes are Python files under the package roots: `deepspeed/**` and the `unit`
  test-helper package at `tests/unit/**`.
- An edge `A → B` means "B imports A". The selector walks **backwards** from each
  changed file to every test that (transitively) imports it.
- **Opaque hub modules** (`OPAQUE_MODULES`: `deepspeed`, `deepspeed.comm`,
  `deepspeed.accelerator`): almost every test imports `unit.common`, which imports
  these. Their `__init__.py` files eagerly pull in huge subtrees, so if treated as
  normal nodes *any* `deepspeed/**` change would fan out to the whole suite. We
  therefore don't expand their `__init__` imports; instead, changes to the hubs
  themselves are caught by the run-all globs in step 5.
- **`conftest.py`** changes select every test under that conftest's directory.
- **New test files** are selected directly (they have no importers yet).

### Dynamic edges (`DYNAMIC_EDGES`)

Some dependencies are wired at runtime (monkey-patching, plugin/registry lookup,
JIT-loaded ops, `deepspeed.initialize()`-time `replace_module` injection), so a
test can depend on code it never `import`s. `DYNAMIC_EDGES` is a curated map of
`changed-file glob → extra test-path globs` that patches these blind spots. It is
additive on top of the static graph (and is only consulted if step 5 didn't
already short-circuit to `all`).


## Driving it (as a contributor)

### Preview what CI will run

The fetcher is pure stdlib — no DeepSpeed/torch install needed, just `git`:

```bash
# Selection for your branch vs. the base branch
python ci/tests_fetcher.py --base origin/master
cat ci/.test_selection/test_list.txt

# Explain *why* each test was selected (prints import chains)
python ci/tests_fetcher.py --base origin/master --explain
```

`--explain` output looks like:

```
deepspeed/shared.py impacts:
    tests/unit/v1/test_shared.py <- deepspeed/shared.py
    tests/unit/v1/moe/test_moe.py <- tests/unit/v1/moe/test_moe.py <- deepspeed/shared.py
```

### Escape hatches

- **Force the full suite for a push:** include `[test all]` (or `[no filter]`)
  anywhere in a commit message on the branch.
- **Touch an infra file:** any change to a run-all glob runs everything.
- **Found a missed test?** It's likely a runtime/dynamic dependency the static
  graph can't see — add a `DYNAMIC_EDGES` entry (see below) and/or report it.


## Changing it (as a maintainer)

All knobs live in `ci/tests_fetcher.py`. After any change, run the self-tests:

```bash
python ci/test_tests_fetcher.py          # standalone
# or: pytest ci/test_tests_fetcher.py
```

### Add a run-all trigger

Add a glob to `TestSelector.COMMON_RUN_ALL_GLOBS` (shared across workflows) or to
a specific workflow's `extra_run_all_globs`. Globs match repo-root-relative POSIX
paths; `dir/**` matches everything under `dir/`.

### Add a dynamic edge

Add to `TestSelector.DYNAMIC_EDGES`:

```python
DYNAMIC_EDGES = {
    # changed-file glob : test-path globs to pull in when it changes
    "deepspeed/module_inject/**": ("tests/unit/v1/moe/**",),
}
```

Keep entries conservative: too broad just wastes GPU time; too narrow misses
coverage. Prefer fixing it here over widening a run-all glob when only a slice of
tests is truly affected.

### Mark a module opaque

If a new universal hub starts fanning every change out to the whole suite, add it
to `OPAQUE_MODULES` and (usually) add the hub file itself to the run-all globs so
real changes to it still run everything.

### Add a new workflow

The engine is config-driven (`WORKFLOWS` registry of `WorkflowConfig`). To drive
another workflow:

1. Add an entry:
   ```python
   WORKFLOWS["my-workflow"] = WorkflowConfig(
       name="my-workflow",
       test_scopes=("tests/unit/v2",),               # dirs this workflow runs
       extra_run_all_globs=(".github/workflows/my-workflow.yml",),
   )
   ```
2. In that workflow's YAML, mirror `modal-torch-latest.yml`'s `collect-tests` job
   and call the fetcher with `--workflow my-workflow`.
3. Point the runner at `ci/.test_selection/test_list.txt`.
4. Add coverage to `ci/test_tests_fetcher.py` if the scope/behavior differs.

### Add a self-test

`ci/test_tests_fetcher.py` builds throwaway git repos (`TmpRepo`) from a synthetic
`BASELINE` tree and asserts the resulting `mode`/`tests`. Add a `test_*` function
for any new behavior; it runs both standalone and under pytest, and executes in
the `collect-tests` CI job, so a broken selector is caught before it mis-picks
tests.


## Security model

The workflow triggers on **`pull_request_target`**, so it runs in the base repo's
context and the `deploy` job has the modal/HF **secrets** in scope. To keep PRs
from abusing that:

- `collect-tests` holds **no secrets** and only **AST-parses** (never executes)
  the PR tree.
- **Both jobs restore `ci/` from the base branch** before using it:
  - `collect-tests` runs the **base** selector + self-tests. This job decides
    whether the Required `deploy` runs, so it must not trust the PR's own
    `ci/tests_fetcher.py` — otherwise a PR could rewrite it to emit `mode=none`
    and skip CI while still going green. The diff is computed from git history, so
    a PR's `ci/` changes still appear in the diff and (via the base selector's
    `ci/**` run-all glob) force a full run.
  - `deploy` runs the **base** orchestration (which drives `modal run` with the
    secrets), so a PR can't repoint it at the secrets to exfiltrate them.
  In both jobs the PR's `deepspeed/` + `tests/` are what gets parsed/tested.
- The `pull_request_target` trigger types are `review_requested`,
  `ready_for_review`, and `synchronize`. Because `synchronize` re-runs on every
  push to an open PR (not just on a maintainer action), the maintainer review is a
  mitigation, **not** an absolute barrier — the base-`ci/` restore above is the
  primary protection for the secrets.

> **Consequence:** changes to `ci/*` (including `tests_fetcher.py` itself) take
> effect under `pull_request_target` only after they're **merged**. Validate
> `ci/*` changes via a `pull_request`-triggered run or the `modal` CLI locally.
>
> **Bootstrap:** when the base branch has no selector yet (the PR that introduces
> it), the restored base `ci/` won't contain `tests_fetcher.py`; `collect-tests`
> detects this and falls back to running the full suite.


## Failure modes & guarantees

The selector is built to **fail safe — to `all`, never to `none`**:

- Missing/unresolvable base, no merge-base, a parse error on a file, or **any
  unexpected exception** in the selector → it falls back to the full suite (the
  top-level handler in `main()` logs the traceback and sets `mode=all`).
- If `collect-tests` fails entirely, `deploy` still runs (and the workflow has an
  explicit "fail if selection failed" guard) so a broken collector can't let a PR
  pass without testing.
- The only way to run *fewer* tests is a clean, well-understood narrow decision;
  every uncertain case widens to everything.

Every run writes a summary to the GitHub job summary (mode, reason, and the
selected files) so the decision is auditable from the Actions UI.


## FAQ / troubleshooting

**My PR shows `mode=none` but I changed code.**
Either the change is non-Python / out of the workflow's scope, or your edits
aren't committed (the fetcher diffs *committed* history, `merge-base..HEAD`).

**A relevant test wasn't selected.**
Likely a runtime/dynamic dependency. Confirm with `--explain`, then add a
`DYNAMIC_EDGES` entry. As a stop-gap, push with `[test all]`.

**Everything runs even for a tiny change.**
You touched a run-all glob (CI/build/shared fixture/core runtime), or a hub module
fanned out. Check the job summary's `reason`. If a hub over-fans, consider
`OPAQUE_MODULES`.

**It ran the full suite and the summary says "shallow clone?" / "no merge-base".**
The runner didn't have enough history to compute the diff. `collect-tests` uses
`fetch-depth: 0`; if you changed that, restore full history for the base.

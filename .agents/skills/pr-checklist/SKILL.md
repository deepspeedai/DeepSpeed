---
name: pr-checklist
description: Prepare a DeepSpeed change for human PR review or re-review, or analyze an open pull request. Use to check design fit, accelerator portability, backward compatibility, on-device validation, diff quality, and closure of previous feedback before requesting maintainer attention.
---

# PR Checklist

This SKILL systematically enforces DeepSpeed's code quality and contribution standards during the
pull request process, to reduce the burden of manual code review on maintainers.

It assumes the repo-level rules in [`AGENTS.md`](../../../AGENTS.md) (duplicated as `CLAUDE.md`) and
[`CONTRIBUTING.md`](../../../CONTRIBUTING.md) are already in context; this checklist adds the
review-time judgement that those rule lists do not cover.

## Usage

Use this checklist to polish a change in preparation for a DeepSpeed pull request, to analyze an
open pull request (as author or as reviewer), or to confirm contribution standards are met before
requesting maintainer attention. Go through each item and verify it has been addressed in the
diff / pull request.

This is intended to be used with a human in the loop. When running programmatically, prepare notes
on every section to produce a thorough report, with a clear header summarizing findings and areas
needing attention. When running interactively, present that report with a concrete list of
actionable follow-ups and design decisions to make, then iterate.

Mark each item **OK**, **Needs work**, or **N/A** — and for anything that cannot be verified from
the repo alone (hardware runs, convergence results, upstream issue context), say so explicitly
rather than assuming it passed. A silent omission reads as a pass and is worse than a flagged gap.

### Re-review

To update a PR following a review, read all comments and threads on the PR. For each concrete
concern, work with your human operator to decide a plan of action: either a direct reply, or a
change plus an acknowledging comment. Keep the PR description fresh and all documentation
(including code comments and tutorials) in the change up-to-date.

`gh pr view --comments` does not include inline review comments. Set `pr` to the PR number, then
fetch all pages of conversation comments, review submissions, and inline comments (including
replies):

```bash
gh api --paginate "repos/deepspeedai/DeepSpeed/issues/$pr/comments"
gh api --paginate "repos/deepspeedai/DeepSpeed/pulls/$pr/reviews"
gh api --paginate "repos/deepspeedai/DeepSpeed/pulls/$pr/comments"
```

Use `in_reply_to_id` to group inline replies with their parent comments. Use GraphQL
`reviewThreads` when resolved or outdated thread status is needed.

Also re-check the mechanical gates on every re-review, since a rebase or a squash can drop them:

```bash
gh api --paginate "repos/deepspeedai/DeepSpeed/pulls/$pr/commits" --jq '.[] | "\(.sha[0:9]) \(.commit.message)"'
gh pr checks "$pr"
```

## Sections

### 1. Design Fit

Does the change align with DeepSpeed's architecture, or does it introduce design inconsistencies,
anti-patterns, or long-term maintenance concerns?

#### 1.1: Impact on core components

DeepSpeed's core runtime is shared by every training deployment. Changes to these files have a very
large blast radius and must be minimally invasive:

- The engine: `deepspeed/runtime/engine.py`, `deepspeed/runtime/pipe/engine.py`
- ZeRO: `deepspeed/runtime/zero/stage_1_and_2.py`, `zero/stage3.py`,
  `zero/partition_parameters.py`, `zero/partitioned_param_coordinator.py`,
  `zero/parameter_offload.py`
- Optimizer wrappers: `deepspeed/runtime/bf16_optimizer.py`, `runtime/fp16/*`,
  `runtime/base_optimizer.py`
- Shared plumbing: `deepspeed/runtime/utils.py`, `runtime/config.py`, `runtime/constants.py`
- Collectives and devices: `deepspeed/comm/*`, `accelerator/abstract_accelerator.py`,
  `accelerator/real_accelerator.py`

This is not a guess at what matters: the `hpu-gaudi2` and `xpu-max1100` workflows enumerate almost
exactly this list in their `pull_request.paths` triggers, because a change to any of them can break
a vendor accelerator. Touching one of these files means your PR is a core-runtime PR.

Several of these files are already very large (`engine.py` is ~6k lines, `stage3.py` ~4k). Adding
another branch to an existing hot method is the cheapest change to write and the most expensive to
maintain. Prefer isolating new behavior behind a small, well-named helper or a separate module, and
keep the core-file delta to the wiring.

Nontrivial logic for non-default functionality must not be added to core/common code paths. If a
feature is opt-in via config, its logic should live outside the always-executed path, not as an
`if self._feature_enabled:` block threaded through the training step.

#### 1.2: Replication

Does the change duplicate code that already exists? DeepSpeed has many near-parallel
implementations by design (ZeRO 1/2 vs 3, fp16 vs bf16 optimizers, per-accelerator op builders), so
"there is already similar code" is not automatically a defect — but it does raise the bar:

- If the new code is a near-copy of an existing path, say why the paths cannot be unified, or unify
  them.
- If existing functionality could be reused (`runtime/utils.py`, `deepspeed/comm`,
  `deepspeed/checkpoint`, `tests/unit/common.py` helpers), reuse it.
- When adding functionality, look for parallels with existing designs and take the opportunity to
  simplify and unify, rather than adding a third variant alongside two others.

Justify any duplication that exists to isolate new functionality from existing workflows.

#### 1.3: Complexity

Scrutinize changes that increase codebase complexity — especially new abstractions, new
dependencies, or new branching in existing components. Consider whether the added complexity could
be mitigated through refactoring or modularization.

As a coarse indicator, estimate complexity by new lines of code added to `deepspeed/`, `csrc/`, and
`op_builder/` (excluding tests and docs):

- **Low**: < 50 lines. Easy to review, unlikely to add significant complexity.
- **Medium**: 50-250 lines. Requires careful review of maintainability impact.
- **High**: > 250 lines. May add significant complexity; requires thorough justification.

Line count is only a flag, not a verdict — use judgement about true semantic complexity and the
review burden implied. 300 lines of a new standalone op builder is far cheaper to review than 40
new lines spread across `engine.py` and both ZeRO optimizers.

Bug fixes should stay below the High bar. Redesigning core components, changing core interfaces, or
adding significant complexity in order to patch a bug is usually discouraged and needs explicit
justification.

A PR that adds significant complexity or a large amount of new code must be well justified: a
feature should matter to enough users to warrant its size, and a performance optimization should
deliver end-to-end gains on workloads that matter, large enough to outweigh what it costs in
maintenance.

New features additionally follow the three-step process in
[`CONTRIBUTING.md`](../../../CONTRIBUTING.md#new-feature-contribution-guidelines): proposal issue
and design discussion *first*, then implementation (code + unit tests + docs + tutorial, plus a
DeepSpeedExamples PR), then maintenance commitment with the author's GitHub username recorded as a
contact. If a feature PR arrives with no linked proposal issue, flag it — that is a process gap the
author should close before maintainers invest review time.

#### 1.4: Accelerator and device portability

This is DeepSpeed-specific and one of the most common sources of review churn. DeepSpeed runs on
CUDA, ROCm, XPU, HPU, NPU, MLU, SDAA, MPS, and CPU.

- **Never call `torch.cuda.*`.** Use `get_accelerator()` from `deepspeed.accelerator`. The
  `check-torchcuda` pre-commit hook enforces this; the `#ignore-cuda` escape is for genuinely
  CUDA-only code only, and using it in shared runtime code is a design smell worth questioning.
- **Never import `torch.distributed` directly.** Use `import deepspeed.comm as dist`, enforced by
  `check-torchdist`.
- Avoid hard-coded `"cuda"` / `"nccl"` strings, `.cuda()` calls, device indices, and
  `torch.cuda.Stream`/`Event` types; route them through the accelerator interface
  (`get_accelerator().device_name()`, `.communication_backend_name()`, `.Stream()`, etc.).
- If you add a method to `accelerator/abstract_accelerator.py`, every concrete accelerator needs an
  implementation — not just `cuda_accelerator.py`. A missing override is an `AttributeError` for a
  vendor at runtime, and their CI is nightly, so it will be found late.
- New kernels: check whether `op_builder/` needs per-accelerator entries
  (`op_builder/{cpu,hpu,npu,xpu,mlu,sdaa,mps}/`) and whether the op degrades gracefully when the
  builder is unavailable. `no-torch` CI builds the sdist with no torch installed, so import-time
  code must not require torch or a device.

#### 1.5: Correctness and compatibility

Ensure changes are sound and do not violate existing contracts. Trace the alternative backends and
call sites that share the modified flow, not just the configuration you tested.

DeepSpeed's compatibility surface is a matrix, and a change that is correct for one cell is often
wrong for another. Walk the axes that your change touches:

- **ZeRO stage**: 0 / 1 / 2 / 3, and `zero_init` / `zero.Init()` contexts
- **Offload**: none / CPU / NVMe, for optimizer state and for parameters
- **Precision**: fp32, fp16 (loss scaling), bf16, `torch_autocast`, fp8/quantized paths
- **Parallelism**: data, pipeline, tensor parallel / AutoTP, sequence parallel (Ulysses), MoE +
  expert parallel
- **Other interacting features**: activation checkpointing, gradient accumulation,
  `torch.compile`, ZenFlow / SuperOffload, flops profiler, monitors
- **Integrations**: HF Transformers, Accelerate, Lightning, MII — the `nv-transformers-v100`,
  `nv-accelerate-v100`, `nv-lightning-v100`, `nv-mii`, and `nv-ds-chat` workflows exist because
  these break in practice

Three contracts deserve explicit attention because breaking them hurts users silently:

1. **Config JSON.** Users have `ds_config.json` files in production. Do not rename or remove a
   config key outright — add the new field and mark the old one `deprecated=True` on the
   `DeepSpeedConfigModel` subclass (see `deepspeed/runtime/config_utils.py` for
   `deprecated_msg` / `new_param` / `new_param_fn`). Document new keys in
   `docs/_pages/config-json.md`.
2. **Checkpoints.** Checkpoints written by an older DeepSpeed must still load, and checkpoints
   written by this change must still convert. If you change what goes into optimizer or parameter
   state, check `deepspeed/checkpoint/` (`zero_checkpoint.py`, `universal_checkpoint.py`,
   `ds_to_universal.py`) and `runtime/state_dict_factory.py`, and test a save/load round-trip
   across a world-size or stage change.
3. **Public API.** `deepspeed.initialize()`, the engine's public methods, and
   `deepspeed/__init__.py` exports are depended on by downstream frameworks. Keep signatures
   backward compatible or state the break and its justification explicitly in the PR.

Identify any incompatibilities or breaking changes and justify them explicitly in the PR notes.

#### 1.6: Tradeoffs and distributed-execution hazards

Assess the downsides of a design change, feature, or optimization. Does a default change that helps
one configuration hurt another? Could an architectural change accelerate one workload while
breaking feature compatibility or regressing another? Note tradeoffs clearly and make sure
validation is broad enough to catch lurking regressions.

For DeepSpeed the recurring hazards are memory, synchronization, and collectives:

- **Memory.** A speedup bought with a persistent extra buffer per parameter is often a bad trade in
  a library whose purpose is to fit large models in limited memory. Report peak memory alongside
  throughput, and keep lifetimes tight — a tensor held past its use (in a dict, a closure, or a
  hook) is a leak the user sees as OOM at scale.
- **Host synchronization in the step.** `.item()`, `.cpu()`, `.tolist()`, `.numpy()`, printing a
  tensor, `assert` on a tensor value, boolean-tensor indexing, and `torch.cuda`-style
  `synchronize()` all stall the device. In a per-step or per-parameter path these are performance
  bugs even when correct.
- **Stream and event ordering.** Offload and prefetch paths use side streams. A copy issued on a
  non-default stream needs explicit event/`record_stream` ordering before the destination buffer is
  reused or freed, or you get a rare, hardware-dependent silent corruption. Review these changes
  against the surrounding stream discipline rather than in isolation.
- **Collective symmetry.** Every rank must reach every collective, in the same order, with matching
  shapes and dtypes. A collective inside an `if` that depends on rank-local state (a local grad
  norm, an empty-bucket check, a local early return) is a hang. Hangs that only appear at larger
  world sizes or on a different backend are the most expensive bugs in this repo — check the
  condition is globally uniform, or make it uniform with an explicit all-reduce.
- **Determinism and seeding.** Changes affecting RNG, partitioning order, or reduction order can
  alter convergence without failing a shape assertion. If reduction order changes, say so.

### 2. Testing and Validation

All changes to DeepSpeed must be **validated**: bug fixes must be exercised on a reproducer, and
new features or performance optimizations must be run locally and/or benchmarked to confirm
correctness and effectiveness. Contributing a fix without running DeepSpeed end-to-end and
verifying that it resolves the issue is not acceptable.

Beyond validation, most changes should also be **tested**: new tests, updated tests, and continued
reliable coverage of the affected functionality.

#### 2.1: On-device validation is mandatory

Per `AGENTS.md`, integration tests must be *executed* on real devices, not merely written, and the
PR must report the execution hardware and results. A green PR-level CI run is not a substitute,
because most accelerator and large-scale workflows do not run on PR events (see 2.4).

Report, in the PR:

- Hardware and software: device model and count, accelerator/driver version, torch version
- The exact command run (`pytest --forked tests/unit/...`, or the training/inference script)
- The result: pass/fail, and for a bug fix, the failure observed *before* the change on the same
  setup
- Any configuration on the affected matrix axes (1.5) that you could **not** run, and why

If you have no device access for the affected path, say that plainly in the PR rather than implying
coverage. An honest coverage gap is reviewable; an unstated one wastes a maintainer's hardware time.

#### 2.2: Test coverage and effectiveness

Consider both coverage (how much of the change is exercised) and effectiveness (whether the tests
would actually catch a regression).

DeepSpeed's test conventions:

- Distributed tests subclass `DistributedTest` from `tests/unit/common.py`, set `world_size`
  (an int, or a list to run several), and may override per-test with
  `@pytest.mark.world_size(n)`. Pick the smallest world size that can observe the behavior — then
  make sure it actually can: a sharding bug is invisible at `world_size=1`, and an uneven-partition
  bug is invisible at a world size that divides evenly.
- End-to-end training behavior uses a minimal loop with `SimpleModel` from
  `tests/unit/simple_model.py`; `tests/unit/common.py` and `tests/unit/util.py` hold the shared
  helpers.
- Tests verify contracts, not implementations: a different correct implementation of the same
  contract must pass. Assert on observable outcomes through public interfaces — loss values,
  parameter/gradient contents after a step, checkpoint round-trips, raised errors — not on private
  method return values or exact internal strings (unless pinning a specific fixed bug, with a
  comment saying so).
- Before writing a test, name the concrete incorrect behavior it would catch. If you cannot name
  one, do not write it.
- Anchor to an external oracle: compare against a plain-torch reference computation, against ZeRO
  stage 0 / a disabled-feature baseline, or against a known-correct analytic value. Do not
  re-implement the logic under test inside the test.
- Mocks stand in for a collaborator's documented contract, never for internals of the module under
  test. Mocking an accelerator or a collective to make a test pass on CPU usually proves nothing
  about the code path that matters.

Verify the test actually exercises the new path rather than a fallback: assert the new
kernel/backend was selected, that offload actually happened, that the feature was not silently
skipped by an unmet precondition. A test that passes identically with the change reverted is worse
than no test.

Testing should not be exhaustive — local validation establishes baseline correctness, and
contributed tests should target areas prone to error, regression, or complex interaction. Recommend
deleting redundant, trivial, or unnecessary tests; excessive test volume is one of the most common
review complaints. Prefer consolidating into existing test files over adding new ones.

#### 2.3: Test reliability and robustness

Flaky tests undermine confidence and block the merge queue. Analyze added tests for flakiness.
Frequent sources in this repo:

- **Bit-exact float assertions** where minor variation is expected. Use `torch_assert_close` /
  `torch_assert_equal` from `tests/unit/util.py` with tolerances appropriate to the dtype; fp16 and
  bf16 need far looser tolerances than fp32, and a changed reduction or kernel-selection order will
  break an exact-equality assert.
- **Shared state across tests.** `reuse_dist_env` reuses the process group between tests in a
  class; a test that leaves global state behind (an initialized engine, a monkeypatched module, a
  changed accelerator, a cached op build) will break its neighbors or itself under `pytest-xdist`.
  Note that `--forked` is required for distributed CUDA tests.
- **Timing, ports, and filesystem.** Do not assume a fixed port or a shared `/tmp` path; use the
  provided helpers and `tmpdir`. Avoid sleeps and wall-clock assertions.
- **Unavailable hardware.** Skip explicitly on unmet requirements (device availability, device
  count, torch version, `required_torch_version`) rather than letting the test fail or, worse,
  silently pass on a fallback path.
- **Convergence assertions** on a handful of steps. Loss-decrease assertions over a short run are
  inherently noisy; prefer deterministic checks against a baseline configuration.

#### 2.4: CI integration, and what CI does not cover

DeepSpeed's CI is a fleet of per-hardware workflows in `.github/workflows/` (34 of them). Know which
one covers your change, and whether it runs on your PR at all:

- `formatting` (pre-commit) and `dco` run on every PR. `python` and `no-torch` cover packaging and
  torch-free import.
- `cpu-torch-latest`, `nv-torch-latest-v100`, `nv-inference`, `nv-torch-nightly-v100`,
  `nv-flash-attn`, `nv-sd`, and the integration workflows (`nv-transformers-v100`,
  `nv-accelerate-v100`, `nv-lightning-v100`, `nv-mii`, `nv-ds-chat`) cover the mainline paths.
- **Vendor accelerator workflows are path-filtered or nightly.** `hpu-gaudi2`, `xpu-max1100`,
  `xpu-compile`, `amd-mi200`, `mps-torch-latest`, `nv-a6000` only run on PRs that touch their
  declared paths (and otherwise on a nightly schedule). If your change can affect a vendor path but
  does not match the filter, it gets **no** PR coverage — consider whether the filter should be
  extended, and say what you validated instead.
- **`modal-torch-latest` does not run its GPU tests on a plain PR push.** It previews a
  diff-selected test list cheaply on the PR and executes in the merge queue, to conserve GPU quota.
  See [`TEST_SELECTION.md`](../../../.github/workflows/TEST_SELECTION.md).

For the diff-based selector (`ci/tests_fetcher.py`), preview what CI will run before asking for
review:

```bash
python ci/tests_fetcher.py --base origin/master
python ci/tests_fetcher.py --base origin/master --explain   # why a test was (de)selected
```

The selector traces static imports. If your change is reached only dynamically — monkey-patching,
plugin/registry lookup, JIT op builds, injection at `deepspeed.initialize()` time — the relevant
tests will not be selected unless there is a `DYNAMIC_EDGES` entry in `ci/tests_fetcher.py`. Adding
such a code path means checking whether `DYNAMIC_EDGES` needs an entry. Use `[test all]` in a commit
message to force the full suite when in doubt. Note that `ci/*` changes only take effect once
merged, so validate them via a `pull_request`-triggered run or the `modal` CLI.

Add or update a workflow when new tests need resources no current workflow provides. Where CI cannot
exercise the changed path at all, document the missing coverage, the resource constraint, and the
validation performed outside CI — a CI gap never waives the validation requirement in 2.1.

### 3. Code Quality and Style

Code quality is a primary consumer of maintainer effort. Audit the diff for standards adherence,
readability, and maintainability.

#### 3.1: Diff hygiene

Minimizing diff noise is critical. Verify, before requesting review:

```bash
pre-commit run --files $(git diff --name-only master)
```

- **No cosmetic or formatting-only changes to existing code.** Only lines that are functionally
  necessary should move. Reflowing a neighboring block, re-sorting imports you did not add, or
  letting an editor reformat a whole file are all review-time friction — revert them.
- Watch for auto-formatter damage from editing tools. For format-sensitive edits, use `sed`,
  `python`, or a heredoc rather than an editor that silently rewrites quotes, blank lines, or
  wrapping.
- New files need the license header:
  ```
  # SPDX-License-Identifier: Apache-2.0
  # DeepSpeed Team
  ```
- Formatting is yapf (`COLUMN_LIMIT = 119`, `.style.yapf`), flake8 (`.flake8`), clang-format for
  C/C++/CUDA, plus codespell and the local `check-*` hooks.

#### 3.2: Blend in

When modifying code, read the surrounding context and match the style of neighboring code — naming,
spacing, patterns, idioms, logging style, how config is read, how distributed state is accessed.
Code that reads as if it were always there is cheaper to review than code that is merely correct.

Write beginner-friendly code: avoid deeply nested expressions or long chained logic; break complex
expressions into clearly named intermediate steps. This matters especially in the ZeRO and engine
paths, where a dense one-liner is where bugs hide.

#### 3.3: Comments and docstrings

Comments explain **why**, not **what**. Describe purpose and reasoning, not mechanics the code
already shows. Keep them concise and sparse — prefer clear code with meaningful names and fewer
comments.

Docstrings: most short helpers do not need one. Complex functions and public API surfaces should
have one. They are especially valuable for custom kernels and performance-optimized
implementations, where documenting input/output shapes, dtypes, device/stream assumptions, and
partitioning invariants is the only way a reader can follow the code.

For bug fixes, do not narrate the original bug in a code comment or explain why the old code was
wrong — that belongs in the PR description. Likewise, design-discussion context from the PR thread
does not belong in the source.

#### 3.4: Dead code and helpers

- Delete dead code decisively. If code is unused at runtime and only referenced by tests, remove it
  together with those tests.
- Inline trivially-simple helper functions with a single call site.
- If the change makes a config option, code path, or compatibility shim obsolete, remove it (with
  deprecation where it is user-visible) rather than leaving both.

#### 3.5: Documentation, tutorials, and examples

New features must include documentation updates — this is a hard requirement, not a follow-up:

- New or changed config keys → `docs/_pages/config-json.md`
- User-facing features → a tutorial under `docs/_tutorials/`
- Public API additions → the API reference under `docs/code-docs/source/`
- Usage examples → `examples/`, and for a full feature, a companion PR to
  [DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples)

Check that existing docs describing the changed behavior were updated too; stale tutorials are a
recurring source of user issues.

### 4. Pull Request Contents

This section is about the pull request itself rather than the code. As author, prepare these; as
reviewer, assess them and report missing context or evidence.

DeepSpeed has no PR template, so the burden is on the description. Use the relevant issue template
under `.github/ISSUE_TEMPLATE/` as a guide to the context maintainers expect
(`training_bug_report.md`, `inference_bug_report.md`, `feature_request.md`,
`ci_failure_report.md`).

#### 4.1: DCO sign-off

Every non-merge commit needs a `Signed-off-by` line (`git commit -s`), checked by the DCO app. Take
the name and email from `git config user.name` / `git config user.email`.

When reviewing: commit trailers are **not** visible in a diff. Never report a missing sign-off from
the diff alone — verify first, and flag it only if the trailer is genuinely absent:

```bash
git log --format='%h %s%n%b' <base>..<head>
```

#### 4.2: Description and context

Lead with a 1-2 line summary so a reviewer can grasp the change immediately. Link the relevant
issue (required for new features, per the proposal process in 1.3), any predecessor or blocking
PRs, and the upstream issue or CI failure that motivated the change.

Keep claims to a few concrete bullets and motivation brief. Keep validation and implementation notes
concise while retaining the evidence, root cause, tradeoffs, and limitations a reviewer needs. Put
lengthy logs behind collapsible `<details>` blocks or links rather than in the main narrative.

#### 4.3: Claims and supporting evidence

Make clear, concise claims about what the PR accomplishes, and supply evidence for each:

- **Bug fix**: what breaks, under what configuration, and how to reproduce it. Show the failure
  before and the pass after, on the same setup.
- **Feature**: what functionality is added, how to enable it (config snippet), and a validated
  usage example.
- **Refactor**: the motivation and expected improvement. Unit coverage suffices for a minor
  refactor; a significant refactor of engine or ZeRO code needs end-to-end training validation.
- **Performance optimization**: the target model, accelerator, parallelism/ZeRO configuration, and
  workload — plus how the gain can be reproduced.

For performance and memory claims, report a compact table with configuration, baseline, changed
result, and the reproduction command — and include **peak memory** next to throughput, since a
throughput win paid for in memory is often not a win for DeepSpeed's users. Give run count and
variability for repeated benchmarks. Microbenchmark-only validation is often insufficient for a
runtime change; if end-to-end evaluation is missing (e.g. no access to sufficient hardware), say so
explicitly.

For anything touching convergence, follow `CONTRIBUTING.md`: a performance-only change needs a
partial-training run showing loss consistent with baseline, while a change that affects convergence
needs full training showing on-par or better final quality.

#### 4.4: Root-cause analysis

For bug fixes, provide a clear root-cause analysis: what caused the problem, how it was identified,
and why this solution addresses it. Be specific about the mechanism — which rank, which stage, which
stream, which ordering — because in distributed code a patch that merely makes a symptom disappear
usually relocates the race rather than removing it. A fix without root-cause analysis invites
recurrence and maintenance overhead.

#### 4.5: Implementation details

Explain the implementation: design decisions, tradeoffs, and relevant technical detail. Highlight
non-obvious choices, limitations, the matrix cells from 1.5 that were and were not validated, and
any area where you specifically want maintainer attention.

<!-- This file is duplicated as CLAUDE.md and AGENTS.md. Keep them in sync. -->
# AGENTS.md — Workspace-level instructions for AI coding agents

> These rules apply to **all** AI-assisted contributions to `deepspeedai/DeepSpeed`.

## Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo deepspeedai/DeepSpeed --comments
gh pr list --repo deepspeedai/DeepSpeed --state open --search "<issue_number> in:body"
gh pr list --repo deepspeedai/DeepSpeed --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.
- DeepSpeed carries long-lived parallel implementations (ZeRO 1/2 vs 3, fp16 vs bf16 optimizers, per-accelerator op builders). Before fixing a bug in one, check whether the sibling paths share the defect, and state in the PR which ones you checked.

### New features need an accepted proposal first

Per [`CONTRIBUTING.md`](CONTRIBUTING.md#new-feature-contribution-guidelines), a new feature starts with a proposal issue -- description, motivation, rough design, and planned experiments -- that maintainers accept before implementation begins. Do not open a feature PR with no accepted proposal; open the proposal issue instead.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (a single typo, an isolated style change, a lone type annotation). Mechanical cleanups are acceptable only when bundled with substantive work in the same area. This is the PR-level form of the rule against cosmetic changes below.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run the relevant tests.
- `Signed-off-by` is a [DCO](https://en.wikipedia.org/wiki/Developer_Certificate_of_Origin) attestation made by a person, not by an agent. Never sign off on a diff that no human has read.
- PR descriptions for AI-assisted work **must** include:
    - Why this does not duplicate an existing PR or issue.
    - Test commands run and their results, including the hardware they ran on: device model and count, accelerator/driver version, and torch version.
    - Convergence, throughput, and peak-memory results when the change affects numerics, performance, or memory -- see [`CONTRIBUTING.md`](CONTRIBUTING.md#step-1-proposal-and-discussion) for the evidence bar.
    - Which cells of the compatibility matrix (ZeRO stage, offload, precision, parallelism) were validated, and which were not.
    - A clear statement that AI assistance was used.
- Before opening or reviewing a PR, work through the [`/pr-checklist`](.agents/skills/pr-checklist/SKILL.md) skill: design fit, accelerator portability, backward compatibility, on-device validation, diff hygiene, and PR contents.

### Fail-closed behavior

If the work is duplicate, trivial busywork, or an unproposed new feature, **do not proceed**. Return a short explanation of what is missing.

---

## DeepSpeed Project Rules

### Commit & CI requirements

- All non-merge commits MUST have a `Signed-off-by` line (use `--signoff`). Get the name and email from `git config user.name` / `git config user.email`.
- Reviewing: commit trailers are NOT visible in a diff. Never report a missing `Signed-off-by` based on the diff alone -- verify with `git log --format='%b' <sha>` (or the commits API) and flag it only if the trailer is actually absent.
- Formatting: yapf (column_limit=119, `.style.yapf`) + flake8 (`.flake8`).
- Always verify changed files pass pre-commit checks before committing: `pre-commit run --files <changed_files>`. Only check modified files, not the entire codebase. Config: `.pre-commit-config.yaml`.
- `check-torchdist` hook: NEVER directly import torch's distributed module. Use `import deepspeed.comm as dist` instead.
- New files require license header:
  ```
  # SPDX-License-Identifier: Apache-2.0
  # DeepSpeed Team
  ```

### Code change discipline

- NEVER make cosmetic/formatting-only changes to existing code. Only add/modify lines that are functionally necessary. Minimizing diff noise is critical for code review.
- Delete dead code decisively — if code is unused at runtime (only referenced in tests), remove it along with its tests.
- Prefer consolidating tests over proliferating test files.
- Blend in: when modifying code, read the surrounding context and match the style of neighboring code (naming, spacing, patterns, idioms).
- Write beginner-friendly code: avoid deeply nested expressions or chained logic. Break complex expressions into clear, named intermediate steps.
- Comments should explain **why**, not **what**. Describe the purpose and reasoning, not the mechanics that the code already shows.
- New features must include corresponding tests and documentation updates.

### Test discipline

- Tests verify contracts, not implementations: a test is well-formed only if a different correct implementation of the same contract passes it.
- Before writing a test, name the concrete incorrect behavior it would catch; if you cannot name one, do not write it.
- Assert on observable outcomes through public/stable interfaces; do not assert private method return values or exact internal strings unless pinning a specific fixed bug (justify in a comment).
- Anchor to an external oracle or an independently derived reference instead of re-implementing the logic under test.
- Mocks must stand in for a collaborator's documented contract (schema, protocol), never for internals of the module under test.
- Changes that affect the external contract at the training-loop or inference level require integration tests, not just unit tests (e.g. a minimal training loop with `SimpleModel`).
- Integration tests must be executed on actual devices, not merely written: report the execution hardware spec and results in the PR.

## Tool Caveats

### Edit tool auto-formatter

The Edit tool has a hidden auto-formatter that silently changes quotes, whitespace, blank lines, and line wrapping. For format-sensitive modifications (e.g., when exact formatting matters for pre-commit), use `bash` with `sed`, `python`, or `cat` instead.

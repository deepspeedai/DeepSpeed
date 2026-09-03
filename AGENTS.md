<!-- This file is duplicated as CLAUDE.md and AGENTS.md. Keep them in sync. -->
# AGENTS.md — Workspace-level instructions for AI coding agents

## DeepSpeed Project Rules

### Commit & CI requirements

- All non-merge commits MUST have a `Signed-off-by` line (use `--signoff`). Get the name and email from `git config user.name` / `git config user.email`.
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

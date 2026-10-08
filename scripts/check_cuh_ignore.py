#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REQUIRED_HEADERS = ("csrc/adam/multi_tensor_apply.cuh",)
FORBIDDEN_PATTERNS = ("*.cuh", )


def _load_patterns(gitignore_path: Path) -> set[str]:
    patterns: set[str] = set()
    for raw_line in gitignore_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.add(line)
    return patterns


def _is_git_tracked(repo_root: Path, relative_path: str) -> bool:
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative_path],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def validate_repo(repo_root: Path) -> list[str]:
    errors: list[str] = []
    gitignore_path = repo_root / ".gitignore"
    patterns = _load_patterns(gitignore_path)

    for pattern in FORBIDDEN_PATTERNS:
        if pattern in patterns:
            errors.append(
                f"Forbidden .gitignore pattern '{pattern}' found in {gitignore_path}. "
                "Do not ignore all CUDA headers globally."
            )

    for header in REQUIRED_HEADERS:
        header_path = repo_root / header
        if not header_path.is_file():
            errors.append(f"Required CUDA header missing: {header}")
            continue
        if not _is_git_tracked(repo_root, header):
            errors.append(f"Required CUDA header is not tracked by git: {header}")

    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    errors = validate_repo(repo_root)
    if not errors:
        return 0

    for error in errors:
        print(error, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Decide whether a diff needs the CPU/GPU CI workflows.

Documentation-only diffs skip them. ``is_docs_path`` is the single definition of
documentation: ``.github/workflows/check-paths.yml`` runs this script as its gate,
and ``ci/tests_fetcher.py`` imports it for the Modal GPU test selection.

The gate copies this file from the base branch and runs it outside the repository,
so a pull request's edits apply only after merge. Keep it a single file that
imports only the standard library.

Preview the gate's decision for your branch::

    python ci/check_paths.py --base "$(git merge-base origin/master HEAD)"
"""

from __future__ import annotations

import argparse
import os
import subprocess

DOC_DIRS = ("docs/", "blogs/")
DOC_SUFFIXES = (".md", )


def is_docs_path(path: str) -> bool:
    """Everything under docs/ or blogs/, plus Markdown files anywhere."""
    return path.startswith(DOC_DIRS) or path.endswith(DOC_SUFFIXES)


def _is_ignored(path: str, ignore: list[str]) -> bool:
    for entry in ignore:
        # An entry ending in "/" is a directory; anything else names exactly one file.
        if entry.endswith("/"):
            if path.startswith(entry):
                return True
        elif path == entry:
            return True
    return False


def _changed_paths(base: str, head: str) -> list[str]:
    # --no-renames reports both sides of a rename, so moving code into docs/ still
    # counts as deleting code. NUL delimiters keep unusual file names intact.
    out = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z", base, head],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [path for path in out.split("\0") if path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="Commit to diff against.")
    parser.add_argument("--head", default="HEAD", help="Commit under test. Default: %(default)s.")
    parser.add_argument(
        "--ignore",
        default="",
        help="Whitespace-separated files, or directories ending in '/', whose changes also skip CI.",
    )
    args = parser.parse_args()

    ignore = args.ignore.split()
    for entry in ignore:
        if any(char in entry for char in "*?["):
            parser.error(f"--ignore takes plain paths, not globs: {entry!r}")

    paths = _changed_paths(args.base, args.head)
    needs_ci = [path for path in paths if not is_docs_path(path) and not _is_ignored(path, ignore)]
    if not paths:
        # An empty diff is no evidence that the tests are unnecessary.
        should_run = True
        reason = "empty diff"
    elif needs_ci:
        should_run = True
        reason = f"{len(needs_ci)} changed path(s) need CI, e.g. {needs_ci[0]!r}"
    else:
        should_run = False
        reason = f"all {len(paths)} changed path(s) are documentation or ignored"

    value = "true" if should_run else "false"
    print(f"should_run={value} ({reason})")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as fh:
            fh.write(f"should_run={value}\n")


if __name__ == "__main__":
    main()

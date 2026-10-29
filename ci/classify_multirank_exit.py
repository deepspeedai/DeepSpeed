# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Classify the multi-rank CPU run's exit code for cpu-multi-triage.yml.

Prints the DS_CI_FAILURE_CLASS sentinel that ci/nightly_classify.sh greps from
the job log. The sentinel lives in this file instead of inline in the
workflow's ``run:`` script because GitHub Actions echoes the whole script text
into the log before executing it, and the classifier's ``head -1`` would pick
up the echoed literal -- routing every failure as a timeout -- instead of the
runtime line.

Run standalone::

    python ci/classify_multirank_exit.py <exit-code-or-none>
"""

from __future__ import annotations

import sys

TIMEOUT_EXIT_CODE = 124  # GNU timeout's exit code when COMMAND times out


def main() -> int:
    raw = sys.argv[1] if len(sys.argv) > 1 else "none"
    if raw == "none":
        # The test step never ran (setup/install died earlier); triage reports
        # this as an unclassified failure, so emit no sentinel.
        print("the test step never ran", file=sys.stderr)
        return 1

    code = int(raw)
    if code == 0:
        print("multi-rank suite passed")
        return 0

    if code == TIMEOUT_EXIT_CODE:
        print("DS_CI_FAILURE_CLASS=timeout")
        return 1

    print("DS_CI_FAILURE_CLASS=test")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

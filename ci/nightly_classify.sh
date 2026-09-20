#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Classify a completed nightly modal-torch-latest run for triage.
#
# Reads NIGHTLY_RUN_ID and NIGHTLY_CONCLUSION from the environment, reads the run's
# logs with gh, and writes `action` and `failure_class` to $GITHUB_OUTPUT:
#
#   green   the nightly passed; the caller moves the nightly-last-green tag
#   infra   no GPU instance was provisioned; no candidate information, do nothing
#   timeout the run died at a time budget; open an issue, never bisect
#   bisect  real test failures; git-bisect between nightly-last-green and the SHA
#
# A failed run whose logs contain no DS_CI_FAILURE_CLASS sentinel was killed before
# the controller could classify itself (job timeout), which is a timeout.

set -u

if [ "${NIGHTLY_CONCLUSION:-}" = "success" ]; then
    echo "action=green" >> "$GITHUB_OUTPUT"
    echo "failure_class=" >> "$GITHUB_OUTPUT"
    echo "nightly passed; will move nightly-last-green"
    exit 0
fi

class=$(gh run view "$NIGHTLY_RUN_ID" --log 2>/dev/null \
    | grep -o 'DS_CI_FAILURE_CLASS=[a-z]*' | head -1 | cut -d= -f2)

case "${class:-none}" in
    test)
        echo "action=bisect" >> "$GITHUB_OUTPUT"
        echo "failure_class=test" >> "$GITHUB_OUTPUT"
        echo "real test failures; will bisect"
        ;;
    infra)
        echo "action=infra" >> "$GITHUB_OUTPUT"
        echo "failure_class=infra" >> "$GITHUB_OUTPUT"
        echo "GPU capacity failure; no candidate information"
        ;;
    timeout | none)
        # No sentinel means the job was killed before the controller could exit.
        echo "action=timeout" >> "$GITHUB_OUTPUT"
        echo "failure_class=${class:-killed}" >> "$GITHUB_OUTPUT"
        echo "time budget exhausted; not bisectable"
        ;;
    *)
        echo "unrecognized DS_CI_FAILURE_CLASS=$class; refusing to route" >&2
        exit 1
        ;;
esac

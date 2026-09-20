#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# One git-bisect step for the nightly regression search
# (.github/workflows/nightly-bisect.yml).
#
# Dispatches modal-torch-latest at the current bisect commit, waits for it, and
# maps the outcome onto git-bisect's run contract:
#   exit 0   the run passed -> this commit is good
#   exit 1   the run finished with real test failures -> this commit is bad
#   exit 33  anything else (infra, timeout, job killed) -> inconclusive; the
#            workflow aborts the whole bisect instead of skipping, because a
#            skip silently shrinks the searched range
#
# Requires GH_TOKEN and GITHUB_REPOSITORY in the environment.

set -u

sha=$(git rev-parse HEAD)

echo "bisect step: dispatching modal-torch-latest at $sha"
gh workflow run modal-torch-latest.yml --repo "$GITHUB_REPOSITORY" --ref "$sha"

# The dispatch run may take a moment to register; find it by head SHA.
run_id=""
for _ in $(seq 1 30); do
    sleep 10
    run_id=$(gh run list --workflow modal-torch-latest.yml --event workflow_dispatch --limit 10 \
        --json databaseId,headSha --jq ".[] | select(.headSha == \"$sha\") | .databaseId" | head -1)
    [ -n "$run_id" ] && break
done
if [ -z "$run_id" ]; then
    echo "bisect step: dispatch at $sha never registered a run" >&2
    exit 33
fi

echo "bisect step: watching run $run_id"
gh run watch "$run_id" --exit-status --interval 60 >/dev/null 2>&1 || true

conclusion=$(gh run view "$run_id" --json conclusion --jq .conclusion)
if [ "$conclusion" = "success" ]; then
    echo "bisect step: $sha is good"
    exit 0
fi

class=$(gh run view "$run_id" --log 2>/dev/null \
    | grep -o 'DS_CI_FAILURE_CLASS=[a-z]*' | head -1 | cut -d= -f2)
if [ "${class:-none}" = "test" ]; then
    echo "bisect step: $sha is bad (real test failures)"
    exit 1
fi

# A killed job leaves no sentinel; old revisions predate the sentinel entirely.
# Both are inconclusive for bisect purposes, so abort rather than mislabel.
echo "bisect step: $sha inconclusive (class: ${class:-no sentinel})" >&2
exit 33

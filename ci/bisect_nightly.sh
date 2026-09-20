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
# When BISECT_TEST_TARGETS_FILE points at the nightly's failing test files, the
# dispatch runs only those files (workflow_dispatch test_targets input) instead
# of the full suite, cutting a step from ~70 to ~15 minutes. Targets that do not
# exist at the step commit are dropped from the dispatch (they cannot fail where
# they do not exist); only when none survive is the commit good without running.
#
# Requires GH_TOKEN and GITHUB_REPOSITORY in the environment.

set -u

sha=$(git rev-parse HEAD)

targets=()
if [ -n "${BISECT_TEST_TARGETS_FILE:-}" ]; then
    while IFS= read -r target; do
        [ -n "$target" ] || continue
        # A target that does not exist at this commit cannot fail here; drop it
        # from the dispatch rather than judging the commit, because a *different*
        # failing target may still condemn it.
        if git cat-file -e "$sha:$target" 2>/dev/null; then
            targets+=("$target")
        fi
    done < "$BISECT_TEST_TARGETS_FILE"
    if [ "${#targets[@]}" -eq 0 ]; then
        echo "bisect step: $sha is good (none of the failing test files exist there yet)"
        exit 0
    fi
fi

echo "bisect step: dispatching modal-torch-latest at $sha"
if [ "${#targets[@]}" -gt 0 ]; then
    target_list=$(printf '%s\n' "${targets[@]}")
    gh workflow run modal-torch-latest.yml --repo "$GITHUB_REPOSITORY" --ref "$sha" \
        -f test_targets="$target_list"
else
    gh workflow run modal-torch-latest.yml --repo "$GITHUB_REPOSITORY" --ref "$sha"
fi

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

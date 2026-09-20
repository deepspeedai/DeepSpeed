#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Deduplicating issue reporter for nightly triage
# (.github/workflows/nightly-bisect.yml).
#
# Usage: nightly_report.sh TITLE BODY_FILE [RUN_URL]
#
# A persistent regression fails the same way every night, so creating a fresh
# issue per night floods the tracker within a week. Every report is labeled
# nightly-triage; when an open issue with the same title and label already
# exists, the new occurrence is posted as a comment instead. Dedup keys on the
# title, so callers must keep the title stable across recurrences (put the
# per-night SHA into the body, not the title) -- except culprit reports, whose
# title carries the culprit SHA precisely because a *different* culprit is a
# different regression.
#
# Requires GH_TOKEN and GITHUB_REPOSITORY in the environment.

set -u

LABEL="nightly-triage"

if [ "$#" -lt 2 ]; then
    echo "usage: $0 TITLE BODY_FILE [RUN_URL]" >&2
    exit 1
fi
title=$1
body_file=$2
run_url=${3:-}

if ! gh label list --repo "$GITHUB_REPOSITORY" --search "$LABEL" --json name --jq ".[].name" 2>/dev/null | grep -qx "$LABEL"; then
    gh label create "$LABEL" --repo "$GITHUB_REPOSITORY" \
        --description "Automated nightly modal-torch-latest triage" >/dev/null
fi

existing=$(gh issue list --repo "$GITHUB_REPOSITORY" --label "$LABEL" --state open \
    --json number,title --jq ".[] | select(.title == \"$(printf '%s' "$title" | sed 's/"/\\"/g')\") | .number" | head -1)

stamp=$(date -u '+%Y-%m-%d %H:%M UTC')
if [ -n "$run_url" ]; then
    occurrence="Recurred $stamp; nightly run: $run_url"
else
    occurrence="Recurred $stamp"
fi

if [ -n "$existing" ]; then
    {
        echo "$occurrence"
        echo
        cat "$body_file"
    } | gh issue comment "$existing" --repo "$GITHUB_REPOSITORY" --body-file - >/dev/null
    echo "commented on existing issue #$existing: $title"
else
    number=$(gh issue create --repo "$GITHUB_REPOSITORY" --label "$LABEL" \
        --title "$title" --body-file "$body_file" 2>/dev/null | grep -o '[0-9]*$')
    echo "created issue #${number:-?}: $title"
fi

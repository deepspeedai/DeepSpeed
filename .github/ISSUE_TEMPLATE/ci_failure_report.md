---
name: CI failure report
about: Report a DeepSpeed CI failure
title: "{{ env.GITHUB_WORKFLOW }} CI test failure"
labels: ci-failure
assignees: ''

---

The Nightly CI for {{ env.GITHUB_SERVER_URL }}/{{ env.GITHUB_REPOSITORY }}/actions/runs/{{ env.GITHUB_RUN_ID }} failed.

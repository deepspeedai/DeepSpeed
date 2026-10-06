# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Small no-thread observer for bounded Modal pytest diagnostics."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence


def _identity() -> dict[str, object]:
    return {
        "pid": os.getpid(),
        "worker": os.environ.get("PYTEST_XDIST_WORKER", "controller"),
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK"),
    }


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _emit(event: str, **fields: object) -> None:
    path_value = os.environ.get("DS_DIAGNOSTIC_EVENTS_FILE")
    if not path_value:
        return
    record = {
        "event": event,
        "time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        **_identity(),
        **{key: _json_safe(value) for key, value in fields.items()},
    }
    line = (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8", errors="replace")
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(line)
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    print("DS_DIAGNOSTIC_EVENT " + line.decode("utf-8").rstrip("\n"), file=sys.__stderr__, flush=True)


def pytest_sessionstart(session: Any) -> None:
    _emit(
        "session_start",
        seed=os.environ.get("DS_DIAGNOSTIC_SEED"),
        argv=list(session.config.invocation_params.args),
    )


def pytest_runtest_logstart(nodeid: str, location: object) -> None:
    _emit("node_start", nodeid=nodeid, location=location)


def pytest_runtest_logreport(report: Any) -> None:
    fields: dict[str, object] = {
        "nodeid": report.nodeid,
        "phase": report.when,
        "outcome": report.outcome,
        "duration_seconds": report.duration,
        "xdist_worker": getattr(report, "worker_id", None),
    }
    for name in ("capstdout", "capstderr", "caplog"):
        value = getattr(report, name, "")
        if value:
            fields[name] = value
    if report.failed:
        fields["longrepr"] = getattr(report, "longreprtext", None) or str(report.longrepr)
    _emit("phase_report", **fields)


def pytest_runtest_logfinish(nodeid: str, location: object) -> None:
    _emit("node_finish", nodeid=nodeid, location=location)


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    _emit("session_finish", exitstatus=int(exitstatus))


class _Collector:

    def __init__(self) -> None:
        self.nodeids: list[str] = []

    def pytest_collection_finish(self, session: Any) -> None:
        self.nodeids = [item.nodeid for item in session.items]


def _read_manifest(path: Path) -> list[str]:
    targets = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")
    ]
    if not targets or any(not target for target in targets):
        raise ValueError("diagnostic manifest must contain non-empty targets")
    if len(targets) != len(set(targets)):
        raise ValueError("diagnostic manifest contains duplicate targets")
    return targets


def collect_nodes(args: argparse.Namespace) -> int:
    import pytest

    requested = _read_manifest(args.manifest)
    collector = _Collector()
    pytest_args = [
        "--collect-only",
        "-q",
        "--ignore=tests/unit/v1/nvme/test_gds.py",
        f"--torch_ver={args.torch_version}",
        f"--cuda_ver={args.cuda_version}",
        "--",
        *requested,
    ]
    pytest_exit = int(pytest.main(pytest_args, plugins=[collector]))
    collected = collector.nodeids
    result = {
        "requested_count": len(requested),
        "requested_unique_count": len(set(requested)),
        "collected_count": len(collected),
        "collected_unique_count": len(set(collected)),
        "expected_count": args.expected_count,
        "missing": sorted(set(requested) - set(collected)),
        "unexpected": sorted(set(collected) - set(requested)),
        "pytest_exit": pytest_exit,
        "requested": requested,
        "collected": collected,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DS_DIAGNOSTIC_COLLECTION " + json.dumps(result, sort_keys=True), flush=True)
    valid = (
        pytest_exit == 0
        and len(requested) == args.expected_count
        and len(collected) == args.expected_count
        and len(set(collected)) == args.expected_count
        and not result["missing"]
        and not result["unexpected"]
    )
    return 0 if valid else 2


def summarize_events(args: argparse.Namespace) -> int:
    events = []
    if args.events.exists():
        for line in args.events.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"event": "malformed", "raw": line})
    active: dict[str, dict[str, object]] = {}
    logical_phases: dict[tuple[str, str], dict[str, object]] = {}
    for event in events:
        kind = event.get("event")
        nodeid = str(event.get("nodeid", ""))
        if kind == "node_start":
            current = active.get(nodeid)
            if current is None or event.get("worker") != "controller":
                active[nodeid] = event
        elif kind == "node_finish":
            active.pop(nodeid, None)
        elif kind == "phase_report":
            phase_key = (nodeid, str(event.get("phase", "")))
            current = logical_phases.get(phase_key)
            if current is None or event.get("worker") != "controller":
                logical_phases[phase_key] = event
    failures = []
    phase_counts: dict[str, int] = {}
    for event in logical_phases.values():
        phase_key = f"{event.get('phase')}:{event.get('outcome')}"
        phase_counts[phase_key] = phase_counts.get(phase_key, 0) + 1
        if event.get("outcome") == "failed":
            failures.append(event)
    summary = {
        "event_count": len(events),
        "phase_counts": phase_counts,
        "failure_count": len(failures),
        "failures": failures,
        "unfinished_count": len(active),
        "unfinished": list(active.values()),
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DS_DIAGNOSTIC_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


def emit_files(paths: Sequence[Path]) -> int:
    for path in paths:
        print(f"===== BEGIN {path.name} =====", flush=True)
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            sys.stdout.write(text)
            if text and not text.endswith("\n"):
                sys.stdout.write("\n")
        else:
            print("<missing>")
        print(f"===== END {path.name} =====", flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--manifest", type=Path, required=True)
    collect.add_argument("--expected-count", type=int, required=True)
    collect.add_argument("--torch-version", required=True)
    collect.add_argument("--cuda-version", required=True)
    collect.add_argument("--output", type=Path, required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--events", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)

    emit = subparsers.add_parser("emit")
    emit.add_argument("paths", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "collect":
        return collect_nodes(args)
    if args.command == "summarize":
        return summarize_events(args)
    return emit_files(args.paths)


if __name__ == "__main__":
    raise SystemExit(main())

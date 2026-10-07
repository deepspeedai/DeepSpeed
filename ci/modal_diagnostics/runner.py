# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Focused pytest diagnostics for the modal-torch-latest CI investigation."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

DEFAULT_STALL_SECONDS = 300.0
DEFAULT_TELEMETRY_SECONDS = 30.0
COMMAND_TIMEOUT_SECONDS = 10.0
MAX_CAPTURE_BYTES = 1024 * 1024
PROGRESS_EVENTS = {"node_start", "phase_outcome", "node_finish"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_line(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _worker_name() -> str:
    return os.environ.get("PYTEST_XDIST_WORKER", "controller")


def _event_path() -> Path | None:
    root = os.environ.get("DS_DIAGNOSTICS_DIR")
    if not root:
        return None
    return Path(root) / "events" / f"events-{_worker_name()}-{os.getpid()}.jsonl"


def _emit_pytest_event(event: str, nodeid: str, **fields: Any) -> None:
    path = _event_path()
    if path is None:
        return
    payload = {
        "event": event,
        "nodeid": nodeid,
        "pid": os.getpid(),
        "worker": _worker_name(),
        "time_monotonic": time.monotonic(),
        "time_utc": _utc_now(),
    }
    payload.update(fields)
    _write_json_line(path, payload)


def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]) -> None:
    del location
    if "PYTEST_XDIST_WORKER" in os.environ:
        _emit_pytest_event("node_start", nodeid)


def pytest_runtest_logreport(report: Any) -> None:
    if "PYTEST_XDIST_WORKER" not in os.environ:
        return
    fields = {
        "duration_seconds": getattr(report, "duration", None),
        "outcome": report.outcome,
        "phase": report.when,
    }
    if report.failed:
        fields["traceback"] = getattr(report, "longreprtext", str(report.longrepr))
    _emit_pytest_event("phase_outcome", report.nodeid, **fields)


def pytest_runtest_logfinish(nodeid: str, location: tuple[str, int | None, str]) -> None:
    del location
    if "PYTEST_XDIST_WORKER" in os.environ:
        _emit_pytest_event("node_finish", nodeid)


class EventReader:
    """Incrementally read the per-process JSONL files written by pytest workers."""

    def __init__(self, event_dir: Path):
        self.event_dir = event_dir
        self.offsets: dict[Path, int] = {}

    def poll(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for path in sorted(self.event_dir.glob("events-*.jsonl")):
            offset = self.offsets.get(path, 0)
            try:
                with path.open("r", encoding="utf-8") as handle:
                    handle.seek(offset)
                    for line in handle:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"[diagnostic] ignoring incomplete event from {path.name}", flush=True)
                    self.offsets[path] = handle.tell()
            except FileNotFoundError:
                continue
        return events


def _run_capture(argv: Sequence[str], timeout: float = COMMAND_TIMEOUT_SECONDS) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(list(argv), check=False, capture_output=True, text=True, timeout=timeout)
        output = (result.stdout + result.stderr)[:MAX_CAPTURE_BYTES]
        return {
            "argv": list(argv),
            "duration_seconds": time.monotonic() - started,
            "output": output,
            "returncode": result.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="backslashreplace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="backslashreplace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        output = (stdout + stderr)[:MAX_CAPTURE_BYTES]
        return {
            "argv": list(argv),
            "duration_seconds": time.monotonic() - started,
            "output": output,
            "returncode": None,
            "timed_out": True,
        }
    except OSError as exc:
        return {
            "argv": list(argv),
            "duration_seconds": time.monotonic() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "returncode": None,
            "timed_out": False,
        }


def _environment_report() -> dict[str, Any]:
    python_probe = (
        "import json, os, torch, torchvision, transformers; "
        "nccl = torch.cuda.nccl.version() if torch.cuda.is_available() else None; "  #ignore-cuda
        "devices = [{'index': i, 'name': torch.cuda.get_device_name(i)} "  #ignore-cuda
        "for i in range(torch.cuda.device_count())]; "  #ignore-cuda
        "visible = os.environ.get('CUDA_VISIBLE_DEVICES'); "
        "print(json.dumps({'torch': torch.__version__, 'torch_cuda': torch.version.cuda, "
        "'torchvision': torchvision.__version__, 'transformers': transformers.__version__, "
        "'nccl': nccl, 'devices': devices, 'cuda_visible_devices_set': visible is not None, "
        "'cuda_visible_device_count': len(visible.split(',')) if visible else None}, sort_keys=True))")
    return {
        "event":
        "environment",
        "time_utc":
        _utc_now(),
        "nvidia_smi":
        _run_capture([
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]),
        "nvcc":
        _run_capture(["nvcc", "--version"]),
        "python_packages_and_devices":
        _run_capture([sys.executable, "-c", python_probe], timeout=20.0),
    }


def _telemetry_sample(reason: str) -> dict[str, Any]:
    return {
        "event":
        "telemetry",
        "reason":
        reason,
        "time_monotonic":
        time.monotonic(),
        "time_utc":
        _utc_now(),
        "gpu":
        _run_capture([
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]),
        "processes":
        _run_capture(["ps", "-eo", "pid,ppid,pgid,stat,etimes,comm,args", "--sort=pid"]),
    }


def _telemetry_loop(root: Path, interval: float, stop: threading.Event) -> None:
    output = root / "telemetry.jsonl"
    while not stop.is_set():
        sample = _telemetry_sample("periodic")
        _write_json_line(output, sample)
        print(f"[diagnostic:telemetry] {json.dumps(sample, sort_keys=True)}", flush=True)
        stop.wait(interval)


def _registered_pids(root: Path) -> list[int]:
    pids: list[int] = []
    for marker in sorted((root / "stacks").glob("registered-*.json")):
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
            pid = int(value["pid"])
            os.kill(pid, 0)
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError, ProcessLookupError,
                PermissionError):
            continue
        pids.append(pid)
    return pids[:256]


def _request_stacks(root: Path) -> list[dict[str, Any]]:
    results = []
    for pid in _registered_pids(root):
        try:
            os.kill(pid, signal.SIGUSR1)
            results.append({"pid": pid, "signal": "SIGUSR1", "status": "sent"})
        except (ProcessLookupError, PermissionError, OSError) as exc:
            results.append({"pid": pid, "signal": "SIGUSR1", "status": f"{type(exc).__name__}: {exc}"})
    _write_json_line(root / "stack-requests.jsonl", {
        "event": "stack_requests",
        "requests": results,
        "time_utc": _utc_now(),
    })
    return results


def _signal_process_group(process: subprocess.Popen[str], requested_signal: signal.Signals) -> None:
    try:
        os.killpg(process.pid, requested_signal)
    except ProcessLookupError:
        pass


def _stop_process_group(
        process: subprocess.Popen[str],
        waits: tuple[float, float, float] = (30.0, 10.0, 5.0),
        sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
    actions: list[str] = []
    for requested_signal, wait_seconds in zip((signal.SIGINT, signal.SIGTERM, signal.SIGKILL), waits):
        if process.poll() is not None:
            break
        _signal_process_group(process, requested_signal)
        actions.append(requested_signal.name)
        deadline = time.monotonic() + wait_seconds
        while process.poll() is None and time.monotonic() < deadline:
            sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    return actions


def _tee_output(process: subprocess.Popen[str], output_path: Path) -> None:
    assert process.stdout is not None
    with output_path.open("w", encoding="utf-8", errors="backslashreplace") as handle:
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(f"[pytest] {line.rstrip()}", flush=True)


def _pytest_command(args: argparse.Namespace, targets: Sequence[str], seed: int, root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        str(args.workers),
        "--verbose",
        "--showlocals",
        "--tb=long",
        "-rA",
        f"--randomly-seed={seed}",
        f"--junitxml={root / 'junit.xml'}",
        f"--torch_ver={args.torch_version}",
        f"--cuda_ver={args.cuda_version}",
        "--ignore=tests/unit/v1/nvme/test_gds.py",
        "-p",
        "ci.modal_diagnostics.runner",
        "--",
        *targets,
    ]


def _run_pytest(args: argparse.Namespace) -> int:
    targets = tuple(target for target in args.targets if target != "--")
    if not targets:
        raise ValueError("at least one pytest node ID is required")
    root = Path(args.artifact_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "events").mkdir(exist_ok=True)
    (root / "stacks").mkdir(exist_ok=True)
    _write_json_line(root / "environment.jsonl", _environment_report())

    seed = secrets.randbits(32)
    command = _pytest_command(args, targets, seed, root)
    start = time.monotonic()
    print(f"[diagnostic:seed] {seed}", flush=True)
    print(f"[diagnostic:selection] count={len(targets)}", flush=True)
    _write_json_line(
        root / "events.jsonl", {
            "event": "session_start",
            "pid": os.getpid(),
            "seed": seed,
            "selection_count": len(targets),
            "time_monotonic": start,
            "time_utc": _utc_now(),
        })

    env = os.environ.copy()
    diagnostic_path = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = diagnostic_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["DS_DIAGNOSTICS_DIR"] = str(root)
    env["PYTHONFAULTHANDLER"] = "1"
    process = subprocess.Popen(
        command,
        env=env,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    output_thread = threading.Thread(target=_tee_output, args=(process, root / "pytest-output.log"), daemon=True)
    output_thread.start()
    telemetry_stop = threading.Event()
    telemetry_thread = threading.Thread(
        target=_telemetry_loop,
        args=(root, args.telemetry_seconds, telemetry_stop),
        daemon=True,
    )
    telemetry_thread.start()

    reader = EventReader(root / "events")
    last_progress = start
    last_heartbeat = start
    stalled = False
    stop_actions: list[str] = []
    try:
        while process.poll() is None:
            now = time.monotonic()
            for event in reader.poll():
                print(f"[diagnostic:event] {json.dumps(event, sort_keys=True)}", flush=True)
                if event.get("event") in PROGRESS_EVENTS:
                    last_progress = now
            if now - last_heartbeat >= 60.0:
                heartbeat = {
                    "event": "heartbeat",
                    "idle_seconds": now - last_progress,
                    "time_monotonic": now,
                    "time_utc": _utc_now(),
                }
                _write_json_line(root / "heartbeat.jsonl", heartbeat)
                print(f"[diagnostic:heartbeat] {json.dumps(heartbeat, sort_keys=True)}", flush=True)
                last_heartbeat = now
            if now - last_progress >= args.stall_seconds:
                stalled = True
                print(f"[diagnostic:stall] no test progress for {now - last_progress:.0f}s", flush=True)
                sample = _telemetry_sample("stall")
                _write_json_line(root / "telemetry.jsonl", sample)
                print(f"[diagnostic:telemetry] {json.dumps(sample, sort_keys=True)}", flush=True)
                requests = _request_stacks(root)
                print(f"[diagnostic:stacks] requested={len(requests)}", flush=True)
                time.sleep(3.0)
                stop_actions = _stop_process_group(process)
                break
            time.sleep(1.0)
        if stalled:
            try:
                returncode = process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                returncode = 137
        else:
            returncode = process.wait()
    finally:
        telemetry_stop.set()
        telemetry_thread.join(15.0)
        output_thread.join(15.0)
        for event in reader.poll():
            print(f"[diagnostic:event] {json.dumps(event, sort_keys=True)}", flush=True)

    summary = {
        "duration_seconds": time.monotonic() - start,
        "event": "session_finish",
        "pytest_returncode": returncode,
        "seed": seed,
        "selection_count": len(targets),
        "stall_seconds": args.stall_seconds,
        "stalled": stalled,
        "stop_actions": stop_actions,
        "time_utc": _utc_now(),
    }
    (root / "run-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[diagnostic:summary] {json.dumps(summary, sort_keys=True)}", flush=True)
    return 124 if stalled else returncode


def _read_manifest(path: Path) -> tuple[str, ...]:
    values = tuple(path.read_text(encoding="utf-8").splitlines())
    if not values or any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError("manifest must contain unique non-empty node IDs")
    return values


def _collection_count(output: str) -> int | None:
    plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
    match = re.search(r"collected (\d+) items|(?:(\d+) tests? collected)", plain)
    if not match:
        return None
    return int(match.group(1) or match.group(2))


def _collect(args: argparse.Namespace) -> int:
    requested = _read_manifest(Path(args.manifest))
    if len(requested) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} requested nodes, found {len(requested)}")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "--collect-only",
        "-q",
        "--disable-warnings",
        f"--torch_ver={args.torch_version}",
        f"--cuda_ver={args.cuda_version}",
        "--",
        *requested,
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    collected_count = _collection_count(result.stdout)
    output = {
        "collected_count": collected_count,
        "pytest_returncode": result.returncode,
        "requested_count": len(requested),
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result.returncode or collected_count != args.expected_count:
        return 1
    print(f"Validated exact collection count={collected_count}", flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-pytest")
    run.add_argument("--artifact-dir", required=True)
    run.add_argument("--stall-seconds", type=float, default=DEFAULT_STALL_SECONDS)
    run.add_argument("--telemetry-seconds", type=float, default=DEFAULT_TELEMETRY_SECONDS)
    run.add_argument("--torch-version", required=True)
    run.add_argument("--cuda-version", required=True)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("targets", nargs=argparse.REMAINDER)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--manifest", required=True)
    collect.add_argument("--expected-count", required=True, type=int)
    collect.add_argument("--torch-version", required=True)
    collect.add_argument("--cuda-version", required=True)
    collect.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "collect":
        return _collect(args)
    return _run_pytest(args)


if __name__ == "__main__":
    raise SystemExit(main())

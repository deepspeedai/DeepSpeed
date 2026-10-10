# Copyright (c) Snowflake.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Trusted controller for the modal-torch-latest workflow.

GitHub runs this file from the trusted base revision. Pull-request code is
identified only by a validated public repository name and exact commit SHA,
then fetched, installed, and tested inside a no-secret Modal Sandbox.

The ``checkout-candidate`` and ``validate-selection`` subcommands are
pure-stdlib so the no-secret selection job can use them without importing
Modal.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

DEFAULT_MODAL_TORCH_PRESET = "2.10.0-cuda12.8"
DEFAULT_MODAL_TRANSFORMERS_SOURCE = "git"
MODAL_TORCH_PRESETS = {
    "2.7.1-cuda12.8": {
        "image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel",
        "torch_package": "torch==2.7.1",
        "torchvision_package": "torchvision==0.22.1",
        "torch_test_version": "2.7",
        "cuda_test_version": "12.8",
    },
    "2.8.0-cuda12.8": {
        "image": "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel",
        "torch_package": "torch==2.8.0",
        "torchvision_package": "torchvision==0.23.0",
        "torch_test_version": "2.8",
        "cuda_test_version": "12.8",
    },
    "2.9.1-cuda12.8": {
        "image": "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel",
        "torch_package": "torch==2.9.1",
        "torchvision_package": "torchvision==0.24.1",
        "torch_test_version": "2.9",
        "cuda_test_version": "12.8",
    },
    "2.10.0-cuda12.8": {
        "image": "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel",
        "torch_package": "torch==2.10.0",
        "torchvision_package": "torchvision==0.25.0",
        "torch_test_version": "2.10",
        "cuda_test_version": "12.8",
    },
    "2.11.0-cuda12.8": {
        "image": "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel",
        "torch_package": "torch==2.11.0",
        "torchvision_package": "torchvision==0.26.0",
        "torch_test_version": "2.11",
        "cuda_test_version": "12.8",
    },
}
PYTORCH_CUDA_128_INDEX_URL = "https://download.pytorch.org/whl/cu128"
APP_NAME = "deepspeedai-torch-latest-ci"
SANDBOX_TIMEOUT_SECONDS = 7200
SANDBOX_ACQUIRE_TIMEOUT_SECONDS = 600
INFRASTRUCTURE_SMOKE_TIMEOUT_SECONDS = 300
NO_PROGRESS_TIMEOUT_SECONDS = 300
AWS_BACKEND_START_TIMEOUT_SECONDS = 600
AWS_COMMAND_POLL_SECONDS = 10
AWS_LOG_POLL_BUDGET_SECONDS = 5
AWS_PREPARE_TIMEOUT_SECONDS = 600
AWS_TEST_TIMEOUT_SECONDS = 3000
AWS_SSM_LOG_GROUP = "/deepspeed-ci/modal-fallback-ssm"
AWS_INSTANCE_TYPE = "g7.12xlarge"
AWS_REGION_ORDER = ("us-east-1", "us-east-2", "us-west-2")
AWS_CAPACITY_ERROR_CODES = frozenset({"InsufficientInstanceCapacity", "InsufficientHostCapacity"})
AWS_CAPACITY_STATE_REASON_CODES = frozenset({"Server.InsufficientInstanceCapacity", "Server.InsufficientHostCapacity"})
AWS_PROJECT_TAG = "deepspeed-ci"
# Exit codes that nightly triage (see .github/workflows/nightly-bisect.yml) keys on. GitHub only
# reports run success/failure, so the controller also prints a DS_CI_FAILURE_CLASS=<class> sentinel
# line that survives into the job logs even when the job is killed before it can exit.
EXIT_TEST_FAILURE = 1
EXIT_INFRA = 75  # EX_TEMPFAIL: no GPU instance was provisioned, so no test ever ran
EXIT_TIMEOUT = 124
# The Sandbox server-side lifetime can kill a run slightly before the local clock crosses the
# nominal budget, so classify a failure as a timeout just inside the limit.
SANDBOX_TIMEOUT_GRACE_SECONDS = 120
MAX_TEST_LIST_BYTES = 64 * 1024
MAX_TEST_TARGETS = 1024
MAX_DISPLAY_BYTES_PER_COMMAND = 16 * 1024 * 1024
SSM_INLINE_STDOUT_CHARS = 24_000
SSM_INLINE_STDERR_CHARS = 8_000
SSM_OUTPUT_TAIL_BYTES = 32 * 1024
REMOTE_ROOT = "/workspace"
REMOTE_REPOSITORY = f"{REMOTE_ROOT}/deepspeed"
REMOTE_TRANSFORMERS = f"{REMOTE_ROOT}/transformers"
AWS_PYTHON_VERSION = "3.10.13"
AWS_UV_VERSION = "0.12.7"
AWS_PYTHON_ROOT = f"{REMOTE_ROOT}/python310"
AWS_CONTAINER_PATH = f"{AWS_PYTHON_ROOT}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
GDS_TEST_TARGET = "tests/unit/v1/nvme/test_gds.py"
AWS_CONFIG_ENV = "AWS_MODAL_FALLBACK_CONFIG"
AWS_RUN_TAG = "DeepSpeedCIRun"

NO_PROGRESS_WATCHDOG = r"""
import os
import selectors
import signal
import subprocess
import sys
import time

timeout_seconds = float(sys.argv[1])
command = sys.argv[2:]
process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
selector = selectors.DefaultSelector()
selector.register(process.stdout, selectors.EVENT_READ)
started = time.monotonic()
last_output = None

def process_group_exists():
    try:
        os.killpg(process.pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def terminate_process_group():
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    # The ordinary test path gets a 30-second graceful teardown. Short-timeout
    # fixtures scale that grace down so they still exercise SIGKILL escalation.
    grace_deadline = time.monotonic() + min(30.0, max(0.1, timeout_seconds))
    while process_group_exists() and time.monotonic() < grace_deadline:
        time.sleep(min(0.1, max(0.0, grace_deadline - time.monotonic())))

    if process_group_exists():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if process.poll() is None:
        process.wait()


while True:
    # Sub-second values are used only by the watchdog tests. Give the child a
    # separate startup bound so host scheduling delay is not mistaken for a
    # no-progress timeout; the production 300-second bound is unchanged.
    current_timeout = timeout_seconds if last_output is not None else max(1.0, timeout_seconds)
    last_progress = last_output if last_output is not None else started
    remaining = current_timeout - (time.monotonic() - last_progress)
    if remaining <= 0:
        print(f"No test output for {timeout_seconds:g}s; terminating pytest", flush=True)
        terminate_process_group()
        raise SystemExit(124)
    events = selector.select(timeout=min(1.0, remaining))
    if not events:
        continue
    chunk = os.read(process.stdout.fileno(), 65536)
    if chunk:
        os.write(sys.stdout.fileno(), chunk)
        last_output = time.monotonic()
        continue
    current_timeout = timeout_seconds if last_output is not None else max(1.0, timeout_seconds)
    last_progress = last_output if last_output is not None else started
    remaining = current_timeout - (time.monotonic() - last_progress)
    if remaining <= 0:
        continue
    try:
        raise SystemExit(process.wait(timeout=remaining))
    except subprocess.TimeoutExpired:
        continue
""".strip()

_REPOSITORY_COMPONENT = r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}"
_REPOSITORY_RE = re.compile(rf"{_REPOSITORY_COMPONENT}/{_REPOSITORY_COMPONENT}\Z")
_SHA_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
_TEST_FILE_RE = re.compile(r"tests/unit/v1/(?:[^/\x00-\x1f\x7f]+/)*test_[^/\x00-\x1f\x7f]+\.py\Z")
_LAUNCH_TEMPLATE_RE = re.compile(r"lt-[0-9a-f]{8,17}\Z")
_SUBNET_RE = re.compile(r"subnet-[0-9a-f]{8,17}\Z")
_S3_BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]\Z")
_S3_PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}\Z")
_INSTANCE_RE = re.compile(r"i-[0-9a-f]{8,17}\Z")
_COMMAND_RE = re.compile(r"[0-9a-f-]{36}\Z")
_AWS_ERROR_RE = re.compile(r"An error occurred \(([^)]+)\)")
_AWS_ERROR_CODE_RE = re.compile(r"[A-Za-z0-9_.-]{1,100}\Z")
_AWS_ACCOUNT_ID_RE = re.compile(r"(?<![0-9])[0-9]{12}(?![0-9])")


def exclude_unsupported_gds_targets(targets: Sequence[str]) -> tuple[str, ...]:
    """Remove explicit GDS targets from runners without GPUDirect Storage."""
    return tuple(target for target in targets if target.split("::", 1)[0].rstrip("/") != GDS_TEST_TARGET)


@dataclass(frozen=True)
class ControllerInputs:
    repository: str
    sha: str
    selection_mode: str
    targets: tuple[str, ...]
    torch_preset: str
    transformers_source: str
    transformers_ref: str
    base_sha: str


@dataclass(frozen=True)
class RemoteCommand:
    label: str
    argv: tuple[str, ...]
    workdir: str | None = None
    expected_line: str | None = None


class RemoteCommandError(RuntimeError):
    """One structurally launched backend command returned a nonzero status."""

    def __init__(self, label: str, return_code: int):
        super().__init__(f"{label} failed with exit code {return_code}")
        self.label = label
        self.return_code = return_code


class ControllerCleanupError(RuntimeError):
    """A primary controller failure accompanied by a cleanup failure."""

    def __init__(self, primary: BaseException, cleanup: BaseException):
        super().__init__(f"controller failed ({primary}); backend cleanup also failed ({cleanup})")
        self.primary = primary
        self.cleanup = cleanup


class SandboxStartTimeout(RuntimeError):
    """The Sandbox never started, so no test ever ran."""

    def __init__(self, timeout_seconds: float):
        super().__init__(f"Sandbox did not start within {timeout_seconds:g}s, so no test ran. This is a capacity "
                         f"problem rather than a test failure: the GPU reservation was never satisfied.")
        self.timeout_seconds = timeout_seconds


class AwsControllerError(RuntimeError):
    """The AWS fallback failed after acquisition or for a non-capacity reason."""


class AwsCommandTimeout(AwsControllerError):
    """One SSM phase reached a provider or controller execution deadline."""

    def __init__(self, phase: str, timeout_seconds: int):
        super().__init__(f"SSM {phase} command exceeded its {timeout_seconds}-second execution bound")
        self.phase = phase
        self.timeout_seconds = timeout_seconds


class AwsCapacityExhausted(RuntimeError):
    """Every configured AWS location returned an explicit capacity error."""


class AwsCapacityUnavailable(RuntimeError):
    """One allocated EC2 request terminated with an explicit capacity reason before startup."""


@dataclass(frozen=True)
class AwsRegionConfig:
    name: str
    launch_template_id: str
    subnet_ids: tuple[str, ...]
    output_bucket: str
    output_prefix: str


@dataclass(frozen=True)
class AwsInstance:
    region: str
    instance_id: str
    config: AwsRegionConfig


def validate_repository(value: str) -> str:
    if not isinstance(value, str) or not _REPOSITORY_RE.fullmatch(value):
        raise ValueError("repository must be an ASCII owner/name pair")
    return value


def validate_sha(value: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise ValueError("commit SHA must contain exactly 40 hexadecimal characters")
    return value.lower()


def validate_transformers_ref(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 200:
        raise ValueError("Transformers ref must contain 1-200 characters")
    if value.startswith("-") or "://" in value or any(char.isspace() or not char.isprintable() for char in value):
        raise ValueError("Transformers ref contains unsafe syntax")
    if _SHA_RE.fullmatch(value):
        return value.lower()
    result = subprocess.run(
        ["git", "check-ref-format", "--branch", value],
        check=False,
        capture_output=True,
        text=True,
        env=build_git_env(),
    )
    if result.returncode:
        raise ValueError("Transformers ref is not a valid branch-like git ref")
    return value


def _validate_target(value: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("-") or "\\" in value:
        raise ValueError(f"invalid pytest target: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid pytest target: {value!r}")
    if not _TEST_FILE_RE.fullmatch(value):
        raise ValueError(f"pytest target is outside tests/unit/v1 or is not a test file: {value!r}")
    return value


def load_test_selection(path: Path, mode: str) -> tuple[str, ...]:
    if mode not in {"all", "subset", "none"}:
        raise ValueError(f"invalid selection mode: {mode!r}")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"test selection file is unavailable: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("test selection must be a non-symlink regular file")
    if metadata.st_size > MAX_TEST_LIST_BYTES:
        raise ValueError("test selection exceeds the 64 KiB limit")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("test selection is not readable UTF-8") from exc
    if any((ord(char) < 32 and char != "\n") or ord(char) == 127 for char in raw):
        raise ValueError("test selection contains control characters")
    lines = raw.splitlines()
    if any(not line for line in lines):
        raise ValueError("test selection contains an empty line")
    if len(lines) > MAX_TEST_TARGETS:
        raise ValueError("test selection exceeds the 1024-target limit")
    if len(lines) != len(set(lines)):
        raise ValueError("test selection contains duplicate targets")
    if mode == "all":
        if lines != ["tests/unit/v1"]:
            raise ValueError("all mode requires exactly tests/unit/v1")
        return tuple(lines)
    if mode == "none":
        if lines:
            raise ValueError("none mode requires an empty selection")
        return ()
    if not lines:
        raise ValueError("subset mode requires at least one test")
    return tuple(_validate_target(line) for line in lines)


def build_git_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    source = source or os.environ
    result = {
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/false",
        "SSH_ASKPASS": "/bin/false",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_LFS_SKIP_SMUDGE": "1",
        "LC_ALL": "C.UTF-8",
    }
    for key in ("PATH", "SYSTEMROOT"):
        if source.get(key):
            result[key] = source[key]
    return result


def _git_command(*args: str) -> list[str]:
    return [
        "git",
        "-c",
        "credential.helper=",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "filter.lfs.smudge=",
        "-c",
        "filter.lfs.required=false",
        *args,
    ]


def _run_local(argv: Sequence[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=build_git_env(),
    ).stdout


def _validate_checkout_symlinks(destination: Path) -> None:
    root = destination.resolve()
    output = _run_local(_git_command("ls-files", "-z", "-s"), cwd=root)
    for record in output.split("\0"):
        if not record:
            continue
        metadata, relative = record.split("\t", 1)
        mode = metadata.split(" ", 1)[0]
        if mode != "120000":
            continue
        link = root / relative
        target = link.resolve(strict=False)
        if not target.is_relative_to(root):
            raise ValueError(f"tracked symlink escapes candidate checkout: {relative!r}")


def _checkout_exact(
    head_url: str,
    head_sha: str,
    base_url: str,
    base_sha: str,
    destination: Path,
) -> None:
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"checkout destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        _run_local(_git_command("init", str(destination)))
        created = True
        fetch_head = _git_command(
            "-C",
            str(destination),
            "fetch",
            "--no-tags",
            "--no-recurse-submodules",
            head_url,
            f"{head_sha}:refs/ci/head",
        )
        _run_local(fetch_head)
        fetch_base = _git_command(
            "-C",
            str(destination),
            "fetch",
            "--no-tags",
            "--no-recurse-submodules",
            base_url,
            f"{base_sha}:refs/ci/base",
        )
        _run_local(fetch_base)
        resolved_head = _run_local(
            _git_command("-C", str(destination), "rev-parse", "--verify", "refs/ci/head^{commit}")).strip()
        resolved_base = _run_local(
            _git_command("-C", str(destination), "rev-parse", "--verify", "refs/ci/base^{commit}")).strip()
        if resolved_head != head_sha or resolved_base != base_sha:
            raise ValueError("fetched commit did not match requested event SHA")
        _run_local(_git_command("-C", str(destination), "checkout", "--detach", "refs/ci/head"))
        checked_out = _run_local(_git_command("-C", str(destination), "rev-parse", "HEAD")).strip()
        if checked_out != head_sha:
            raise ValueError("checked-out HEAD did not match requested event SHA")
        _validate_checkout_symlinks(destination)
    except BaseException:
        if created:
            shutil.rmtree(destination, ignore_errors=True)
        raise


def checkout_candidate(
    head_repository: str,
    head_sha: str,
    base_repository: str,
    base_sha: str,
    destination: Path,
) -> None:
    head_repository = validate_repository(head_repository)
    base_repository = validate_repository(base_repository)
    head_sha = validate_sha(head_sha)
    base_sha = validate_sha(base_sha)
    _checkout_exact(
        f"https://github.com/{head_repository}.git",
        head_sha,
        f"https://github.com/{base_repository}.git",
        base_sha,
        destination,
    )


def resolve_controller_inputs(env: Mapping[str, str]) -> ControllerInputs:
    event_name = env.get("GITHUB_EVENT_NAME", "")
    repository = env.get("DS_CI_REPOSITORY", "")
    sha = env.get("DS_CI_SHA", "")
    if not repository or not sha:
        if event_name == "pull_request_target":
            raise ValueError("pull_request_target requires explicit PR repository and SHA metadata")
        repository = repository or env.get("GITHUB_REPOSITORY", "")
        sha = sha or env.get("GITHUB_SHA", "")
    repository = validate_repository(repository)
    sha = validate_sha(sha)
    # The base SHA keys the baked requirements layer: merge-group bases move slowly, so the
    # layer cache stays hot, while the candidate SHA changes every run. Events without a base
    # (push, dispatch) reuse the candidate SHA, which only lowers the hit rate, never correctness.
    base_sha = validate_sha(env.get("DS_CI_BASE_SHA", "") or sha)

    selection_mode = env.get("DS_TEST_SELECTION_MODE", "")
    selection_file = env.get("DS_TEST_LIST_FILE", "")
    if not selection_file:
        raise ValueError("DS_TEST_LIST_FILE is required")
    targets = load_test_selection(Path(selection_file), selection_mode)

    torch_preset = env.get("MODAL_TORCH_PRESET") or DEFAULT_MODAL_TORCH_PRESET
    if torch_preset not in MODAL_TORCH_PRESETS:
        supported = ", ".join(sorted(MODAL_TORCH_PRESETS))
        raise ValueError(f"unsupported MODAL_TORCH_PRESET={torch_preset!r}; supported values: {supported}")
    transformers_source = env.get("MODAL_TRANSFORMERS_SOURCE") or DEFAULT_MODAL_TRANSFORMERS_SOURCE
    if transformers_source not in {"requirements", "git"}:
        raise ValueError("MODAL_TRANSFORMERS_SOURCE must be 'requirements' or 'git'")
    transformers_ref = env.get("MODAL_TRANSFORMERS_REF", "")
    if transformers_source == "git":
        transformers_ref = validate_transformers_ref(transformers_ref or "main")
    else:
        transformers_ref = ""

    return ControllerInputs(
        repository=repository,
        sha=sha,
        selection_mode=selection_mode,
        targets=targets,
        torch_preset=torch_preset,
        transformers_source=transformers_source,
        transformers_ref=transformers_ref,
        base_sha=base_sha,
    )


def build_sandbox_env() -> dict[str, str]:
    return {
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/false",
        "SSH_ASKPASS": "/bin/false",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_LFS_SKIP_SMUDGE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
    }


def _build_sandbox_image(modal_module: Any, preset: dict[str, str], inputs: ControllerInputs) -> Any:
    """Bake the static dependency chain into content-addressed image layers.

    Layers apply in chain order, mirroring the previous runtime sequence: requirements first,
    then the Torch pin, so a transitive dependency cannot displace the intended CUDA build.
    Modal caches each layer by its inputs, so a warm run skips both installs entirely; the
    runtime `pip install -r` commands remain as cheap correctness guards -- they are a no-op
    unless the candidate branch changed a requirements file, in which case they install the
    difference. The requirements layer is keyed by the base SHA, which moves far slower than
    the candidate SHA the controller tests.
    """
    requirements_url = f"https://raw.githubusercontent.com/{inputs.repository}/{inputs.base_sha}/requirements"
    image = modal_module.Image.from_registry(preset["image"], add_python="3.10")
    image = image.run_commands(
        f"python -m pip install -r {requirements_url}/requirements.txt "
        f"-r {requirements_url}/requirements-dev.txt -r {requirements_url}/requirements-deepcompile.txt")
    return image.pip_install(preset["torch_package"],
                             preset["torchvision_package"],
                             index_url=PYTORCH_CUDA_128_INDEX_URL)


def build_sandbox_kwargs(image: Any, *, timeout_seconds: int = SANDBOX_TIMEOUT_SECONDS) -> dict[str, Any]:
    return {
        "image": image,
        "env": build_sandbox_env(),
        "secrets": [],
        "network_file_systems": {},
        "volumes": {},
        "encrypted_ports": [],
        "h2_ports": [],
        "unencrypted_ports": [],
        "proxy": None,
        "block_network": False,
        "cloud": "oci",
        "gpu": "l40s:2",
        "timeout": timeout_seconds,
    }


def _remote_git(*args: str) -> tuple[str, ...]:
    return tuple(_git_command(*args))


def _with_no_progress_watchdog(argv: Sequence[str]) -> tuple[str, ...]:
    return ("python", "-u", "-c", NO_PROGRESS_WATCHDOG, str(NO_PROGRESS_TIMEOUT_SECONDS), *argv)


def build_remote_commands(inputs: ControllerInputs) -> tuple[RemoteCommand, ...]:
    preset = MODAL_TORCH_PRESETS[inputs.torch_preset]
    repository_url = f"https://github.com/{inputs.repository}.git"
    commands = [
        RemoteCommand("install system prerequisites", ("apt-get", "update")),
        RemoteCommand("install system packages", ("apt-get", "install", "-y", "git", "libaio-dev")),
        RemoteCommand("create work root", ("mkdir", "-p", REMOTE_ROOT)),
        RemoteCommand("initialize candidate repository", _remote_git("init", REMOTE_REPOSITORY)),
        RemoteCommand(
            "fetch candidate SHA",
            _remote_git(
                "-C",
                REMOTE_REPOSITORY,
                "fetch",
                "--no-tags",
                "--no-recurse-submodules",
                "--depth=1",
                repository_url,
                f"{inputs.sha}:refs/ci/candidate",
            ),
        ),
        RemoteCommand(
            "checkout candidate SHA",
            _remote_git("-C", REMOTE_REPOSITORY, "checkout", "--detach", "refs/ci/candidate"),
        ),
        RemoteCommand(
            "verify candidate SHA",
            _remote_git("-C", REMOTE_REPOSITORY, "rev-parse", "--verify", "HEAD^{commit}"),
            expected_line=inputs.sha,
        ),
        RemoteCommand(
            "install runtime requirements",
            ("python", "-m", "pip", "install", "-r", "requirements/requirements.txt"),
            REMOTE_REPOSITORY,
        ),
        RemoteCommand(
            "install development requirements",
            ("python", "-m", "pip", "install", "-r", "requirements/requirements-dev.txt"),
            REMOTE_REPOSITORY,
        ),
        RemoteCommand(
            "install DeepCompile requirements",
            ("python", "-m", "pip", "install", "-r", "requirements/requirements-deepcompile.txt"),
            REMOTE_REPOSITORY,
        ),
        # The image layer already pins Torch after its requirements layer. Repeating the pin at
        # runtime is normally a no-op and keeps the same manifest correct on the AWS container.
        RemoteCommand(
            "pin PyTorch runtime",
            (
                "python",
                "-m",
                "pip",
                "install",
                preset["torch_package"],
                preset["torchvision_package"],
                "--index-url",
                PYTORCH_CUDA_128_INDEX_URL,
            ),
            REMOTE_REPOSITORY,
        ),
    ]
    if inputs.transformers_source == "git":
        commands.extend([
            RemoteCommand("initialize Transformers repository", _remote_git("init", REMOTE_TRANSFORMERS)),
            RemoteCommand(
                "fetch Transformers ref",
                _remote_git(
                    "-C",
                    REMOTE_TRANSFORMERS,
                    "fetch",
                    "--no-tags",
                    "--no-recurse-submodules",
                    "--depth=1",
                    "https://github.com/huggingface/transformers.git",
                    inputs.transformers_ref,
                ),
            ),
            RemoteCommand(
                "checkout Transformers ref",
                _remote_git("-C", REMOTE_TRANSFORMERS, "checkout", "--detach", "FETCH_HEAD"),
            ),
            RemoteCommand(
                "report Transformers commit",
                _remote_git("-C", REMOTE_TRANSFORMERS, "rev-parse", "HEAD"),
            ),
            RemoteCommand(
                "install Transformers",
                ("python", "-m", "pip", "install", "."),
                REMOTE_TRANSFORMERS,
            ),
        ])
    commands.extend([
        RemoteCommand("install candidate DeepSpeed", ("python", "-m", "pip", "install", "."), REMOTE_REPOSITORY),
        RemoteCommand(
            "report package versions",
            (
                "python",
                "-c",
                "import json, torch, torchvision, transformers; "
                "print(json.dumps({'torch': torch.__version__, 'torch_cuda': torch.version.cuda, "
                "'torchvision': torchvision.__version__, 'transformers': transformers.__version__}, "
                "sort_keys=True))",
            ),
            REMOTE_REPOSITORY,
        ),
        RemoteCommand(
            "verify two visible GPUs",
            (
                "python",
                "-c",
                "import json, torch; "
                "count = torch.cuda.device_count(); "  #ignore-cuda
                "print(json.dumps({'device_count': count, 'devices': "
                "[torch.cuda.get_device_name(index) for index in range(count)]}, sort_keys=True)); "  #ignore-cuda
                "assert count == 2, f'expected exactly 2 visible GPUs, observed {count}'",
            ),
            REMOTE_REPOSITORY,
        ),
        RemoteCommand(
            "run pytest",
            _with_no_progress_watchdog((
                "pytest",
                "-n",
                "4",
                "--verbose",
                # GDS tests require GPUDirect Storage support unavailable on these runners.
                "--ignore=tests/unit/v1/nvme/test_gds.py",
                f"--torch_ver={preset['torch_test_version']}",
                f"--cuda_ver={preset['cuda_test_version']}",
                "--",
                *inputs.targets,
            )),
            REMOTE_REPOSITORY,
        ),
    ])
    return tuple(commands)


def _single_line(value: object) -> str:
    return "".join(char if char.isprintable() else f"\\x{ord(char):02x}" for char in str(value))


def run_sandbox_command(sandbox: Any, modal_module: Any, command: RemoteCommand) -> str:
    process = sandbox.exec(
        *command.argv,
        stderr=modal_module.stream_type.StreamType.STDOUT,
        workdir=command.workdir,
    )
    displayed = 0
    last_line = ""
    truncated = False
    for raw_line in process.stdout:
        line = _single_line(raw_line.rstrip("\r\n"))
        last_line = line
        rendered = f"[sandbox:{command.label}] {line}"
        encoded_size = len(rendered.encode("utf-8", errors="replace")) + 1
        if displayed + encoded_size <= MAX_DISPLAY_BYTES_PER_COMMAND:
            print(rendered)
            displayed += encoded_size
        else:
            truncated = True
    if truncated:
        print(f"[sandbox:{command.label}] output truncated after {MAX_DISPLAY_BYTES_PER_COMMAND} bytes")
    return_code = process.wait()
    if return_code:
        raise RemoteCommandError(command.label, return_code)
    if command.expected_line is not None:
        actual = last_line.strip()
        if actual != command.expected_line:
            raise RuntimeError(
                f"{command.label} returned {_single_line(actual)!r}, expected {command.expected_line!r}")
    return last_line


def load_aws_config(raw: str) -> tuple[AwsRegionConfig, ...]:
    if not raw or len(raw.encode("utf-8")) > 16 * 1024:
        raise ValueError(f"{AWS_CONFIG_ENV} must contain a bounded regional JSON value")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{AWS_CONFIG_ENV} is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"regions"} or not isinstance(payload["regions"], list):
        raise ValueError(f"{AWS_CONFIG_ENV} must contain only a regions list")

    regions = []
    for expected_name, item in zip(AWS_REGION_ORDER, payload["regions"], strict=False):
        required = {"name", "launch_template_id", "subnet_ids", "output_bucket", "output_prefix"}
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("each AWS region entry must contain exactly the documented fields")
        if item["name"] != expected_name:
            raise ValueError(f"AWS regions must use fixed order {AWS_REGION_ORDER}")
        if not isinstance(item["launch_template_id"], str) or not _LAUNCH_TEMPLATE_RE.fullmatch(
                item["launch_template_id"]):
            raise ValueError(f"invalid launch template for {expected_name}")
        subnet_ids = item["subnet_ids"]
        if (not isinstance(subnet_ids, list) or not subnet_ids or len(subnet_ids) > 8
                or any(not isinstance(value, str) or not _SUBNET_RE.fullmatch(value) for value in subnet_ids)
                or len(subnet_ids) != len(set(subnet_ids))):
            raise ValueError(f"invalid subnet list for {expected_name}")
        bucket = item["output_bucket"]
        prefix = item["output_prefix"]
        if not isinstance(bucket, str) or not _S3_BUCKET_RE.fullmatch(bucket) or ".." in bucket:
            raise ValueError(f"invalid output bucket for {expected_name}")
        if (not isinstance(prefix, str) or not _S3_PREFIX_RE.fullmatch(prefix) or ".." in prefix or "//" in prefix):
            raise ValueError(f"invalid output prefix for {expected_name}")
        regions.append(
            AwsRegionConfig(
                name=expected_name,
                launch_template_id=item["launch_template_id"],
                subnet_ids=tuple(subnet_ids),
                output_bucket=bucket,
                output_prefix=prefix.rstrip("/"),
            ))
    if len(regions) != len(AWS_REGION_ORDER) or len(payload["regions"]) != len(AWS_REGION_ORDER):
        raise ValueError(f"AWS config must contain exactly {len(AWS_REGION_ORDER)} regions")
    return tuple(regions)


def _aws_run_identity(env: Mapping[str, str]) -> str:
    run_id = env.get("GITHUB_RUN_ID", "")
    attempt = env.get("GITHUB_RUN_ATTEMPT", "")
    if not run_id.isdigit() or not attempt.isdigit():
        raise ValueError("GitHub run ID and attempt are required for run-owned AWS cleanup")
    suffix = env.get("DS_CI_AWS_RUN_SUFFIX", "")
    if suffix and suffix not in AWS_REGION_ORDER:
        raise ValueError("AWS run suffix must be one of the configured regions")
    return "-".join(value for value in (run_id, attempt, suffix) if value)


def _default_command(argv: Sequence[str], *, timeout: float | None = None) -> subprocess.CompletedProcess:
    command_env = dict(os.environ)
    command_env.update({"AWS_MAX_ATTEMPTS": "1", "AWS_RETRY_MODE": "standard", "AWS_PAGER": ""})
    return subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=command_env,
    )


def _run_aws(run_command: Any, args: Sequence[str], *, timeout: float | None = None) -> Any:
    try:
        return run_command(("aws", "--no-cli-pager", *args), timeout=timeout)
    except subprocess.TimeoutExpired:
        # TimeoutExpired renders the full argv, including private resource IDs.
        # The stable message preserves the failure while suppressing that cause.
        raise AwsControllerError("AWS CLI operation exceeded its time bound") from None


def _aws_error_code(result: Any) -> str:
    match = _AWS_ERROR_RE.search(result.stderr or "")
    return match.group(1) if match and _AWS_ERROR_CODE_RE.fullmatch(match.group(1)) else ""


def _aws_text(run_command: Any, args: Sequence[str], *, timeout: float | None = None) -> str:
    result = _run_aws(run_command, args, timeout=timeout)
    if result.returncode:
        code = _aws_error_code(result) or f"exit {result.returncode}"
        raise AwsControllerError(f"AWS CLI failed ({code})")
    return result.stdout.strip()


def _aws_json(run_command: Any, args: Sequence[str], *, timeout: float | None = None) -> dict[str, Any]:
    output = _aws_text(run_command, (*args, "--output", "json"), timeout=timeout)
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise AwsControllerError("AWS CLI returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise AwsControllerError("AWS CLI returned an unexpected JSON value")
    return payload


def acquire_aws_instance(
    regions: Sequence[AwsRegionConfig],
    run_identity: str,
    run_command: Any = _default_command,
    *,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
    start_timeout_seconds: int = AWS_BACKEND_START_TIMEOUT_SECONDS,
) -> AwsInstance:
    for config in regions:
        for location_index, subnet_id in enumerate(config.subnet_ids, start=1):
            print(f"AWS_STATE=acquiring region={config.name} location={location_index}", flush=True)
            tag_spec = ("ResourceType=instance,Tags=["
                        f"{{Key=Project,Value={AWS_PROJECT_TAG}}},"
                        f"{{Key={AWS_RUN_TAG},Value={run_identity}}}"
                        "]")
            result = _run_aws(
                run_command,
                (
                    "ec2",
                    "run-instances",
                    "--region",
                    config.name,
                    "--launch-template",
                    f"LaunchTemplateId={config.launch_template_id}",
                    "--instance-type",
                    AWS_INSTANCE_TYPE,
                    "--subnet-id",
                    subnet_id,
                    "--instance-market-options",
                    "MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}",
                    "--tag-specifications",
                    tag_spec,
                    "--count",
                    "1",
                    "--query",
                    "Instances[0].InstanceId",
                    "--output",
                    "text",
                ),
                timeout=120,
            )
            if result.returncode:
                code = _aws_error_code(result)
                if code in AWS_CAPACITY_ERROR_CODES:
                    print(
                        f"AWS_CAPACITY_UNAVAILABLE region={config.name} location={location_index} code={code}",
                        flush=True,
                    )
                    continue
                raise AwsControllerError(f"AWS launch failed without a capacity classification ({code or 'unknown'})")
            instance_id = result.stdout.strip()
            if not _INSTANCE_RE.fullmatch(instance_id):
                raise AwsControllerError("AWS launch did not return one valid instance ID")
            print(f"AWS_INSTANCE_ALLOCATED region={config.name}", flush=True)
            instance = AwsInstance(config.name, instance_id, config)
            try:
                wait_for_aws_backend(
                    instance,
                    run_command,
                    sleep=sleep,
                    monotonic=monotonic,
                    timeout_seconds=start_timeout_seconds,
                )
            except AwsCapacityUnavailable as exc:
                print(
                    f"AWS_CAPACITY_UNAVAILABLE region={config.name} location={location_index} "
                    f"reason={_single_line(exc)}",
                    flush=True,
                )
                terminate_aws_instance(instance, run_command)
                continue
            except BaseException as primary:
                try:
                    terminate_aws_instance(instance, run_command)
                except BaseException as cleanup:
                    raise ControllerCleanupError(primary, cleanup) from primary
                raise
            return instance
    raise AwsCapacityExhausted("all configured AWS G7.12 Spot locations reported insufficient capacity")


def wait_for_aws_backend(
    instance: AwsInstance,
    run_command: Any = _default_command,
    *,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
    timeout_seconds: int = AWS_BACKEND_START_TIMEOUT_SECONDS,
) -> None:
    deadline = monotonic() + timeout_seconds

    def remaining_timeout() -> float:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise AwsControllerError("AWS backend did not become ready before the startup bound")
        return remaining

    try:
        _aws_text(
            run_command,
            ("ec2", "wait", "instance-running", "--region", instance.region, "--instance-ids", instance.instance_id),
            timeout=remaining_timeout(),
        )
    except AwsControllerError as exc:
        if monotonic() >= deadline or str(exc) == "AWS CLI operation exceeded its time bound":
            raise AwsControllerError("AWS instance did not reach a running state before the startup bound") from None
        state_reason = _aws_text(
            run_command,
            (
                "ec2",
                "describe-instances",
                "--region",
                instance.region,
                "--instance-ids",
                instance.instance_id,
                "--query",
                "Reservations[0].Instances[0].[State.Name,StateReason.Code]",
                "--output",
                "text",
            ),
            timeout=min(60, remaining_timeout()),
        ).split()
        if len(state_reason) == 2 and state_reason[0] == "terminated" and state_reason[1] in \
                AWS_CAPACITY_STATE_REASON_CODES:
            raise AwsCapacityUnavailable(state_reason[1]) from exc
        raise
    try:
        _aws_text(
            run_command,
            ("ec2", "wait", "instance-status-ok", "--region", instance.region, "--instance-ids", instance.instance_id),
            timeout=remaining_timeout(),
        )
    except AwsControllerError as exc:
        if monotonic() >= deadline or str(exc) == "AWS CLI operation exceeded its time bound":
            raise AwsControllerError("AWS instance did not become healthy before the startup bound") from None
        raise

    while monotonic() < deadline:
        status = _aws_text(
            run_command,
            (
                "ssm",
                "describe-instance-information",
                "--region",
                instance.region,
                "--filters",
                f"Key=InstanceIds,Values={instance.instance_id}",
                "--query",
                "InstanceInformationList[0].PingStatus",
                "--output",
                "text",
            ),
            timeout=min(60, remaining_timeout()),
        )
        if status == "Online":
            print(f"AWS_STATE=backend_started region={instance.region}", flush=True)
            return
        sleep(min(AWS_COMMAND_POLL_SECONDS, max(0.0, deadline - monotonic())))
    raise AwsControllerError("AWS instance started but did not become SSM-online before the startup bound")


def _docker_exec(container_name: str, command: RemoteCommand) -> str:
    argv = ["docker", "exec"]
    if command.workdir is not None:
        argv.extend(("--workdir", command.workdir))
    argv.extend((container_name, *command.argv))
    rendered = shlex.join(argv)
    if command.expected_line is None:
        return rendered
    expected = shlex.quote(command.expected_line)
    return f"actual=$({rendered}); printf '%s\\n' \"$actual\"; test \"$actual\" = {expected}"


def build_aws_bootstrap_commands() -> tuple[RemoteCommand, ...]:
    uv_root = f"{REMOTE_ROOT}/uv-bootstrap"
    return (
        RemoteCommand(
            "bootstrap fixed uv",
            (
                "python3",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--break-system-packages",
                "--no-deps",
                "--target",
                uv_root,
                f"uv=={AWS_UV_VERSION}",
            ),
        ),
        RemoteCommand(
            "create fixed Python runtime",
            (
                "env",
                f"UV_CACHE_DIR={REMOTE_ROOT}/cache/uv",
                f"UV_PYTHON_INSTALL_DIR={REMOTE_ROOT}/uv-python",
                f"{uv_root}/bin/uv",
                "venv",
                "--python",
                AWS_PYTHON_VERSION,
                "--seed",
                AWS_PYTHON_ROOT,
            ),
        ),
        RemoteCommand(
            "verify fixed Python runtime",
            ("python", "-c", "import platform; print(platform.python_version())"),
            expected_line=AWS_PYTHON_VERSION,
        ),
    )


def build_aws_scripts(inputs: ControllerInputs, run_identity: str) -> tuple[str, str]:
    preset = MODAL_TORCH_PRESETS[inputs.torch_preset]
    container_name = f"ds-ci-{run_identity}"
    run_root = f"/var/lib/devds/runs/{run_identity}"
    commands = build_remote_commands(inputs)
    prepare_commands = [
        *build_aws_bootstrap_commands(), *(command for command in commands if command.label != "run pytest")
    ]
    pytest_commands = [command for command in commands if command.label == "run pytest"]
    if len(pytest_commands) != 1:
        raise ValueError("the shared manifest must contain exactly one pytest command")

    docker_run = [
        "docker",
        "run",
        "--detach",
        "--name",
        container_name,
        "--gpus",
        '"device=0,1"',
        "--shm-size",
        "32G",
        "--volume",
        f"{run_root}:{REMOTE_ROOT}",
    ]
    for key, value in build_sandbox_env().items():
        docker_run.extend(("--env", f"{key}={value}"))
    docker_run.extend(("--env", f"PATH={AWS_CONTAINER_PATH}"))
    docker_run.extend((preset["image"], "sleep", "infinity"))

    prepare_lines = [
        "set -euo pipefail",
        f"install -d -m 0755 {shlex.quote(run_root)}",
        shlex.join(("docker", "pull", preset["image"])),
        shlex.join(docker_run),
    ]
    prepare_lines.extend(_docker_exec(container_name, command) for command in prepare_commands)
    test_lines = [
        "set -euo pipefail",
        "printf 'AWS_STATE=test_started backend=aws\\n'",
        _docker_exec(container_name, pytest_commands[0]),
    ]
    return "\n".join(prepare_lines), "\n".join(test_lines)


def build_aws_infrastructure_smoke_script(
    torch_preset: str,
    run_identity: str,
    config: AwsRegionConfig,
    *,
    run_root: str | None = None,
) -> str:
    if torch_preset not in MODAL_TORCH_PRESETS:
        raise ValueError("unsupported PyTorch preset for infrastructure smoke")
    image = MODAL_TORCH_PRESETS[torch_preset]["image"]
    run_root = run_root or f"/var/lib/devds/runs/{run_identity}"
    proof_file = f"{run_root}/artifacts/stdout-proof.txt"
    proof_key = f"{config.output_prefix}/{run_identity}/infra-smoke/stdout-proof.txt"
    container_check = (
        "import torch; "
        "count = torch.cuda.device_count(); "  #ignore-cuda
        "assert count == 2, f'expected exactly 2 visible GPUs, observed {count}'; "
        "print('AWS_INFRA_SMOKE=ready device_count=2 container=true')")
    docker_run = (
        "docker",
        "run",
        "--rm",
        "--gpus",
        '"device=0,1"',
        "--volume",
        f"{run_root}:{REMOTE_ROOT}",
        image,
        "python",
        "-c",
        container_check,
    )
    return "\n".join((
        "set -euo pipefail",
        f"install -d -m 0755 {shlex.quote(run_root)} {shlex.quote(str(PurePosixPath(proof_file).parent))}",
        "host_gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)",
        'test "$host_gpu_count" -eq 2',
        f"{shlex.join(('docker', 'pull', '--quiet', image))} >/dev/null",
        f"{shlex.join(docker_run)} | tee {shlex.quote(proof_file)}",
        f"grep -Fxq 'AWS_INFRA_SMOKE=ready device_count=2 container=true' {shlex.quote(proof_file)}",
        f"proof_checksum=$(openssl dgst -sha256 -binary {shlex.quote(proof_file)} | openssl base64 -A)",
        "uploaded_checksum=$(aws --no-cli-pager s3api put-object "
        f"--region {shlex.quote(config.name)} --bucket {shlex.quote(config.output_bucket)} "
        f"--key {shlex.quote(proof_key)} --body {shlex.quote(proof_file)} "
        "--checksum-sha256 \"$proof_checksum\" --query ChecksumSHA256 --output text 2>/dev/null) || "
        "{ printf 'AWS_INFRA_SMOKE=private_log_upload_failed\\n'; exit 1; }",
        'test "$uploaded_checksum" = "$proof_checksum" || '
        "{ printf 'AWS_INFRA_SMOKE=private_log_checksum_failed\\n'; exit 1; }",
        "printf 'AWS_INFRA_SMOKE=private_log_retained checksum=sha256\\n'",
    ))


def send_ssm_command(
    instance: AwsInstance,
    run_identity: str,
    phase: str,
    script: str,
    timeout_seconds: int,
    run_command: Any = _default_command,
) -> str:
    command_id = _aws_text(
        run_command,
        (
            "ssm",
            "send-command",
            "--region",
            instance.region,
            "--instance-ids",
            instance.instance_id,
            "--document-name",
            "AWS-RunShellScript",
            "--parameters",
            json.dumps({"commands": [shlex.join(("bash", "-lc", script))]}, separators=(",", ":")),
            "--timeout-seconds",
            str(timeout_seconds),
            "--output-s3-bucket-name",
            instance.config.output_bucket,
            "--output-s3-key-prefix",
            f"{instance.config.output_prefix}/{run_identity}/{phase}",
            "--output-s3-region",
            instance.region,
            "--cloud-watch-output-config",
            json.dumps({
                "CloudWatchLogGroupName": AWS_SSM_LOG_GROUP,
                "CloudWatchOutputEnabled": True,
            },
                       separators=(",", ":")),
            "--query",
            "Command.CommandId",
            "--output",
            "text",
        ),
        timeout=60,
    )
    if not _COMMAND_RE.fullmatch(command_id):
        raise AwsControllerError("SSM did not return one valid command ID")
    print(f"AWS_SSM_COMMAND phase={phase} submitted=true", flush=True)
    return command_id


def _sanitize_ssm_output_line(
    value: object,
    instance: AwsInstance,
    command_id: str,
    run_identity: str,
) -> str:
    public_line = str(value)
    replacements = (
        (instance.instance_id, "<instance-id>"),
        (command_id, "<command-id>"),
        (instance.config.output_bucket, "<output-bucket>"),
        (f"/var/lib/devds/runs/{run_identity}", "<run-root>"),
    )
    for private_value, replacement in replacements:
        public_line = public_line.replace(private_value, replacement)
    public_line = _AWS_ACCOUNT_ID_RE.sub("<account-id>", public_line)
    error_match = _AWS_ERROR_RE.search(public_line)
    if error_match:
        error_code = error_match.group(1)
        if not _AWS_ERROR_CODE_RE.fullmatch(error_code):
            error_code = "unknown"
        return f"AWS CLI error ({error_code})"
    return _single_line(public_line)


def _ssm_output_key(run_identity: str, phase: str, command_id: str, instance_id: str, stream: str) -> str:
    return str(
        PurePosixPath(run_identity) / phase / command_id / instance_id / "awsrunShellScript" / "0.awsrunShellScript" /
        stream)


def _read_ssm_output_tail(
    instance: AwsInstance,
    run_identity: str,
    phase: str,
    command_id: str,
    stream: str,
    run_command: Any,
) -> str | None:
    key = str(
        PurePosixPath(instance.config.output_prefix) /
        _ssm_output_key(run_identity, phase, command_id, instance.instance_id, stream))
    with tempfile.TemporaryDirectory(prefix="ds-ssm-tail-") as temp_root:
        output_path = Path(temp_root) / stream
        try:
            result = _run_aws(
                run_command,
                (
                    "s3api",
                    "get-object",
                    "--region",
                    instance.region,
                    "--bucket",
                    instance.config.output_bucket,
                    "--key",
                    key,
                    "--range",
                    f"bytes=-{SSM_OUTPUT_TAIL_BYTES}",
                    "--output",
                    "json",
                    str(output_path),
                ),
                timeout=60,
            )
        except AwsControllerError:
            return None
        if result.returncode:
            return None
        try:
            return output_path.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            return None


def _emit_ssm_output(
    phase: str,
    payload: Mapping[str, Any],
    instance: AwsInstance,
    command_id: str,
    run_identity: str,
    run_command: Any,
    emitted_lines: Mapping[str, Counter[str]],
    incomplete_streams: set[str],
) -> None:
    streams = (
        ("StandardOutputContent", "StandardOutputUrl", "stdout", SSM_INLINE_STDOUT_CHARS),
        ("StandardErrorContent", "StandardErrorUrl", "stderr", SSM_INLINE_STDERR_CHARS),
    )
    for content_field, url_field, stream, inline_limit in streams:
        live_counts = emitted_lines.get(stream, Counter())
        if live_counts and stream not in incomplete_streams:
            continue
        content = str(payload.get(content_field, ""))
        remaining_live = live_counts.copy()
        sanitized_content = []
        for line in content.splitlines():
            public_line = _sanitize_ssm_output_line(line, instance, command_id, run_identity)
            sanitized_content.append(public_line)
            if remaining_live[public_line]:
                remaining_live[public_line] -= 1
                continue
            print(f"[aws:{phase}] {public_line}", flush=True)
        if (phase == "test" and payload.get("Status") != "Success" and len(content) >= inline_limit
                and payload.get(url_field)):
            tail = _read_ssm_output_tail(instance, run_identity, phase, command_id, stream, run_command)
            if tail is None:
                print(f"[aws:{phase}:tail] complete {stream} unavailable", flush=True)
                continue
            print(f"[aws:{phase}:tail] complete {stream}", flush=True)
            already_seen = live_counts + Counter(sanitized_content)
            for line in tail.splitlines():
                public_line = _sanitize_ssm_output_line(line, instance, command_id, run_identity)
                if already_seen[public_line]:
                    already_seen[public_line] -= 1
                    continue
                print(f"[aws:{phase}:tail] {public_line}", flush=True)


def _poll_ssm_cloudwatch_output(
    phase: str,
    instance: AwsInstance,
    command_id: str,
    run_identity: str,
    run_command: Any,
    tokens: dict[str, str],
    emitted_lines: dict[str, Counter[str]],
    poll_deadline: float,
    monotonic: Any,
) -> set[str]:
    incomplete_streams: set[str] = set()
    streams = ("stdout", "stderr")
    for index, stream in enumerate(streams):
        if monotonic() >= poll_deadline:
            incomplete_streams.update(streams[index:])
            break
        token = tokens.get(stream)
        stream_name = f"{command_id}/{instance.instance_id}/awsrunShellScript/{stream}"
        end_time = str(int(time.time() * 1000) + 1)
        while True:
            remaining = poll_deadline - monotonic()
            if remaining <= 0:
                incomplete_streams.add(stream)
                break
            args = [
                "logs",
                "get-log-events",
                "--region",
                instance.region,
                "--log-group-name",
                AWS_SSM_LOG_GROUP,
                "--log-stream-name",
                stream_name,
                "--start-from-head",
                "--end-time",
                end_time,
                "--output",
                "json",
            ]
            if token is not None:
                args.extend(("--next-token", token))
            try:
                result = _run_aws(run_command, args, timeout=remaining)
            except AwsControllerError:
                print(f"[aws:{phase}:{stream}] live output temporarily unavailable (timeout)", flush=True)
                incomplete_streams.add(stream)
                break
            if result.returncode:
                code = _aws_error_code(result)
                if code != "ResourceNotFoundException":
                    print(f"[aws:{phase}:{stream}] live output temporarily unavailable ({code or 'unknown'})",
                          flush=True)
                incomplete_streams.add(stream)
                break
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                print(f"[aws:{phase}:{stream}] live output temporarily unavailable (invalid-response)", flush=True)
                incomplete_streams.add(stream)
                break
            events = payload.get("events") if isinstance(payload, dict) else None
            next_token = payload.get("nextForwardToken") if isinstance(payload, dict) else None
            if not isinstance(events, list) or not isinstance(next_token, str) or not next_token:
                print(f"[aws:{phase}:{stream}] live output temporarily unavailable (invalid-response)", flush=True)
                incomplete_streams.add(stream)
                break
            for event in events:
                message = event.get("message") if isinstance(event, dict) else None
                if not isinstance(message, str):
                    continue
                for line in message.splitlines():
                    public_line = _sanitize_ssm_output_line(line, instance, command_id, run_identity)
                    print(f"[aws:{phase}:{stream}] {public_line}", flush=True)
                    counts = emitted_lines.setdefault(stream, Counter())
                    counts[public_line] += 1
            previous_token = token
            token = next_token
            tokens[stream] = token
            if token == previous_token:
                break
    return incomplete_streams


def wait_for_ssm_command(
    instance: AwsInstance,
    command_id: str,
    phase: str,
    timeout_seconds: int,
    run_command: Any = _default_command,
    *,
    run_identity: str = "unknown-run",
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
) -> int:
    deadline = monotonic() + timeout_seconds
    log_tokens: dict[str, str] = {}
    emitted_lines: dict[str, Counter[str]] = {}
    final_observation = False
    while True:
        if monotonic() >= deadline:
            if final_observation:
                raise AwsCommandTimeout(phase, timeout_seconds)
            final_observation = True
        result = _run_aws(
            run_command,
            (
                "ssm",
                "get-command-invocation",
                "--region",
                instance.region,
                "--command-id",
                command_id,
                "--instance-id",
                instance.instance_id,
                "--output",
                "json",
            ),
            timeout=60,
        )
        if result.returncode:
            if _aws_error_code(result) == "InvocationDoesNotExist":
                if final_observation:
                    raise AwsCommandTimeout(phase, timeout_seconds)
                sleep(min(AWS_COMMAND_POLL_SECONDS, max(0.0, deadline - monotonic())))
                continue
            raise AwsControllerError(f"SSM observation failed ({_aws_error_code(result) or 'unknown'})")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise AwsControllerError("SSM observation returned invalid JSON") from exc
        status = payload.get("Status")
        if status in {"Pending", "InProgress", "Delayed"} and final_observation:
            raise AwsCommandTimeout(phase, timeout_seconds)
        poll_budget = AWS_LOG_POLL_BUDGET_SECONDS
        if status in {"Pending", "InProgress", "Delayed"}:
            poll_budget = min(poll_budget, max(0.0, deadline - monotonic()))
        incomplete_streams = _poll_ssm_cloudwatch_output(
            phase,
            instance,
            command_id,
            run_identity,
            run_command,
            log_tokens,
            emitted_lines,
            monotonic() + poll_budget,
            monotonic,
        )
        if status in {"Pending", "InProgress", "Delayed"}:
            sleep(min(AWS_COMMAND_POLL_SECONDS, max(0.0, deadline - monotonic())))
            continue
        _emit_ssm_output(
            phase,
            payload,
            instance,
            command_id,
            run_identity,
            run_command,
            emitted_lines,
            incomplete_streams,
        )
        if status == "Success":
            return 0
        if status == "TimedOut":
            raise AwsCommandTimeout(phase, timeout_seconds)
        response_code = payload.get("ResponseCode")
        return response_code if isinstance(response_code, int) and response_code >= 0 else EXIT_TEST_FAILURE


def terminate_aws_instance(instance: AwsInstance, run_command: Any = _default_command) -> None:
    _aws_text(
        run_command,
        ("ec2", "terminate-instances", "--region", instance.region, "--instance-ids", instance.instance_id),
        timeout=60,
    )
    _aws_text(
        run_command,
        ("ec2", "wait", "instance-terminated", "--region", instance.region, "--instance-ids", instance.instance_id),
        timeout=AWS_BACKEND_START_TIMEOUT_SECONDS,
    )
    print(f"AWS_CLEANUP=terminated region={instance.region}", flush=True)


def cleanup_aws_instances(env: Mapping[str, str], run_command: Any = _default_command) -> int:
    regions = load_aws_config(env.get(AWS_CONFIG_ENV, ""))
    run_identity = _aws_run_identity(env)
    for config in regions:
        instance_ids = _aws_text(
            run_command,
            (
                "ec2",
                "describe-instances",
                "--region",
                config.name,
                "--filters",
                f"Name=tag:{AWS_RUN_TAG},Values={run_identity}",
                "Name=instance-state-name,Values=pending,running,stopping,stopped",
                "--query",
                "Reservations[].Instances[].InstanceId",
                "--output",
                "text",
            ),
            timeout=60,
        ).split()
        for instance_id in instance_ids:
            if not _INSTANCE_RE.fullmatch(instance_id):
                raise AwsControllerError("cleanup query returned an invalid instance ID")
            terminate_aws_instance(AwsInstance(config.name, instance_id, config), run_command)
    return 0


def _cleanup_sandbox(sandbox: Any) -> None:
    termination_error = None
    observation_error = None
    try:
        sandbox.terminate()
    except BaseException as exc:
        termination_error = exc
    try:
        sandbox.wait(raise_on_termination=False)
    except BaseException as exc:
        observation_error = exc
    if termination_error is not None and observation_error is not None:
        raise RuntimeError(f"Sandbox termination failed ({termination_error}); terminal-state observation also failed "
                           f"({observation_error})") from termination_error
    if termination_error is not None:
        raise termination_error.with_traceback(termination_error.__traceback__)
    if observation_error is not None:
        raise observation_error.with_traceback(observation_error.__traceback__)


def await_sandbox_start(sandbox: Any, timeout_seconds: float | None = None) -> float:
    """Block until the Sandbox container is running, and return how long that took.

    ``Sandbox.create`` returns before the container exists, so the wait for a free GPU surfaces on the first
    ``exec`` instead. Bounding that wait on its own keeps an unsatisfied reservation from consuming the whole
    job budget, and keeps the Sandbox lifetime budget available for the tests that follow.
    """
    if timeout_seconds is None:
        timeout_seconds = SANDBOX_ACQUIRE_TIMEOUT_SECONDS
    started_at = time.monotonic()
    probe_result: list[BaseException | None] = []

    def probe() -> None:
        try:
            sandbox.exec("true").wait()
            probe_result.append(None)
        except BaseException as exc:  # surfaced on the calling thread below
            probe_result.append(exc)

    probe_thread = threading.Thread(target=probe, daemon=True)
    probe_thread.start()
    probe_thread.join(timeout_seconds)
    if probe_thread.is_alive():
        raise SandboxStartTimeout(timeout_seconds)
    if probe_result and probe_result[0] is not None:
        raise probe_result[0]
    return time.monotonic() - started_at


def _resolve_execution_inputs(env: Mapping[str, str]) -> ControllerInputs | None:
    inputs = resolve_controller_inputs(env)
    if inputs.selection_mode == "none":
        print("No impacted tests; no GPU backend was created.")
        return None
    supported_targets = exclude_unsupported_gds_targets(inputs.targets)
    if not supported_targets:
        print("Skipping the selected GDS tests because this runner has no GPUDirect Storage support")
        return None
    if supported_targets != inputs.targets:
        inputs = replace(inputs, targets=supported_targets)
    return inputs


def run_controller(env: Mapping[str, str], modal_module: Any | None = None) -> int:
    inputs = _resolve_execution_inputs(env)
    if inputs is None:
        return 0

    if modal_module is None:
        modal_module = importlib.import_module("modal")
    preset = MODAL_TORCH_PRESETS[inputs.torch_preset]
    image = _build_sandbox_image(modal_module, preset, inputs)
    app = modal_module.App.lookup(APP_NAME, create_if_missing=True)
    sandbox = None
    sandbox_started_at: float | None = None
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        sandbox = modal_module.Sandbox.create(app=app, **build_sandbox_kwargs(image))
        startup_seconds = await_sandbox_start(sandbox)
        sandbox_started_at = time.monotonic()
        print(f"Sandbox started after {startup_seconds:.0f}s", flush=True)
        for command in build_remote_commands(inputs):
            run_sandbox_command(sandbox, modal_module, command)
    except BaseException as exc:
        primary_error = exc
    finally:
        if sandbox is not None:
            try:
                _cleanup_sandbox(sandbox)
            except BaseException as exc:
                cleanup_error = exc

    if primary_error is not None and cleanup_error is not None:
        raise ControllerCleanupError(primary_error, cleanup_error) from primary_error
    if primary_error is not None:
        return _report_primary_failure(primary_error, sandbox_started_at)
    if cleanup_error is not None:
        raise RuntimeError(f"Sandbox cleanup failed: {cleanup_error}") from cleanup_error
    return 0


def run_modal_infrastructure_smoke(env: Mapping[str, str], modal_module: Any | None = None) -> int:
    torch_preset = env.get("MODAL_TORCH_PRESET") or DEFAULT_MODAL_TORCH_PRESET
    if torch_preset not in MODAL_TORCH_PRESETS:
        raise ValueError("unsupported PyTorch preset for infrastructure smoke")
    if modal_module is None:
        modal_module = importlib.import_module("modal")

    preset = MODAL_TORCH_PRESETS[torch_preset]
    image = modal_module.Image.from_registry(preset["image"])
    app = modal_module.App.lookup(APP_NAME, create_if_missing=True)
    sandbox = None
    primary_error = None
    cleanup_error = None
    try:
        sandbox = modal_module.Sandbox.create(
            app=app,
            **build_sandbox_kwargs(image, timeout_seconds=INFRASTRUCTURE_SMOKE_TIMEOUT_SECONDS * 2),
        )
        startup_seconds = await_sandbox_start(sandbox, INFRASTRUCTURE_SMOKE_TIMEOUT_SECONDS)
        command = RemoteCommand(
            "infrastructure smoke",
            (
                "python",
                "-c",
                "import torch; "
                "count = torch.cuda.device_count(); "  #ignore-cuda
                "assert count == 2, f'expected exactly 2 visible GPUs, observed {count}'; "
                "print('MODAL_INFRA_SMOKE=ready device_count=2 container=true')",
            ),
        )
        run_sandbox_command(sandbox, modal_module, command)
        print(f"MODAL_INFRA_SMOKE=complete startup_seconds={startup_seconds:.0f}", flush=True)
    except BaseException as exc:
        primary_error = exc
    finally:
        if sandbox is not None:
            try:
                _cleanup_sandbox(sandbox)
            except BaseException as exc:
                cleanup_error = exc

    if primary_error is not None and cleanup_error is not None:
        raise ControllerCleanupError(primary_error, cleanup_error) from primary_error
    if primary_error is not None:
        raise primary_error.with_traceback(primary_error.__traceback__)
    if cleanup_error is not None:
        raise RuntimeError(f"Sandbox cleanup failed: {cleanup_error}") from cleanup_error
    return 0


def _report_primary_failure(error: BaseException, sandbox_started_at: float | None) -> int:
    """Map a controller failure to a triage class for nightly regression tooling.

    A Sandbox that never started is a capacity problem and returns the workflow's private
    fallback code without a final triage sentinel. A run that dies at the Sandbox lifetime
    budget is an operational timeout and prints a final sentinel. Anything else is a candidate
    failure and still raises for the full traceback.
    """
    if isinstance(error, SandboxStartTimeout):
        print(f"MODAL_FALLBACK=capacity: no test ran ({error})", flush=True)
        return EXIT_INFRA
    if isinstance(error, RemoteCommandError) and error.label == "run pytest" and error.return_code == EXIT_TIMEOUT:
        print("DS_CI_FAILURE_CLASS=timeout: no test progress for 300 seconds", flush=True)
        return EXIT_TIMEOUT
    if sandbox_started_at is None:
        print("DS_CI_FAILURE_CLASS=test: Modal failed before capacity was established", flush=True)
        raise error.with_traceback(error.__traceback__)
    elapsed = time.monotonic() - sandbox_started_at
    if elapsed >= SANDBOX_TIMEOUT_SECONDS - SANDBOX_TIMEOUT_GRACE_SECONDS:
        print(f"DS_CI_FAILURE_CLASS=timeout: Sandbox lifetime exhausted after {elapsed:.0f}s ({error})", flush=True)
        return EXIT_TIMEOUT
    print("DS_CI_FAILURE_CLASS=test: candidate failed", flush=True)
    raise error.with_traceback(error.__traceback__)


def run_aws_controller(
    env: Mapping[str, str],
    run_command: Any = _default_command,
    *,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
) -> int:
    inputs = _resolve_execution_inputs(env)
    if inputs is None:
        return 0
    regions = load_aws_config(env.get(AWS_CONFIG_ENV, ""))
    run_identity = _aws_run_identity(env)
    instance = None
    primary_error = None
    cleanup_error = None
    outcome = 0
    try:
        instance = acquire_aws_instance(
            regions,
            run_identity,
            run_command,
            sleep=sleep,
            monotonic=monotonic,
        )
        prepare_script, test_script = build_aws_scripts(inputs, run_identity)

        print("AWS_STATE=preparing", flush=True)
        prepare_id = send_ssm_command(
            instance,
            run_identity,
            "prepare",
            prepare_script,
            AWS_PREPARE_TIMEOUT_SECONDS,
            run_command,
        )
        prepare_code = wait_for_ssm_command(
            instance,
            prepare_id,
            "prepare",
            AWS_PREPARE_TIMEOUT_SECONDS,
            run_command,
            run_identity=run_identity,
            sleep=sleep,
            monotonic=monotonic,
        )
        if prepare_code:
            raise AwsControllerError(f"AWS prepare failed with exit code {prepare_code}")

        print("AWS_STATE=test_started", flush=True)
        test_id = send_ssm_command(
            instance,
            run_identity,
            "test",
            test_script,
            AWS_TEST_TIMEOUT_SECONDS,
            run_command,
        )
        test_code = wait_for_ssm_command(
            instance,
            test_id,
            "test",
            AWS_TEST_TIMEOUT_SECONDS,
            run_command,
            run_identity=run_identity,
            sleep=sleep,
            monotonic=monotonic,
        )
        if test_code == EXIT_TIMEOUT:
            print("DS_CI_FAILURE_CLASS=timeout: no test progress for 300 seconds", flush=True)
            outcome = EXIT_TIMEOUT
        elif test_code:
            print(f"DS_CI_FAILURE_CLASS=test: AWS pytest failed with exit code {test_code}", flush=True)
            outcome = EXIT_TEST_FAILURE
    except AwsCapacityExhausted as exc:
        print(f"DS_CI_FAILURE_CLASS=infra: no backend ran ({exc})", flush=True)
        outcome = EXIT_INFRA
    except AwsCommandTimeout as exc:
        if exc.phase == "test":
            print(f"DS_CI_FAILURE_CLASS=timeout: AWS test command exceeded {exc.timeout_seconds} seconds", flush=True)
            outcome = EXIT_TIMEOUT
        else:
            print(f"DS_CI_FAILURE_CLASS=test: AWS backend failed ({_single_line(exc)})", flush=True)
            primary_error = exc
    except BaseException as exc:
        print(f"DS_CI_FAILURE_CLASS=test: AWS backend failed ({_single_line(exc)})", flush=True)
        primary_error = exc
    finally:
        if instance is not None:
            try:
                terminate_aws_instance(instance, run_command)
            except BaseException as exc:
                cleanup_error = exc

    if primary_error is not None and cleanup_error is not None:
        raise ControllerCleanupError(primary_error, cleanup_error) from primary_error
    if primary_error is not None:
        raise primary_error.with_traceback(primary_error.__traceback__)
    if cleanup_error is not None:
        raise RuntimeError(f"AWS cleanup failed: {cleanup_error}") from cleanup_error
    return outcome


def run_aws_infrastructure_smoke(
    env: Mapping[str, str],
    region_name: str,
    run_command: Any = _default_command,
    *,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
) -> int:
    regions = load_aws_config(env.get(AWS_CONFIG_ENV, ""))
    matching = tuple(config for config in regions if config.name == region_name)
    if len(matching) != 1:
        raise ValueError("infrastructure smoke region must be one configured AWS region")
    torch_preset = env.get("MODAL_TORCH_PRESET") or DEFAULT_MODAL_TORCH_PRESET
    run_identity = _aws_run_identity(env)
    instance = None
    primary_error = None
    cleanup_error = None
    try:
        instance = acquire_aws_instance(
            matching,
            run_identity,
            run_command,
            sleep=sleep,
            monotonic=monotonic,
            start_timeout_seconds=INFRASTRUCTURE_SMOKE_TIMEOUT_SECONDS,
        )
        script = build_aws_infrastructure_smoke_script(torch_preset, run_identity, instance.config)
        command_id = send_ssm_command(
            instance,
            run_identity,
            "infra-smoke",
            script,
            INFRASTRUCTURE_SMOKE_TIMEOUT_SECONDS,
            run_command,
        )
        return_code = wait_for_ssm_command(
            instance,
            command_id,
            "infra-smoke",
            INFRASTRUCTURE_SMOKE_TIMEOUT_SECONDS,
            run_command,
            run_identity=run_identity,
            sleep=sleep,
            monotonic=monotonic,
        )
        if return_code:
            raise AwsControllerError(f"AWS infrastructure smoke failed with exit code {return_code}")
        print(f"AWS_INFRA_SMOKE=complete region={region_name}", flush=True)
    except BaseException as exc:
        primary_error = exc
    finally:
        if instance is not None:
            try:
                terminate_aws_instance(instance, run_command)
            except BaseException as exc:
                cleanup_error = exc

    if primary_error is not None and cleanup_error is not None:
        raise ControllerCleanupError(primary_error, cleanup_error) from primary_error
    if primary_error is not None:
        raise primary_error.with_traceback(primary_error.__traceback__)
    if cleanup_error is not None:
        raise RuntimeError(f"AWS cleanup failed: {cleanup_error}") from cleanup_error
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    checkout = subparsers.add_parser("checkout-candidate", help="Fetch an exact public PR SHA as data")
    checkout.add_argument("--head-repository", required=True)
    checkout.add_argument("--head-sha", required=True)
    checkout.add_argument("--base-repository", required=True)
    checkout.add_argument("--base-sha", required=True)
    checkout.add_argument("--destination", type=Path, required=True)

    selection = subparsers.add_parser("validate-selection", help="Validate a mode/list artifact pair")
    selection.add_argument("--mode", required=True)
    selection.add_argument("--path", type=Path, required=True)

    subparsers.add_parser("controller", help="Try the no-secret Modal Sandbox and run the selected tests")
    subparsers.add_parser("modal-infrastructure-smoke", help="Check bounded Modal OCI two-GPU startup and cleanup")
    subparsers.add_parser("aws-controller", help="Run the selected tests on the bounded AWS fallback")
    aws_smoke = subparsers.add_parser("aws-infrastructure-smoke",
                                      help="Check one AWS region's launch, SSM, two-GPU container, and cleanup")
    aws_smoke.add_argument("--region", required=True, choices=AWS_REGION_ORDER)
    subparsers.add_parser("cleanup-aws", help="Terminate run-owned AWS fallback instances")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "checkout-candidate":
        checkout_candidate(
            args.head_repository,
            args.head_sha,
            args.base_repository,
            args.base_sha,
            args.destination,
        )
        return 0
    if args.command == "validate-selection":
        targets = load_test_selection(args.path, args.mode)
        print(f"Validated selection mode={args.mode} count={len(targets)}")
        return 0
    if args.command == "controller":
        return run_controller(os.environ)
    if args.command == "modal-infrastructure-smoke":
        return run_modal_infrastructure_smoke(os.environ)
    if args.command == "aws-controller":
        return run_aws_controller(os.environ)
    if args.command == "aws-infrastructure-smoke":
        return run_aws_infrastructure_smoke(os.environ, args.region)
    return cleanup_aws_instances(os.environ)


if __name__ == "__main__":
    raise SystemExit(main())

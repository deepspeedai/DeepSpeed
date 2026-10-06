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
import base64
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import stat
import subprocess
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
SANDBOX_TIMEOUT_SECONDS = 5400
SANDBOX_ACQUIRE_TIMEOUT_SECONDS = 1800
DIAGNOSTIC_SANDBOX_TIMEOUT_SECONDS = 2400
# Exit codes that nightly triage (see .github/workflows/nightly-bisect.yml) keys on. GitHub only
# reports run success/failure, so the controller also prints a DS_CI_FAILURE_CLASS=<class> sentinel
# line that survives into the job logs even when the job is killed before it can exit.
EXIT_TEST_FAILURE = 1
EXIT_INFRA = 75  # EX_TEMPFAIL: no GPU instance was provisioned, so no test ever ran
EXIT_TIMEOUT = 124
EXIT_DIAGNOSTIC_SETUP = 78
# The Sandbox server-side lifetime can kill a run slightly before the local clock crosses the
# nominal budget, so classify a failure as a timeout just inside the limit.
SANDBOX_TIMEOUT_GRACE_SECONDS = 120
MAX_TEST_LIST_BYTES = 64 * 1024
MAX_TEST_TARGETS = 1024
MAX_DISPLAY_BYTES_PER_COMMAND = 16 * 1024 * 1024
REMOTE_ROOT = "/workspace"
REMOTE_REPOSITORY = f"{REMOTE_ROOT}/deepspeed"
REMOTE_TRANSFORMERS = f"{REMOTE_ROOT}/transformers"
REMOTE_DIAGNOSTIC_ROOT = f"{REMOTE_ROOT}/modal-diagnostic"
REMOTE_DIAGNOSTIC_HELPER = f"{REMOTE_DIAGNOSTIC_ROOT}/modal_diagnostic.py"
REMOTE_DIAGNOSTIC_MANIFEST = f"{REMOTE_DIAGNOSTIC_ROOT}/targets.txt"
REMOTE_DIAGNOSTIC_CONSTRAINTS = f"{REMOTE_DIAGNOSTIC_ROOT}/constraints.txt"
REMOTE_DIAGNOSTIC_COLLECTION = f"{REMOTE_DIAGNOSTIC_ROOT}/collection.json"
REMOTE_DIAGNOSTIC_EVENTS = f"{REMOTE_DIAGNOSTIC_ROOT}/events.jsonl"
REMOTE_DIAGNOSTIC_SUMMARY = f"{REMOTE_DIAGNOSTIC_ROOT}/summary.json"
REMOTE_DIAGNOSTIC_JUNIT = f"{REMOTE_DIAGNOSTIC_ROOT}/junit.xml"
GDS_TEST_TARGET = "tests/unit/v1/nvme/test_gds.py"

_REPOSITORY_COMPONENT = r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}"
_REPOSITORY_RE = re.compile(rf"{_REPOSITORY_COMPONENT}/{_REPOSITORY_COMPONENT}\Z")
_SHA_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
_TEST_FILE_RE = re.compile(r"tests/unit/v1/(?:[^/\x00-\x1f\x7f]+/)*test_[^/\x00-\x1f\x7f]+\.py\Z")
_TEST_NODE_SEGMENT_RE = re.compile(r"[A-Za-z0-9_.\-\[\],=]+\Z")
_SEED_RE = re.compile(r"[0-9]{1,10}\Z")


@dataclass(frozen=True)
class DiagnosticSuite:
    manifest: str
    constraints: str
    expected_count: int
    target_sha: str
    base_sha: str
    transformers_sha: str


DIAGNOSTIC_SUITES = {
    "pr8654-71":
    DiagnosticSuite(
        manifest="modal_diagnostics/pr8654_71_nodes.txt",
        constraints="modal_diagnostics/pr8654_constraints.txt",
        expected_count=71,
        target_sha="34704e111bd480d89aa505461a91ae027b38731f",
        base_sha="a77aeb676507beb9fe0604bc31c2bf075daf7c46",
        transformers_sha="080c288fe607ef54cd4a469608f4d82ac4dd4d4d",
    ),
}


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
    diagnostic_suite: str
    diagnostic_seed: int | None
    diagnostic_timeout_seconds: int | None
    diagnostic_controller_sha: str
    diagnostic_artifact_dir: str


@dataclass(frozen=True)
class RemoteCommand:
    label: str
    argv: tuple[str, ...]
    workdir: str | None = None
    expected_line: str | None = None


class RemoteCommandError(RuntimeError):
    """A Sandbox command returned a nonzero exit code."""

    def __init__(self, command: RemoteCommand, return_code: int):
        super().__init__(f"{command.label} failed with exit code {return_code}")
        self.command = command
        self.return_code = return_code


class DiagnosticSetupError(RuntimeError):
    """The diagnostic environment or exact-node collection was invalid."""

    def __init__(self, command: RemoteCommand, error: BaseException):
        super().__init__(f"diagnostic setup failed during {command.label}: {error}")
        self.command = command
        self.error = error


class ControllerCleanupError(RuntimeError):
    """A primary controller failure accompanied by a cleanup failure."""

    def __init__(self, primary: BaseException, cleanup: BaseException):
        super().__init__(f"controller failed ({primary}); Sandbox cleanup also failed ({cleanup})")
        self.primary = primary
        self.cleanup = cleanup


class SandboxStartTimeout(RuntimeError):
    """The Sandbox never started, so no test ever ran."""

    def __init__(self, timeout_seconds: float):
        super().__init__(f"Sandbox did not start within {timeout_seconds:g}s, so no test ran. This is a capacity "
                         f"problem rather than a test failure: the GPU reservation was never satisfied.")
        self.timeout_seconds = timeout_seconds


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
    file_target, separator, node_target = value.partition("::")
    path = PurePosixPath(file_target)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid pytest target: {value!r}")
    if not _TEST_FILE_RE.fullmatch(file_target):
        raise ValueError(f"pytest target is outside tests/unit/v1 or is not a test file: {value!r}")
    if separator:
        segments = node_target.split("::")
        if not all(segment and _TEST_NODE_SEGMENT_RE.fullmatch(segment) for segment in segments):
            raise ValueError(f"invalid parametrized pytest node: {value!r}")
    return value


def _diagnostic_suite(name: str) -> DiagnosticSuite:
    try:
        return DIAGNOSTIC_SUITES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(DIAGNOSTIC_SUITES))
        raise ValueError(f"unsupported diagnostic suite {name!r}; supported values: {supported}") from exc


def _diagnostic_manifest_targets(suite: DiagnosticSuite) -> tuple[str, ...]:
    path = Path(__file__).resolve().parent / suite.manifest
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    targets = tuple(_validate_target(line) for line in raw_lines if line and not line.startswith("#"))
    if len(targets) != len(set(targets)):
        raise ValueError(f"diagnostic manifest {suite.manifest} contains duplicate targets")
    if len(targets) != suite.expected_count:
        raise ValueError(
            f"diagnostic manifest {suite.manifest} contains {len(targets)} targets; expected {suite.expected_count}")
    return targets


def validate_diagnostic_seed(value: str) -> int:
    if not _SEED_RE.fullmatch(value):
        raise ValueError("diagnostic seed must be an unsigned 32-bit decimal integer")
    seed = int(value)
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("diagnostic seed must be an unsigned 32-bit decimal integer")
    return seed


def validate_diagnostic_request(
    suite_name: str,
    target_sha: str,
    base_sha: str,
    seed: str,
    timeout_minutes: str,
    transformers_ref: str,
) -> tuple[DiagnosticSuite, int, int]:
    suite = _diagnostic_suite(suite_name)
    if validate_sha(target_sha) != suite.target_sha:
        raise ValueError(f"diagnostic target SHA must be {suite.target_sha}")
    if validate_sha(base_sha) != suite.base_sha:
        raise ValueError(f"diagnostic base SHA must be {suite.base_sha}")
    if validate_sha(transformers_ref) != suite.transformers_sha:
        raise ValueError(f"diagnostic Transformers SHA must be {suite.transformers_sha}")
    validated_seed = validate_diagnostic_seed(seed)
    if timeout_minutes != "15":
        raise ValueError("diagnostic pytest budget must be exactly 15 minutes")
    _diagnostic_manifest_targets(suite)
    return suite, validated_seed, int(timeout_minutes) * 60


def prepare_diagnostic_selection(
    suite_name: str,
    target_sha: str,
    base_sha: str,
    seed: str,
    timeout_minutes: str,
    transformers_ref: str,
    output: Path,
) -> None:
    suite, validated_seed, timeout_seconds = validate_diagnostic_request(
        suite_name,
        target_sha,
        base_sha,
        seed,
        timeout_minutes,
        transformers_ref,
    )
    targets = _diagnostic_manifest_targets(suite)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(targets) + "\n", encoding="utf-8")
    print(f"Prepared diagnostic suite={suite_name} count={len(targets)} seed={validated_seed} "
          f"pytest_budget_seconds={timeout_seconds} target_sha={suite.target_sha}")


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

    diagnostic_suite = env.get("DS_DIAGNOSTIC_SUITE", "")
    if diagnostic_suite == "none":
        diagnostic_suite = ""
    diagnostic_seed = None
    diagnostic_timeout_seconds = None
    diagnostic_controller_sha = ""
    diagnostic_artifact_dir = ""
    if diagnostic_suite:
        suite, diagnostic_seed, diagnostic_timeout_seconds = validate_diagnostic_request(
            diagnostic_suite,
            sha,
            base_sha,
            env.get("DS_DIAGNOSTIC_SEED", ""),
            env.get("DS_DIAGNOSTIC_TIMEOUT_MINUTES", ""),
            transformers_ref,
        )
        if selection_mode != "subset" or targets != _diagnostic_manifest_targets(suite):
            raise ValueError("diagnostic selection does not exactly match its trusted manifest")
        diagnostic_controller_sha = validate_sha(env.get("DS_DIAGNOSTIC_CONTROLLER_SHA", ""))
        if diagnostic_controller_sha == sha:
            raise ValueError("diagnostic controller revision must be separate from the code-under-test SHA")
        diagnostic_artifact_dir = env.get("DS_DIAGNOSTIC_ARTIFACT_DIR", "")
        if diagnostic_artifact_dir != "ci/.modal_diagnostic_artifacts":
            raise ValueError("unexpected diagnostic artifact directory")

    return ControllerInputs(
        repository=repository,
        sha=sha,
        selection_mode=selection_mode,
        targets=targets,
        torch_preset=torch_preset,
        transformers_source=transformers_source,
        transformers_ref=transformers_ref,
        base_sha=base_sha,
        diagnostic_suite=diagnostic_suite,
        diagnostic_seed=diagnostic_seed,
        diagnostic_timeout_seconds=diagnostic_timeout_seconds,
        diagnostic_controller_sha=diagnostic_controller_sha,
        diagnostic_artifact_dir=diagnostic_artifact_dir,
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


def build_sandbox_kwargs(image: Any, *, diagnostic: bool = False) -> dict[str, Any]:
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
        "gpu": "l40s:2",
        "timeout": DIAGNOSTIC_SANDBOX_TIMEOUT_SECONDS if diagnostic else SANDBOX_TIMEOUT_SECONDS,
    }


def _remote_git(*args: str) -> tuple[str, ...]:
    return tuple(_git_command(*args))


def _remote_file_command(path: str, content: bytes) -> tuple[str, ...]:
    encoded = base64.b64encode(content).decode("ascii")
    source = ("import base64,pathlib,sys; "
              "path=pathlib.Path(sys.argv[1]); path.parent.mkdir(parents=True,exist_ok=True); "
              "path.write_bytes(base64.b64decode(sys.argv[2]))")
    return ("python", "-c", source, path, encoded)


def _diagnostic_runtime_probe() -> str:
    return ("import importlib.metadata as m,json,platform,torch; "
            "actual={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,"
            "'pytest':m.version('pytest'),'xdist':m.version('pytest-xdist'),'randomly':m.version('pytest-randomly'),"
            "'nccl':m.version('nvidia-nccl-cu12')}; "
            "expected={'python':'3.10.13','torch':'2.10.0+cu128','cuda':'12.8','pytest':'8.3.5',"
            "'xdist':'3.8.0','randomly':'5.0.0','nccl':'2.27.5'}; "
            "print(json.dumps({'actual':actual,'expected':expected},sort_keys=True)); "
            "assert actual==expected,(actual,expected)")


def _diagnostic_bootstrap_commands(inputs: ControllerInputs) -> tuple[RemoteCommand, ...]:
    suite = _diagnostic_suite(inputs.diagnostic_suite)
    constraints = (Path(__file__).resolve().parent / suite.constraints).read_bytes()
    return (
        RemoteCommand("create diagnostic evidence directory", ("mkdir", "-p", REMOTE_DIAGNOSTIC_ROOT)),
        RemoteCommand("write trusted diagnostic constraints",
                      _remote_file_command(REMOTE_DIAGNOSTIC_CONSTRAINTS, constraints)),
    )


def _diagnostic_setup_commands(inputs: ControllerInputs, preset: dict[str, str]) -> tuple[RemoteCommand, ...]:
    suite = _diagnostic_suite(inputs.diagnostic_suite)
    helper = Path(__file__).resolve().with_name("modal_diagnostic.py").read_bytes()
    manifest = ("\n".join(inputs.targets) + "\n").encode("utf-8")
    return (
        RemoteCommand("write trusted diagnostic helper", _remote_file_command(REMOTE_DIAGNOSTIC_HELPER, helper)),
        RemoteCommand("write trusted diagnostic manifest", _remote_file_command(REMOTE_DIAGNOSTIC_MANIFEST, manifest)),
        RemoteCommand("verify diagnostic dependency consistency", ("python", "-m", "pip", "check"), REMOTE_REPOSITORY),
        RemoteCommand(
            "verify diagnostic runtime",
            ("python", "-c", _diagnostic_runtime_probe()),
            REMOTE_REPOSITORY,
        ),
        RemoteCommand("record diagnostic package freeze", ("python", "-m", "pip", "freeze", "--all"),
                      REMOTE_REPOSITORY),
        RemoteCommand(
            "collect diagnostic nodes",
            (
                "python",
                REMOTE_DIAGNOSTIC_HELPER,
                "collect",
                "--manifest",
                REMOTE_DIAGNOSTIC_MANIFEST,
                "--expected-count",
                str(suite.expected_count),
                "--torch-version",
                preset["torch_test_version"],
                "--cuda-version",
                preset["cuda_test_version"],
                "--output",
                REMOTE_DIAGNOSTIC_COLLECTION,
            ),
            REMOTE_REPOSITORY,
        ),
    )


def _diagnostic_pytest_command(inputs: ControllerInputs, preset: dict[str, str]) -> RemoteCommand:
    assert inputs.diagnostic_seed is not None
    assert inputs.diagnostic_timeout_seconds is not None
    return RemoteCommand(
        "run diagnostic pytest",
        (
            "timeout",
            "--signal=TERM",
            "--kill-after=30s",
            f"{inputs.diagnostic_timeout_seconds}s",
            "env",
            f"PYTHONPATH={REMOTE_DIAGNOSTIC_ROOT}",
            "PYTHONFAULTHANDLER=1",
            f"PYTHONHASHSEED={inputs.diagnostic_seed}",
            f"DS_DIAGNOSTIC_SEED={inputs.diagnostic_seed}",
            f"DS_DIAGNOSTIC_EVENTS_FILE={REMOTE_DIAGNOSTIC_EVENTS}",
            "pytest",
            "-p",
            "modal_diagnostic",
            "-n",
            "4",
            "--verbose",
            "--tb=long",
            "--capture=tee-sys",
            "-ra",
            "--durations=0",
            f"--randomly-seed={inputs.diagnostic_seed}",
            f"--junitxml={REMOTE_DIAGNOSTIC_JUNIT}",
            "--ignore=tests/unit/v1/nvme/test_gds.py",
            f"--torch_ver={preset['torch_test_version']}",
            f"--cuda_ver={preset['cuda_test_version']}",
            "--",
            *inputs.targets,
        ),
        REMOTE_REPOSITORY,
    )


def _diagnostic_recovery_commands() -> tuple[RemoteCommand, ...]:
    return (
        RemoteCommand(
            "summarize diagnostic events",
            (
                "python",
                REMOTE_DIAGNOSTIC_HELPER,
                "summarize",
                "--events",
                REMOTE_DIAGNOSTIC_EVENTS,
                "--output",
                REMOTE_DIAGNOSTIC_SUMMARY,
            ),
            REMOTE_REPOSITORY,
        ),
        RemoteCommand(
            "emit diagnostic artifacts",
            (
                "python",
                REMOTE_DIAGNOSTIC_HELPER,
                "emit",
                REMOTE_DIAGNOSTIC_COLLECTION,
                REMOTE_DIAGNOSTIC_EVENTS,
                REMOTE_DIAGNOSTIC_SUMMARY,
                REMOTE_DIAGNOSTIC_JUNIT,
            ),
            REMOTE_REPOSITORY,
        ),
    )


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
    ]
    constraint_args: tuple[str, ...] = ()
    if inputs.diagnostic_suite:
        commands.extend(_diagnostic_bootstrap_commands(inputs))
        constraint_args = ("-c", REMOTE_DIAGNOSTIC_CONSTRAINTS)
    commands.extend([
        RemoteCommand(
            "install runtime requirements",
            ("python", "-m", "pip", "install", *constraint_args, "-r", "requirements/requirements.txt"),
            REMOTE_REPOSITORY,
        ),
        RemoteCommand(
            "install development requirements",
            ("python", "-m", "pip", "install", *constraint_args, "-r", "requirements/requirements-dev.txt"),
            REMOTE_REPOSITORY,
        ),
        RemoteCommand(
            "install DeepCompile requirements",
            ("python", "-m", "pip", "install", *constraint_args, "-r", "requirements/requirements-deepcompile.txt"),
            REMOTE_REPOSITORY,
        ),
        # Torch itself is pinned in the image (see _build_sandbox_image), after the requirements
        # layers, so a transitive dependency cannot displace the intended CUDA build.
    ])
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
                expected_line=inputs.transformers_ref if inputs.diagnostic_suite else None,
            ),
            RemoteCommand(
                "install Transformers",
                ("python", "-m", "pip", "install", *constraint_args, "."),
                REMOTE_TRANSFORMERS,
            ),
        ])
    commands.extend([
        RemoteCommand("install candidate DeepSpeed", ("python", "-m", "pip", "install", *constraint_args, "."),
                      REMOTE_REPOSITORY),
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
    ])
    if inputs.diagnostic_suite:
        commands.extend(_diagnostic_setup_commands(inputs, preset))
        commands.append(_diagnostic_pytest_command(inputs, preset))
    else:
        commands.append(
            RemoteCommand(
                "run pytest",
                (
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
                ),
                REMOTE_REPOSITORY,
            ))
    return tuple(commands)


def _single_line(value: object) -> str:
    return "".join(char if char.isprintable() else f"\\x{ord(char):02x}" for char in str(value))


def _command_artifact_path(artifact_dir: Path, command: RemoteCommand) -> Path:
    safe_label = re.sub(r"[^a-z0-9]+", "-", command.label.lower()).strip("-")
    return artifact_dir / f"sandbox-{safe_label}.log"


def run_sandbox_command(
    sandbox: Any,
    modal_module: Any,
    command: RemoteCommand,
    artifact_dir: Path | None = None,
) -> str:
    process = sandbox.exec(
        *command.argv,
        stderr=modal_module.stream_type.StreamType.STDOUT,
        workdir=command.workdir,
    )
    displayed = 0
    last_line = ""
    truncated = False
    artifact_stream = None
    try:
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_stream = _command_artifact_path(artifact_dir, command).open("a",
                                                                                 encoding="utf-8",
                                                                                 errors="replace")
        for raw_line in process.stdout:
            if artifact_stream is not None:
                artifact_stream.write(raw_line)
                artifact_stream.flush()
            line = _single_line(raw_line.rstrip("\r\n"))
            last_line = line
            rendered = f"[sandbox:{command.label}] {line}"
            encoded_size = len(rendered.encode("utf-8", errors="replace")) + 1
            if displayed + encoded_size <= MAX_DISPLAY_BYTES_PER_COMMAND:
                print(rendered, flush=True)
                displayed += encoded_size
            else:
                truncated = True
    finally:
        if artifact_stream is not None:
            artifact_stream.close()
    if truncated:
        print(f"[sandbox:{command.label}] output truncated after {MAX_DISPLAY_BYTES_PER_COMMAND} bytes")
    return_code = process.wait()
    if return_code:
        raise RemoteCommandError(command, return_code)
    if command.expected_line is not None:
        actual = last_line.strip()
        if actual != command.expected_line:
            raise RuntimeError(
                f"{command.label} returned {_single_line(actual)!r}, expected {command.expected_line!r}")
    return last_line


def _run_remote_plan(
    sandbox: Any,
    modal_module: Any,
    inputs: ControllerInputs,
    artifact_dir: Path | None,
) -> None:
    for command in build_remote_commands(inputs):
        if command.label != "run diagnostic pytest":
            try:
                run_sandbox_command(sandbox, modal_module, command, artifact_dir)
            except BaseException as exc:
                if inputs.diagnostic_suite:
                    raise DiagnosticSetupError(command, exc) from exc
                raise
            continue

        test_error = None
        recovery_error = None
        try:
            run_sandbox_command(sandbox, modal_module, command, artifact_dir)
        except BaseException as exc:
            test_error = exc
        for recovery in _diagnostic_recovery_commands():
            try:
                run_sandbox_command(sandbox, modal_module, recovery, artifact_dir)
            except BaseException as exc:
                if recovery_error is None:
                    recovery_error = exc
        if test_error is not None and recovery_error is not None:
            raise RuntimeError(f"diagnostic pytest failed ({test_error}); evidence recovery failed ({recovery_error})"
                               ) from test_error
        if test_error is not None:
            raise test_error.with_traceback(test_error.__traceback__)
        if recovery_error is not None:
            raise RuntimeError(f"diagnostic evidence recovery failed: {recovery_error}") from recovery_error


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


def run_controller(env: Mapping[str, str], modal_module: Any | None = None) -> int:
    inputs = resolve_controller_inputs(env)
    if inputs.selection_mode == "none":
        print("No impacted tests; Modal Sandbox was not created.")
        return 0
    supported_targets = exclude_unsupported_gds_targets(inputs.targets)
    if not supported_targets:
        print("Skipping the selected GDS tests because this runner has no GPUDirect Storage support")
        return 0
    if supported_targets != inputs.targets:
        inputs = replace(inputs, targets=supported_targets)

    if modal_module is None:
        modal_module = importlib.import_module("modal")
    preset = MODAL_TORCH_PRESETS[inputs.torch_preset]
    image = _build_sandbox_image(modal_module, preset, inputs)
    app = modal_module.App.lookup(APP_NAME, create_if_missing=True)
    artifact_dir = Path(inputs.diagnostic_artifact_dir) if inputs.diagnostic_artifact_dir else None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        request = {
            "controller_sha": inputs.diagnostic_controller_sha,
            "target_sha": inputs.sha,
            "base_sha": inputs.base_sha,
            "suite": inputs.diagnostic_suite,
            "target_count": len(inputs.targets),
            "seed": inputs.diagnostic_seed,
            "pytest_timeout_seconds": inputs.diagnostic_timeout_seconds,
            "sandbox_timeout_seconds": DIAGNOSTIC_SANDBOX_TIMEOUT_SECONDS,
            "transformers_sha": inputs.transformers_ref,
            "modal_version": importlib.metadata.version("modal"),
        }
        (artifact_dir / "request.json").write_text(json.dumps(request, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
        print("DS_DIAGNOSTIC_REQUEST " + json.dumps(request, sort_keys=True), flush=True)
    sandbox = None
    sandbox_started_at: float | None = None
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        sandbox = modal_module.Sandbox.create(app=app,
                                              **build_sandbox_kwargs(image, diagnostic=bool(inputs.diagnostic_suite)))
        startup_seconds = await_sandbox_start(sandbox)
        sandbox_started_at = time.monotonic()
        print(f"Sandbox started after {startup_seconds:.0f}s", flush=True)
        _run_remote_plan(sandbox, modal_module, inputs, artifact_dir)
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
        sandbox_timeout = DIAGNOSTIC_SANDBOX_TIMEOUT_SECONDS if inputs.diagnostic_suite else SANDBOX_TIMEOUT_SECONDS
        return _report_primary_failure(primary_error, sandbox_started_at, sandbox_timeout)
    if cleanup_error is not None:
        raise RuntimeError(f"Sandbox cleanup failed: {cleanup_error}") from cleanup_error
    return 0


def _report_primary_failure(
    error: BaseException,
    sandbox_started_at: float | None,
    sandbox_timeout_seconds: float = SANDBOX_TIMEOUT_SECONDS,
) -> int:
    """Map a controller failure to a triage class for nightly regression tooling.

    A Sandbox that never started is a capacity problem and a run that died at the Sandbox
    lifetime budget is an operational timeout; both print a DS_CI_FAILURE_CLASS sentinel and
    return a dedicated exit code instead of raising, so nightly triage can route them away from
    git-bisect. Anything else is a candidate failure and still raises for the full traceback.
    """
    if sandbox_started_at is None:
        print(f"DS_CI_FAILURE_CLASS=infra: no test ran ({error})", flush=True)
        return EXIT_INFRA
    if isinstance(error, DiagnosticSetupError):
        print(f"DS_CI_FAILURE_CLASS=diagnostic_setup: no product test ran ({error})", flush=True)
        return EXIT_DIAGNOSTIC_SETUP
    if isinstance(error,
                  RemoteCommandError) and error.command.label == "run diagnostic pytest" and error.return_code == 124:
        print(f"DS_CI_FAILURE_CLASS=timeout: diagnostic pytest budget exhausted ({error})", flush=True)
        return EXIT_TIMEOUT
    elapsed = time.monotonic() - sandbox_started_at
    if elapsed >= sandbox_timeout_seconds - SANDBOX_TIMEOUT_GRACE_SECONDS:
        print(f"DS_CI_FAILURE_CLASS=timeout: Sandbox lifetime exhausted after {elapsed:.0f}s ({error})", flush=True)
        return EXIT_TIMEOUT
    print("DS_CI_FAILURE_CLASS=test: candidate failed", flush=True)
    raise error.with_traceback(error.__traceback__)


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

    diagnostic = subparsers.add_parser("prepare-diagnostic",
                                       help="Validate and materialize a trusted diagnostic suite")
    diagnostic.add_argument("--suite", required=True)
    diagnostic.add_argument("--target-sha", required=True)
    diagnostic.add_argument("--base-sha", required=True)
    diagnostic.add_argument("--seed", required=True)
    diagnostic.add_argument("--timeout-minutes", required=True)
    diagnostic.add_argument("--transformers-ref", required=True)
    diagnostic.add_argument("--output", type=Path, required=True)

    subparsers.add_parser("controller", help="Create the no-secret Modal Sandbox and run the selected tests")
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
    if args.command == "prepare-diagnostic":
        prepare_diagnostic_selection(
            args.suite,
            args.target_sha,
            args.base_sha,
            args.seed,
            args.timeout_minutes,
            args.transformers_ref,
            args.output,
        )
        return 0
    return run_controller(os.environ)


if __name__ == "__main__":
    raise SystemExit(main())

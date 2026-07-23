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
import importlib
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
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
SANDBOX_TIMEOUT_SECONDS = 3600
MAX_TEST_LIST_BYTES = 64 * 1024
MAX_TEST_TARGETS = 1024
MAX_DISPLAY_BYTES_PER_COMMAND = 16 * 1024 * 1024
REMOTE_ROOT = "/workspace"
REMOTE_REPOSITORY = f"{REMOTE_ROOT}/deepspeed"
REMOTE_TRANSFORMERS = f"{REMOTE_ROOT}/transformers"

_REPOSITORY_COMPONENT = r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}"
_REPOSITORY_RE = re.compile(rf"{_REPOSITORY_COMPONENT}/{_REPOSITORY_COMPONENT}\Z")
_SHA_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
_TEST_FILE_RE = re.compile(r"tests/unit/v1/(?:[^/\x00-\x1f\x7f]+/)*test_[^/\x00-\x1f\x7f]+\.py\Z")


@dataclass(frozen=True)
class ControllerInputs:
    repository: str
    sha: str
    selection_mode: str
    targets: tuple[str, ...]
    torch_preset: str
    transformers_source: str
    transformers_ref: str


@dataclass(frozen=True)
class RemoteCommand:
    label: str
    argv: tuple[str, ...]
    workdir: str | None = None
    expected_line: str | None = None


class ControllerCleanupError(RuntimeError):
    """A primary controller failure accompanied by a cleanup failure."""

    def __init__(self, primary: BaseException, cleanup: BaseException):
        super().__init__(f"controller failed ({primary}); Sandbox cleanup also failed ({cleanup})")
        self.primary = primary
        self.cleanup = cleanup


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
            "--depth=1",
            head_url,
            f"{head_sha}:refs/devds/head",
        )
        _run_local(fetch_head)
        fetch_base = _git_command(
            "-C",
            str(destination),
            "fetch",
            "--no-tags",
            "--no-recurse-submodules",
            "--depth=1",
            base_url,
            f"{base_sha}:refs/devds/base",
        )
        _run_local(fetch_base)
        resolved_head = _run_local(
            _git_command("-C", str(destination), "rev-parse", "--verify", "refs/devds/head^{commit}")).strip()
        resolved_base = _run_local(
            _git_command("-C", str(destination), "rev-parse", "--verify", "refs/devds/base^{commit}")).strip()
        if resolved_head != head_sha or resolved_base != base_sha:
            raise ValueError("fetched commit did not match requested event SHA")
        _run_local(_git_command("-C", str(destination), "checkout", "--detach", "refs/devds/head"))
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
    elif transformers_ref:
        raise ValueError("MODAL_TRANSFORMERS_REF is only valid when source is 'git'")

    return ControllerInputs(
        repository=repository,
        sha=sha,
        selection_mode=selection_mode,
        targets=targets,
        torch_preset=torch_preset,
        transformers_source=transformers_source,
        transformers_ref=transformers_ref,
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


def build_sandbox_kwargs(image: Any) -> dict[str, Any]:
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
        "timeout": SANDBOX_TIMEOUT_SECONDS,
    }


def _remote_git(*args: str) -> tuple[str, ...]:
    return tuple(_git_command(*args))


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
                f"{inputs.sha}:refs/devds/candidate",
            ),
        ),
        RemoteCommand(
            "checkout candidate SHA",
            _remote_git("-C", REMOTE_REPOSITORY, "checkout", "--detach", "refs/devds/candidate"),
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
        RemoteCommand(
            "reinstall Torch packages",
            (
                "python",
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-cache-dir",
                "--index-url",
                PYTORCH_CUDA_128_INDEX_URL,
                preset["torch_package"],
                preset["torchvision_package"],
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
            "run pytest",
            (
                "pytest",
                "-n",
                "4",
                "--verbose",
                f"--torch_ver={preset['torch_test_version']}",
                f"--cuda_ver={preset['cuda_test_version']}",
                "--",
                *inputs.targets,
            ),
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
    captured: list[str] = []
    truncated = False
    for raw_line in process.stdout:
        line = _single_line(raw_line.rstrip("\r\n"))
        encoded_size = len(line.encode("utf-8", errors="replace")) + 1
        if displayed + encoded_size <= MAX_DISPLAY_BYTES_PER_COMMAND:
            print(f"[sandbox:{command.label}] {line}")
            displayed += encoded_size
            captured.append(line)
        else:
            truncated = True
    if truncated:
        print(f"[sandbox:{command.label}] output truncated after {MAX_DISPLAY_BYTES_PER_COMMAND} bytes")
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{command.label} failed with exit code {return_code}")
    if command.expected_line is not None:
        actual = captured[-1].strip() if captured else ""
        if actual != command.expected_line:
            raise RuntimeError(
                f"{command.label} returned {_single_line(actual)!r}, expected {command.expected_line!r}")
    return "\n".join(captured)


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


def run_controller(env: Mapping[str, str], modal_module: Any | None = None) -> int:
    inputs = resolve_controller_inputs(env)
    if inputs.selection_mode == "none":
        print("No impacted tests; Modal Sandbox was not created.")
        return 0

    if modal_module is None:
        modal_module = importlib.import_module("modal")
    preset = MODAL_TORCH_PRESETS[inputs.torch_preset]
    image = modal_module.Image.from_registry(preset["image"], add_python="3.10")
    app = modal_module.App.lookup(APP_NAME, create_if_missing=True)
    sandbox = None
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        sandbox = modal_module.Sandbox.create(app=app, **build_sandbox_kwargs(image))
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
        raise primary_error.with_traceback(primary_error.__traceback__)
    if cleanup_error is not None:
        raise RuntimeError(f"Sandbox cleanup failed: {cleanup_error}") from cleanup_error
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
    return run_controller(os.environ)


if __name__ == "__main__":
    raise SystemExit(main())

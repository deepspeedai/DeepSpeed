# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Pure-stdlib security tests for ci/torch_latest.py."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import signal
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch_latest  # noqa: E402
import test_tests_fetcher  # noqa: E402
from modal_diagnostics import runner as modal_runner  # noqa: E402


def _expect_error(function, *args, exception=ValueError, **kwargs):
    try:
        function(*args, **kwargs)
    except exception as exc:
        return exc
    raise AssertionError(f"{function.__name__} unexpectedly succeeded")


def _selection_file(content: str) -> tuple[Path, Path]:
    root = Path(tempfile.mkdtemp(prefix="ds-modal-selection-")).resolve()
    path = root / "test_list.txt"
    path.write_text(content, encoding="utf-8")
    return root, path


def _valid_env(path: Path, **overrides: str) -> dict[str, str]:
    values = {
        "GITHUB_EVENT_NAME": "pull_request_target",
        "DS_CI_REPOSITORY": "example/DeepSpeed",
        "DS_CI_SHA": "a" * 40,
        "DS_TEST_SELECTION_MODE": "all",
        "DS_TEST_LIST_FILE": str(path),
        "MODAL_TORCH_PRESET": "2.10.0-cuda12.8",
        "MODAL_TRANSFORMERS_SOURCE": "git",
        "MODAL_TRANSFORMERS_REF": "main",
        "DS_DIAGNOSTICS_DIR": str(path.parent / "diagnostics"),
    }
    values.update(overrides)
    return values


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class LocalHistory:

    def __init__(self, *, escaping_symlink: bool = False):
        self.root = Path(tempfile.mkdtemp(prefix="ds-modal-git-")).resolve()
        _git(self.root, "init", "-q", "-b", "master")
        _git(self.root, "config", "user.email", "ci@example.com")
        _git(self.root, "config", "user.name", "ci")
        (self.root / "README.md").write_text("base\n", encoding="utf-8")
        _git(self.root, "add", "README.md")
        _git(self.root, "commit", "-q", "-m", "base")
        self.base = _git(self.root, "rev-parse", "HEAD")
        if escaping_symlink:
            (self.root / "unsafe-link").symlink_to("../../outside")
            _git(self.root, "add", "unsafe-link")
        else:
            (self.root / "README.md").write_text("head\n", encoding="utf-8")
            _git(self.root, "add", "README.md")
        _git(self.root, "commit", "-q", "-m", "head")
        self.head = _git(self.root, "rev-parse", "HEAD")

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


class FakeProcess:

    def __init__(self, lines=None, return_code=0):
        self.stdout = list(lines or [])
        self.return_code = return_code
        self.waited = False

    def wait(self):
        self.waited = True
        return self.return_code


class FakeSandbox:

    def __init__(
        self,
        candidate_sha: str,
        fail_label: str | None = None,
        cleanup_failure: bool = False,
        wait_failure: bool = False,
        never_starts: bool = False,
        transfer_failure: bool = False,
    ):
        self.candidate_sha = candidate_sha
        self.fail_label = fail_label
        self.cleanup_failure = cleanup_failure
        self.wait_failure = wait_failure
        self.never_starts = never_starts
        self.transfer_failure = transfer_failure
        self.exec_calls = []
        self.processes = []
        self.terminated = False
        self.wait_calls = []

    def exec(self, *args, **kwargs):
        self.exec_calls.append((args, kwargs))
        if self.never_starts:
            # A container that never gets a GPU never returns from its first exec.
            threading.Event().wait()
        if args[0] == "cat":
            lines = [b"diagnostics"]
        else:
            lines = [self.candidate_sha + "\n"] if "rev-parse" in args and "HEAD^{commit}" in args else ["ok\n"]
        label_failure = (self.fail_label and self.fail_label in " ".join(args)) or (self.transfer_failure
                                                                                    and args[0] == "cat")
        process = FakeProcess(lines, return_code=9 if label_failure else 0)
        self.processes.append(process)
        return process

    def terminate(self):
        self.terminated = True
        if self.cleanup_failure:
            raise RuntimeError("terminate failed")

    def wait(self, raise_on_termination=True):
        self.wait_calls.append(raise_on_termination)
        if self.wait_failure:
            raise RuntimeError("wait failed")


def _fake_modal(
    candidate_sha: str,
    fail_label: str | None = None,
    cleanup_failure: bool = False,
    wait_failure: bool = False,
    create_failure: bool = False,
    never_starts: bool = False,
    transfer_failure: bool = False,
):
    state = SimpleNamespace(image_calls=[], app_calls=[], create_calls=[])
    sandbox = FakeSandbox(candidate_sha, fail_label, cleanup_failure, wait_failure, never_starts, transfer_failure)

    class FakeImage:

        def __init__(self):
            self.layers = []

        def run_commands(self, *commands):
            self.layers.append(("run_commands", commands))
            return self

        def pip_install(self, *packages, index_url=None):
            self.layers.append(("pip_install", packages, index_url))
            return self

    image = FakeImage()

    class Image:

        @staticmethod
        def from_registry(registry, add_python=None):
            state.image_calls.append((registry, add_python))
            return image

    class App:

        @staticmethod
        def lookup(name, create_if_missing=False):
            state.app_calls.append((name, create_if_missing))
            return ("app", name)

    class Sandbox:

        @staticmethod
        def create(*args, **kwargs):
            state.create_calls.append((args, kwargs))
            if create_failure:
                raise RuntimeError("create failed")
            return sandbox

    stream_type = SimpleNamespace(StreamType=SimpleNamespace(STDOUT=object()))
    return SimpleNamespace(Image=Image, App=App, Sandbox=Sandbox, stream_type=stream_type), state, sandbox


def test_module_import_is_modal_free():
    assert "modal" not in torch_latest.__dict__, "Modal was imported at module load time"


def test_repository_and_sha_validation():
    assert torch_latest.validate_repository("owner/repo.name-1") == "owner/repo.name-1"
    assert torch_latest.validate_sha("A" * 40) == "a" * 40
    for value in ("owner", "https://github.com/owner/repo", "../repo", "-owner/repo", "owner/repo/sub"):
        _expect_error(torch_latest.validate_repository, value)
    for value in ("a" * 39, "g" * 40, "-a" * 20, "a" * 40 + "\n"):
        _expect_error(torch_latest.validate_sha, value)


def test_transformers_ref_validation():
    assert torch_latest.validate_transformers_ref("main") == "main"
    assert torch_latest.validate_transformers_ref("A" * 40) == "a" * 40
    for value in ("-main", "https://example.test/repo", "bad ref", "bad\nref", "branch..name"):
        _expect_error(torch_latest.validate_transformers_ref, value)


def test_selection_modes_and_path_validation():
    cases = [
        ("all", "tests/unit/v1\n", ("tests/unit/v1", )),
        ("subset", "tests/unit/v1/test_one.py\ntests/unit/v1/sub/test_two.py\n", ("tests/unit/v1/test_one.py",
                                                                                  "tests/unit/v1/sub/test_two.py")),
        ("subset", "tests/unit/v1/test_one.py::TestOne::test_value[param]\n",
         ("tests/unit/v1/test_one.py::TestOne::test_value[param]", )),
        ("none", "", ()),
    ]
    for mode, content, expected in cases:
        root, path = _selection_file(content)
        try:
            assert torch_latest.load_test_selection(path, mode) == expected
        finally:
            shutil.rmtree(root, ignore_errors=True)

    invalid = [
        ("all", ""),
        ("all", "tests/unit/v1/test_one.py\n"),
        ("subset", ""),
        ("subset", "tests/unit/v1/../test_bad.py\n"),
        ("subset", "/tests/unit/v1/test_bad.py\n"),
        ("subset", "tests\\unit\\v1\\test_bad.py\n"),
        ("subset", "--collect-only\n"),
        ("subset", "tests/unit/v1/helper.py\n"),
        ("subset", "tests/unit/v1/test_one.py::\n"),
        ("subset", "tests/unit/v1/test_one.py::TestOne/test_value\n"),
        ("none", "tests/unit/v1\n"),
        ("bogus", ""),
    ]
    for mode, content in invalid:
        root, path = _selection_file(content)
        try:
            _expect_error(torch_latest.load_test_selection, path, mode)
        finally:
            shutil.rmtree(root, ignore_errors=True)


def test_selection_rejects_duplicate_symlink_size_count_and_controls():
    invalid_contents = [
        "tests/unit/v1/test_one.py\ntests/unit/v1/test_one.py\n",
        "tests/unit/v1/test_\x01bad.py\n",
        "\n",
    ]
    for content in invalid_contents:
        root, path = _selection_file(content)
        try:
            _expect_error(torch_latest.load_test_selection, path, "subset")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    root, path = _selection_file("tests/unit/v1/test_one.py\n")
    link = root / "link"
    link.symlink_to(path)
    try:
        _expect_error(torch_latest.load_test_selection, link, "subset")
        path.write_text("x" * (torch_latest.MAX_TEST_LIST_BYTES + 1), encoding="utf-8")
        _expect_error(torch_latest.load_test_selection, path, "subset")
        path.write_text("\n".join(f"tests/unit/v1/test_{index}.py" for index in range(1025)), encoding="utf-8")
        _expect_error(torch_latest.load_test_selection, path, "subset")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_exact_checkout_uses_requested_commits_and_detached_head():
    history = LocalHistory()
    destination = history.root.parent / f"{history.root.name}-checkout"
    try:
        torch_latest._checkout_exact(str(history.root), history.head, str(history.root), history.base, destination)
        assert _git(destination, "rev-parse", "HEAD") == history.head
        detached = subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"],
            cwd=destination,
            check=False,
            capture_output=True,
            text=True,
        )
        assert detached.returncode == 1
        _expect_error(
            torch_latest._checkout_exact,
            str(history.root),
            history.head,
            str(history.root),
            history.base,
            destination,
        )
    finally:
        shutil.rmtree(destination, ignore_errors=True)
        history.cleanup()


def test_collector_checkout_preserves_merge_base_for_subset_selection():
    repo = test_tests_fetcher.TmpRepo()
    destination = repo.root.parent / f"{repo.root.name}-collector"
    try:
        repo.write("deepspeed/leaf.py", "VALUE = 11\n")
        repo.commit("touch leaf")
        head = repo._git("rev-parse", "HEAD").strip()
        base = repo._git("rev-parse", "master").strip()
        torch_latest._checkout_exact(str(repo.root), head, str(repo.root), base, destination)
        selection = test_tests_fetcher.TestSelector(destination, test_tests_fetcher.CONFIG).select("refs/ci/base")
        assert selection.mode == "subset", selection.reason
        assert {path.relative_to(destination).as_posix() for path in selection.tests} == {"tests/unit/v1/test_leaf.py"}
    finally:
        shutil.rmtree(destination, ignore_errors=True)
        repo.cleanup()


def test_exact_checkout_rejects_escaping_symlink_and_cleans_destination():
    history = LocalHistory(escaping_symlink=True)
    destination = history.root.parent / f"{history.root.name}-checkout"
    try:
        _expect_error(
            torch_latest._checkout_exact,
            str(history.root),
            history.head,
            str(history.root),
            history.base,
            destination,
        )
        assert not destination.exists()
    finally:
        shutil.rmtree(destination, ignore_errors=True)
        history.cleanup()


def test_candidate_checkout_builds_public_urls_from_validated_metadata():
    captured = []
    original = torch_latest._checkout_exact
    try:
        torch_latest._checkout_exact = lambda *args: captured.append(args)
        destination = Path("fixed-candidate")
        torch_latest.checkout_candidate(
            "fork-owner/DeepSpeed",
            "A" * 40,
            "deepspeedai/DeepSpeed",
            "B" * 40,
            destination,
        )
    finally:
        torch_latest._checkout_exact = original
    assert captured == [(
        "https://github.com/fork-owner/DeepSpeed.git",
        "a" * 40,
        "https://github.com/deepspeedai/DeepSpeed.git",
        "b" * 40,
        destination,
    )]


def test_git_environment_is_positive_allowlist():
    env = torch_latest.build_git_env({
        "PATH": "/safe/path",
        "MODAL_TOKEN_SECRET": "secret",
        "GITHUB_TOKEN": "token",
        "HF_TOKEN": "hf",
        "HOME": "/untrusted",
    })
    assert env["PATH"] == "/safe/path"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    for key in ("MODAL_TOKEN_SECRET", "GITHUB_TOKEN", "HF_TOKEN", "HOME"):
        assert key not in env


def test_controller_inputs_and_push_manual_fallback():
    root, path = _selection_file("tests/unit/v1\n")
    try:
        explicit = torch_latest.resolve_controller_inputs(_valid_env(path))
        assert explicit.repository == "example/DeepSpeed"
        assert explicit.sha == "a" * 40

        fallback = _valid_env(path)
        fallback.update({
            "GITHUB_EVENT_NAME": "push",
            "DS_CI_REPOSITORY": "",
            "DS_CI_SHA": "",
            "GITHUB_REPOSITORY": "deepspeedai/DeepSpeed",
            "GITHUB_SHA": "B" * 40,
        })
        resolved = torch_latest.resolve_controller_inputs(fallback)
        assert resolved.repository == "deepspeedai/DeepSpeed"
        assert resolved.sha == "b" * 40

        requirements = torch_latest.resolve_controller_inputs(
            _valid_env(
                path,
                GITHUB_EVENT_NAME="workflow_dispatch",
                MODAL_TRANSFORMERS_SOURCE="requirements",
                MODAL_TRANSFORMERS_REF="main",
            ))
        assert requirements.transformers_source == "requirements"
        assert requirements.transformers_ref == ""
        assert not any("Transformers" in command.label for command in torch_latest.build_remote_commands(requirements))

        missing_pr = dict(fallback)
        missing_pr["GITHUB_EVENT_NAME"] = "pull_request_target"
        _expect_error(torch_latest.resolve_controller_inputs, missing_pr)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_remote_plan_is_structural_and_preserves_order_and_scope():
    root, path = _selection_file("tests/unit/v1/test_one.py\n")
    try:
        inputs = torch_latest.resolve_controller_inputs(
            _valid_env(path, DS_TEST_SELECTION_MODE="subset", DS_CI_SHA="C" * 40))
        commands = torch_latest.build_remote_commands(inputs)
        assert all(isinstance(command.argv, tuple) for command in commands)
        labels = [command.label for command in commands]
        assert labels.index("install runtime requirements") < labels.index("install candidate DeepSpeed")
        pytest_command = next(command for command in commands if command.label == "run pytest")
        separator = pytest_command.argv.index("--")
        assert pytest_command.argv[separator + 1:] == ("tests/unit/v1/test_one.py", )
        assert pytest_command.argv[:3] == ("python3", "ci/modal_diagnostics/runner.py", "run-pytest")
        assert pytest_command.argv[pytest_command.argv.index("--workers") + 1] == "4"
        assert pytest_command.argv[pytest_command.argv.index("--stall-seconds") + 1] == "300"
        fetch = next(command for command in commands if command.label == "fetch candidate SHA")
        assert "https://github.com/example/DeepSpeed.git" in fetch.argv
        assert f"{'c' * 40}:refs/ci/candidate" in fetch.argv
        assert not any("sh" == argument or "bash" == argument for command in commands for argument in command.argv)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_sandbox_kwargs_are_fixed_and_secret_free():
    kwargs = torch_latest.build_sandbox_kwargs("image")
    assert kwargs["cloud"] == "oci"
    assert kwargs["gpu"] == "l40s:2"
    assert kwargs["timeout"] == 7200
    assert torch_latest.SANDBOX_ACQUIRE_TIMEOUT_SECONDS == 1800
    assert kwargs["secrets"] == []
    assert kwargs["network_file_systems"] == {}
    assert kwargs["volumes"] == {}
    assert kwargs["encrypted_ports"] == []
    assert kwargs["unencrypted_ports"] == []
    assert kwargs["proxy"] is None
    joined = repr(kwargs).upper()
    for forbidden in ("MODAL_TOKEN", "GITHUB_TOKEN", "HF_TOKEN", "OIDC", "CONNECT_TOKEN"):
        assert forbidden not in joined


def test_controller_creates_one_sandbox_without_forwarding_secrets_and_cleans_up():
    root, path = _selection_file("tests/unit/v1\n")
    try:
        env = _valid_env(
            path,
            MODAL_TOKEN_ID="controller-only",
            MODAL_TOKEN_SECRET="controller-only",
            GITHUB_TOKEN="not-forwarded",
            HF_TOKEN="not-forwarded",
        )
        fake, state, sandbox = _fake_modal("a" * 40)
        assert torch_latest.run_controller(env, fake) == 0
        assert state.app_calls == [(torch_latest.APP_NAME, True)]
        assert len(state.create_calls) == 1
        create_kwargs = state.create_calls[0][1]
        assert create_kwargs["gpu"] == "l40s:2"
        assert create_kwargs["secrets"] == []
        assert set(create_kwargs["env"]) == set(torch_latest.build_sandbox_env())
        assert sandbox.terminated
        assert sandbox.wait_calls == [False]
        assert all(process.waited for process in sandbox.processes)
        transfer_args, transfer_kwargs = next(call for call in sandbox.exec_calls if call[0][0] == "cat")
        assert transfer_args == ("cat", torch_latest.REMOTE_DIAGNOSTICS_ARCHIVE)
        assert transfer_kwargs["text"] is False
        assert (path.parent / "diagnostics" / "modal-torch-latest-diagnostics.tar.gz").read_bytes() == b"diagnostics"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_diagnostic_transfer_round_trip_uses_exec_without_legacy_filesystem_api():
    root = Path(tempfile.mkdtemp(prefix="ds-modal-transfer-"))
    remote_root = root / "remote"
    diagnostics = remote_root / "diagnostics"
    destination = root / "download"
    diagnostics.mkdir(parents=True)
    payload = bytes(range(256)) * 257
    (diagnostics / "events.bin").write_bytes(payload)

    class LocalExecSandbox:

        def __init__(self):
            self.exec_calls = []

        def exec(self, *args, **kwargs):
            self.exec_calls.append((args, dict(kwargs)))
            text = kwargs.pop("text", True)
            kwargs.pop("stderr", None)
            kwargs.pop("timeout", None)
            return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=text, **kwargs)

    sandbox = LocalExecSandbox()
    modal = SimpleNamespace(stream_type=SimpleNamespace(StreamType=SimpleNamespace(STDOUT=object())))
    old_root = torch_latest.REMOTE_ROOT
    old_archive = torch_latest.REMOTE_DIAGNOSTICS_ARCHIVE
    torch_latest.REMOTE_ROOT = str(remote_root)
    torch_latest.REMOTE_DIAGNOSTICS_ARCHIVE = str(remote_root / "diagnostics.tar.gz")
    try:
        archive = torch_latest._retrieve_diagnostics(sandbox, modal, destination)
        with tarfile.open(archive, "r:gz") as downloaded:
            assert "diagnostics/events.bin" in downloaded.getnames()
            extracted = downloaded.extractfile("diagnostics/events.bin")
            assert extracted is not None
            assert extracted.read() == payload
        assert not hasattr(sandbox, "open")
        assert any(args[0] == "cat" and kwargs["text"] is False for args, kwargs in sandbox.exec_calls)
    finally:
        torch_latest.REMOTE_ROOT = old_root
        torch_latest.REMOTE_DIAGNOSTICS_ARCHIVE = old_archive
        shutil.rmtree(root, ignore_errors=True)


def test_await_sandbox_start_gives_up_when_the_container_never_runs():
    # Catches a controller that blocks forever on a GPU reservation that is never satisfied.
    sandbox = FakeSandbox("a" * 40, never_starts=True)
    error = _expect_error(
        torch_latest.await_sandbox_start,
        sandbox,
        0.05,
        exception=torch_latest.SandboxStartTimeout,
    )
    assert "no test ran" in str(error)


def test_await_sandbox_start_reports_startup_duration():
    sandbox = FakeSandbox("a" * 40)
    assert torch_latest.await_sandbox_start(sandbox, 30) >= 0


def test_controller_aborts_without_running_tests_when_sandbox_never_starts():
    # Catches a controller that spends the whole job budget waiting, or that runs commands
    # against a Sandbox that never started, or that leaks the Sandbox when startup times out.
    root, path = _selection_file("tests/unit/v1\n")
    original = torch_latest.SANDBOX_ACQUIRE_TIMEOUT_SECONDS
    torch_latest.SANDBOX_ACQUIRE_TIMEOUT_SECONDS = 0.05
    try:
        env = _valid_env(path)
        fake, _, sandbox = _fake_modal("a" * 40, never_starts=True)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = torch_latest.run_controller(env, fake)
        assert code == torch_latest.EXIT_INFRA
        assert "DS_CI_FAILURE_CLASS=infra" in stdout.getvalue()
        assert sandbox.terminated
        assert not any("pytest" in " ".join(args) for args, _ in sandbox.exec_calls)
    finally:
        torch_latest.SANDBOX_ACQUIRE_TIMEOUT_SECONDS = original
        shutil.rmtree(root, ignore_errors=True)


def test_controller_reports_sandbox_lifetime_exhaustion_as_timeout():
    # Catches a lifetime-budget death being misreported as a candidate regression:
    # a run that dies at the Sandbox ceiling must classify as a timeout, not a test failure.
    root, path = _selection_file("tests/unit/v1\n")
    original = torch_latest.SANDBOX_TIMEOUT_SECONDS
    torch_latest.SANDBOX_TIMEOUT_SECONDS = 0.05
    try:
        env = _valid_env(path)
        fake, _, sandbox = _fake_modal("a" * 40, fail_label="pytest")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = torch_latest.run_controller(env, fake)
        assert code == torch_latest.EXIT_TIMEOUT
        assert "DS_CI_FAILURE_CLASS=timeout" in stdout.getvalue()
        assert sandbox.terminated
    finally:
        torch_latest.SANDBOX_TIMEOUT_SECONDS = original
        shutil.rmtree(root, ignore_errors=True)


def test_controller_propagates_command_and_cleanup_failures():
    root, path = _selection_file("tests/unit/v1\n")
    try:
        env = _valid_env(path)
        fake, _, sandbox = _fake_modal("a" * 40, fail_label="pytest")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            _expect_error(torch_latest.run_controller, env, fake, exception=RuntimeError)
        assert "DS_CI_FAILURE_CLASS=test" in stdout.getvalue()
        assert sandbox.terminated

        fake, _, sandbox = _fake_modal("a" * 40, fail_label="pytest", cleanup_failure=True)
        error = _expect_error(
            torch_latest.run_controller,
            env,
            fake,
            exception=torch_latest.ControllerCleanupError,
        )
        assert isinstance(error.primary, RuntimeError)
        assert isinstance(error.cleanup, RuntimeError)

        fake, _, sandbox = _fake_modal("a" * 40, wait_failure=True)
        _expect_error(torch_latest.run_controller, env, fake, exception=RuntimeError)
        assert sandbox.terminated
        assert sandbox.wait_calls == [False]

        fake, _, sandbox = _fake_modal("a" * 40, fail_label="cat")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            error = _expect_error(torch_latest.run_controller,
                                  env,
                                  fake,
                                  exception=torch_latest.DiagnosticRetrievalError)
        assert isinstance(error.cause, RuntimeError)
        assert "DS_CI_FAILURE_CLASS=artifact" in stdout.getvalue()
        assert "DS_CI_FAILURE_CLASS=test" not in stdout.getvalue()
        assert sandbox.terminated
        assert not (path.parent / "diagnostics" / "modal-torch-latest-diagnostics.tar.gz.partial").exists()

        fake, _, sandbox = _fake_modal("a" * 40, fail_label="pytest", transfer_failure=True)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            error = _expect_error(torch_latest.run_controller, env, fake, exception=RuntimeError)
        assert not isinstance(error, torch_latest.DiagnosticRetrievalError)
        assert "DS_CI_FAILURE_CLASS=test" in stdout.getvalue()
        assert "DS_CI_FAILURE_CLASS=artifact" not in stdout.getvalue()
        assert sandbox.terminated

        fake, state, sandbox = _fake_modal("a" * 40, create_failure=True)
        assert torch_latest.run_controller(env, fake) == torch_latest.EXIT_INFRA
        assert len(state.create_calls) == 1
        assert not sandbox.terminated
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_none_mode_creates_no_modal_resources():
    root, path = _selection_file("")
    try:
        fake, state, _ = _fake_modal("a" * 40)
        assert torch_latest.run_controller(_valid_env(path, DS_TEST_SELECTION_MODE="none"), fake) == 0
        assert not state.create_calls
        assert not state.app_calls
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_remote_output_is_prefixed_escaped_capped_and_drained():
    process = FakeProcess(["::warning:: first\n", "second\n", "third\n"])

    class Sandbox:

        @staticmethod
        def exec(*args, **kwargs):
            return process

    modal = SimpleNamespace(stream_type=SimpleNamespace(StreamType=SimpleNamespace(STDOUT=object())))
    original_limit = torch_latest.MAX_DISPLAY_BYTES_PER_COMMAND
    output = io.StringIO()
    try:
        torch_latest.MAX_DISPLAY_BYTES_PER_COMMAND = 40
        with contextlib.redirect_stdout(output):
            last_line = torch_latest.run_sandbox_command(Sandbox(), modal,
                                                         torch_latest.RemoteCommand("test", ("command", )))
    finally:
        torch_latest.MAX_DISPLAY_BYTES_PER_COMMAND = original_limit
    text = output.getvalue()
    assert "\n::warning::" not in text
    assert "[sandbox:test] ::warning:: first" in text
    assert "[sandbox:test] second" not in text
    assert "output truncated" in text
    assert last_line == "third"
    assert process.waited


def test_validate_selection_cli_needs_no_modal_install():
    root, path = _selection_file("tests/unit/v1\n")
    try:
        script = Path(torch_latest.__file__).resolve()
        env = {
            "PATH": os.environ["PATH"],
            "PYTHONPATH": "",
        }
        result = subprocess.run(
            [sys.executable, str(script), "validate-selection", "--mode", "all", "--path",
             str(path)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "Validated selection mode=all count=1"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_workflow_keeps_github_execution_trusted_and_preserves_modes():
    workflow = Path(torch_latest.__file__).resolve().parents[1] / ".github/workflows/modal-torch-latest.yml"
    text = workflow.read_text(encoding="utf-8")
    trusted_ref = "ref: ${{ github.event.pull_request.base.sha || github.event.merge_group.base_sha || github.sha }}"
    assert text.count(trusted_ref) == 2
    assert "ref: ${{ github.event.pull_request.head.sha" not in text
    assert "allow-unsafe-pr-checkout" not in text
    assert "Use base-branch CI scripts" not in text
    assert "HF_TOKEN" not in text
    assert "modal==1.2.6" in text
    assert "timeout-minutes: 20" in text
    assert "timeout-minutes: 135" in text
    assert text.count("persist-credentials: false") == 2
    assert text.count("lfs: false") == 2
    assert text.count("submodules: false") == 2
    assert "github.event.pull_request.head.repo.full_name" in text
    assert "github.event.pull_request.head.sha" in text
    assert "github.event.pull_request.base.repo.full_name" in text
    assert "github.event.pull_request.base.sha" in text
    assert "refs/ci/base" in text
    assert "refs/" + "dev" + "ds" not in text
    assert text.count("modal-torch-latest-test-selection") == 2
    assert "cp ci/modal_diagnostics/modal_master_71_nodes.txt ci/.test_selection/test_list.txt" not in text
    assert 'echo "mode=subset" >> "$GITHUB_OUTPUT"' not in text
    assert '--repo-root "$GITHUB_WORKSPACE"' in text
    assert '--base ""' in text
    assert "if: always()" in text
    assert "modal-torch-latest-focused-diagnostics" in text
    assert "needs.collect-tests.outputs.mode != 'none'" in text
    assert "needs.collect-tests.result != 'success'" in text
    assert 'python3 ci/torch_latest.py controller' in text

    deploy = text.split("\n  deploy:\n", 1)[1]
    assert "CANDIDATE_ROOT" not in deploy
    assert "checkout-candidate" not in deploy
    assert "pull_request.head.sha || github.sha" in deploy
    assert "pull_request.head.repo.full_name || github.repository" in deploy


def test_focused_manifest_is_exact_and_unique():
    manifest = Path(torch_latest.__file__).resolve().parent / "modal_diagnostics/modal_master_71_nodes.txt"
    content = manifest.read_bytes()
    lines = content.decode("utf-8").splitlines()
    assert len(lines) == 71
    assert len(set(lines)) == 71
    assert hashlib.sha256(content).hexdigest() == "7005c702faeb3a9bb89d1335a3aea326d293b5df00869a836621ce31ef5ff9da"
    assert torch_latest.load_test_selection(manifest, "subset") == tuple(lines)


def test_diagnostic_events_are_incremental_and_failure_details_are_immediate():
    root = Path(tempfile.mkdtemp(prefix="ds-modal-events-"))
    old_root = os.environ.get("DS_DIAGNOSTICS_DIR")
    old_worker = os.environ.get("PYTEST_XDIST_WORKER")
    try:
        os.environ["DS_DIAGNOSTICS_DIR"] = str(root)
        os.environ["PYTEST_XDIST_WORKER"] = "gw2"
        nodeid = "tests/unit/v1/test_one.py::test_value"
        modal_runner.pytest_runtest_logstart(nodeid, ("test_one.py", 1, "test_value"))
        report = SimpleNamespace(nodeid=nodeid,
                                 outcome="failed",
                                 when="call",
                                 duration=1.25,
                                 failed=True,
                                 longrepr="traceback",
                                 longreprtext="detailed traceback")
        modal_runner.pytest_runtest_logreport(report)
        modal_runner.pytest_runtest_logfinish(nodeid, ("test_one.py", 1, "test_value"))
        events = modal_runner.EventReader(root / "events")
        values = events.poll()
        assert [value["event"] for value in values] == ["node_start", "phase_outcome", "node_finish"]
        assert all(value["worker"] == "gw2" and value["pid"] == os.getpid() for value in values)
        assert values[1]["traceback"] == "detailed traceback"
        assert events.poll() == []
        assert "heartbeat" not in modal_runner.PROGRESS_EVENTS
    finally:
        if old_root is None:
            os.environ.pop("DS_DIAGNOSTICS_DIR", None)
        else:
            os.environ["DS_DIAGNOSTICS_DIR"] = old_root
        if old_worker is None:
            os.environ.pop("PYTEST_XDIST_WORKER", None)
        else:
            os.environ["PYTEST_XDIST_WORKER"] = old_worker


def test_diagnostic_command_and_bounded_process_stop():
    root = Path(tempfile.mkdtemp(prefix="ds-modal-command-"))
    args = SimpleNamespace(workers=4, torch_version="2.10", cuda_version="12.8")
    targets = ("tests/unit/v1/test_one.py::test_a", "tests/unit/v1/test_two.py::test_b")
    command = modal_runner._pytest_command(args, targets, 12345, root)
    assert command[command.index("-n") + 1] == "4"
    assert "--randomly-seed=12345" in command
    assert "-x" not in command
    assert command[command.index("--") + 1:] == list(targets)

    process = SimpleNamespace(pid=123, poll=lambda: None)
    sent = []
    original_signal = modal_runner._signal_process_group
    original_alive = modal_runner._process_group_alive
    try:
        modal_runner._signal_process_group = lambda target, requested_signal: sent.append(requested_signal)
        modal_runner._process_group_alive = lambda process_group_id: True
        actions = modal_runner._stop_process_group(process, waits=(0.0, 0.0, 0.0), sleep=lambda _: None)
    finally:
        modal_runner._signal_process_group = original_signal
        modal_runner._process_group_alive = original_alive
    assert actions == ["SIGINT", "SIGTERM", "SIGKILL"]
    assert sent == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]


def test_diagnostic_stop_escalates_after_parent_exits():
    child_source = """import os
import signal
import time

signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print(os.getpid(), flush=True)
time.sleep(30)
"""
    parent_source = f"""import signal
import subprocess
import sys
import time

signal.signal(signal.SIGINT, lambda *args: sys.exit(0))
child = subprocess.Popen([sys.executable, "-c", {child_source!r}], stdout=subprocess.PIPE, text=True)
print(child.stdout.readline().strip(), flush=True)
time.sleep(30)
"""
    process = subprocess.Popen([sys.executable, "-c", parent_source],
                               stdout=subprocess.PIPE,
                               text=True,
                               start_new_session=True)
    assert process.stdout is not None
    child_pid = int(process.stdout.readline())
    try:
        actions = modal_runner._stop_process_group(process, waits=(1.0, 0.2, 0.2))
        assert actions == ["SIGINT", "SIGTERM", "SIGKILL"]
        assert process.poll() == 0
        deadline = time.monotonic() + 5.0
        while modal_runner._process_group_alive(process.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not modal_runner._process_group_alive(process.pid)
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError(f"child process {child_pid} survived bounded process-group shutdown")
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        process.stdout.close()


def test_diagnostic_registered_pids_timeout_and_collection_count():
    root = Path(tempfile.mkdtemp(prefix="ds-modal-stacks-"))
    stack_dir = root / "stacks"
    stack_dir.mkdir()
    (stack_dir / "registered-self.json").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    (stack_dir / "registered-bad.json").write_text("not json", encoding="utf-8")
    assert modal_runner._registered_pids(root) == [os.getpid()]
    probe = modal_runner._run_capture([sys.executable, "-c", "import time; time.sleep(1)"], timeout=0.01)
    assert probe["timed_out"] is True
    assert modal_runner._collection_count("collected 71 items") == 71
    assert modal_runner._collection_count("71 tests collected") == 71


def test_diagnostic_stack_registration_after_forkserver_fork():
    root = Path(tempfile.mkdtemp(prefix="ds-modal-forkserver-"))
    probe = root / "probe.py"
    probe.write_text(
        """import json
import multiprocessing
import os
import signal
import time
from pathlib import Path

def child(connection):
    connection.send(os.getpid())
    connection.close()
    time.sleep(10)

if __name__ == "__main__":
    context = multiprocessing.get_context("forkserver")
    parent, child_connection = context.Pipe(duplex=False)
    process = context.Process(target=child, args=(child_connection,))
    process.start()
    child_connection.close()
    pid = parent.recv()
    stack_dir = Path(os.environ["DS_DIAGNOSTICS_DIR"]) / "stacks"
    marker = stack_dir / f"registered-{pid}.json"
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    os.kill(pid, signal.SIGUSR1)
    stack = stack_dir / f"stack-{pid}.log"
    deadline = time.monotonic() + 5
    while (not stack.exists() or "Current thread" not in stack.read_text()) and time.monotonic() < deadline:
        time.sleep(0.05)
    result = {"marker": marker.exists(), "stack": stack.exists() and "Current thread" in stack.read_text()}
    process.terminate()
    process.join(5)
    print(json.dumps(result, sort_keys=True))
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    diagnostics = root / "diagnostics"
    env["DS_DIAGNOSTICS_DIR"] = str(diagnostics)
    helper_dir = str(Path(modal_runner.__file__).resolve().parent)
    env["PYTHONPATH"] = helper_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run([sys.executable, str(probe)],
                            env=env,
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {"marker": True, "stack": True}


def test_launcher_source_has_no_local_packaging_or_shell_execution():
    source = Path(torch_latest.__file__).read_text(encoding="utf-8")
    for forbidden in ("add_local_dir", "modal.Function", "@app.function", "shell=True", "os.system(", "HF_TOKEN"):
        assert forbidden not in source


def _all_test_functions():
    return sorted((name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj))


def main() -> int:
    failures = 0
    tests = _all_test_functions()
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Pure-stdlib security tests for ci/torch_latest.py."""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch_latest  # noqa: E402
import test_tests_fetcher  # noqa: E402


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
    }
    values.update(overrides)
    return values


def _aws_config() -> str:
    return json.dumps({
        "regions": [
            {
                "name": "us-east-1",
                "launch_template_id": "lt-11111111",
                "subnet_ids": ["subnet-11111111", "subnet-22222222"],
                "output_bucket": "ds-ci-east-1",
                "output_prefix": "modal-fallback",
            },
            {
                "name": "us-east-2",
                "launch_template_id": "lt-22222222",
                "subnet_ids": ["subnet-33333333"],
                "output_bucket": "ds-ci-east-2",
                "output_prefix": "modal-fallback",
            },
            {
                "name": "us-west-2",
                "launch_template_id": "lt-33333333",
                "subnet_ids": ["subnet-44444444"],
                "output_bucket": "ds-ci-west-2",
                "output_prefix": "modal-fallback",
            },
        ]
    })


def _command_result(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class FakeAwsCli:

    def __init__(
        self,
        *,
        capacity_failures=0,
        prepare_code=0,
        test_code=0,
        test_status=None,
        prepare_stdout="phase output\n",
        test_stdout="phase output\n",
        test_stderr="",
        non_capacity_launch=False,
        start_state_reason=None,
        cleanup_instances=None,
    ):
        self.capacity_failures = capacity_failures
        self.prepare_code = prepare_code
        self.test_code = test_code
        self.test_status = test_status
        self.prepare_stdout = prepare_stdout
        self.test_stdout = test_stdout
        self.test_stderr = test_stderr
        self.non_capacity_launch = non_capacity_launch
        self.start_state_reason = start_state_reason
        self.cleanup_instances = cleanup_instances or {}
        self.failed_start = False
        self.calls = []
        self.sent_scripts = []
        self.send_count = 0

    def __call__(self, argv, *, timeout=None):
        self.calls.append((tuple(argv), timeout))
        service, operation = argv[2:4]
        if (service, operation) == ("ec2", "run-instances"):
            if self.non_capacity_launch:
                return _command_result(255,
                                       stderr="An error occurred (UnauthorizedOperation) when calling RunInstances")
            if self.capacity_failures:
                self.capacity_failures -= 1
                return _command_result(
                    255,
                    stderr="An error occurred (InsufficientInstanceCapacity) when calling RunInstances",
                )
            return _command_result(stdout="i-1234567890abcdef0\n")
        if (service, operation) == ("ec2", "wait"):
            waiter = argv[4]
            if waiter == "instance-running" and self.start_state_reason is not None and not self.failed_start:
                self.failed_start = True
                return _command_result(255, stderr="Waiter InstanceRunning failed: terminal failure state")
            return _command_result()
        if (service, operation) == ("ssm", "describe-instance-information"):
            return _command_result(stdout="Online\n")
        if (service, operation) == ("ssm", "send-command"):
            parameters = argv[argv.index("--parameters") + 1]
            self.sent_scripts.append(json.loads(parameters)["commands"][0])
            self.send_count += 1
            command_id = "11111111-1111-1111-1111-111111111111" if self.send_count == 1 else \
                "22222222-2222-2222-2222-222222222222"
            return _command_result(stdout=command_id + "\n")
        if (service, operation) == ("ssm", "get-command-invocation"):
            command_id = argv[argv.index("--command-id") + 1]
            code = self.prepare_code if command_id.startswith("1") else self.test_code
            stdout = self.prepare_stdout if command_id.startswith("1") else self.test_stdout
            stderr = "" if command_id.startswith("1") else self.test_stderr
            status = self.test_status if not command_id.startswith("1") and self.test_status else \
                ("Success" if code == 0 else "Failed")
            payload = {
                "Status": status,
                "ResponseCode": code,
                "StandardOutputContent": stdout[:torch_latest.SSM_INLINE_STDOUT_CHARS],
                "StandardErrorContent": stderr[:torch_latest.SSM_INLINE_STDERR_CHARS],
                "StandardOutputUrl": "https://ignored.invalid/stdout",
                "StandardErrorUrl": "https://ignored.invalid/stderr" if stderr else "",
            }
            return _command_result(stdout=json.dumps(payload))
        if (service, operation) == ("s3api", "get-object"):
            key = argv[argv.index("--key") + 1]
            stream = self.test_stderr if key.endswith("/stderr") else self.test_stdout
            Path(argv[-1]).write_bytes(stream.encode("utf-8")[-torch_latest.SSM_OUTPUT_TAIL_BYTES:])
            return _command_result(stdout="{}\n")
        if (service, operation) == ("ec2", "terminate-instances"):
            return _command_result(stdout="{}\n")
        if (service, operation) == ("ec2", "describe-instances"):
            if "--instance-ids" in argv and self.failed_start:
                reason = self.start_state_reason
                self.start_state_reason = None
                return _command_result(stdout=f"terminated\t{reason}\n")
            region = argv[argv.index("--region") + 1]
            return _command_result(stdout="\t".join(self.cleanup_instances.get(region, ())))
        raise AssertionError(f"unexpected AWS CLI call: {argv}")


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
        fail_code: int = 9,
    ):
        self.candidate_sha = candidate_sha
        self.fail_label = fail_label
        self.cleanup_failure = cleanup_failure
        self.wait_failure = wait_failure
        self.never_starts = never_starts
        self.fail_code = fail_code
        self.exec_calls = []
        self.processes = []
        self.terminated = False
        self.wait_calls = []

    def exec(self, *args, **kwargs):
        self.exec_calls.append((args, kwargs))
        if self.never_starts:
            # A container that never gets a GPU never returns from its first exec.
            threading.Event().wait()
        lines = [self.candidate_sha + "\n"] if "rev-parse" in args and "HEAD^{commit}" in args else ["ok\n"]
        label_failure = self.fail_label and self.fail_label in " ".join(args)
        process = FakeProcess(lines, return_code=self.fail_code if label_failure else 0)
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
    fail_code: int = 9,
):
    state = SimpleNamespace(image_calls=[], app_calls=[], create_calls=[])
    sandbox = FakeSandbox(candidate_sha, fail_label, cleanup_failure, wait_failure, never_starts, fail_code)

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
    assert torch_latest.SANDBOX_ACQUIRE_TIMEOUT_SECONDS == 600
    assert kwargs["secrets"] == []
    assert kwargs["network_file_systems"] == {}
    assert kwargs["volumes"] == {}
    assert kwargs["encrypted_ports"] == []
    assert kwargs["unencrypted_ports"] == []
    assert kwargs["proxy"] is None
    joined = repr(kwargs).upper()
    for forbidden in ("MODAL_TOKEN", "GITHUB_TOKEN", "HF_TOKEN", "OIDC", "CONNECT_TOKEN"):
        assert forbidden not in joined


def test_modal_infrastructure_smoke_uses_production_oci_shape_and_cleans_up():
    fake, state, sandbox = _fake_modal("a" * 40)
    assert torch_latest.run_modal_infrastructure_smoke({"MODAL_TORCH_PRESET": "2.10.0-cuda12.8"}, fake) == 0
    assert state.image_calls == [("pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel", None)]
    assert state.app_calls == [(torch_latest.APP_NAME, True)]
    assert len(state.create_calls) == 1
    create_kwargs = state.create_calls[0][1]
    assert create_kwargs["cloud"] == "oci"
    assert create_kwargs["gpu"] == "l40s:2"
    assert create_kwargs["timeout"] == torch_latest.INFRASTRUCTURE_SMOKE_TIMEOUT_SECONDS * 2
    assert create_kwargs["secrets"] == []
    assert len(sandbox.exec_calls) == 2
    cuda_count = "torch." + "cuda.device_count"
    assert cuda_count in " ".join(sandbox.exec_calls[1][0])
    assert not any("pytest" in " ".join(args) for args, _ in sandbox.exec_calls)
    assert sandbox.terminated
    assert sandbox.wait_calls == [False]

    failed, _, failed_sandbox = _fake_modal("a" * 40, fail_label=cuda_count)
    _expect_error(torch_latest.run_modal_infrastructure_smoke, {}, failed, exception=RuntimeError)
    assert failed_sandbox.terminated


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
    finally:
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
        assert "MODAL_FALLBACK=capacity" in stdout.getvalue()
        assert "DS_CI_FAILURE_CLASS" not in stdout.getvalue()
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


def test_controller_reports_modal_no_progress_as_terminal_timeout():
    root, path = _selection_file("tests/unit/v1\n")
    try:
        fake, _, sandbox = _fake_modal("a" * 40, fail_label="pytest", fail_code=torch_latest.EXIT_TIMEOUT)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = torch_latest.run_controller(_valid_env(path), fake)
        assert code == torch_latest.EXIT_TIMEOUT
        assert "DS_CI_FAILURE_CLASS=timeout: no test progress for 300 seconds" in stdout.getvalue()
        assert sandbox.terminated
    finally:
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

        fake, state, sandbox = _fake_modal("a" * 40, create_failure=True)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            _expect_error(torch_latest.run_controller, env, fake, exception=RuntimeError)
        assert "DS_CI_FAILURE_CLASS=test" in stdout.getvalue()
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


def test_aws_config_requires_fixed_complete_region_order():
    regions = torch_latest.load_aws_config(_aws_config())
    assert tuple(region.name for region in regions) == torch_latest.AWS_REGION_ORDER
    assert regions[0].subnet_ids == ("subnet-11111111", "subnet-22222222")

    payload = json.loads(_aws_config())
    payload["regions"][0], payload["regions"][1] = payload["regions"][1], payload["regions"][0]
    _expect_error(torch_latest.load_aws_config, json.dumps(payload))
    payload = json.loads(_aws_config())
    payload["regions"][2]["subnet_ids"] = []
    _expect_error(torch_latest.load_aws_config, json.dumps(payload))


def test_aws_acquisition_advances_only_for_explicit_capacity():
    regions = torch_latest.load_aws_config(_aws_config())
    fake = FakeAwsCli(capacity_failures=1)
    instance = torch_latest.acquire_aws_instance(regions, "123-1", fake)
    assert instance.region == "us-east-1"
    launch_calls = [call for call, _ in fake.calls if call[2:4] == ("ec2", "run-instances")]
    assert len(launch_calls) == 2
    assert launch_calls[0][launch_calls[0].index("--subnet-id") + 1] == "subnet-11111111"
    assert launch_calls[1][launch_calls[1].index("--subnet-id") + 1] == "subnet-22222222"
    for call in launch_calls:
        assert call[call.index("--instance-type") + 1] == "g7.12xlarge"
        assert "MarketType=spot" in call[call.index("--instance-market-options") + 1]
        tags = call[call.index("--tag-specifications") + 1]
        assert "Key=Project,Value=deepspeed-ci" in tags
        assert "Key=DeepSpeedCIRun,Value=123-1" in tags

    unauthorized = FakeAwsCli(non_capacity_launch=True)
    _expect_error(torch_latest.acquire_aws_instance,
                  regions,
                  "123-1",
                  unauthorized,
                  exception=torch_latest.AwsControllerError)
    assert len([call for call, _ in unauthorized.calls if call[2:4] == ("ec2", "run-instances")]) == 1

    state_capacity = FakeAwsCli(start_state_reason="Server.InsufficientInstanceCapacity")
    instance = torch_latest.acquire_aws_instance(regions, "123-1", state_capacity)
    assert instance.region == "us-east-1"
    state_operations = [call[2:4] for call, _ in state_capacity.calls]
    assert state_operations.count(("ec2", "run-instances")) == 2
    assert state_operations.count(("ec2", "terminate-instances")) == 1

    arbitrary_start_failure = FakeAwsCli(start_state_reason="Client.InvalidSnapshot.NotFound")
    _expect_error(
        torch_latest.acquire_aws_instance,
        regions,
        "123-1",
        arbitrary_start_failure,
        exception=torch_latest.AwsControllerError,
    )
    arbitrary_operations = [call[2:4] for call, _ in arbitrary_start_failure.calls]
    assert arbitrary_operations.count(("ec2", "run-instances")) == 1
    assert arbitrary_operations.count(("ec2", "terminate-instances")) == 1


def test_aws_scripts_reuse_full_selection_manifest_and_two_gpu_contract():
    root, path = _selection_file("tests/unit/v1\n")
    try:
        inputs = torch_latest.resolve_controller_inputs(_valid_env(path))
        prepare, test = torch_latest.build_aws_scripts(inputs, "123-1")
        assert "/var/lib/devds/runs/123-1" in prepare
        assert "--gpus '\"device=0,1\"'" in prepare
        assert "uv==0.12.7" in prepare
        assert "uv venv --python 3.10.13 --seed /workspace/python310" in prepare
        assert "PATH=/workspace/python310/bin:" in prepare
        assert "expected exactly 2 visible GPUs" in prepare
        assert "ci/modal_diagnostics/pr8654_71_nodes.txt" not in prepare + test
        assert "tests/unit/v1" in test
        assert "pytest -n 4 --verbose" in test
        assert str(torch_latest.NO_PROGRESS_TIMEOUT_SECONDS) in test
        assert torch_latest.AWS_PREPARE_TIMEOUT_SECONDS + torch_latest.AWS_TEST_TIMEOUT_SECONDS == 3600
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_aws_infrastructure_smoke_is_region_scoped_and_has_no_test_execution():
    env = {
        "AWS_MODAL_FALLBACK_CONFIG": _aws_config(),
        "GITHUB_RUN_ID": "321",
        "GITHUB_RUN_ATTEMPT": "2",
        "DS_CI_AWS_RUN_SUFFIX": "us-east-2",
        "MODAL_TORCH_PRESET": "2.10.0-cuda12.8",
    }
    fake = FakeAwsCli()
    assert torch_latest.run_aws_infrastructure_smoke(env, "us-east-2", fake) == 0
    launch_calls = [call for call, _ in fake.calls if call[2:4] == ("ec2", "run-instances")]
    assert len(launch_calls) == 1
    assert launch_calls[0][launch_calls[0].index("--region") + 1] == "us-east-2"
    tags = launch_calls[0][launch_calls[0].index("--tag-specifications") + 1]
    assert "Key=DeepSpeedCIRun,Value=321-2-us-east-2" in tags
    assert len(fake.sent_scripts) == 1
    script = shlex.split(fake.sent_scripts[0])[2]
    assert "/var/lib/devds/runs/321-2-us-east-2" in script
    assert "nvidia-smi" in script
    assert "docker pull --quiet pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel >/dev/null" in script
    assert 'device=0,1' in script
    assert "torch." + "cuda.device_count" in script
    assert "pytest" not in script
    assert "pip install" not in script
    assert "s3api put-object" in script
    assert "fileb://" not in script
    assert "--checksum-sha256" in script
    assert "AWS_INFRA_SMOKE=private_log_upload_failed" in script
    assert "AWS_INFRA_SMOKE=private_log_checksum_failed" in script
    operations = [call[2:4] for call, _ in fake.calls]
    assert operations.count(("ec2", "terminate-instances")) == 1
    start_waits = [timeout for call, timeout in fake.calls if call[2:4] == ("ec2", "wait")][:2]
    assert len(start_waits) == 2
    assert all(0 < timeout <= torch_latest.INFRASTRUCTURE_SMOKE_TIMEOUT_SECONDS for timeout in start_waits)

    failed = FakeAwsCli(prepare_code=7)
    _expect_error(torch_latest.run_aws_infrastructure_smoke,
                  env,
                  "us-east-2",
                  failed,
                  exception=torch_latest.AwsControllerError)
    failed_operations = [call[2:4] for call, _ in failed.calls]
    assert failed_operations.count(("ec2", "terminate-instances")) == 1

    invalid = dict(env, DS_CI_AWS_RUN_SUFFIX="not-a-region")
    _expect_error(torch_latest._aws_run_identity, invalid)


def test_aws_infrastructure_smoke_log_upload_is_required_and_sanitized():
    root = Path(tempfile.mkdtemp(prefix="ds-infra-smoke-")).resolve()
    try:
        fake_bin = root / "bin"
        fake_bin.mkdir()
        commands = {
            "nvidia-smi":
            "#!/bin/sh\nprintf 'GPU one\\nGPU two\\n'\n",
            "docker": ("#!/bin/sh\n"
                       "if [ \"$1\" = pull ]; then exit 0; fi\n"
                       "printf 'AWS_INFRA_SMOKE=ready device_count=2 container=true\\n'\n"),
            "aws": ("#!/bin/sh\n"
                    "if [ \"${FAKE_AWS_FAIL:-}\" = 1 ]; then "
                    "printf 'private backend detail\\n' >&2; exit 2; fi\n"
                    "while [ \"$#\" -gt 0 ]; do\n"
                    "  if [ \"$1\" = --checksum-sha256 ]; then printf '%s\\n' \"$2\"; exit 0; fi\n"
                    "  shift\n"
                    "done\n"
                    "exit 3\n"),
        }
        for name, source in commands.items():
            path = fake_bin / name
            path.write_text(source, encoding="utf-8")
            path.chmod(0o755)
        config = torch_latest.load_aws_config(_aws_config())[1]
        script = torch_latest.build_aws_infrastructure_smoke_script(
            "2.10.0-cuda12.8",
            "321-2-us-east-2",
            config,
            run_root=str(root / "run"),
        )
        run_env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}")
        success = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=run_env)
        assert success.returncode == 0, success.stderr
        assert "AWS_INFRA_SMOKE=private_log_retained checksum=sha256" in success.stdout

        failed = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            env=dict(run_env, FAKE_AWS_FAIL="1"),
        )
        assert failed.returncode != 0
        assert "AWS_INFRA_SMOKE=private_log_upload_failed" in failed.stdout
        assert "private backend detail" not in failed.stdout + failed.stderr
        assert config.output_bucket not in failed.stdout + failed.stderr
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_aws_backend_startup_uses_one_absolute_deadline():
    clock = SimpleNamespace(now=0.0)

    class SlowAwsCli(FakeAwsCli):

        def __call__(self, argv, *, timeout=None):
            if argv[2:4] == ("ec2", "wait") and argv[4] in {"instance-running", "instance-status-ok"}:
                clock.now += min(290.0, timeout)
            return super().__call__(argv, timeout=timeout)

    config = torch_latest.load_aws_config(_aws_config())[1]
    fake = SlowAwsCli()
    _expect_error(
        torch_latest.wait_for_aws_backend,
        torch_latest.AwsInstance(config.name, "i-1234567890abcdef0", config),
        fake,
        sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
        monotonic=lambda: clock.now,
        timeout_seconds=torch_latest.INFRASTRUCTURE_SMOKE_TIMEOUT_SECONDS,
        exception=torch_latest.AwsControllerError,
    )
    wait_timeouts = [timeout for call, timeout in fake.calls if call[2:4] == ("ec2", "wait")]
    assert wait_timeouts == [300.0, 10.0]
    assert clock.now == 300.0


def test_aws_controller_runs_one_backend_and_always_terminates():
    root, path = _selection_file("tests/unit/v1/test_one.py\n")
    try:
        env = _valid_env(
            path,
            DS_TEST_SELECTION_MODE="subset",
            AWS_MODAL_FALLBACK_CONFIG=_aws_config(),
            GITHUB_RUN_ID="123",
            GITHUB_RUN_ATTEMPT="1",
        )
        fake = FakeAwsCli()
        assert torch_latest.run_aws_controller(env, fake) == 0
        assert len(fake.sent_scripts) == 2
        for sent_script in fake.sent_scripts:
            shell = shlex.split(sent_script)
            assert shell[:2] == ["bash", "-lc"]
            assert shell[2].startswith("set -euo pipefail\n")
        operations = [call[2:4] for call, _ in fake.calls]
        assert operations.count(("ec2", "run-instances")) == 1
        assert operations.count(("ec2", "terminate-instances")) == 1
        assert operations.count(("ec2", "wait")) == 3

        failed = FakeAwsCli(test_code=9)
        assert torch_latest.run_aws_controller(env, failed) == torch_latest.EXIT_TEST_FAILURE
        failed_operations = [call[2:4] for call, _ in failed.calls]
        assert failed_operations.count(("ec2", "run-instances")) == 1
        assert failed_operations.count(("ec2", "terminate-instances")) == 1

        stalled = FakeAwsCli(test_code=torch_latest.EXIT_TIMEOUT)
        assert torch_latest.run_aws_controller(env, stalled) == torch_latest.EXIT_TIMEOUT
        assert len([call for call, _ in stalled.calls if call[2:4] == ("ec2", "run-instances")]) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_aws_controller_capacity_exhaustion_and_prepare_failure_are_not_retried_as_tests():
    root, path = _selection_file("tests/unit/v1/test_one.py\n")
    try:
        env = _valid_env(
            path,
            DS_TEST_SELECTION_MODE="subset",
            AWS_MODAL_FALLBACK_CONFIG=_aws_config(),
            GITHUB_RUN_ID="456",
            GITHUB_RUN_ATTEMPT="2",
        )
        capacity = FakeAwsCli(capacity_failures=4)
        assert torch_latest.run_aws_controller(env, capacity) == torch_latest.EXIT_INFRA
        capacity_operations = [call[2:4] for call, _ in capacity.calls]
        assert capacity_operations.count(("ec2", "run-instances")) == 4
        assert ("ec2", "terminate-instances") not in capacity_operations

        prepare_failure = FakeAwsCli(prepare_code=7)
        _expect_error(torch_latest.run_aws_controller, env, prepare_failure, exception=torch_latest.AwsControllerError)
        failure_operations = [call[2:4] for call, _ in prepare_failure.calls]
        assert failure_operations.count(("ec2", "run-instances")) == 1
        assert failure_operations.count(("ec2", "terminate-instances")) == 1
        assert prepare_failure.send_count == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _pid_is_running(pid: int, proc_root: Path = Path("/proc")) -> bool:
    try:
        stat_fields = (proc_root / str(pid) / "stat").read_text(encoding="utf-8").rpartition(")")[2].split()
    except OSError:
        stat_fields = []
    if stat_fields and stat_fields[0] == "Z":
        return False
    return _pid_exists(pid)


def _assert_pid_gone(pid: int) -> None:
    deadline = time.monotonic() + 2
    while _pid_is_running(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    if _pid_is_running(pid):
        os.kill(pid, signal.SIGKILL)
        raise AssertionError(f"watchdog left descendant {pid} running")


def test_pid_running_treats_linux_zombies_as_terminated():
    root = Path(tempfile.mkdtemp(prefix="ds-proc-state-"))
    try:
        pid = os.getpid()
        proc = root / str(pid)
        proc.mkdir()
        (proc / "stat").write_text(f"{pid} (python worker) Z 1 2 3\n", encoding="utf-8")
        assert not _pid_is_running(pid, root)
        (proc / "stat").write_text(f"{pid} (python worker) S 1 2 3\n", encoding="utf-8")
        assert _pid_is_running(pid, root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_no_progress_watchdog_enforces_deadline_after_parent_exit():
    original = torch_latest.NO_PROGRESS_TIMEOUT_SECONDS
    torch_latest.NO_PROGRESS_TIMEOUT_SECONDS = 0.1
    try:
        descendant = "import time; time.sleep(5)"
        parent = ("import subprocess, sys; "
                  f"child=subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
                  "print(f'descendant={child.pid}', flush=True)")
        argv = torch_latest._with_no_progress_watchdog((sys.executable, "-c", parent))
        result = subprocess.run((sys.executable, *argv[1:]), check=False, capture_output=True, text=True, timeout=2)
        assert result.returncode == torch_latest.EXIT_TIMEOUT
        assert "No test output" in result.stdout
        _assert_pid_gone(int(result.stdout.split("descendant=", 1)[1].splitlines()[0]))
    finally:
        torch_latest.NO_PROGRESS_TIMEOUT_SECONDS = original


def test_no_progress_watchdog_enforces_deadline_after_output_eof():
    original = torch_latest.NO_PROGRESS_TIMEOUT_SECONDS
    torch_latest.NO_PROGRESS_TIMEOUT_SECONDS = 0.1
    try:
        child = "import os, time; os.close(1); os.close(2); time.sleep(5)"
        argv = torch_latest._with_no_progress_watchdog((sys.executable, "-c", child))
        result = subprocess.run((sys.executable, *argv[1:]), check=False, capture_output=True, text=True, timeout=2)
        assert result.returncode == torch_latest.EXIT_TIMEOUT
        assert "No test output" in result.stdout
    finally:
        torch_latest.NO_PROGRESS_TIMEOUT_SECONDS = original


def test_no_progress_watchdog_kills_sigterm_resistant_descendant():
    original = torch_latest.NO_PROGRESS_TIMEOUT_SECONDS
    torch_latest.NO_PROGRESS_TIMEOUT_SECONDS = 0.1
    try:
        descendant = ("import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                      "print(f'descendant={os.getpid()}', flush=True); time.sleep(5)")
        parent = ("import subprocess, sys, time; time.sleep(0.2); "
                  f"subprocess.Popen([sys.executable, '-c', {descendant!r}]); time.sleep(5)")
        argv = torch_latest._with_no_progress_watchdog((sys.executable, "-c", parent))
        result = subprocess.run((sys.executable, *argv[1:]), check=False, capture_output=True, text=True, timeout=2)
        assert result.returncode == torch_latest.EXIT_TIMEOUT
        assert "descendant=" in result.stdout
        _assert_pid_gone(int(result.stdout.split("descendant=", 1)[1].splitlines()[0]))
    finally:
        torch_latest.NO_PROGRESS_TIMEOUT_SECONDS = original


def test_ssm_diagnostic_paths_redact_known_instance_and_command_ids():
    instance_id = "i-1234567890abcdef0"
    command_id = "11111111-1111-1111-1111-111111111111"
    config = torch_latest.load_aws_config(_aws_config())[0]
    instance = torch_latest.AwsInstance(config.name, instance_id, config)
    diagnostic_path = (f"/var/lib/amazon/ssm/{instance_id}/document/orchestration/{command_id}/"
                       "awsrunShellScript/0.awsrunShellScript/_script.sh")
    payload = {
        "Status": "Failed",
        "ResponseCode": 17,
        "StandardOutputContent": f"running {diagnostic_path}\n",
        "StandardErrorContent": f"{diagnostic_path}: line 5: test failed\n",
    }

    def run_command(_argv, *, timeout=None):
        return _command_result(stdout=json.dumps(payload))

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert torch_latest.wait_for_ssm_command(
            instance,
            command_id,
            "test",
            60,
            run_command,
            run_identity="123-1",
        ) == 17
    public_log = output.getvalue()
    assert instance_id not in public_log
    assert command_id not in public_log
    redacted_path = ("/var/lib/amazon/ssm/<instance-id>/document/orchestration/<command-id>/"
                     "awsrunShellScript/0.awsrunShellScript/_script.sh")
    assert f"running {redacted_path}" in public_log
    assert f"{redacted_path}: line 5: test failed" in public_log


def test_aws_test_failure_emits_bounded_s3_tail_for_nightly_triage():
    root, path = _selection_file("tests/unit/v1/test_one.py\n")
    try:
        late_failure = "FAILED unit/v1/test_late.py::test_case - AssertionError"
        private_line = ("i-1234567890abcdef0 22222222-2222-2222-2222-222222222222 ds-ci-east-1 "
                        "arn:aws:iam::123456789012:role/private")
        raw_error = "An error occurred (AccessDenied) when calling GetObject: private detail"
        full_output = "early output\n" * 2500 + f"{private_line}\n{raw_error}\n{late_failure}\n"
        env = _valid_env(
            path,
            DS_TEST_SELECTION_MODE="subset",
            AWS_MODAL_FALLBACK_CONFIG=_aws_config(),
            GITHUB_RUN_ID="123",
            GITHUB_RUN_ATTEMPT="1",
        )
        fake = FakeAwsCli(test_code=9, test_stdout=full_output)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            assert torch_latest.run_aws_controller(env, fake) == torch_latest.EXIT_TEST_FAILURE
        public_log = output.getvalue()
        assert late_failure in public_log
        assert "[aws:test:tail] complete stdout" in public_log
        assert "AWS CLI error (AccessDenied)" in public_log
        assert "private detail" not in public_log
        assert not any(value in public_log for value in (
            "i-1234567890abcdef0",
            "22222222-2222-2222-2222-222222222222",
            "ds-ci-east-1",
            "123456789012",
            "ignored.invalid",
        ))
        tail_calls = [call for call, _ in fake.calls if call[2:4] == ("s3api", "get-object")]
        assert len(tail_calls) == 1
        assert tail_calls[0][tail_calls[0].index("--range") + 1] == f"bytes=-{torch_latest.SSM_OUTPUT_TAIL_BYTES}"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_aws_s3_tail_timeout_preserves_test_timeout_and_cleanup():
    root, path = _selection_file("tests/unit/v1/test_one.py\n")
    try:
        env = _valid_env(
            path,
            DS_TEST_SELECTION_MODE="subset",
            AWS_MODAL_FALLBACK_CONFIG=_aws_config(),
            GITHUB_RUN_ID="123",
            GITHUB_RUN_ATTEMPT="1",
        )

        class TailTimeoutAws(FakeAwsCli):

            def __call__(self, argv, *, timeout=None):
                if argv[2:4] == ("s3api", "get-object"):
                    self.calls.append((tuple(argv), timeout))
                    raise subprocess.TimeoutExpired(argv, timeout)
                return super().__call__(argv, timeout=timeout)

        fake = TailTimeoutAws(
            test_code=torch_latest.EXIT_TIMEOUT,
            test_stdout="progress\n" * 4000,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            assert torch_latest.run_aws_controller(env, fake) == torch_latest.EXIT_TIMEOUT
        public_log = output.getvalue()
        assert "[aws:test:tail] complete stdout unavailable" in public_log
        assert "DS_CI_FAILURE_CLASS=timeout: no test progress for 300 seconds" in public_log
        assert "DS_CI_FAILURE_CLASS=test" not in public_log
        operations = [call[2:4] for call, _ in fake.calls]
        assert operations.count(("ec2", "terminate-instances")) == 1
        assert len([call for call, _ in fake.calls if call[2:5] == ("ec2", "wait", "instance-terminated")]) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_aws_test_deadlines_and_terminal_timeouts_are_classified_as_timeouts():
    root, path = _selection_file("tests/unit/v1/test_one.py\n")
    try:
        env = _valid_env(
            path,
            DS_TEST_SELECTION_MODE="subset",
            AWS_MODAL_FALLBACK_CONFIG=_aws_config(),
            GITHUB_RUN_ID="123",
            GITHUB_RUN_ATTEMPT="1",
        )
        clock = SimpleNamespace(now=0.0)

        class InProgressAws(FakeAwsCli):

            def __call__(self, argv, *, timeout=None):
                if (argv[2:4] == ("ssm", "get-command-invocation")
                        and not argv[argv.index("--command-id") + 1].startswith("1")):
                    self.calls.append((tuple(argv), timeout))
                    return _command_result(stdout=json.dumps({
                        "Status": "InProgress",
                        "ResponseCode": -1,
                        "StandardOutputContent": "still running\n",
                        "StandardErrorContent": "",
                    }))
                return super().__call__(argv, timeout=timeout)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            assert torch_latest.run_aws_controller(
                env,
                InProgressAws(),
                sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
                monotonic=lambda: clock.now,
            ) == torch_latest.EXIT_TIMEOUT
        assert "DS_CI_FAILURE_CLASS=timeout: AWS test command exceeded 3000 seconds" in output.getvalue()
        assert "DS_CI_FAILURE_CLASS=test" not in output.getvalue()

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            assert torch_latest.run_aws_controller(
                env,
                FakeAwsCli(test_code=-1, test_status="TimedOut"),
            ) == torch_latest.EXIT_TIMEOUT
        assert "DS_CI_FAILURE_CLASS=timeout: AWS test command exceeded 3000 seconds" in output.getvalue()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_aws_logs_redact_private_identifiers_and_cli_text():
    root, path = _selection_file("tests/unit/v1/test_one.py\n")
    try:
        env = _valid_env(
            path,
            DS_TEST_SELECTION_MODE="subset",
            AWS_MODAL_FALLBACK_CONFIG=_aws_config(),
            GITHUB_RUN_ID="789",
            GITHUB_RUN_ATTEMPT="1",
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            assert torch_latest.run_aws_controller(env, FakeAwsCli()) == 0
        public_log = output.getvalue()
        private_values = (
            "lt-11111111",
            "subnet-11111111",
            "ds-ci-east-1",
            "i-1234567890abcdef0",
            "11111111-1111-1111-1111-111111111111",
        )
        assert not any(value in public_log for value in private_values)

        private_stderr = ("An error occurred (UnauthorizedOperation) while using "
                          "subnet-11111111 and arn:aws:iam::123456789012:role/private")
        error = _expect_error(
            torch_latest._aws_text,
            lambda _argv, timeout=None: _command_result(255, stderr=private_stderr),
            ("ec2", "describe-instances"),
            exception=torch_latest.AwsControllerError,
        )
        assert str(error) == "AWS CLI failed (UnauthorizedOperation)"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_aws_controller_entrypoint_timeout_tracebacks_redact_command_arguments():
    harness = r"""
import os
import subprocess
import sys

sys.path.insert(0, "ci")
import torch_latest as controller
import test_torch_latest as fixtures

os.environ.update(fixtures._valid_env(
    "/unused",
    DS_TEST_SELECTION_MODE="subset",
    AWS_MODAL_FALLBACK_CONFIG=fixtures._aws_config(),
    GITHUB_RUN_ID="999",
    GITHUB_RUN_ATTEMPT="2",
))
controller.load_test_selection = lambda *args: ("tests/unit/v1/test_one.py",)
controller.validate_transformers_ref = lambda value: value
fake = fixtures.FakeAwsCli()
operation = sys.argv[1]
real_run = controller.subprocess.run

def run_command(argv, **kwargs):
    if argv[0] != "aws":
        return real_run(argv, **kwargs)
    if argv[3] == operation:
        raise controller.subprocess.TimeoutExpired(argv, kwargs["timeout"])
    return fake(argv, timeout=kwargs.get("timeout"))

controller.subprocess.run = run_command
raise SystemExit(controller.main(["aws-controller"]))
"""
    private_values = ("lt-11111111", "subnet-11111111", "i-1234567890abcdef0", "ds-ci-east-1")
    for operation in ("run-instances", "send-command"):
        result = subprocess.run(
            [sys.executable, "-B", "-c", textwrap.dedent(harness), operation],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == torch_latest.EXIT_TEST_FAILURE, (operation, result.stdout, result.stderr)
        assert "AWS CLI operation exceeded its time bound" in result.stderr, (operation, result.stderr)
        assert "subprocess.TimeoutExpired: Command" not in result.stderr, (operation, result.stderr)
        assert not any(value in result.stderr for value in private_values), (operation, result.stderr)


def test_cleanup_entrypoint_terminates_only_run_owned_instances():
    instance_id = "i-0abcdef1234567890"
    fake = FakeAwsCli(cleanup_instances={"us-east-1": (instance_id, )})
    env = {
        "AWS_MODAL_FALLBACK_CONFIG": _aws_config(),
        "GITHUB_RUN_ID": "999",
        "GITHUB_RUN_ATTEMPT": "2",
    }
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert torch_latest.cleanup_aws_instances(env, fake) == 0
    operations = [call[2:4] for call, _ in fake.calls]
    assert operations.count(("ec2", "describe-instances")) == 3
    assert operations.count(("ec2", "terminate-instances")) == 1
    assert instance_id not in output.getvalue()


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
    assert "timeout-minutes: 180" in text
    assert text.count("persist-credentials: false") == 4
    assert text.count("lfs: false") == 4
    assert text.count("submodules: false") == 4
    assert "github.event.pull_request.head.repo.full_name" in text
    assert "github.event.pull_request.head.sha" in text
    assert "github.event.pull_request.base.repo.full_name" in text
    assert "github.event.pull_request.base.sha" in text
    assert "refs/ci/base" in text
    assert "refs/" + "dev" + "ds" not in text
    assert text.count("modal-torch-latest-test-selection") == 2
    assert "needs.collect-tests.outputs.mode != 'none'" in text
    assert "needs.collect-tests.result != 'success'" in text
    assert 'python3 ci/torch_latest.py controller' in text
    assert 'python3 ci/torch_latest.py aws-controller' in text
    assert 'python3 ci/torch_latest.py cleanup-aws' in text
    assert "steps.modal.outputs.fallback == 'true'" in text
    assert 'if [ "$status" -eq 75 ]' in text
    assert "aws-actions/configure-aws-credentials" not in text
    assert "sts assume-role-with-web-identity" in text
    assert "Credentials.[AccessKeyId,SecretAccessKey,SessionToken]" in text
    assert "AssumedRoleId" not in text
    assert "::add-mask::$web_identity_token" in text
    assert "--duration-seconds 7200" in text
    assert text.count("AWS_ROLE_ARN: ${{ secrets.AWS_MODAL_FALLBACK_ROLE_ARN }}") == 2
    assert text.count("AWS_MODAL_FALLBACK_CONFIG: ${{ secrets.AWS_MODAL_FALLBACK_CONFIG }}") == 4
    assert "vars.AWS_MODAL_FALLBACK" not in text
    assert "infrastructure_smoke:" in text
    assert "default: false" in text
    assert text.count("inputs.infrastructure_smoke == true") == 2
    assert text.count("inputs.infrastructure_smoke != true") == 2
    assert "region: [us-east-1, us-east-2, us-west-2]" in text
    assert "fail-fast: false" in text
    assert 'python3 ci/torch_latest.py modal-infrastructure-smoke' in text
    assert 'python3 ci/torch_latest.py aws-infrastructure-smoke --region "${{ matrix.region }}"' in text
    assert "needs: modal-infrastructure-smoke" not in text
    assert "DS_CI_AWS_RUN_SUFFIX: ${{ matrix.region }}" in text

    deploy = text.split("\n  deploy:\n", 1)[1]
    collect = text.split("\n  collect-tests:\n", 1)[1].split("\n  deploy:\n", 1)[0]
    assert "id-token: write" in deploy
    assert "id-token: write" not in collect
    assert "CANDIDATE_ROOT" not in deploy
    assert "checkout-candidate" not in deploy
    assert "pull_request.head.sha || github.sha" in deploy
    assert "pull_request.head.repo.full_name || github.repository" in deploy


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

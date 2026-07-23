# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Pure-stdlib security tests for ci/torch_latest.py."""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch_latest  # noqa: E402


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
    ):
        self.candidate_sha = candidate_sha
        self.fail_label = fail_label
        self.cleanup_failure = cleanup_failure
        self.wait_failure = wait_failure
        self.exec_calls = []
        self.processes = []
        self.terminated = False
        self.wait_calls = []

    def exec(self, *args, **kwargs):
        self.exec_calls.append((args, kwargs))
        lines = [self.candidate_sha + "\n"] if "rev-parse" in args and "HEAD^{commit}" in args else ["ok\n"]
        label_failure = self.fail_label and self.fail_label in " ".join(args)
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
):
    state = SimpleNamespace(image_calls=[], app_calls=[], create_calls=[])
    sandbox = FakeSandbox(candidate_sha, fail_label, cleanup_failure, wait_failure)

    class Image:

        @staticmethod
        def from_registry(image, add_python=None):
            state.image_calls.append((image, add_python))
            return ("image", image, add_python)

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
        ("subset", "tests/unit/v1/test_one.py\ntests/unit/v1/sub/test_two.py\n",
         ("tests/unit/v1/test_one.py", "tests/unit/v1/sub/test_two.py")),
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
        assert labels.index("install runtime requirements") < labels.index("reinstall Torch packages")
        assert labels.index("reinstall Torch packages") < labels.index("install candidate DeepSpeed")
        pytest_command = next(command for command in commands if command.label == "run pytest")
        separator = pytest_command.argv.index("--")
        assert pytest_command.argv[separator + 1:] == ("tests/unit/v1/test_one.py", )
        fetch = next(command for command in commands if command.label == "fetch candidate SHA")
        assert "https://github.com/example/DeepSpeed.git" in fetch.argv
        assert f"{'c' * 40}:refs/devds/candidate" in fetch.argv
        assert not any("sh" == argument or "bash" == argument for command in commands for argument in command.argv)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_sandbox_kwargs_are_fixed_and_secret_free():
    kwargs = torch_latest.build_sandbox_kwargs("image")
    assert kwargs["gpu"] == "l40s:2"
    assert kwargs["timeout"] == 3600
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
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_controller_propagates_command_and_cleanup_failures():
    root, path = _selection_file("tests/unit/v1\n")
    try:
        env = _valid_env(path)
        fake, _, sandbox = _fake_modal("a" * 40, fail_label="pytest")
        _expect_error(torch_latest.run_controller, env, fake, exception=RuntimeError)
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
        _expect_error(torch_latest.run_controller, env, fake, exception=RuntimeError)
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
        torch_latest.MAX_DISPLAY_BYTES_PER_COMMAND = 20
        with contextlib.redirect_stdout(output):
            torch_latest.run_sandbox_command(Sandbox(), modal, torch_latest.RemoteCommand("test", ("command", )))
    finally:
        torch_latest.MAX_DISPLAY_BYTES_PER_COMMAND = original_limit
    text = output.getvalue()
    assert "\n::warning::" not in text
    assert "[sandbox:test] ::warning:: first" in text
    assert "output truncated" in text
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
            [sys.executable, str(script), "validate-selection", "--mode", "all", "--path", str(path)],
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
    trusted_ref = "ref: ${{ github.event.pull_request.base.sha || github.sha }}"
    assert text.count(trusted_ref) == 2
    assert "ref: ${{ github.event.pull_request.head.sha" not in text
    assert "allow-unsafe-pr-checkout" not in text
    assert "Use base-branch CI scripts" not in text
    assert "HF_TOKEN" not in text
    assert "modal==1.2.6" in text
    assert "timeout-minutes: 20" in text
    assert "timeout-minutes: 75" in text
    assert text.count("persist-credentials: false") == 2
    assert text.count("lfs: false") == 2
    assert text.count("submodules: false") == 2
    assert "github.event.pull_request.head.repo.full_name" in text
    assert "github.event.pull_request.head.sha" in text
    assert "github.event.pull_request.base.repo.full_name" in text
    assert "github.event.pull_request.base.sha" in text
    assert text.count("modal-torch-latest-test-selection") == 2
    assert "needs.collect-tests.outputs.mode != 'none'" in text
    assert "needs.collect-tests.result != 'success'" in text
    assert 'python3 ci/torch_latest.py controller' in text

    deploy = text.split("\n  deploy:\n", 1)[1]
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

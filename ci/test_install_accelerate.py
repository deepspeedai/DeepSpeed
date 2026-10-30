# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import install_accelerate  # noqa: E402


def test_install_requires_preinstalled_framework(monkeypatch, tmp_path):
    versions = {package: "1.0" for package in install_accelerate.PROTECTED_PACKAGES}
    versions["datasets"] = None
    monkeypatch.setattr(install_accelerate, "package_versions", lambda packages: versions)

    with pytest.raises(RuntimeError, match="datasets"):
        install_accelerate.install(tmp_path / "accelerate", install_accelerate.ACCELERATE_REF)


def test_install_constrains_and_preserves_framework(monkeypatch, tmp_path):
    versions = {package: "1.0" for package in install_accelerate.PROTECTED_PACKAGES}
    commands = []

    monkeypatch.setattr(install_accelerate, "package_versions", lambda packages: versions)
    monkeypatch.setattr(install_accelerate, "run", lambda command, **kwargs: commands.append(tuple(command)))
    monkeypatch.setattr(install_accelerate.subprocess, "check_output",
                        lambda *args, **kwargs: f"{install_accelerate.ACCELERATE_REF}\n")

    install_accelerate.install(tmp_path / "accelerate", install_accelerate.ACCELERATE_REF)

    install_command = next(command for command in commands if command[:4] == (sys.executable, "-m", "pip", "install"))
    assert "-c" in install_command
    assert install_command[-1].endswith("[testing]")
    assert (sys.executable, "-m", "pip", "check") in commands

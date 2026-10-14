# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("use_installed_package", [False, True])
def test_deepspeed_import_source(use_installed_package, tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    conftest = repo_root / "tests" / "conftest.py"
    installed_package = tmp_path / "deepspeed"
    installed_package.mkdir()
    (installed_package / "__init__.py").write_text("")
    program = f"""
import importlib.util
import runpy

runpy.run_path({str(conftest)!r})
print(importlib.util.find_spec('deepspeed').origin)
"""
    env = os.environ.copy()
    env["DS_TEST_USE_INSTALLED_DEEPSPEED"] = "1" if use_installed_package else "0"
    env["PYTHONPATH"] = str(tmp_path)

    result = subprocess.run([sys.executable, "-c", program],
                            cwd=repo_root / "tests",
                            capture_output=True,
                            text=True,
                            env=env,
                            check=True)

    expected_package = installed_package if use_installed_package else repo_root / "deepspeed"
    assert Path(result.stdout.strip()).resolve() == expected_package / "__init__.py"

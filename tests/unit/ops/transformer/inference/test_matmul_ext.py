# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import ast
import os
import subprocess
from pathlib import Path
from typing import Callable, cast
from unittest.mock import patch

MATMUL_EXT_PATH = Path(
    __file__).resolve().parents[5] / "deepspeed" / "ops" / "transformer" / "inference" / "triton" / "matmul_ext.py"


def load_is_nfs_path() -> Callable[[Path], bool]:
    tree = ast.parse(MATMUL_EXT_PATH.read_text(), filename=str(MATMUL_EXT_PATH))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "is_nfs_path")
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"os": os, "subprocess": subprocess}
    exec(compile(module, str(MATMUL_EXT_PATH), "exec"), namespace)
    return cast(Callable[[Path], bool], namespace["is_nfs_path"])


def test_is_nfs_path_handles_wrapped_device_name(tmp_path):
    is_nfs_path = load_is_nfs_path()
    busybox_output = """Filesystem           Type       1K-blocks      Used Available Use% Mounted on
/dev/dvol0123456789abcdef0
                     ext4       2112647088   3439616 2109191088   0% /mount
"""

    with patch.object(subprocess, "check_output", return_value=busybox_output) as check_output:
        assert not is_nfs_path(tmp_path)

    check_output.assert_called_once_with(['df', '-PT', str(tmp_path)], encoding='utf-8', stderr=subprocess.DEVNULL)

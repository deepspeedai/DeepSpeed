# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import subprocess
from unittest.mock import patch

import pytest

try:
    import torch  # noqa: F401
    import triton  # noqa: F401
    from deepspeed.ops.transformer.inference.triton.matmul_ext import is_nfs_path
except ImportError:
    pytest.skip("Triton matmul extension is not available on this system", allow_module_level=True)


def test_is_nfs_path_handles_wrapped_device_name(tmp_path):
    # BusyBox df wraps device names longer than ~20 characters onto their own
    # row; the malformed short row must read as not-NFS instead of raising.
    busybox_output = """Filesystem           Type       1K-blocks      Used Available Use% Mounted on
/dev/dvol0123456789abcdef0
                     ext4       2112647088   3439616 2109191088   0% /mount
"""

    with patch.object(subprocess, "check_output", return_value=busybox_output) as check_output:
        assert not is_nfs_path(tmp_path)

    check_output.assert_called_once_with(['df', '-PT', str(tmp_path)], encoding='utf-8', stderr=subprocess.DEVNULL)


def test_is_nfs_path_detects_nfs_mount(tmp_path):
    nfs_output = """Filesystem           Type      1K-blocks      Used Available Use% Mounted on
fileserver:/export  nfs       2112647088  3439616 2109191088   0% /mount
"""

    with patch.object(subprocess, "check_output", return_value=nfs_output):
        assert is_nfs_path(tmp_path)

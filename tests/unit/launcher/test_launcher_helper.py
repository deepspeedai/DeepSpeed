# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.launcher import launcher_helper


def test_launcher_helper_propagates_exit_code(tmp_path, monkeypatch):
    # mpirun only sees the helper's exit status, so a failing user script has to fail the helper too.
    for name in ('PMIX_RANK', 'PALS_LOCAL_RANKID'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('PMI_RANK', '0')
    monkeypatch.setenv('MPI_LOCALRANKID', '0')
    script = tmp_path / 'failing_script.py'
    script.write_text('import sys\nsys.exit(3)\n')
    with pytest.raises(SystemExit) as exc_info:
        launcher_helper.main(['--launcher', 'mpich', str(script)])
    assert exc_info.value.code == 3

# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Binding ranks to cores is built on numactl, which is missing on Windows.

``--bind_cores_to_rank`` makes the launcher prefix every rank command with ``numactl``,
so the absence of that binary decides whether binding can work at all.
"""

import shutil

import pytest

from deepspeed.utils.numa import get_numactl_cmd


def test_bind_cores_without_numactl_raises(monkeypatch):
    """Without numactl, ``get_numactl_cmd`` must fail fast with a clear error.

    Pins the fixed bug: the returned command started with a ``numactl`` token that could
    not be spawned, surfacing later as a confusing file-not-found error per rank.
    """
    monkeypatch.setattr(shutil, 'which', lambda name: None)

    with pytest.raises(ValueError, match='numactl'):
        get_numactl_cmd(None, 2, 0)

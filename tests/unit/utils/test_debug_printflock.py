# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""``printflock`` must print where ``fcntl`` does not exist, as on Windows.

``fcntl`` is a Unix-only module, so the lazy import in ``printflock`` decides whether the
debug helper works at all on a given platform.
"""

import importlib.util
import sys
from unittest import mock

import pytest

from deepspeed.utils import debug


def test_prints_without_fcntl(capsys, monkeypatch):
    """A platform without ``fcntl`` must not stop ``printflock`` from printing.

    Pins the fixed bug: the lazy import was unguarded, so the first call on Windows
    raised ``ModuleNotFoundError`` instead of printing the message.
    """
    monkeypatch.setattr(debug, 'fcntl', None)

    # A ``None`` entry in sys.modules makes ``import fcntl`` raise ImportError, which is
    # how an absent module behaves. See module_inject/test_auto_ep_comm.py.
    with mock.patch.dict(sys.modules, {'fcntl': None}):
        debug.printflock('printed without fcntl')

    assert 'printed without fcntl' in capsys.readouterr().out


def test_prints_with_fcntl(capsys):
    """The locked path must keep printing, so the fallback above cannot swallow output."""
    if importlib.util.find_spec('fcntl') is None:
        pytest.skip('fcntl is not available on this platform')

    debug.printflock('printed with fcntl')

    assert 'printed with fcntl' in capsys.readouterr().out

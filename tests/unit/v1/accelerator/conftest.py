# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import importlib
import re
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest


@contextmanager
def mock_missing_modules(module_name):
    """Import *module_name*, auto-stubbing any missing third-party packages.

    Yields the imported module.  On exit every stub is removed from
    ``sys.modules`` so other tests are not affected.

    The stubs use :class:`~unittest.mock.MagicMock` so that arbitrary
    attribute access (e.g. ``torch_xmlir.some_api``) succeeds without
    raising ``AttributeError``.
    """
    mocked = []
    while True:
        try:
            module = importlib.import_module(module_name)
            break
        except ModuleNotFoundError as exc:
            missing = re.search(r"'(.*)'", str(exc)).group(1)
            sys.modules[missing] = MagicMock()
            mocked.append(missing)
    try:
        yield module
    finally:
        for name in mocked:
            sys.modules.pop(name, None)


@pytest.fixture
def accel_class_name(module_name):
    """Yield the single ``*_Accelerator`` class name exported by *module_name*.

    Missing third-party SDKs are auto-stubbed for the duration of the
    fixture so that every accelerator module can be imported in any CI
    environment.
    """
    with mock_missing_modules(module_name) as module:
        classes = [n for n in dir(module) if n.endswith("_Accelerator")]
        assert len(classes) == 1, f"Expected one *_Accelerator class in {module_name}, found {classes}"
        yield classes[0]

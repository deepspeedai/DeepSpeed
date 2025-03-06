# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

import os
import sys
import importlib
import re

import deepspeed

DS_ACCEL_PATH = "deepspeed.accelerator"
IGNORE_FILES = ["abstract_accelerator.py", "real_accelerator.py"]


@pytest.fixture
def accel_class_name(module_name):
    class_list = []
    mocked_modules = []

    # Get the accelerator class name for a given module
    while True:
        try:
            module = importlib.import_module(module_name)
            break
        except ModuleNotFoundError as e:
            # If the environment is missing a module, mock it so we can still
            # test importing the accelerator class
            missing_module = re.search(r"\'(.*)\'", e.msg).group().strip("'")
            sys.modules[missing_module] = lambda x: None
            mocked_modules.append(missing_module)
    for name in dir(module):
        if name.endswith("_Accelerator"):
            class_list.append(name)

    assert len(class_list) == 1, f"Multiple accelerator classes found in {module_name}"

    yield class_list[0]

    # Clean up mocked modules so as to not impact other tests
    for module in mocked_modules:
        del sys.modules[module]


@pytest.mark.parametrize(
    "module_name",
    [
        DS_ACCEL_PATH + "." + f.rstrip(".py") for f in os.listdir(deepspeed.accelerator.__path__[0])
        if f.endswith("_accelerator.py") and f not in IGNORE_FILES
    ],
)
def test_abstract_methods_defined(module_name, accel_class_name):
    module = importlib.import_module(module_name)
    accel_class = getattr(module, accel_class_name)
    accel_class.__init__ = lambda self: None
    _ = accel_class()

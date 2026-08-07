# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

import ast
import os
import sys
import importlib
import inspect
import re
import textwrap

import deepspeed
from deepspeed.accelerator.abstract_accelerator import DeepSpeedAccelerator

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


def _positional_params(func):
    params = list(inspect.signature(func).parameters.values())
    keep = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return [p for p in params if p.kind in keep]


@pytest.mark.parametrize(
    "module_name",
    [
        DS_ACCEL_PATH + "." + f.rstrip(".py") for f in os.listdir(deepspeed.accelerator.__path__[0])
        if f.endswith("_accelerator.py") and f not in IGNORE_FILES
    ],
)
def test_abstract_method_signatures(module_name, accel_class_name):
    """An override may widen the abstract signature, never narrow it: callers hold a
    DeepSpeedAccelerator, so any call the ABC permits has to work on every backend."""
    module = importlib.import_module(module_name)
    accel_class = getattr(module, accel_class_name)

    errors = []
    for name, abstract in inspect.getmembers(DeepSpeedAccelerator, inspect.isfunction):
        if name.startswith("__") or not getattr(abstract, "__isabstractmethod__", False):
            continue
        override = getattr(accel_class, name, None)
        if override is None or not inspect.isfunction(override):
            continue

        abstract_params = _positional_params(abstract)
        override_params = _positional_params(override)
        accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD
                             for p in inspect.signature(override).parameters.values())

        abstract_required = [p for p in abstract_params if p.default is inspect.Parameter.empty]
        override_required = [p for p in override_params if p.default is inspect.Parameter.empty]
        if len(override_required) > len(abstract_required):
            errors.append(f"{name}: requires {len(override_required)} args, ABC requires "
                          f"{len(abstract_required)} ({inspect.signature(override)} vs "
                          f"{inspect.signature(abstract)})")
            continue

        if not accepts_kwargs:
            override_names = {p.name for p in override_params}
            for p in abstract_params:
                if p.default is not inspect.Parameter.empty and p.name not in override_names:
                    errors.append(f"{name}: does not accept optional ABC parameter "
                                  f"'{p.name}' ({inspect.signature(override)} vs "
                                  f"{inspect.signature(abstract)})")

    assert not errors, f"{accel_class_name} narrows the DeepSpeedAccelerator contract: " + "; ".join(errors)


@pytest.mark.parametrize(
    "module_name",
    [
        DS_ACCEL_PATH + "." + f.rstrip(".py") for f in os.listdir(deepspeed.accelerator.__path__[0])
        if f.endswith("_accelerator.py") and f not in IGNORE_FILES
    ],
)
def test_create_graph_returns(module_name, accel_class_name):
    """create_graph() must return its graph object, or None to opt out: graph_process()
    feeds the result straight to capture_to_graph() and replay_graph(). Checked
    statically because instantiating a vendor graph type needs the vendor runtime."""
    module = importlib.import_module(module_name)
    accel_class = getattr(module, accel_class_name)

    source = textwrap.dedent(inspect.getsource(accel_class.create_graph))
    body = ast.parse(source).body[0].body
    assert isinstance(
        body[-1], ast.Return), (f"{accel_class_name}.create_graph does not return; graph_process() would pass None "
                                f"to capture_to_graph() and replay_graph(). Source:\n{source}")

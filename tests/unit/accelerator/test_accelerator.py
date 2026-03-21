# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

import os
import sys
import importlib
import re
import types

import deepspeed
import torch

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


def _install_fake_torch_xla(monkeypatch, local_ordinal=0, device_count=2):
    torch_xla = types.ModuleType("torch_xla")
    torch_xla_core = types.ModuleType("torch_xla.core")
    torch_xla_xla_model = types.ModuleType("torch_xla.core.xla_model")
    torch_xla_distributed = types.ModuleType("torch_xla.distributed")
    torch_xla_backend = types.ModuleType("torch_xla.distributed.xla_backend")
    selected_device = {"index": local_ordinal}

    class FakeDevice:

        def __init__(self, index):
            self.type = "xla"
            self.index = index

        def __str__(self):
            return f"xla:{self.index}"

    def xla_device(n=None, devkind=None):
        if n is not None:
            selected_device["index"] = n
        return FakeDevice(selected_device["index"])

    torch_xla.devices = lambda: [FakeDevice(idx) for idx in range(device_count)]
    torch_xla_xla_model.xla_device = xla_device
    torch_xla_xla_model.get_local_ordinal = lambda: local_ordinal
    torch_xla_xla_model.get_xla_supported_devices = lambda devkind=None: [f"xla:{idx}" for idx in range(device_count)]
    torch_xla_xla_model.mark_step = lambda: None
    torch_xla_xla_model.wait_device_ops = lambda: None

    monkeypatch.setitem(sys.modules, "torch_xla", torch_xla)
    monkeypatch.setitem(sys.modules, "torch_xla.core", torch_xla_core)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", torch_xla_xla_model)
    monkeypatch.setitem(sys.modules, "torch_xla.distributed", torch_xla_distributed)
    monkeypatch.setitem(sys.modules, "torch_xla.distributed.xla_backend", torch_xla_backend)


def test_xla_override_selects_xla_accelerator(monkeypatch):
    _install_fake_torch_xla(monkeypatch)

    import deepspeed.accelerator.real_accelerator as real_accelerator

    monkeypatch.setenv("DS_ACCELERATOR", "xla")
    monkeypatch.setattr(real_accelerator, "ds_accelerator", None)

    accelerator = real_accelerator.get_accelerator()

    assert accelerator.device_name() == "xla"
    assert accelerator.device_name(1) == "xla:1"
    assert accelerator.communication_backend_name() == "xla"
    assert accelerator.is_bf16_supported()
    assert not accelerator.is_fp16_supported()

    monkeypatch.setattr(real_accelerator, "ds_accelerator", None)


def test_xla_device_mapping_uses_addressable_devices(monkeypatch):
    _install_fake_torch_xla(monkeypatch, local_ordinal=0, device_count=1)

    import accelerator.xla_accelerator as xla_accelerator

    importlib.reload(xla_accelerator)
    accelerator = xla_accelerator.XLA_Accelerator()

    accelerator.set_device(3)

    assert accelerator.device_name(3) == "xla:0"
    assert accelerator.current_device_name() == "xla:0"


def test_zero_split_half_float_double_groups_xla_tensors(monkeypatch):
    from deepspeed.runtime.zero import stage_1_and_2

    class FakeAccelerator:

        @staticmethod
        def on_accelerator(tensor):
            return tensor.device.type == 'xla'

    class FakeTensor:

        def __init__(self, dtype, device_type='xla'):
            self.dtype = dtype
            self.device = types.SimpleNamespace(type=device_type)

    tensors = [
        FakeTensor(torch.float16),
        FakeTensor(torch.float32),
        FakeTensor(torch.bfloat16),
        FakeTensor(torch.float16, device_type='cpu'),
    ]

    monkeypatch.setattr(stage_1_and_2, "get_accelerator", lambda: FakeAccelerator())

    buckets = stage_1_and_2.split_half_float_double(tensors)

    assert len(buckets) == 3
    assert [tensor.dtype for tensor in buckets[0]] == [torch.float16]
    assert [tensor.dtype for tensor in buckets[1]] == [torch.float32]
    assert [tensor.dtype for tensor in buckets[2]] == [torch.bfloat16]

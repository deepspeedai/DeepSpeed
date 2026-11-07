# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

import pytest

from accelerator import cuda_accelerator
from accelerator.cuda_accelerator import CUDA_Accelerator


def accelerator_without_nvml_init():
    return object.__new__(CUDA_Accelerator)


@pytest.mark.parametrize("selector", ["GPU-1234", "MIG-5678"])
def test_available_memory_uses_uuid_lookup(monkeypatch, selector):
    accelerator = accelerator_without_nvml_init()
    calls = []
    fake_pynvml = SimpleNamespace(
        nvmlDeviceGetHandleByIndex=lambda index: calls.append(("index", index)) or "index-handle",
        nvmlDeviceGetHandleByUUID=lambda uuid: calls.append(("uuid", uuid)) or "uuid-handle",
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(free=42 if handle == "uuid-handle" else 0),
    )
    monkeypatch.setattr(cuda_accelerator, "pynvml", fake_pynvml)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", selector)

    assert accelerator.available_memory(0) == 42
    assert calls == [("uuid", selector)]


def test_available_memory_preserves_index_lookup(monkeypatch):
    accelerator = accelerator_without_nvml_init()
    calls = []
    fake_pynvml = SimpleNamespace(
        nvmlDeviceGetHandleByIndex=lambda index: calls.append(("index", index)) or "index-handle",
        nvmlDeviceGetHandleByUUID=lambda uuid: calls.append(("uuid", uuid)) or "uuid-handle",
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(free=84 if handle == "index-handle" else 0),
    )
    monkeypatch.setattr(cuda_accelerator, "pynvml", fake_pynvml)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,7")

    assert accelerator.available_memory(1) == 84
    assert calls == [("index", 7)]

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

from accelerator import cuda_accelerator
from accelerator.cuda_accelerator import CUDA_Accelerator


def accelerator_without_nvml_init():
    return object.__new__(CUDA_Accelerator)


def test_get_nvml_gpu_id_supports_indices_and_uuids(monkeypatch):
    accelerator = accelerator_without_nvml_init()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " 7, GPU-1234, MIG-5678 ")

    assert accelerator._get_nvml_gpu_id(0) == 7
    assert accelerator._get_nvml_gpu_id(1) == "GPU-1234"
    assert accelerator._get_nvml_gpu_id(2) == "MIG-5678"


def test_available_memory_uses_uuid_lookup(monkeypatch):
    accelerator = accelerator_without_nvml_init()
    calls = []
    fake_pynvml = SimpleNamespace(
        nvmlDeviceGetHandleByIndex=lambda index: calls.append(("index", index)) or "index-handle",
        nvmlDeviceGetHandleByUUID=lambda uuid: calls.append(("uuid", uuid)) or "uuid-handle",
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(free=42 if handle == "uuid-handle" else 0),
    )
    monkeypatch.setattr(cuda_accelerator, "pynvml", fake_pynvml)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1234")

    assert accelerator.available_memory(0) == 42
    assert calls == [("uuid", "GPU-1234")]


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

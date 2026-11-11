# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

import deepspeed.ops.triton_ops as triton_ops
from deepspeed.accelerator.xpu_accelerator import XPU_Accelerator


@pytest.mark.parametrize("device_available, backend_active, expected", [
    (True, True, True),
    (True, False, False),
    (False, True, False),
])
def test_triton_gate_follows_device_and_backend(monkeypatch, device_available, backend_active, expected):
    # The gate opens only when an XPU device is present and the Intel Triton backend is active.
    accelerator = XPU_Accelerator()
    monkeypatch.setattr(accelerator, "is_available", lambda: device_available)
    monkeypatch.setattr(triton_ops, "is_triton_available", lambda: backend_active)
    assert accelerator.is_triton_supported() is expected

# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

try:
    import torch.mps  # noqa: F401
except ImportError:
    pytest.skip("this torch build has no torch.mps module", allow_module_level=True)

from deepspeed.accelerator.mps_accelerator import MPS_Accelerator


def test_construction_succeeds_on_supported_torch():
    # torch>=2.3 exposes recommended_max_memory on every platform, so this runs on Linux CI too.
    accelerator = MPS_Accelerator()
    assert accelerator.device_name() == "mps"


def test_construction_fails_clearly_without_memory_query(monkeypatch):
    # Simulate torch<2.3: the constructor must raise a self-explanatory error instead of an
    # AttributeError surfacing later from inside ZeRO's buffer sizing.
    monkeypatch.delattr(torch.mps, "recommended_max_memory")
    with pytest.raises(ValueError, match="torch>=2.3"):
        MPS_Accelerator()

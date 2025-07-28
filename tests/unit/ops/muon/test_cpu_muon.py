# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import numpy as np
import pytest

from deepspeed.accelerator import get_accelerator
from muon import MuonWithAuxAdam
from unit.common import DistributedTest


def check_equal(first, second, atol=1e-2, verbose=False):
    """Assert |first-second| <= atol element-wise (on CPU)."""
    x = first.detach().float().cpu().numpy()
    y = second.detach().float().cpu().numpy()
    if verbose:
        print("ATOL", atol)
        print("x =", x.flatten())
        print("y =", y.flatten())
        print("-" * 80)
    np.testing.assert_allclose(x, y, atol=atol, err_msg="param-update mismatch!")


def _run_steps(param1, opt1, param2, opt2, steps=10):
    for _ in range(steps):
        grad = torch.randn_like(param1)
        param1.grad = grad
        param2.grad = grad.to(param2.device, dtype=param2.dtype)
        opt1.step()
        opt2.step()


# Test matrix (dtype, model_size)
_dtypes = [torch.float]
for dt in (torch.half, torch.bfloat16):
    if dt in get_accelerator().supported_dtypes():
        _dtypes.append(dt)

_model_sizes = [64, 22, 128, 1024, 8192, 1048576]


@pytest.mark.parametrize("dtype", _dtypes, ids=[str(dt).split('.')[-1] for dt in _dtypes])
@pytest.mark.parametrize("model_size", _model_sizes)
class TestCPUMuon(DistributedTest):
    world_size = 1
    reuse_dist_env = True
    requires_cuda_env = False
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    def _build_optimizer(self, param):
        # Simple param_group with use_muon flag
        pg = [dict(params=[param], use_muon=True, lr=1e-3)]
        return MuonWithAuxAdam(pg)

    @pytest.mark.skipif(not get_accelerator().is_available(), reason="GPU required for reference path")
    def test_cpu_gpu_parity(self, dtype, model_size):
        """Compare CPU vs GPU Muon updates for same gradient stream."""
        if dtype == torch.half and not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 not supported on accelerator")
        if dtype == torch.bfloat16 and not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 not supported on accelerator")

        cpu_param = torch.nn.Parameter(torch.randn(model_size, 32, device="cpu", dtype=dtype))
        gpu_param = torch.nn.Parameter(cpu_param.detach().to(get_accelerator().device_name()))

        cpu_opt = self._build_optimizer(cpu_param)
        gpu_opt = self._build_optimizer(gpu_param)

        _run_steps(cpu_param, cpu_opt, gpu_param, gpu_opt)

        tolerance = cpu_param.float().norm().item() * 1e-2
        check_equal(cpu_param.float().norm(), gpu_param.float().norm(), atol=tolerance, verbose=False)

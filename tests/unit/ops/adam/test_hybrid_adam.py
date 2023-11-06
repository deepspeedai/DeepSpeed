# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import numpy as np
import pytest

from cpuinfo import get_cpu_info

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from deepspeed.ops.op_builder import CPUAdamBuilder
from unit.common import DistributedTest

if not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
    pytest.skip("hybrid-adam is not compatible", allow_module_level=True)

pytest.cpu_vendor = get_cpu_info()["vendor_id_raw"].lower()


def check_equal(first, second, atol=1e-2, verbose=False):
    x = first.detach().numpy()
    y = second.detach().numpy()
    print("ATOL", atol)
    if verbose:
        print("x = {}".format(x.flatten()))
        print("y = {}".format(y.flatten()))
        print('-' * 80)
    np.testing.assert_allclose(x, y, err_msg="param-update mismatch!", atol=atol)


@pytest.mark.parametrize('dtype', [torch.half, torch.float], ids=["fp16", "fp32"])
@pytest.mark.parametrize('model_size', [8, 16])
class TestHybridAdam(DistributedTest):
    world_size = 1
    reuse_dist_env = True
    requires_cuda_env = False
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    @pytest.mark.skipif(not get_accelerator().is_available(), reason="only supported in CUDA environments.")
    def test_hybrid_adam_equal(self, dtype, model_size):
        if ("amd" in pytest.cpu_vendor) and (dtype == torch.half):
            pytest.skip("cpu-adam with half precision not supported on AMD CPUs")

        ref_data = torch.randn(model_size).to(dtype)
        total_data = ref_data.clone().detach()

        ref_param = torch.nn.Parameter(ref_data)
        ref_optimizer = DeepSpeedCPUAdam([ref_param])

        cpu_data, cuda_data = total_data.chunk(2)
        cpu_param = torch.nn.Parameter(cpu_data)
        cuda_param = torch.nn.Parameter(cuda_data.to(get_accelerator().device_name()))

        cpu_optimizer = DeepSpeedCPUAdam([cpu_param])
        cuda_optimizer = FusedAdam([cuda_param])

        ref_grad = torch.randn(model_size).to(dtype)
        cpu_grad, cuda_grad = ref_grad.clone().detach().chunk(2)

        ref_param.grad = ref_grad
        cpu_param.grad = cpu_grad
        cuda_param.grad = cuda_grad.to(get_accelerator().device_name())

        ref_optimizer.step()
        cpu_optimizer.step()
        cuda_optimizer.step()

        cuda_param_copy = cuda_param.cpu()

        total_param = torch.cat((cpu_param, cuda_param_copy))

        check_equal(ref_param, total_param)

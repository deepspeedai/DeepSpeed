# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import numpy as np
import pytest
from cpuinfo import get_cpu_info

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.lion import FusedLion
from deepspeed.ops.op_builder import CPULionBuilder, FusedLionBuilder
from unit.common import DistributedTest

pytest.cpu_vendor = get_cpu_info()["vendor_id_raw"].lower()


def check_equal(first, second, atol=1e-2, verbose=False):
    x = first.detach().float().numpy()
    y = second.detach().float().numpy()
    print("ATOL", atol)
    if verbose:
        print("x = {}".format(x.flatten()))
        print("y = {}".format(y.flatten()))
        print('-' * 80)
    np.testing.assert_allclose(x, y, err_msg="param-update mismatch!", atol=atol)


def _compare_optimizers(model_size, param1, optimizer1, param2, optimizer2):
    for i in range(10):
        param1.grad = torch.randn(model_size, device=param1.device).to(param1.dtype)
        param2.grad = param1.grad.clone().detach().to(device=param2.device, dtype=param2.dtype)

        optimizer1.step()
        optimizer2.step()

    tolerance = param1.float().norm().detach().numpy() * 1e-2
    check_equal(param1.float().norm(), param2.float().cpu().norm(), atol=tolerance, verbose=True)


def _reference_lion_step(param, grad, exp_avg, lr, betas, weight_decay):
    # Algorithm 2 of https://arxiv.org/abs/2302.06675, with decoupled weight decay.
    beta1, beta2 = betas
    update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1).sign_()
    param.mul_(1 - lr * weight_decay).add_(update, alpha=-lr)
    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)


# The sizes reach every SIMD unroll span for AVX2 and AVX512 as well as the scalar tail.
@pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[CPULionBuilder.NAME],
                    reason="CPULionBuilder has not been implemented on this system.")
@pytest.mark.parametrize('model_size', [100, 200, 1027])
@pytest.mark.parametrize('weight_decay', [0.0, 0.1])
def test_cpu_lion_matches_reference(model_size, weight_decay):
    from deepspeed.ops.lion import DeepSpeedCPULion

    generator = torch.Generator().manual_seed(model_size)
    initial = torch.randn(model_size, generator=generator)
    param = torch.nn.Parameter(initial.clone())
    ref_param = initial.clone()
    ref_exp_avg = torch.zeros_like(ref_param)
    lr = 1e-3
    betas = (0.9, 0.99)

    optimizer = DeepSpeedCPULion([param], lr=lr, betas=betas, weight_decay=weight_decay)

    for _ in range(5):
        grad = torch.randn(model_size, generator=generator)
        param.grad = grad.clone()
        optimizer.step()
        _reference_lion_step(ref_param, grad, ref_exp_avg, lr, betas, weight_decay)

        # The sign term moves every element by lr, so a tolerance far below lr still catches a wrong direction.
        torch.testing.assert_close(param.detach(), ref_param, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(optimizer.state[param]['exp_avg'], ref_exp_avg, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('dtype', [torch.half, torch.bfloat16, torch.float], ids=["fp16", "bf16", "fp32"])
@pytest.mark.parametrize('model_size',
                         [
                             (64),
                             (22),
                             #(55),
                             (128),
                             (1024),
                             (1048576),
                         ]) # yapf: disable
class TestCPULion(DistributedTest):
    world_size = 1
    reuse_dist_env = True
    requires_cuda_env = False
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    @pytest.mark.skipif(not get_accelerator().is_available(), reason="only supported in CUDA environments.")
    @pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[CPULionBuilder.NAME],
                        reason="CPULionBuilder has not been implemented on this system.")
    @pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedLionBuilder.NAME],
                        reason="FusedLion is not compatible")
    def test_fused_lion_equal(self, dtype, model_size):
        if ("amd" in pytest.cpu_vendor) and (dtype == torch.half):
            pytest.skip("cpu-lion with half precision not supported on AMD CPUs")

        from deepspeed.ops.lion import DeepSpeedCPULion

        cpu_data = torch.randn(model_size, device='cpu').to(dtype)
        cpu_param = torch.nn.Parameter(cpu_data)
        cuda_param = torch.nn.Parameter(cpu_data.to(get_accelerator().device_name()))

        cpu_optimizer = DeepSpeedCPULion([cpu_param])
        cuda_optimizer = FusedLion([cuda_param])

        _compare_optimizers(model_size=model_size,
                            param1=cpu_param,
                            optimizer1=cpu_optimizer,
                            param2=cuda_param,
                            optimizer2=cuda_optimizer)


class TestCPULionGPUError(DistributedTest):

    @pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[CPULionBuilder.NAME],
                        reason="CPULionBuilder has not been implemented on this system.")
    @pytest.mark.skipif(get_accelerator().device_name() == "cpu",
                        reason="Needs a non-CPU accelerator device for CPULion to reject.")
    def test_cpu_lion_gpu_error(self):
        model_size = 64
        from deepspeed.ops.lion import DeepSpeedCPULion
        device = get_accelerator().device_name(0)  # 'cuda:0' or 'xpu:0'
        param = torch.nn.Parameter(torch.randn(model_size, device=device))
        optimizer = DeepSpeedCPULion([param])

        param.grad = torch.randn(model_size, device=device)
        with pytest.raises(AssertionError):
            optimizer.step()

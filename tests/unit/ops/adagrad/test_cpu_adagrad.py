# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import numpy as np
import pytest

import deepspeed
from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import CPUAdagradBuilder
from unit.common import DistributedTest

if not deepspeed.ops.__compatible_ops__[CPUAdagradBuilder.NAME]:
    pytest.skip("cpu-adagrad is not compatible", allow_module_level=True)


def check_equal(first, second, atol=1e-2, verbose=False):
    x = first.detach().float().numpy()
    y = second.detach().float().numpy()
    if verbose:
        print("x = {}".format(x.flatten()))
        print("y = {}".format(y.flatten()))
        print('-' * 80)
    np.testing.assert_allclose(x, y, err_msg="param-update mismatch!", atol=atol)


@pytest.mark.parametrize('model_size', [129, 1024, 8000])
@pytest.mark.parametrize('weight_decay', [0.0, 0.1])
def test_cpu_adagrad_weight_decay(model_size, weight_decay):
    param = torch.nn.Parameter(torch.ones(model_size, device='cpu', dtype=torch.float32))
    ref_param = torch.nn.Parameter(param.detach().clone())
    optimizer_kwargs = dict(lr=0.01, eps=1e-8, weight_decay=weight_decay)
    optimizer = DeepSpeedCPUAdagrad([param], **optimizer_kwargs)
    ref_optimizer = torch.optim.Adagrad([ref_param], **optimizer_kwargs)

    for grad_value in (0.25, 0.0, -0.5):
        param.grad = torch.full_like(param, grad_value)
        ref_param.grad = param.grad.clone()
        optimizer.step()
        ref_optimizer.step()

        torch.testing.assert_close(param, ref_param, rtol=1e-6, atol=1e-7)


class TestCPUAdagrad(DistributedTest):
    world_size = 1
    requires_cuda_env = False
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    def test_cpu_adagrad_opt(self, model_size=64):
        device = 'cpu'
        rng_state = torch.get_rng_state()
        param = torch.nn.Parameter(torch.randn(model_size, device=device))
        torch.set_rng_state(rng_state)
        param1 = torch.nn.Parameter(torch.randn(model_size, device=device))
        torch.set_rng_state(rng_state)

        optimizer = DeepSpeedCPUAdagrad([param])
        optimizer1 = torch.optim.Adagrad([param1])

        for i in range(10):
            rng_state = torch.get_rng_state()
            param.grad = torch.randn(model_size, device=device)
            torch.set_rng_state(rng_state)
            param1.grad = torch.randn(model_size, device=device)
            optimizer.step()
            optimizer1.step()

        check_equal(param, param1, atol=1e-2, verbose=True)

    def test_cpu_adagrad_opt_sparse_embedding(self, model_size=32, vocabulary_size=64, dim=16):
        device = 'cpu'
        rng_state = torch.get_rng_state()

        def gen_sparse_grad(vocabulary_size, dim, num_indices, dtype, device):
            i = torch.randint(vocabulary_size, size=(1, num_indices), dtype=torch.int64, device=device)
            v = torch.randn(num_indices, dim, dtype=dtype, device=device)
            t = torch.sparse_coo_tensor(i, v, (vocabulary_size, dim), device=device)
            t = t.coalesce()
            new_i = (t.indices().view(-1, 1).repeat(1, dim) * dim + torch.tensor(range(dim))).flatten().unsqueeze(0)
            new_v = t.values().flatten()
            new_t = torch.sparse_coo_tensor(new_i, new_v, (vocabulary_size * dim, ), device=device)
            new_t = new_t.coalesce()
            new_t.requires_grad = False
            return new_t

        voc_size = vocabulary_size
        dim = dim
        num_indices = int(model_size // dim)
        dtype = torch.float32

        param = torch.nn.Parameter(torch.randn((voc_size * dim, ), dtype=dtype, device=device), requires_grad=True)
        torch.set_rng_state(rng_state)
        param1 = torch.nn.Parameter(torch.randn((voc_size * dim, ), dtype=dtype, device=device), requires_grad=True)
        torch.set_rng_state(rng_state)

        optimizer = DeepSpeedCPUAdagrad([param])
        optimizer1 = torch.optim.Adagrad([param1])

        for i in range(10):
            torch.set_rng_state(rng_state)
            param.grad = gen_sparse_grad(voc_size, dim, num_indices, dtype=dtype, device=device)
            torch.set_rng_state(rng_state)
            param1.grad = gen_sparse_grad(voc_size, dim, num_indices, dtype=dtype, device=device)
            optimizer.step()
            optimizer1.step()

        check_equal(param, param1, atol=1e-2, verbose=True)


class TestCPUAdagradGPUError(DistributedTest):

    def test_cpu_adagrad_gpu_error(self):
        model_size = 64
        device = get_accelerator().device_name(0)  # 'cuda:0' or 'xpu:0'
        param = torch.nn.Parameter(torch.randn(model_size, device=device))
        optimizer = DeepSpeedCPUAdagrad([param])

        param.grad = torch.randn(model_size, device=device)
        with pytest.raises(AssertionError):
            optimizer.step()

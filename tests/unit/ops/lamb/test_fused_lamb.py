# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.lamb import FusedLamb
from deepspeed.ops.op_builder import FusedLambBuilder

if not deepspeed.ops.__compatible_ops__[FusedLambBuilder.NAME]:
    pytest.skip("FusedLamb is not compatible", allow_module_level=True)


def reference_lamb_step(p, m, v, g, step, lr, beta1, beta2, eps, weight_decay, eps_inside_sqrt, min_coeff, max_coeff):
    # LAMB, Algorithm 2 of https://arxiv.org/abs/1904.00962: bias-correct m and v, then
    # x -= lr * clamp(||x|| / ||r + wd * x||) * (r + wd * x).
    m.mul_(beta1).add_(g, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
    m_hat = m / (1 - beta1**step)
    v_hat = v / (1 - beta2**step)
    denom = (v_hat + eps).sqrt() if eps_inside_sqrt else v_hat.sqrt() + eps
    update = m_hat / denom + weight_decay * p
    coeff = (p.norm() / update.norm()).clamp(min_coeff, max_coeff)
    p.sub_(lr * coeff * update)


@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
@pytest.mark.parametrize("eps_inside_sqrt", [False, True])
def test_fused_lamb_matches_reference(weight_decay, eps_inside_sqrt):
    device = get_accelerator().device_name()
    lr, betas, eps = 1e-2, (0.9, 0.999), 1e-6
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(1000, device=device))
    ref = param.detach().clone()
    m, v = torch.zeros_like(ref), torch.zeros_like(ref)
    optimizer = FusedLamb([param],
                          lr=lr,
                          betas=betas,
                          eps=eps,
                          eps_inside_sqrt=eps_inside_sqrt,
                          weight_decay=weight_decay)

    for step in range(1, 6):
        param.grad = torch.randn_like(ref)
        reference_lamb_step(ref, m, v, param.grad, step, lr, *betas, eps, weight_decay, eps_inside_sqrt, 0.01, 10.0)
        optimizer.step()
        torch.testing.assert_close(param.detach(), ref, rtol=1e-5, atol=1e-6)

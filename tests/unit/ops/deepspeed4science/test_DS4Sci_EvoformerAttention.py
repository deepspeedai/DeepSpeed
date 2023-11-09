# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List

import pytest
import torch
from torch.nn import functional as F
import deepspeed
from deepspeed.ops.op_builder import EvoformerAttnBuilder
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
from deepspeed.accelerator import get_accelerator
from unit.util import skip_on_arch

if not deepspeed.ops.__compatible_ops__[EvoformerAttnBuilder.NAME]:
    pytest.skip("DS4Sci_EvoformerAttention ops are not available on this system", allow_module_level=True)


def attention_reference(
        q_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
        k_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
        v_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
        biases: List[torch.Tensor],
        sm_scale: float) -> torch.Tensor:
    q = q_input.transpose(-2, -3)
    k = k_input.transpose(-2, -3)
    v = v_input.transpose(-2, -3)
    k_t = k.transpose(-1, -2)
    a = torch.matmul(q, k_t) * sm_scale

    for b in biases:
        a += b

    a = F.softmax(a, dim=-1)
    a_v = torch.matmul(a, v)
    o = a_v.transpose(-2, -3)

    return o


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tensor_shape", [(1, 256, 256, 4, 32), (1, 512, 256, 8, 8)])
def test_DS4Sci_EvoformerAttention(dtype, tensor_shape):
    skip_on_arch(8 if dtype == torch.bfloat16 else 7)
    batch, n, seq_len, heads, dim = tensor_shape
    Q = torch.randn(batch,
                    n,
                    seq_len,
                    heads,
                    dim,
                    dtype=dtype,
                    device=get_accelerator().device_name(),
                    requires_grad=True)
    K = torch.randn(batch,
                    n,
                    seq_len,
                    heads,
                    dim,
                    dtype=dtype,
                    device=get_accelerator().device_name(),
                    requires_grad=True)
    V = torch.randn(batch,
                    n,
                    seq_len,
                    heads,
                    dim,
                    dtype=dtype,
                    device=get_accelerator().device_name(),
                    requires_grad=True)
    mask = torch.randint(0, 2, (batch, n, 1, 1, seq_len), dtype=dtype, device=get_accelerator().device_name())
    mask_bias = 1e9 * (mask - 1)
    bias = torch.randn(batch,
                       1,
                       heads,
                       seq_len,
                       seq_len,
                       dtype=dtype,
                       device=get_accelerator().device_name(),
                       requires_grad=True)
    dummy_out = torch.rand_like(Q, dtype=dtype, device=get_accelerator().device_name())
    ref_out = attention_reference(Q, K, V, [mask_bias, bias], 1 / (dim**0.5))
    ref_out.backward(dummy_out)
    ref_dv, V.grad = V.grad.clone(), None
    ref_dk, K.grad = K.grad.clone(), None
    ref_dq, Q.grad = Q.grad.clone(), None
    ref_db, bias.grad = bias.grad.clone(), None

    out = DS4Sci_EvoformerAttention(Q, K, V, [mask_bias, bias])
    out.backward(dummy_out)
    dv, v_grad = V.grad.clone(), None
    dk, k_grad = K.grad.clone(), None
    dq, q_grad = Q.grad.clone(), None
    db, bias.grad = bias.grad.clone(), None

    eps = 1e-2 if dtype == torch.float16 else 5e-2

    assert torch.max(torch.abs(ref_out - out)).item() < eps, f"out eps: {torch.max(torch.abs(ref_out - out))}"
    assert torch.max(torch.abs(ref_dv - dv)) < eps, f"dv eps: {torch.max(torch.abs(ref_dv - dv))}"
    assert torch.max(torch.abs(ref_dk - dk)) < eps, f"dk eps: {torch.max(torch.abs(ref_dk - dk))}"
    assert torch.max(torch.abs(ref_dq - dq)) < eps, f"dq eps: {torch.max(torch.abs(ref_dq - dq))}"
    assert torch.max(torch.abs(ref_db - db)) < 2 * eps, f"db eps: {torch.max(torch.abs(ref_db - db))}"

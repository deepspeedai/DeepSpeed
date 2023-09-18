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
    bias1 = torch.randn(batch,
                        n,
                        1,
                        1,
                        seq_len,
                        dtype=dtype,
                        device=get_accelerator().device_name(),
                        requires_grad=True)
    bias2 = torch.randn(batch,
                        1,
                        heads,
                        seq_len,
                        seq_len,
                        dtype=dtype,
                        device=get_accelerator().device_name(),
                        requires_grad=True)
    dummy_out = torch.rand_like(Q, dtype=dtype, device=get_accelerator().device_name())
    ref_out = attention_reference(Q, K, V, [bias1, bias2], 1 / (dim**0.5))
    ref_out.backward(dummy_out)
    ref_dv, V.grad = V.grad.clone(), None
    ref_dk, K.grad = K.grad.clone(), None
    ref_dq, Q.grad = Q.grad.clone(), None
    ref_db1, bias1.grad = bias1.grad.clone(), None
    ref_db2, bias2.grad = bias2.grad.clone(), None

    out = DS4Sci_EvoformerAttention(Q, K, V, [bias1, bias2])
    out.backward(dummy_out)
    dv, v_grad = V.grad.clone(), None
    dk, k_grad = K.grad.clone(), None
    dq, q_grad = Q.grad.clone(), None
    db1, bias1.grad = bias1.grad.clone(), None
    db2, bias2.grad = bias2.grad.clone(), None

    assert torch.allclose(ref_out, out, atol=2e-2, rtol=0), f"\n{ref_out} \n {out}"
    assert torch.allclose(ref_dv, dv, atol=2e-2, rtol=0), f"\n{ref_dv} \n {dv}"
    assert torch.allclose(ref_dk, dk, atol=2e-2, rtol=0), f"\n{ref_dk} \n {dk}"
    assert torch.allclose(ref_dq, dq, atol=2e-2, rtol=0), f"\n{ref_dq} \n {dq}"
    assert torch.allclose(ref_db1, db1, atol=2e-2, rtol=1e-2), f"{ref_db1} \n {db1}"
    assert torch.allclose(ref_db2, db2, atol=2e-2, rtol=1e-2), f"{ref_db2} \n {db2}"

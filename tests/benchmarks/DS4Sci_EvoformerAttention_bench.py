# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
This script is to test the performance of the DS4Sci_EvoformerAttention op.
To run the script,
1. Clone the CUTLASS repo. E.g. git clone https://github.com/NVIDIA/cutlass.git
2. Specify the CUTLASS_PATH environment variable. E.g. export CUTLASS_PATH=$(pwd)/cutlass
3. Run the script. E.g. python DS4Sci_EvoformerAttention_bench.py
"""

import contextlib
import torch
from typing import List
from torch.nn import functional as F
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
from deepspeed.accelerator import get_accelerator


def attention_reference(
        q_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
        k_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
        v_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
        biases: List[torch.Tensor],
        sm_scale: float) -> torch.Tensor:
    # Original shape: [*, Dim_Q, H, C_hid] -> Transpose to: [*, H, Dim_Q, C_hid]
    q = q_input.transpose(-2, -3)
    k = k_input.transpose(-2, -3)
    v = v_input.transpose(-2, -3)

    # Now, q, k, v are in shape: [*, H, Dim_Q, C_hid]

    # Transpose k to shape [*, H, C_hid, Dim_Q]
    k_t = k.transpose(-1, -2)

    # Now, q and k_t are in shapes: [*, H, Dim_Q, C_hid] and [*, H, C_hid, Dim_Q] respectively

    # [*, H, Dim_Q, Dim_Q]
    a = torch.matmul(q, k_t) * sm_scale

    for b in biases:
        a += b

    a = F.softmax(a, dim=-1)

    # Now, a is in shape [*, H, Dim_Q, Dim_Q], v is in shape [*, H, Dim_Q, C_hid]

    # Matmul operation results in [*, H, Dim_Q, C_hid]
    a_v = torch.matmul(a, v)

    # [*, Dim_Q, H, C_hid]
    o = a_v.transpose(-2, -3)

    return o


dtype = torch.float16

N = 256
heads = 4
dim = 32
seq_len = 256


@contextlib.contextmanager
def cuda_timer(res_list):
    start = get_accelerator().Event(enable_timing=True)
    end = get_accelerator().Event(enable_timing=True)
    start.record()
    yield
    end.record()
    get_accelerator().synchronize()
    res_list.append(start.elapsed_time(end))


def benchmark():
    ours_fw = []
    ours_bw = []
    baseline_fw = []
    baseline_bw = []
    for batch in range(1, 17):
        Q = torch.randn(batch, N, seq_len, heads, dim, dtype=dtype, device="cuda", requires_grad=True)
        K = torch.randn(batch, N, seq_len, heads, dim, dtype=dtype, device="cuda", requires_grad=True)
        V = torch.randn(batch, N, seq_len, heads, dim, dtype=dtype, device="cuda", requires_grad=True)
        bias1 = torch.randn(batch, N, 1, 1, seq_len, dtype=dtype, device="cuda", requires_grad=False)
        bias2 = torch.randn(batch, 1, heads, seq_len, seq_len, dtype=dtype, device="cuda", requires_grad=True)
        # warm up
        DS4Sci_EvoformerAttention(Q, K, V, [bias1, bias2])
        with cuda_timer(ours_fw):
            out = DS4Sci_EvoformerAttention(Q, K, V, [bias1, bias2])
        d_out = torch.rand_like(out)
        with cuda_timer(ours_bw):
            out.backward(d_out)
        # warm up
        attention_reference(Q, K, V, [bias1, bias2], 1 / (dim**0.5))
        with cuda_timer(baseline_fw):
            ref_out = attention_reference(Q, K, V, [bias1, bias2], 1 / (dim**0.5))
        with cuda_timer(baseline_bw):
            ref_out.backward(d_out)

    print(f"batch size\tours (FW)\tbaseline (FW)\tours (BW)\tbaseline (BW)")
    for i in range(len(ours_fw)):
        print(f"{i+1}\t{ours_fw[i]}\t{baseline_fw[i]}\t{ours_bw[i]}\t{baseline_bw[i]}")


benchmark()

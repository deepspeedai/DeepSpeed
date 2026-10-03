# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder
from .inference_test_utils import allclose


# reference timplementation
def ref_torch_attention(q, k, v, mask, sm_scale):
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float() + mask, dim=-1).half()
    ref_out = torch.matmul(p, v)
    return ref_out


# test attention operator
@pytest.mark.inference_ops
@pytest.mark.parametrize("BATCH", [1])  # batch
@pytest.mark.parametrize("H", [12])  # heads
@pytest.mark.parametrize("N_CTX", [16, 128])  # sequence length
@pytest.mark.parametrize("D_HEAD", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("use_flash", [True, False])
def test_attention(BATCH, H, N_CTX, D_HEAD, causal, use_flash, dtype=torch.float16):
    if not deepspeed.get_accelerator().is_triton_supported():
        pytest.skip("triton is not supported on this system")
    if not deepspeed.HAS_TRITON:
        pytest.skip("triton is not installed")

    minus_inf = -65504.0
    dev = deepspeed.accelerator.get_accelerator().device_name()
    # skip autotune in testing
    from deepspeed.ops.transformer.inference.triton.matmul_ext import fp16_matmul
    fp16_matmul.skip_autotune()

    from deepspeed.ops.transformer.inference.triton.attention import _triton_attention, _triton_packed_flash
    torch.manual_seed(20)
    q = torch.empty((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=dev).normal_(mean=0, std=.5)
    k = torch.empty((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=dev).normal_(mean=0, std=.5)
    v = torch.empty((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=dev).normal_(mean=0, std=.5)
    sm_scale = 0.3

    # reference implementation
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    score = p
    mask = torch.zeros((BATCH, H, N_CTX, N_CTX), dtype=dtype, device=dev)
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=dev))
    if causal:
        for z in range(BATCH):
            for h in range(H):
                mask[:, :, M == 0] = minus_inf
    p = torch.softmax(p.float() + mask, dim=-1).half()
    softmax_out = p
    ref_out = torch.matmul(p, v)
    context = ref_out

    # adjust it to expected tensor format and run test
    qkv = torch.randn((BATCH, N_CTX, 3 * H * D_HEAD), dtype=dtype, device=dev, requires_grad=False)
    qkv[:, :, :H * D_HEAD] = q.permute(0, 2, 1, 3).contiguous().reshape((BATCH, N_CTX, H * D_HEAD))
    qkv[:, :, 1 * H * D_HEAD:2 * H * D_HEAD] = k.permute(0, 2, 1, 3).contiguous().reshape((BATCH, N_CTX, H * D_HEAD))
    qkv[:, :, 2 * H * D_HEAD:] = v.permute(0, 2, 1, 3).contiguous().reshape((BATCH, N_CTX, H * D_HEAD))

    if use_flash:
        if not get_accelerator().is_triton_supported():
            pytest.skip("triton flash attention is supported when the compute capability > 8.0")
        triton_mask = torch.zeros((BATCH, 1, 1, N_CTX), dtype=dtype, device=dev)
        if not causal:
            lengths = torch.randint(N_CTX - 8, N_CTX, (BATCH, 1), device=dev)
            for i, l in enumerate(lengths):
                triton_mask[i, ..., l:] = minus_inf
            mask = torch.zeros((BATCH, H, N_CTX, N_CTX), dtype=dtype, device=dev)
            for b in range(BATCH):
                mask[b, :, :, lengths[b]:] = minus_inf
            ref_out = ref_torch_attention(q, k, v, mask, sm_scale)
        tri_out = _triton_packed_flash(qkv, D_HEAD, triton_mask, sm_scale, causal=causal, add_mask=(not causal))
    else:
        tri_out = _triton_attention(qkv,
                                    input_mask=mask,
                                    layer_past=None,
                                    alibi=None,
                                    scale=sm_scale,
                                    head_size=D_HEAD,
                                    triangular=False,
                                    use_cuda_flash=False,
                                    use_triton_flash=False,
                                    use_ds_attention=False)
    tri_out = tri_out.reshape((BATCH, N_CTX, H, D_HEAD)).permute(0, 2, 1, 3)
    assert (allclose(ref_out, tri_out))


# ds_attention passes 1/norm_factor where norm_factor = sqrt(sqrt(head_dim)) in the default
# configuration, so the binding must square it to reach the 1/sqrt(head_dim) softmax scale.
# This legacy binding derives seq_len from query.size(1), so it is only addressable with a
# single head; the int8 KV cache also stores keys transposed ([batch, head_dim, seq]).
def run_softmax_context_int8_reference(q, k, v, norm_factor):
    softmax_scale = norm_factor * norm_factor
    scores = torch.matmul(q, k.transpose(2, 3)) * softmax_scale
    probs = torch.softmax(scores.float(), dim=-1).half()
    return torch.matmul(probs, v)


@pytest.mark.inference_ops
@pytest.mark.parametrize("BATCH", [1])
@pytest.mark.parametrize("H", [1])
@pytest.mark.parametrize("N_CTX", [16, 128])
@pytest.mark.parametrize("D_HEAD", [64, 128, 256])
def test_softmax_context_int8(BATCH, H, N_CTX, D_HEAD):
    if not torch.cuda.is_available():
        pytest.skip("softmax_context_int8 requires CUDA")

    device = deepspeed.accelerator.get_accelerator().device_name()
    inference_module = InferenceBuilder().load()
    torch.manual_seed(20)

    head_dim_4th_root = math.sqrt(math.sqrt(D_HEAD))
    norm_factor = 1.0 / head_dim_4th_root

    q = torch.empty((BATCH, H, N_CTX, D_HEAD), dtype=torch.float16, device=device).normal_(mean=0, std=0.5)
    k = torch.empty((BATCH, H, N_CTX, D_HEAD), dtype=torch.float16, device=device).normal_(mean=0, std=0.5)
    v = torch.empty((BATCH, H, N_CTX, D_HEAD), dtype=torch.float16, device=device).normal_(mean=0, std=0.5)
    ref_out = run_softmax_context_int8_reference(q, k, v, norm_factor)

    query = q.permute(0, 2, 1, 3).contiguous().reshape((BATCH, N_CTX, H * D_HEAD))
    # keys are consumed in transposed cache layout, values in [batch, seq, head_dim] layout
    prev_key = k.transpose(-1, -2).contiguous()
    prev_value = v.permute(0, 2, 1, 3).contiguous().reshape((BATCH, N_CTX, H * D_HEAD))

    ds_out = inference_module.softmax_context_int8(query, prev_key, torch.empty(1, device=device),
                                                   torch.empty(1, device=device), prev_value,
                                                   torch.empty(1, device=device), H, norm_factor, False, False, False,
                                                   -1, True)[0]

    ds_out = ds_out.reshape((BATCH, H, N_CTX, D_HEAD))
    assert allclose(ref_out, ds_out)

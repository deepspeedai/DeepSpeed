# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed


# reference timplementation
def ref_torch_attention(q, k, v, mask, sm_scale):
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float() + mask, dim=-1).half()
    ref_out = torch.matmul(p, v)
    return ref_out


# test attention operator
@pytest.mark.inference_ops
@pytest.mark.parametrize("Z", [1])  # batch
@pytest.mark.parametrize("H", [12])  # heads
@pytest.mark.parametrize("N_CTX", [4, 128])  # sequence length
@pytest.mark.parametrize("D_HEAD", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_attention(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):
    if not deepspeed.HAS_TRITON:
        pytest.skip("triton has to be installed for the test")

    # skip autotune in testing
    from deepspeed.ops.transformer.inference.triton.matmul_ext import fp16_matmul
    fp16_matmul.skip_autotune()

    import triton
    from deepspeed.ops.transformer.inference.triton.attention import compute_attention
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5)
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5)
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5)
    sm_scale = 0.3

    # reference implementation
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    score = p
    mask = torch.zeros((Z, H, N_CTX, N_CTX), dtype=dtype, device="cuda")
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    if causal:
        for z in range(Z):
            for h in range(H):
                mask[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float() + mask, dim=-1).half()
    softmax_out = p
    ref_out = torch.matmul(p, v)
    context = ref_out

    # adjust it to expected tensor format and run test
    qkv = torch.randn((Z, N_CTX, 3 * H * D_HEAD), dtype=dtype, device='cuda', requires_grad=False)
    qkv[:, :, :H * D_HEAD] = q.permute(0, 2, 1, 3).contiguous().reshape((Z, N_CTX, H * D_HEAD))
    qkv[:, :, 1 * H * D_HEAD:2 * H * D_HEAD] = k.permute(0, 2, 1, 3).contiguous().reshape((Z, N_CTX, H * D_HEAD))
    qkv[:, :, 2 * H * D_HEAD:] = v.permute(0, 2, 1, 3).contiguous().reshape((Z, N_CTX, H * D_HEAD))
    tri_out = compute_attention(qkv,
                                input_mask=mask,
                                layer_past=None,
                                alibi=None,
                                scale=sm_scale,
                                head_size=D_HEAD,
                                triangular=False,
                                use_cuda_flash=False,
                                use_triton_flash=False,
                                use_ds_attention=False)
    tri_out = tri_out.reshape((Z, N_CTX, H, D_HEAD)).permute(0, 2, 1, 3)
    triton.testing.allclose(ref_out, tri_out)
    triton.testing.assert_almost_equal(ref_out, tri_out)

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.ops.op_builder import InferenceBuilder
from deepspeed.accelerator import get_accelerator

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system", allow_module_level=True)


@pytest.mark.inference_ops
@pytest.mark.parametrize("num_heads", [64, 32, 16, 8])
def test_rope_warp_size_alignment(num_heads):
    if get_accelerator().device_name() != "cuda":
        pytest.skip("This test runs only on GPU")

    batch = 1
    head = 8
    seq_len = 1024
    head_dim = 32
    rotary_dim = 32
    offset = 8
    rotate_half = False
    rope_theta = 2

    cuda0 = torch.device('cuda:0')
    query = torch.randn(batch, head, seq_len, head_dim, device=cuda0)
    key = torch.randn(batch, head, seq_len, head_dim, device=cuda0)

    inference = InferenceBuilder().load()
    # For num_heads values of 64, 32, 16, 8
    # corresponding threads_per_head (defined in apply_rotary_pos_emb.cu) values are 4, 8, 16, 32
    inference.apply_rotary_pos_emb(query, key, rotary_dim, offset, num_heads, rotate_half, rope_theta)

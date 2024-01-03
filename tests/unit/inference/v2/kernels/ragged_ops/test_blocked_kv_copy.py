# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.kernels.ragged_ops import LinearBlockedKVCopy
from .ragged_testing_utils import build_batch_and_manager, validate_kv_cache


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("n_tokens, history_size", [(1, 0), (17, 0), (33, 8), (63, 1)])
@pytest.mark.parametrize("head_size", [64, 80, 128])
def test_single_sequence_single_block(n_tokens: int, history_size: int, head_size: int):
    """
    Validate that the copy works correctly
    """
    n_heads_q = 16
    n_heads_kv = 16
    kv_block_size = 64
    device = get_accelerator().current_device()

    batch, state_manager, seq_descs = build_batch_and_manager([(n_tokens, history_size)], head_size, n_heads_kv,
                                                              kv_block_size)

    assert batch.current_sequences == 1
    assert batch.current_tokens == n_tokens

    qkv = torch.randn((batch.current_tokens, (n_heads_q + 2 * n_heads_kv) * head_size),
                      device=device,
                      dtype=torch.float16)

    kv_cache = state_manager.get_cache(0)

    copy_impl = LinearBlockedKVCopy(head_size, n_heads_q, n_heads_kv, torch.float16)
    copy_impl(kv_cache, qkv, batch)

    k = qkv[:, head_size * n_heads_q:head_size * (n_heads_q + n_heads_kv)]
    v = qkv[:, head_size * (n_heads_q + n_heads_kv):]

    validate_kv_cache(kv_cache, k, v, seq_descs, batch)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("n_tokens, history_size", [(128, 0), (177, 0), (169, 8), (117, 88)])
@pytest.mark.parametrize("head_size", [64, 80, 128])
def test_single_sequence_multiple_blocks(n_tokens: int, history_size: int, head_size: int):
    """
    Validate that the copy works correctly
    """
    n_heads_q = 16
    n_heads_kv = 16
    kv_block_size = 64
    device = get_accelerator().current_device()

    batch, state_manager, seq_descs = build_batch_and_manager([(n_tokens, history_size)], head_size, n_heads_kv,
                                                              kv_block_size)

    assert batch.current_sequences == 1
    assert batch.current_tokens == n_tokens

    qkv = torch.randn((batch.current_tokens, (n_heads_q + 2 * n_heads_kv) * head_size),
                      device=device,
                      dtype=torch.float16)

    kv_cache = state_manager.get_cache(0)

    copy_impl = LinearBlockedKVCopy(head_size, n_heads_q, n_heads_kv, torch.float16)
    copy_impl(kv_cache, qkv, batch)

    k = qkv[:, head_size * n_heads_q:head_size * (n_heads_q + n_heads_kv)]
    v = qkv[:, head_size * (n_heads_q + n_heads_kv):]

    validate_kv_cache(kv_cache, k, v, seq_descs, batch)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("head_size", [64, 80, 128])
def test_multi_sequence(head_size: int) -> None:
    n_heads_q = 16
    n_heads_kv = 16
    kv_block_size = 64
    device = get_accelerator().current_device()

    batch_config = [
        (128, 0),
        (177, 0),
        (169, 8),
        (117, 88),
        (1, 293),
        (1, 733),
        (1, 33),
    ]

    batch, state_manager, seq_descs = build_batch_and_manager(batch_config, head_size, n_heads_kv, kv_block_size)

    qkv = torch.randn((batch.current_tokens, (n_heads_q + 2 * n_heads_kv) * head_size),
                      device=device,
                      dtype=torch.float16)

    kv_cache = state_manager.get_cache(0)

    copy_impl = LinearBlockedKVCopy(head_size, n_heads_q, n_heads_kv, torch.float16)
    copy_impl(kv_cache, qkv, batch)

    k = qkv[:, head_size * n_heads_q:head_size * (n_heads_q + n_heads_kv)]
    v = qkv[:, head_size * (n_heads_q + n_heads_kv):]

    validate_kv_cache(kv_cache, k, v, seq_descs, batch)

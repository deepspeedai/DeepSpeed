# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.model_implementations.sharding import *

# None of the logic should be dependent on head size.
HEAD_SIZE = 64


def fill_with_head_ids(head_size: int, n_heads: int) -> torch.Tensor:
    """
    Fills a tensor with the associated head ids. All columns should have the same value.
    """
    head_ids = torch.arange(n_heads, dtype=torch.half, device=get_accelerator().current_device())

    head_ids = head_ids.repeat_interleave(head_size).repeat(head_size * n_heads).reshape(n_heads * head_size, -1)
    return head_ids


@pytest.mark.inference_v2
@pytest.mark.parametrize("n_heads, n_shards", [(1, 1), (8, 4), (32, 8)])
def test_mha_even_sharding(n_heads: int, n_shards: int):
    """
    Even head sharding for MHA.

    Args:
        n_heads (int): The number QKV heads.
        n_shards (int): The number of shards to test for.
    """
    param = fill_with_head_ids(HEAD_SIZE, n_heads)

    n_local_heads = n_heads // n_shards
    sharded_shape = (HEAD_SIZE * n_heads, HEAD_SIZE * n_local_heads)

    for shard_rank in range(n_shards):
        sharded_param = shard_attn_out_param(param, shard_rank, n_shards, HEAD_SIZE)
        n_heads_local_q, _ = get_local_heads(shard_rank, n_shards, n_heads)

        assert sharded_param.shape[-1] == HEAD_SIZE * n_heads_local_q
        assert sharded_param.shape == sharded_shape

        heads = torch.chunk(sharded_param, n_local_heads, dim=1)

        for i, head in enumerate(heads):
            assert torch.all(head == i + shard_rank * n_local_heads)


@pytest.mark.inference_v2
@pytest.mark.parametrize("n_heads, n_shards", [(3, 2), (20, 8)])
def test_mha_unbalanced_sharding(n_heads: int, n_shards: int):
    """
    Unbalanced head sharding for MHA.

    Args:
        n_heads (int): The number QKV heads.
        n_shards (int): The number of shards to test for.
    """
    param = fill_with_head_ids(HEAD_SIZE, n_heads)

    max_heads = 0
    min_heads = n_heads
    seen_heads = set()
    total_heads = 0

    for shard_rank in range(n_shards):
        sharded_param = shard_attn_out_param(param, shard_rank, n_shards, HEAD_SIZE)
        n_heads_local_q, _ = get_local_heads(shard_rank, n_shards, n_heads)

        assert sharded_param.shape[-1] == HEAD_SIZE * n_heads_local_q

        n_local_heads = sharded_param.shape[1] // HEAD_SIZE
        total_heads += n_local_heads
        max_heads = max(max_heads, n_local_heads)
        min_heads = min(min_heads, n_local_heads)

        for i in range(n_local_heads):
            head_ids = torch.unique_consecutive(sharded_param[:, i * HEAD_SIZE:(i + 1) * HEAD_SIZE])
            assert len(head_ids) == 1
            seen_heads.add(head_ids.item())

    assert max_heads == min_heads + 1
    assert total_heads == n_heads
    assert len(seen_heads) == n_heads


@pytest.mark.inference_v2
@pytest.mark.parametrize("n_heads_q, n_heads_kv, n_shards", [(20, 4, 8)])
def test_gqa_uneven_sharding(n_heads_q: int, n_heads_kv: int, n_shards: int):
    """
    We only test the uneven GQA test case because even GQA shards the attention output
    in the exact same manner as MHA.

    Args:
        n_heads_q (int): The number of query heads.
        n_heads_kv (int): The number of key/value heads.
        n_shards (int): The number of shards to test for.
    """
    param = fill_with_head_ids(HEAD_SIZE, n_heads_q)

    min_heads = n_heads_q
    max_heads = 0
    seen_heads = set()
    total_heads = 0

    for shard_rank in range(n_shards):
        sharded_param = shard_attn_out_param(param, shard_rank, n_shards, HEAD_SIZE, n_heads_q, n_heads_kv)
        n_heads_local_q, _ = get_local_heads(shard_rank, n_shards, n_heads_q, n_heads_kv)

        assert sharded_param.shape[-1] == HEAD_SIZE * n_heads_local_q

        n_local_heads = sharded_param.shape[1] // HEAD_SIZE
        total_heads += n_local_heads
        max_heads = max(max_heads, n_local_heads)
        min_heads = min(min_heads, n_local_heads)

        for i in range(n_local_heads):
            head_id = torch.unique_consecutive(sharded_param[:, i * HEAD_SIZE:(i + 1) * HEAD_SIZE])
            assert len(head_id) == 1
            seen_heads.add(head_id.item())

    assert max_heads == min_heads + 1
    assert total_heads == n_heads_q
    assert len(seen_heads) == n_heads_q

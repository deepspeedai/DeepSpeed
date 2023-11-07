# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.model_implementations.sharding import *


def fill_with_head_ids(head_size: int, n_heads_q: int, n_heads_kv: Optional[int] = None) -> torch.Tensor:
    """

    """
    head_ids_q = torch.arange(n_heads_q, dtype=torch.half, device=get_accelerator().current_device())
    head_vals_q = head_ids_q.repeat_interleave(head_size * head_size * n_heads_q).reshape(n_heads_q * head_size, -1)

    if n_heads_kv is None:
        return torch.cat([head_vals_q, head_vals_q, head_vals_q], dim=0)

    head_ids_k = torch.arange(n_heads_kv, dtype=torch.half, device=get_accelerator().current_device())
    head_vals_k = head_ids_k.repeat_interleave(head_size * head_size * n_heads_q).reshape(n_heads_kv * head_size, -1)

    return torch.cat([head_vals_q, head_vals_k, head_vals_k], dim=0)


def validate_inferred_shape(shard: torch.Tensor, head_size: int, n_local_q_heads: int, n_local_kv_heads: int):
    """
    Validate that the leading dim of the shard is of the expected size and aligns with the sharding
    logic for the attention computation itself.
    """
    inferred_leading_dim = head_size * (n_local_q_heads + 2 * n_local_kv_heads)
    assert shard.shape[0] == inferred_leading_dim


@pytest.mark.inference_v2
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("n_heads,n_shards", [(1, 1), (32, 1), (32, 8)])
def test_even_mha_sharding(head_size: int, n_heads: int, n_shards: int):
    """
    Test for MHA sharding. In these scenarios, we expect that each of the shards
    should be the same size.
    """
    param = fill_with_head_ids(head_size, n_heads)

    heads_per_shard = n_heads // n_shards

    for shard_rank in range(n_shards):

        shard = shard_qkv_param(param, shard_rank, n_shards, head_size, n_heads, n_heads)
        n_local_q_heads, n_local_kv_heads = get_local_heads(shard_rank, n_shards, n_heads, n_heads)
        validate_inferred_shape(shard, head_size, n_local_q_heads, n_local_kv_heads)

        assert shard.shape == (3 * head_size * heads_per_shard, head_size * n_heads)

        heads = shard.chunk(heads_per_shard * 3, dim=0)
        for i in range(heads_per_shard):
            assert torch.all(heads[i] == i + shard_rank * heads_per_shard)
            assert torch.all(heads[i + heads_per_shard] == i + shard_rank * heads_per_shard)
            assert torch.all(heads[i + heads_per_shard * 2] == i + shard_rank * heads_per_shard)


@pytest.mark.inference_v2
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("n_heads, n_shards", [(3, 2), (20, 8)])
def test_unbalanced_mha_sharding(head_size: int, n_heads: int, n_shards: int):
    """
    Test MHA sharding when the distribution of heads will not be equal across all ranks.
    """
    param = fill_with_head_ids(head_size, n_heads)

    max_heads = 0
    min_heads = n_heads
    total_heads = 0
    seen_heads = set()

    for shard_rank in range(n_shards):
        shard = shard_qkv_param(param, shard_rank, n_shards, head_size, n_heads, n_heads)
        n_local_q_heads, n_local_kv_heads = get_local_heads(shard_rank, n_shards, n_heads, n_heads)
        validate_inferred_shape(shard, head_size, n_local_q_heads, n_local_kv_heads)

        n_heads_in_shard = shard.shape[0] // head_size // 3

        max_heads = max(max_heads, n_heads_in_shard)
        min_heads = min(min_heads, n_heads_in_shard)
        total_heads += n_heads_in_shard

        heads = shard.chunk(n_heads_in_shard * 3, dim=0)

        for local_head_id in range(n_heads_in_shard):
            head_qkv = torch.cat([
                heads[local_head_id], heads[local_head_id + n_heads_in_shard],
                heads[local_head_id + 2 * n_heads_in_shard]
            ],
                                 dim=0)
            assert head_qkv.shape == (3 * head_size, head_size * n_heads)

            global_head_id = torch.unique_consecutive(head_qkv)
            assert len(global_head_id) == 1

            seen_heads.add(global_head_id.item())

    assert max_heads - min_heads <= 1
    assert total_heads == n_heads
    assert len(seen_heads) == n_heads


@pytest.mark.inference_v2
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("n_heads_q, n_heads_kv, n_shards", [(4, 2, 1), (8, 2, 1), (64, 16, 8)])
def test_gqa_even_sharding(head_size: int, n_heads_q: int, n_heads_kv: int, n_shards: int):
    """
    Test GQA sharding when the KV heads are evenly divisible by the number of shards.
    """
    param = fill_with_head_ids(head_size, n_heads_q, n_heads_kv)

    n_kv_heads_in_shard = n_heads_kv // n_shards
    n_q_heads_in_shard = n_heads_q // n_shards

    for shard_rank in range(n_shards):
        shard = shard_qkv_param(param, shard_rank, n_shards, head_size, n_heads_q, n_heads_kv)
        n_local_q_heads, n_local_kv_heads = get_local_heads(shard_rank, n_shards, n_heads_q, n_heads_kv)
        validate_inferred_shape(shard, head_size, n_local_q_heads, n_local_kv_heads)

        assert shard.shape[0] == (n_q_heads_in_shard + n_kv_heads_in_shard * 2) * head_size

        q = shard[:n_q_heads_in_shard * head_size]
        k = shard[n_q_heads_in_shard * head_size:(n_q_heads_in_shard + n_kv_heads_in_shard) * head_size]
        v = shard[(n_q_heads_in_shard + n_kv_heads_in_shard) * head_size:]

        for local_head_id in range(n_q_heads_in_shard):
            assert torch.all(q[local_head_id * head_size:(local_head_id + 1) * head_size] == local_head_id +
                             shard_rank * n_q_heads_in_shard)

        for local_head_id in range(n_kv_heads_in_shard):
            assert torch.all(k[local_head_id * head_size:(local_head_id + 1) * head_size] == local_head_id +
                             shard_rank * n_kv_heads_in_shard)
            assert torch.all(v[local_head_id * head_size:(local_head_id + 1) * head_size] == local_head_id +
                             shard_rank * n_kv_heads_in_shard)


@pytest.mark.inference_v2
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("n_heads_q, n_heads_kv, n_shards", [(4, 2, 4), (20, 4, 8)])
def test_gqa_uneven_sharding(head_size: int, n_heads_q: int, n_heads_kv: int, n_shards: int):
    """
    Test GQA sharding when there are more shards than KV heads.
    """
    param = fill_with_head_ids(head_size, n_heads_q, n_heads_kv)

    n_kv_heads_in_shard = 1
    n_shards_per_kv_head = n_shards // n_heads_kv

    max_heads = 0
    min_heads = n_heads_q
    total_heads = 0
    seen_heads = set()

    for shard_rank in range(n_shards):
        shard = shard_qkv_param(param, shard_rank, n_shards, head_size, n_heads_q, n_heads_kv)
        n_local_q_heads, n_local_kv_heads = get_local_heads(shard_rank, n_shards, n_heads_q, n_heads_kv)
        validate_inferred_shape(shard, head_size, n_local_q_heads, n_local_kv_heads)

        local_n_heads_q = (shard.shape[0] - 2 * n_kv_heads_in_shard * head_size) // head_size

        max_heads = max(max_heads, local_n_heads_q)
        min_heads = min(min_heads, local_n_heads_q)
        total_heads += local_n_heads_q

        q = shard[:local_n_heads_q * head_size]
        kv = shard[local_n_heads_q * head_size:]

        for local_head_id in range(local_n_heads_q):
            q_head_id = torch.unique_consecutive(q[local_head_id * head_size:(local_head_id + 1) * head_size])
            assert len(q_head_id) == 1

            seen_heads.add(q_head_id.item())

        kv_id_calc = shard_rank // n_shards_per_kv_head
        kv_id = torch.unique_consecutive(kv)
        assert len(kv_id) == 1
        assert kv_id.item() == kv_id_calc

    assert max_heads - min_heads <= 1
    assert total_heads == n_heads_q
    assert len(seen_heads) == n_heads_q


@pytest.mark.inference_v2
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("n_heads, n_shards", [(6, 8)])
def test_unsupported_mha_configs(head_size: int, n_heads: int, n_shards: int):
    """
    Sharding should fail if there are fewer heads than shards.

    TODO(cmikeh2): Look to support this configuration.
    """
    param = fill_with_head_ids(head_size, n_heads)

    for shard_rank in range(n_shards):
        with pytest.raises(ValueError):
            shard_qkv_param(param, shard_rank, n_shards, head_size, n_heads, n_heads)


@pytest.mark.inference_v2
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("n_heads_q, n_heads_kv, n_shards", [(5, 2, 1), (40, 10, 8), (30, 5, 8)])
def test_unsupported_gqa_configs(head_size: int, n_heads_q: int, n_heads_kv: int, n_shards: int):
    """
    GQA has stricter requirements. We must be able to evenly shard or distribute the KV heads.

    Test cases are to test the following preconditions specifically:
        1. n_heads_q % n_heads_kv == 0
        2. We must be able to evenly distribute KV heads
        3. We must be able to evely split KV heads
    """
    param = fill_with_head_ids(head_size, n_heads_q, n_heads_kv)

    for shard_rank in range(n_shards):
        with pytest.raises(ValueError):
            shard_qkv_param(param, shard_rank, n_shards, head_size, n_heads_q, n_heads_kv)


@pytest.mark.inference_v2
def test_mha_input_shape_error():

    param = torch.empty(256, 128)

    n_heads = 2
    head_size = 64

    with pytest.raises(ValueError):
        shard_qkv_param(param, 0, 1, 64)


@pytest.mark.inference_v2
def test_gqa_input_shape_error():

    head_size = 64
    n_heads_q = 16
    n_heads_kv = 4

    # Correct shape is 1536 (=16 * 64 + 2 * 4 * 64), 1024
    param = torch.empty(2048, 1024)

    with pytest.raises(ValueError):
        shard_qkv_param(param, 0, 1, head_size, n_heads_q, n_heads_kv)

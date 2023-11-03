# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.model_implementations.sharding import *


def round_up_to_256(x: int) -> int:
    """
    Round up to the nearest multiple of 256.
    """
    return x + (256 - x % 256)


def make_params(model_dim: int, ffn_multiplier: int, n_experts: int, gated: bool = False) -> torch.Tensor:
    """

    """
    if gated:
        mlp_1_intermediate = round_up_to_256(int(model_dim * ffn_multiplier * 4 / 3))
        mlp_2_intermediate = mlp_1_intermediate // 2
    else:
        mlp_1_intermediate = ffn_multiplier * model_dim
        mlp_2_intermediate = ffn_multiplier * model_dim

    mlp_1_shared_dim = torch.arange(mlp_1_intermediate, dtype=torch.float32, device=get_accelerator().current_device())

    mlp_1_w = mlp_1_shared_dim.repeat_interleave(model_dim).reshape(mlp_1_intermediate, model_dim)
    mlp_1_b = mlp_1_shared_dim

    mlp_2_shared_dim = torch.arange(mlp_2_intermediate, dtype=torch.float32, device=get_accelerator().current_device())
    mlp_2_w = mlp_2_shared_dim.repeat(model_dim).reshape(model_dim, mlp_2_intermediate)
    mlp_2_b = torch.ones(model_dim, dtype=torch.float32, device=get_accelerator().current_device())

    if n_experts > 1:
        mlp_1_w = mlp_1_w.expand(n_experts, -1, -1)
        mlp_1_b = mlp_1_b.expand(n_experts, -1)
        mlp_2_w = mlp_2_w.expand(n_experts, -1, -1)
        mlp_2_b = mlp_2_b.expand(n_experts, -1)

    return (mlp_1_w, mlp_1_b, mlp_2_w, mlp_2_b)


@pytest.mark.inference_v2
@pytest.mark.parametrize("model_dim, ffn_multiplier, n_shards", [(1024, 4, 1), (1024, 4, 8), (1024, 4, 6)])
@pytest.mark.parametrize("n_experts", [1, 16])
def test_even_ffn_sharding(model_dim: int, ffn_multiplier: int, n_shards: int, n_experts: int):
    """
    FFN sharding tends to be much simpler than attention sharding since it works on larger granularities.
    While the test case of (1024, 4, 6) is not a use case we're likely to see, this does ensure that
    the sharding logic will round correctly for the alignments we care about.
    """
    mlp_1_w, mlp_1_b, mlp_2_w, mlp_2_b = make_params(model_dim, ffn_multiplier, n_experts)

    total_ffn_dim = model_dim * ffn_multiplier
    mapped_neurons = 0

    is_moe = n_experts > 1

    for shard_rank in range(n_shards):
        shard_1_w = shard_mlp_1_param(mlp_1_w, shard_rank, n_shards, is_moe=is_moe)
        shard_1_b = shard_mlp_1_param(mlp_1_b, shard_rank, n_shards, is_moe=is_moe)
        shard_2_w = shard_mlp_2_param(mlp_2_w, shard_rank, n_shards, is_moe=is_moe)
        shard_2_b = shard_mlp_2_param(mlp_2_b, shard_rank, n_shards, is_moe=is_moe)

        assert shard_1_w.shape[-2] == shard_2_w.shape[-1]
        assert shard_1_w.shape[-2] % DEFAULT_SHARD_GRANULARITY == 0
        assert shard_1_w.shape[-2] == shard_1_b.shape[-1]

        mapped_neurons += shard_1_w.shape[-2]

        if shard_rank != 0:
            assert shard_2_b is None
        else:
            assert shard_2_b.shape[-1] == model_dim

    assert mapped_neurons == total_ffn_dim


@pytest.mark.inference_v2
@pytest.mark.parametrize("model_dim, ffn_multiplier, n_shards", [(1024, 4, 1), (1024, 4, 8), (1024, 4, 6)])
@pytest.mark.parametrize("n_experts", [1, 16])
def test_gated_ffn_sharding(model_dim: int, ffn_multiplier: int, n_shards: int, n_experts: int):
    """
    Test the same cases assuming a gated regime.
    """
    mlp_1_w, mlp_1_b, mlp_2_w, mlp_2_b = make_params(model_dim, ffn_multiplier, n_experts, gated=True)

    total_ffn_dim = round_up_to_256(int(model_dim * ffn_multiplier * 4 / 3))
    mapped_neurons = 0

    is_moe = n_experts > 1

    for shard_rank in range(n_shards):
        shard_1_w = shard_mlp_1_param(mlp_1_w, shard_rank, n_shards, gated=True, is_moe=is_moe)
        shard_1_b = shard_mlp_1_param(mlp_1_b, shard_rank, n_shards, gated=True, is_moe=is_moe)
        shard_2_w = shard_mlp_2_param(mlp_2_w, shard_rank, n_shards, is_moe=is_moe)
        shard_2_b = shard_mlp_2_param(mlp_2_b, shard_rank, n_shards, is_moe=is_moe)

        assert shard_1_w.shape[-2] == shard_2_w.shape[-1] * 2
        assert shard_1_w.shape[-2] % DEFAULT_SHARD_GRANULARITY == 0
        assert shard_1_w.shape[-2] == shard_1_b.shape[-1]

        mapped_neurons += shard_1_w.shape[-2]

        if shard_rank != 0:
            assert shard_2_b is None
        else:
            assert shard_2_b.shape[-1] == model_dim

    assert mapped_neurons == total_ffn_dim

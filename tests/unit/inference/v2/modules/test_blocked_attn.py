# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import itertools

from typing import List, Tuple

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.modules import ConfigBundle
from deepspeed.inference.v2.modules.configs import DSSelfAttentionConfig, PositionalEmbeddingType, RotateHalfConfig
from deepspeed.inference.v2.modules.interfaces import DSSelfAttentionRegistry, DSSelfAttentionBase

from ..kernels.ragged_ops.ragged_testing_utils import build_batch_and_manager
from ...v2.inference_test_utils import allclose

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    validate_accuracy = True
except ImportError:
    validate_accuracy = False


def _blocked_flash_testing_helper(head_size: int,
                                  n_heads_q: int,
                                  n_heads_kv: int,
                                  seq_params: List[Tuple[int, int]],
                                  trained_freqs: bool = None) -> None:
    """
    Helper function for testing blocked flash attention. This implementation is based on
    the implemnentation in ``unit.inference.kernels.ragged_ops.test_blocked_flash`` but
    integrates functionality to validate the composability.
    """
    if trained_freqs is None:
        embed_type = PositionalEmbeddingType.none
        embed_args = None
    else:
        embed_type = PositionalEmbeddingType.rotate_half
        embed_args = RotateHalfConfig(use_trained_freqs=trained_freqs)

    attn_config = DSSelfAttentionConfig(max_tokens=2048,
                                        n_heads_q=n_heads_q,
                                        n_heads_kv=n_heads_kv,
                                        head_size=head_size,
                                        max_sequences=32,
                                        positional_embedding_type=embed_type,
                                        positional_embedding_config=embed_args)

    config = ConfigBundle(name='dense_blocked_attention', config=attn_config)
    attn_module: DSSelfAttentionBase = DSSelfAttentionRegistry.instantiate_config(config)

    kv_block_size = attn_module.kv_block_size

    kvs = []
    for _, history_len in seq_params:
        if history_len > 0:
            kvs.append(
                torch.randn((history_len, 2 * n_heads_kv * head_size),
                            device=get_accelerator().current_device(),
                            dtype=torch.float16))
        else:
            kvs.append(None)

    batch, state_manager, _ = build_batch_and_manager(seq_params, head_size, n_heads_kv, kv_block_size, kv_fill=kvs)

    qkv = torch.randn((batch.current_tokens, (n_heads_q + 2 * n_heads_kv) * head_size),
                      device=get_accelerator().current_device(),
                      dtype=torch.float16)

    kv_cache = state_manager.get_cache(0)

    attn_module.build_atoms(batch)
    if not trained_freqs:
        out = attn_module(qkv, kv_cache, batch)
    else:
        inv_freqs = torch.randn((head_size // 2, ), device=get_accelerator().current_device(), dtype=torch.float16)
        out = attn_module(qkv, kv_cache, batch, inv_freqs)

    if validate_accuracy and trained_freqs is None:
        cu_seqlens_q = torch.tensor([0] + list(itertools.accumulate([seq[0] for seq in seq_params])),
                                    dtype=torch.int32,
                                    device=get_accelerator().current_device())
        cu_seqlens_kv = torch.tensor([0] + list(itertools.accumulate([seq[1] + seq[0] for seq in seq_params])),
                                     dtype=torch.int32,
                                     device=get_accelerator().current_device())

        inflight_kv = qkv[:, head_size * n_heads_q:]
        full_kvs = []
        for i, kv in enumerate(kvs):
            if kv is not None:
                full_kvs.append(torch.cat([kv, inflight_kv[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]], dim=0))
            else:
                full_kvs.append(inflight_kv[cu_seqlens_q[i]:cu_seqlens_q[i + 1]])
        run_kvs = torch.cat(full_kvs, dim=0)
        k = run_kvs[:, :head_size * n_heads_kv]
        v = run_kvs[:, head_size * n_heads_kv:]

        q = qkv[:, :head_size * n_heads_q]
        q_ref = q.reshape((batch.current_tokens, n_heads_q, head_size))
        k_ref = k.reshape((k.shape[0], n_heads_kv, head_size))
        v_ref = v.reshape((v.shape[0], n_heads_kv, head_size))

        max_seqlen_q = max([seq[0] for seq in seq_params])
        max_seqlen_kv = max([seq[1] + seq[0] for seq in seq_params])

        ref_o = flash_attn_varlen_func(q_ref,
                                       k_ref,
                                       v_ref,
                                       cu_seqlens_q,
                                       cu_seqlens_kv,
                                       max_seqlen_q,
                                       max_seqlen_kv,
                                       softmax_scale=1.0,
                                       causal=True)

        ref_o = ref_o.reshape(batch.current_tokens, head_size * n_heads_q)

        assert allclose(out, ref_o)

    get_accelerator().synchronize()


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("n_tokens", [2, 33, 65, 128, 256, 2037])
def test_single_prompt(n_tokens: int) -> None:
    head_size = 64
    n_heads_q = 16
    n_heads_kv = 16

    seq_params = [(n_tokens, 0)]
    _blocked_flash_testing_helper(head_size, n_heads_q, n_heads_kv, seq_params)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("prompt_lengths", [(128, 128), (192, 38), (514, 713), (83, 312, 610)])
def test_multiple_prompts(prompt_lengths: Tuple[int, int]) -> None:
    """
    Test multiple prompts in a single batch.
    """
    head_size = 64
    n_heads_q = 16
    n_heads_kv = 16

    seq_params = [(prompt_lengths[i], 0) for i in range(len(prompt_lengths))]
    _blocked_flash_testing_helper(head_size, n_heads_q, n_heads_kv, seq_params)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("seq_params", [(1, 34), (43, 40), (1, 144), (64, 128), (332, 628)])
def test_continuation(seq_params: Tuple[int, int]) -> None:
    """
    Test continued generation/prompt processing.
    """
    head_size = 64
    n_heads_q = 32
    n_heads_kv = 32

    _blocked_flash_testing_helper(head_size, n_heads_q, n_heads_kv, [seq_params])


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("head_size", [64, 128])
def test_head_size(head_size: int) -> None:
    n_heads_q = 16
    n_heads_kv = 16
    seq_params = [(128, 128), (192, 38), (1, 814)]

    _blocked_flash_testing_helper(head_size, n_heads_q, n_heads_kv, seq_params)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("head_config", [(32, 8), (64, 16), (40, 8)])
def test_gqa(head_config: Tuple[int, int]) -> None:
    head_size = 128
    n_heads_q = head_config[0]
    n_heads_kv = head_config[1]

    seq_params = [(128, 128), (192, 38), (1, 814)]

    _blocked_flash_testing_helper(head_size, n_heads_q, n_heads_kv, seq_params)


@pytest.mark.inference_v2_ops
def test_fully_composed() -> None:
    head_size = 64
    n_heads_q = 16
    n_heads_kv = 16

    seq_params = [(332, 628), (1, 718), (1, 323), (180, 5), (224, 0)]

    _blocked_flash_testing_helper(head_size, n_heads_q, n_heads_kv, seq_params)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("trained_freqs", [True, False])
def test_rotary_emb(trained_freqs: bool) -> None:
    head_size = 64
    n_heads_q = 16
    n_heads_kv = 16

    seq_params = [(332, 628), (1, 718), (1, 323), (180, 5), (224, 0)]

    _blocked_flash_testing_helper(head_size, n_heads_q, n_heads_kv, seq_params, trained_freqs=trained_freqs)

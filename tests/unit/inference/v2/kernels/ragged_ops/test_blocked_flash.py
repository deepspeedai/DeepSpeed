# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import itertools

from typing import List, Tuple

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.inference_utils import DtypeEnum
from deepspeed.inference.v2.kernels.ragged_ops import (
    AtomBuilder,
    BlockedFlashAttn,
    get_q_block_size,
    get_kv_block_size,
    LinearBlockedKVCopy,
)
from deepspeed.inference.v2.ragged import split_kv
from deepspeed.ops.op_builder import RaggedUtilsBuilder

from .ragged_testing_utils import build_batch_and_manager
from ....v2.inference_test_utils import allclose

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    validate_accuracy = True
except ImportError:
    validate_accuracy = False
"""
NOTE(cmikeh2): These tests depend on atom construction and KV-cache copying to behave correctly.
If one or the other of those is not working, then these tests will fail. Before debugging here,
make sure that the atom construction and KV-cache copying tests are passing.
"""


def _blocked_flash_testing_helper(head_size: int, n_heads_q: int, n_heads_kv: int,
                                  seq_params: List[Tuple[int, int]]) -> None:
    """
    Helper function for testing blocked flash attention. Used to enable parametrize to only set up
    a subset of parameters before being passed to the unified test function.
    """
    q_block_size = get_q_block_size(head_size)
    kv_block_size = get_kv_block_size(head_size)

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

    atom_builder = AtomBuilder()
    kv_copy = LinearBlockedKVCopy(head_size, n_heads_q, n_heads_kv, DtypeEnum.fp16)
    atom_flash = BlockedFlashAttn(head_size, DtypeEnum.fp16)

    total_atoms = sum((seq[0] + q_block_size - 1) // q_block_size for seq in seq_params)
    atoms = torch.empty((total_atoms, 8), dtype=torch.int32, device=get_accelerator().current_device())
    alloc_func = RaggedUtilsBuilder().load().allocate_fast_host_buffer
    atoms_host = alloc_func(atoms)

    qkv = torch.randn((batch.current_tokens, (n_heads_q + 2 * n_heads_kv) * head_size),
                      device=get_accelerator().current_device(),
                      dtype=torch.float16)

    atoms_host, n_atoms = atom_builder(atoms_host, batch, q_block_size, kv_block_size)
    atoms.copy_(atoms_host[:n_atoms])

    kv_cache = state_manager.get_cache(0)
    kv_copy(kv_cache, qkv, batch)

    out = torch.empty((batch.current_tokens, head_size * n_heads_q),
                      device=get_accelerator().current_device(),
                      dtype=torch.float16)
    k_cache, v_cache = split_kv(kv_cache)
    q = qkv[:, :head_size * n_heads_q]

    atom_flash(out, q, k_cache, v_cache, atoms, 1.0)

    if validate_accuracy:
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

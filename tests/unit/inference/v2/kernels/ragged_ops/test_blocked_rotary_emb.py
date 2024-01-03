# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Tuple

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.kernels.ragged_ops import BlockedRotaryEmbeddings, BlockedTrainedRotaryEmbeddings
from deepspeed.inference.v2.ragged import RaggedBatchWrapper, DSSequenceDescriptor
from .ragged_testing_utils import build_batch_and_manager, validate_kv_cache
from ....v2.inference_test_utils import allclose
"""
NOTE(cmikeh2): It is very possible to see unit test failures (even on FP16) depending on when
certain values are casted up to or down from float32. If we are seeing accuracy issues, we should
make sure we are aligning on the training implementation's cast pattern here, given these tolerances
tend to be sufficient elsewhere.
"""


def rotary_pos_embs(q: torch.Tensor,
                    k: torch.Tensor,
                    seq_descs: List[DSSequenceDescriptor],
                    batch: RaggedBatchWrapper,
                    head_size: int,
                    rotary_dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    rotary_dim = rotary_dim if rotary_dim >= 0 else head_size

    def make_cos_sin_emb(seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, dtype=torch.float32, device=get_accelerator().current_device())
        inv_freq = (1.0 / (10000.0**(torch.arange(
            0, rotary_dim, 2, dtype=torch.float32, device=get_accelerator().current_device()) / rotary_dim))).half()

        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        return torch.cos(emb)[:, None, :], torch.sin(emb)[:, None, :], inv_freq

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)

    cos, sin, freqs = make_cos_sin_emb(1024)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    n_heads_q = q.shape[1] // head_size
    n_heads_kv = k.shape[1] // head_size

    inflight_descs = batch.inflight_seq_descriptors(on_device=False)[:batch.current_sequences]

    if inflight_descs.shape[0] != len(seq_descs):
        raise ValueError("The number of sequence descriptors does not match the number of sequences in the batch.")

    for seq_desc, inflight_seq in zip(seq_descs, inflight_descs):
        start_idx = inflight_seq[0]
        n_tokens = seq_desc.in_flight_tokens

        q_src = q[start_idx:start_idx + n_tokens].reshape(n_tokens, n_heads_q, head_size).float()
        k_src = k[start_idx:start_idx + n_tokens].reshape(n_tokens, n_heads_kv, head_size).float()
        freq_start_offset = seq_desc.seen_tokens

        q_src_rot = q_src[:, :, :rotary_dim]
        k_src_rot = k_src[:, :, :rotary_dim]

        cos_chunk = cos[range(freq_start_offset, freq_start_offset + n_tokens)]
        sin_chunk = sin[range(freq_start_offset, freq_start_offset + n_tokens)]

        q_rot = q_src_rot * cos_chunk + rotate_half(q_src_rot) * sin_chunk
        k_rot = k_src_rot * cos_chunk + rotate_half(k_src_rot) * sin_chunk

        q_emb = torch.cat((q_rot, q_src[:, :, rotary_dim:]), dim=-1)
        k_emb = torch.cat((k_rot, k_src[:, :, rotary_dim:]), dim=-1)

        q_out[start_idx:start_idx + n_tokens] = q_emb.reshape(n_tokens, n_heads_q * head_size).to(q_out.dtype)
        k_out[start_idx:start_idx + n_tokens] = k_emb.reshape(n_tokens, n_heads_kv * head_size).to(k_out.dtype)

    return q_out, k_out, freqs


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("n_tokens, history_size", [(1, 0), (17, 0), (33, 15), (1, 63)])
@pytest.mark.parametrize("trained_emb", [False, True])
@pytest.mark.parametrize("head_size", [64, 80])
def test_single_sequence_single_block(n_tokens: int, history_size: int, trained_emb: bool, head_size: int):
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
    qkv_ref = qkv.clone()

    q = qkv_ref[:, :head_size * n_heads_q]
    k = qkv_ref[:, head_size * n_heads_q:head_size * (n_heads_q + n_heads_kv)]
    v = qkv_ref[:, head_size * (n_heads_q + n_heads_kv):]

    q_ref, k, freqs = rotary_pos_embs(q, k, seq_descs, batch, head_size)
    freqs = freqs.half()

    kv_cache = state_manager.get_cache(0)

    if trained_emb:
        copy_impl = BlockedTrainedRotaryEmbeddings(head_size, n_heads_q, n_heads_kv, torch.float16)
        copy_impl(kv_cache, qkv, batch, freqs)
    else:
        copy_impl = BlockedRotaryEmbeddings(head_size, n_heads_q, n_heads_kv, torch.float16, head_size, 10000.0)
        copy_impl(kv_cache, qkv, batch)

    assert allclose(qkv[:, :head_size * n_heads_q], q_ref)
    validate_kv_cache(kv_cache, k, v, seq_descs, batch, exact=False)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("n_tokens, history_size", [(128, 0), (177, 0), (169, 8), (117, 88)])
@pytest.mark.parametrize("trained_emb", [False, True])
@pytest.mark.parametrize("head_size", [64, 80])
def test_single_sequence_multiple_blocks(n_tokens: int, history_size: int, trained_emb: bool, head_size: int):
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
    qkv_ref = qkv.clone()

    q = qkv_ref[:, :head_size * n_heads_q]
    k = qkv_ref[:, head_size * n_heads_q:head_size * (n_heads_q + n_heads_kv)]
    v = qkv_ref[:, head_size * (n_heads_q + n_heads_kv):]

    q_ref, k, freqs = rotary_pos_embs(q, k, seq_descs, batch, head_size)
    freqs = freqs.half()

    kv_cache = state_manager.get_cache(0)

    if trained_emb:
        copy_impl = BlockedTrainedRotaryEmbeddings(head_size, n_heads_q, n_heads_kv, torch.float16)
        copy_impl(kv_cache, qkv, batch, freqs)
    else:
        copy_impl = BlockedRotaryEmbeddings(head_size, n_heads_q, n_heads_kv, torch.float16, head_size, 10000.0)
        copy_impl(kv_cache, qkv, batch)

    assert allclose(qkv[:, :head_size * n_heads_q], q_ref)
    validate_kv_cache(kv_cache, k, v, seq_descs, batch, exact=False)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("trained_emb", [False, True])
@pytest.mark.parametrize("head_size", [64, 80])
def test_multi_sequences(trained_emb: bool, head_size: int) -> None:
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
    qkv_ref = qkv.clone()

    q = qkv_ref[:, :head_size * n_heads_q]
    k = qkv_ref[:, head_size * n_heads_q:head_size * (n_heads_q + n_heads_kv)]
    v = qkv_ref[:, head_size * (n_heads_q + n_heads_kv):]

    q_ref, k, freqs = rotary_pos_embs(q, k, seq_descs, batch, head_size)
    freqs = freqs.half()

    kv_cache = state_manager.get_cache(0)

    if trained_emb:
        copy_impl = BlockedTrainedRotaryEmbeddings(head_size, n_heads_q, n_heads_kv, torch.float16)
        copy_impl(kv_cache, qkv, batch, freqs)
    else:
        copy_impl = BlockedRotaryEmbeddings(head_size, n_heads_q, n_heads_kv, torch.float16, head_size, 10000.0)
        copy_impl(kv_cache, qkv, batch)

    assert allclose(qkv[:, :head_size * n_heads_q], q_ref)
    validate_kv_cache(kv_cache, k, v, seq_descs, batch, exact=False)


@pytest.mark.inference_v2_ops
def test_rotary_dim() -> None:
    trained_emb = False
    head_size = 80
    rotary_dim = 64
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
    qkv_ref = qkv.clone()

    q = qkv_ref[:, :head_size * n_heads_q]
    k = qkv_ref[:, head_size * n_heads_q:head_size * (n_heads_q + n_heads_kv)]
    v = qkv_ref[:, head_size * (n_heads_q + n_heads_kv):]

    q_ref, k, freqs = rotary_pos_embs(q, k, seq_descs, batch, head_size, rotary_dim=rotary_dim)
    freqs = freqs.half()

    kv_cache = state_manager.get_cache(0)

    copy_impl = BlockedRotaryEmbeddings(head_size, n_heads_q, n_heads_kv, torch.float16, rotary_dim, 10000.0)
    copy_impl(kv_cache, qkv, batch)

    assert allclose(qkv[:, :head_size * n_heads_q], q_ref)
    validate_kv_cache(kv_cache, k, v, seq_descs, batch, exact=False)

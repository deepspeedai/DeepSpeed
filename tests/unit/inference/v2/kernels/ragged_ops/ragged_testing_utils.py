# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import random
from typing import List, Optional, Tuple

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.ragged import (
    AllocationMode,
    DSSequenceDescriptor,
    DSStateManager,
    DSStateManagerConfig,
    KVCacheConfig,
    MemoryConfig,
    PlaceholderSequenceDescriptor,
    RaggedBatchWrapper,
)
from ....v2.inference_test_utils import allclose


def build_simple_batch(seq_lens: List[int],
                       vocab_range: Optional[int] = 100,
                       padding: Optional[bool] = False) -> RaggedBatchWrapper:
    """
    Construct a simple batch with the given sequence lengths. This method should not
    be used for for testing scenarios that require information about KV or sequence
    history.
    """
    total_tokens = max(sum(seq_lens), 1024)
    n_seqs = max(len(seq_lens), 128)

    config = DSStateManagerConfig(max_tracked_sequences=n_seqs,
                                  max_ragged_sequence_count=n_seqs,
                                  max_ragged_batch_size=total_tokens)
    batch = RaggedBatchWrapper(config)

    batch.clear()

    for seq_len in seq_lens:
        seq_desc = PlaceholderSequenceDescriptor()
        tokens = torch.randint(0, vocab_range, (seq_len, ))
        batch.insert_sequence(seq_desc, tokens)

    batch.finalize(padding=padding)

    return batch


def build_complex_batch(seq_params: List[Tuple[int, int, int]],
                        kv_block_size: int,
                        vocab_range: Optional[int] = 100,
                        padding: Optional[bool] = False) -> Tuple[RaggedBatchWrapper, int]:
    """
    Construct a fully paramtrized batch with the given sequence lengths. This method
    can be used to construct more realistic inputs for testing scenarios that will interact
    with all the members of the RaggedBatchWrapper.
    """
    seq_lens = [seq_param[0] for seq_param in seq_params]
    total_tokens = max(sum(seq_lens), 1024)
    n_seqs = max(len(seq_lens), 128)

    config = DSStateManagerConfig(max_tracked_sequences=n_seqs,
                                  max_ragged_sequence_count=n_seqs,
                                  max_ragged_batch_size=total_tokens)
    batch = RaggedBatchWrapper(config)

    batch.clear()

    total_kv_blocks = 0

    for seq_len, n_seen_tokens, kv_ptr in seq_params:
        n_kv_blocks = (seq_len + n_seen_tokens + kv_block_size - 1) // kv_block_size
        seq_desc = PlaceholderSequenceDescriptor(seen_tokens=n_seen_tokens,
                                                 cur_allocated_blocks=n_kv_blocks,
                                                 kv_blocks_ptr=kv_ptr)
        tokens = torch.randint(0, vocab_range, (seq_len, ))
        batch.insert_sequence(seq_desc, tokens)
        total_kv_blocks += n_kv_blocks

    batch.finalize(padding=padding)

    return batch, total_kv_blocks


def build_batch_and_manager(
    seq_params: List[Tuple[int, int]],
    head_size: int,
    n_heads_kv: int,
    kv_block_size: int,
    vocab_range: Optional[int] = 100,
    padding: Optional[bool] = False,
    kv_fill: Optional[List[torch.Tensor]] = None
) -> Tuple[RaggedBatchWrapper, DSStateManager, List[DSSequenceDescriptor]]:
    """
    Will construct and populate a batch and KVCache with the given sequence parameters.

    Arguments:
        seq_params (List[Tuple[int, int]]): A list of tuples containing the sequence length and
            the number of tokens that have already been seen for that sequence.
        head_size (int): The size of each attention head.
        n_heads_kv (int): The number of attention heads for the KV-cache.
        kv_block_size (int): The size of each block in the KV-cache.
        vocab_range (Optional[int]): The range of the vocabulary. Defaults to 100.
        padding (Optional[bool]): Whether to pad the batch. Defaults to False.
        kv_fill (Optional[List[torch.Tensor]]): A list of tensors to use to populate the KV-cache.
            If this is not provided, the KV-cache will be treated as empty and the contents should
            not be relied upon. NOTE(cmikeh2): This functionality relies on the functionality
            of LinearBlockedKVCopy. If tests relying on this feature are failing, make sure that
            LinearBlockedKVCopy is working correctly.
    """
    seq_lens = [seq_param[0] for seq_param in seq_params]
    fill_lens = [seq_param[1] for seq_param in seq_params]
    max_created_batch_len = max(sum(seq_lens), sum(fill_lens))
    total_tokens = max(max_created_batch_len, 1024)
    n_seqs = max(len(seq_lens), 128)

    req_kv_blocks = [None] * n_seqs
    total_kv_blocks = 0
    for i, (seq_len, n_seen_tokens) in enumerate(seq_params):
        req_kv_blocks[i] = (seq_len + n_seen_tokens + kv_block_size - 1) // kv_block_size
        total_kv_blocks += req_kv_blocks[i]

    kv_config = KVCacheConfig(block_size=kv_block_size,
                              num_allocation_groups=1,
                              cache_shape=(1, n_heads_kv, head_size))
    memory_config = MemoryConfig(mode=AllocationMode.ALLOCATE, size=total_kv_blocks)

    config = DSStateManagerConfig(max_tracked_sequences=n_seqs,
                                  max_ragged_sequence_count=n_seqs,
                                  max_ragged_batch_size=total_tokens,
                                  memory_config=memory_config)

    batch = RaggedBatchWrapper(config)
    state_manager = DSStateManager(config, (kv_config, ))

    # At the beginning of operation, the design of the allocator is such that it will return
    # linear blocks of memory. The following will "warm up" the allocator so that we can be
    # more certain that code is not dependent on this behavior.
    all_allocs = []
    for _ in range(20):
        decision = random.randint(0, 1)

        if decision == 0:
            blocks_to_allocate = random.randint(0, total_kv_blocks)
            if blocks_to_allocate <= state_manager.free_blocks[0] and blocks_to_allocate > 0:
                all_allocs.append(state_manager.allocate_blocks(blocks_to_allocate))
        else:
            if len(all_allocs) > 0:
                idx = random.randint(0, len(all_allocs) - 1)
                state_manager._kv_cache.free(all_allocs[idx])

                del all_allocs[idx]

    for alloc in all_allocs:
        state_manager._kv_cache.free(alloc)

    assert state_manager.free_blocks[0] == total_kv_blocks

    batch.clear()
    seq_descs = []

    if kv_fill is None or sum(fill_lens) == 0:
        for i, (seq_len, n_seen_tokens) in enumerate(seq_params):
            # Create empty descriptor
            seq_desc = state_manager.get_or_create_sequence(i)

            # Update `seen_tokens` in the descriptor
            seq_desc.pre_forward(n_seen_tokens)
            seq_desc.post_forward()

            # Ensure there's enough KV-cache for the sequence
            kv_block_ids = state_manager.allocate_blocks(req_kv_blocks[i])
            print(f"Allocated {req_kv_blocks[i]} blocks for sequence {i}: {kv_block_ids}")
            seq_desc.extend_kv_cache(kv_block_ids)

            # Insert sequence into batch
            tokens = torch.randint(0, vocab_range, (seq_len, ))
            batch.insert_sequence(seq_desc, tokens)
            seq_desc.pre_forward(seq_len)
            seq_descs.append(seq_desc)
    else:
        qkv = torch.empty((total_tokens, (n_heads_kv * 3) * head_size),
                          dtype=torch.float16,
                          device=get_accelerator().current_device())
        fills_as_tensor = torch.tensor(fill_lens, dtype=torch.int32)
        fill_cumsum = torch.cat((torch.tensor([0], dtype=torch.int32), torch.cumsum(fills_as_tensor, dim=0)))

        for i, (_, n_seen_tokens) in enumerate(seq_params):
            # Create empty descriptor
            seq_desc = state_manager.get_or_create_sequence(i)

            # Update `seen_tokens` in the descriptor
            if n_seen_tokens > 0:
                dummy_fill_toks = torch.randint(0, vocab_range, (n_seen_tokens, ))
                batch.insert_sequence(seq_desc, dummy_fill_toks)
                seq_desc.pre_forward(n_seen_tokens)

            # Ensure there's enough KV-cache for the sequence
            kv_block_ids = state_manager.allocate_blocks(req_kv_blocks[i])
            print(f"Allocated {req_kv_blocks[i]} blocks for sequence {i}: {kv_block_ids}")
            seq_desc.extend_kv_cache(kv_block_ids)
            seq_descs.append(seq_desc)

            if n_seen_tokens == 0:
                continue

            assert kv_fill[i].shape[0] == n_seen_tokens
            assert kv_fill[i].shape[1] == n_heads_kv * head_size * 2

            local_q = torch.randn((n_seen_tokens, n_heads_kv * head_size), dtype=torch.float16, device=qkv.device)
            local_qkv = torch.cat((local_q, kv_fill[i]), dim=1)
            qkv[fill_cumsum[i]:fill_cumsum[i + 1]] = local_qkv

        batch.finalize(padding=padding)

        from deepspeed.inference.v2.kernels.ragged_ops import LinearBlockedKVCopy
        kv_copy = LinearBlockedKVCopy(head_size, n_heads_kv, n_heads_kv, torch.float16)
        kv_cache = state_manager.get_cache(0)
        kv_copy(kv_cache, qkv, batch)

        for seq_desc in seq_descs:
            if seq_desc.in_flight_tokens > 0:
                seq_desc.post_forward()

        batch.clear()

        for i, (seq_len, _) in enumerate(seq_params):
            seq_desc = state_manager.get_or_create_sequence(i)
            tokens = torch.randint(0, vocab_range, (seq_len, ))
            batch.insert_sequence(seq_desc, tokens)
            seq_desc.pre_forward(seq_len)

            # We will skip KV cache allocation here because we did a lump allocation above
            # for both the fill and the sequence itself.

    batch.finalize(padding=padding)

    return batch, state_manager, seq_descs


def validate_kv_cache(kv_cache: torch.Tensor,
                      k: torch.Tensor,
                      v: torch.Tensor,
                      seq_descs: List[DSSequenceDescriptor],
                      batch: RaggedBatchWrapper,
                      exact: bool = True) -> None:
    """
    Given a QKV tensor and a KV cache, validate that the cache contains the correct values.
    """
    block_size = kv_cache.shape[1]
    n_kv_heads = kv_cache.shape[3]
    head_size = kv_cache.shape[4]

    inflight_descs = batch.inflight_seq_descriptors(on_device=False)[:batch.current_sequences]

    if inflight_descs.shape[0] != len(seq_descs):
        raise ValueError("The number of sequence descriptors does not match the number of sequences in the batch.")

    for seq_desc, inflight_seq in zip(seq_descs, inflight_descs):
        start_idx = inflight_seq[0]
        assigned_kv_blocks = seq_desc.kv_cache_ids(on_device=False)

        real_k_values = k[start_idx:start_idx + seq_desc.in_flight_tokens]
        real_v_values = v[start_idx:start_idx + seq_desc.in_flight_tokens]

        start_block_idx = seq_desc.seen_tokens // block_size
        local_start_idx = 0
        cur_start_idx = seq_desc.seen_tokens

        for block_idx in range(start_block_idx, seq_desc.cur_allocated_blocks):
            block = kv_cache[assigned_kv_blocks[0, block_idx].item()]
            block_start_idx = cur_start_idx % block_size
            n_tokens_to_check = min(block_size - block_start_idx, seq_desc.in_flight_tokens - local_start_idx)
            block_end_idx = block_start_idx + n_tokens_to_check

            if exact:
                assert torch.equal(
                    block[block_start_idx:block_end_idx, 0, :, :],
                    real_k_values[local_start_idx:local_start_idx + n_tokens_to_check].reshape(
                        n_tokens_to_check, n_kv_heads, head_size))
                assert torch.equal(
                    block[block_start_idx:block_end_idx, 1, :, :],
                    real_v_values[local_start_idx:local_start_idx + n_tokens_to_check].reshape(
                        n_tokens_to_check, n_kv_heads, head_size))
            else:
                assert allclose(
                    block[block_start_idx:block_end_idx, 0, :, :],
                    real_k_values[local_start_idx:local_start_idx + n_tokens_to_check].reshape(
                        n_tokens_to_check, n_kv_heads, head_size))
                assert allclose(
                    block[block_start_idx:block_end_idx, 1, :, :],
                    real_v_values[local_start_idx:local_start_idx + n_tokens_to_check].reshape(
                        n_tokens_to_check, n_kv_heads, head_size))

            local_start_idx += n_tokens_to_check
            cur_start_idx += n_tokens_to_check

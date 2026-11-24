# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import pytest
import torch

from types import SimpleNamespace

from deepspeed.inference.v2.kernels.ragged_ops.blocked_flash.blocked_flash import get_kv_block_size
from deepspeed.runtime.rollout.paged_cache import PagedAttentionCache


def _config(num_kv_heads, num_heads, head_dim):
    return SimpleNamespace(
        num_hidden_layers=1,
        num_key_value_heads=num_kv_heads,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        hidden_size=num_heads * head_dim,
    )


def _dense_oracle(query, row_keys, row_values, num_heads, num_kv_heads, is_causal=False):
    """Grouped-query attention over one row's dense K/V, the reference the kernel must match."""
    ratio = num_heads // num_kv_heads
    dense_keys = row_keys.unsqueeze(0).repeat_interleave(ratio, dim=1)
    dense_values = row_values.unsqueeze(0).repeat_interleave(ratio, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(query, dense_keys, dense_values, is_causal=is_causal)


def test_paged_decode_matches_dense_oracle():
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim, num_kv_heads, num_heads = 64, 2, 4
    block_size = get_kv_block_size(head_dim)
    cache = PagedAttentionCache(_config(num_kv_heads, num_heads, head_dim),
                                batch_size=2,
                                max_cache_len=2 * block_size,
                                device=device,
                                dtype=dtype,
                                block_size=block_size)

    prompt_len = 5
    keys = torch.randn(2, num_kv_heads, prompt_len, head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    valid = torch.tensor([True, False, True, True, True], device=device)
    cache.copy_from_dense(0, keys[0], values[0], 0)
    cache.copy_from_dense(0, keys[1], values[1], 1, valid)

    positions = torch.tensor([prompt_len, int(valid.sum())], device=device, dtype=torch.long)
    cache.set_write_position(positions)
    cache.set_active_rows([0, 1])
    new_keys = torch.randn(2, num_kv_heads, 1, head_dim, device=device, dtype=dtype)
    new_values = torch.randn_like(new_keys)
    cache.write(0, new_keys, new_values)

    query = torch.randn(2, num_heads, 1, head_dim, device=device, dtype=dtype)
    actual = cache.attention(0, query, 1.0 / (head_dim**0.5))
    for row, length in ((0, prompt_len + 1), (1, int(valid.sum()) + 1)):
        row_keys, row_values = cache.materialize(0, row, length)
        expected = _dense_oracle(query[row:row + 1], row_keys, row_values, num_heads, num_kv_heads)
        # attention() returns [rows, q_len, heads * head_dim] head-major; restore the
        # [rows, heads, q_len, head_dim] layout the dense oracle produces.
        actual_row = actual[row:row + 1].view(1, num_heads, 1, head_dim)
        torch.testing.assert_close(actual_row, expected, atol=0.25, rtol=0.25)


def test_paged_compact_releases_and_reorders_rows():
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim, num_kv_heads = 64, 2
    block_size = get_kv_block_size(head_dim)
    cache = PagedAttentionCache(_config(num_kv_heads, 4, head_dim),
                                batch_size=2,
                                max_cache_len=2 * block_size,
                                device=device,
                                dtype=dtype,
                                block_size=block_size)
    keys = torch.randn(2, num_kv_heads, 3, head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    cache.copy_from_dense(0, keys[0], values[0], 0)
    cache.copy_from_dense(0, keys[1], values[1], 1)

    cache.compact(torch.tensor([1], device=device))

    assert cache.row_len.tolist()[:1] == [3]
    assert cache.blocks_in_use == 1
    row_keys, row_values = cache.materialize(0, 0, 3)
    torch.testing.assert_close(row_keys, keys[1])
    torch.testing.assert_close(row_values, values[1])


def test_graph_safe_paged_decode_matches_dense_oracle():
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(2)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim, num_kv_heads, num_heads = 64, 2, 4
    block_size = get_kv_block_size(head_dim)
    cache = PagedAttentionCache(_config(num_kv_heads, num_heads, head_dim),
                                batch_size=2,
                                max_cache_len=2 * block_size,
                                device=device,
                                dtype=dtype,
                                block_size=block_size)
    cache.enable_graph_safe()
    prompt_len = 5
    keys = torch.randn(2, num_kv_heads, prompt_len, head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    cache.copy_from_dense(0, keys[0], values[0], 0)
    cache.copy_from_dense(0, keys[1], values[1], 1)
    positions = torch.tensor([prompt_len, prompt_len], device=device, dtype=torch.long)
    cache.set_write_position(positions)
    cache.set_graph_mode()
    new_keys = torch.randn(2, num_kv_heads, 1, head_dim, device=device, dtype=dtype)
    new_values = torch.randn_like(new_keys)
    cache.write(0, new_keys, new_values)
    cache.update_graph_atoms()

    query = torch.randn(2, num_heads, 1, head_dim, device=device, dtype=dtype)
    actual = cache.attention(0, query, 1.0 / (head_dim**0.5))
    for row in range(2):
        row_keys, row_values = cache.materialize(0, row, prompt_len + 1)
        expected = _dense_oracle(query[row:row + 1], row_keys, row_values, num_heads, num_kv_heads)
        actual_row = actual[row:row + 1].view(1, num_heads, 1, head_dim)
        torch.testing.assert_close(actual_row, expected, atol=0.25, rtol=0.25)


def test_set_graph_mode_keeps_persistent_row_and_atom_views():
    # Pins the two graph-mode bugs fixed in the paged cache: set_graph_mode() used to
    # (a) point the captured decode at the whole atom table instead of the per-row
    # view, and (b) allocate a fresh arange row-index every call that the captured
    # selects read by address; that buffer was freed before the first replay, so the
    # replay indexed with allocator garbage ("index out of bounds" in IndexKernel).
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    device = torch.device("cuda")
    block_size = get_kv_block_size(64)
    cache = PagedAttentionCache(_config(2, 4, 64),
                                batch_size=2,
                                max_cache_len=4 * block_size,
                                device=device,
                                dtype=torch.bfloat16,
                                block_size=block_size)
    cache.enable_graph_safe()
    cache.set_graph_mode()
    rows = cache._active_rows
    # Decode uses one atom per row; the wider table also holds eager-prefill atoms.
    assert cache._graph_atoms.shape[0] == 2
    # The row-index buffer is the cache's persistent allocation; re-entering graph
    # mode must not hand out a fresh tensor that a captured kernel would outlive.
    assert rows is cache._graph_rows
    cache.set_graph_mode()
    assert cache._active_rows is rows


def test_graph_safe_multi_block_decode_matches_dense_oracle():
    # Blocked-flash atom decode across a multi-block prompt in graph-atom mode, at
    # both head_dim 64 (kv block 128) and 128 (kv block 64, the Qwen3-4B/8B
    # geometry): the configuration the rollout benches run, which the earlier
    # single-block suite did not cover.
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_kv_heads, num_heads = 2, 4
    for head_dim in (64, 128):
        block_size = get_kv_block_size(head_dim)
        cache = PagedAttentionCache(_config(num_kv_heads, num_heads, head_dim),
                                    batch_size=2,
                                    max_cache_len=8 * block_size,
                                    device=device,
                                    dtype=dtype,
                                    block_size=block_size)
        cache.enable_graph_safe()
        prompt_len = 5 * block_size + 3
        keys = torch.randn(2, num_kv_heads, prompt_len, head_dim, device=device, dtype=dtype)
        values = torch.randn_like(keys)
        cache.copy_from_dense(0, keys[0], values[0], 0)
        cache.copy_from_dense(0, keys[1], values[1], 1)
        cache.set_write_position(torch.full((2, ), prompt_len, device=device, dtype=torch.long))
        cache.set_graph_mode()
        new_keys = torch.randn(2, num_kv_heads, 1, head_dim, device=device, dtype=dtype)
        new_values = torch.randn_like(new_keys)
        cache.write(0, new_keys, new_values)
        cache.update_graph_atoms()

        query = torch.randn(2, num_heads, 1, head_dim, device=device, dtype=dtype)
        actual = cache.attention(0, query, 1.0 / (head_dim**0.5))
        for row in range(2):
            row_keys, row_values = cache.materialize(0, row, prompt_len + 1)
            expected = _dense_oracle(query[row:row + 1], row_keys, row_values, num_heads, num_kv_heads)
            actual_row = actual[row:row + 1].view(1, num_heads, 1, head_dim)
            torch.testing.assert_close(actual_row, expected, atol=0.25, rtol=0.25)


def test_paged_prefill_attention_matches_dense_oracle():
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(3)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim, num_kv_heads, num_heads = 64, 2, 4
    block_size = get_kv_block_size(head_dim)
    cache = PagedAttentionCache(_config(num_kv_heads, num_heads, head_dim),
                                batch_size=1,
                                max_cache_len=512,
                                device=device,
                                dtype=dtype,
                                block_size=block_size)
    prompt_len = 260
    keys = torch.randn(1, num_kv_heads, prompt_len, head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    cache.set_write_position(torch.tensor([0], device=device, dtype=torch.long))
    cache.set_active_rows([0])
    cache.write(0, keys, values)
    query = torch.randn(1, num_heads, prompt_len, head_dim, device=device, dtype=dtype)
    actual = cache.attention(0, query, 1.0 / (head_dim**0.5))
    expected = _dense_oracle(query, keys[0], values[0], num_heads, num_kv_heads, is_causal=True)
    actual = actual.view(1, prompt_len, num_heads, head_dim).permute(0, 2, 1, 3)
    torch.testing.assert_close(actual, expected, atol=0.25, rtol=0.25)


def test_flashinfer_decode_matches_dense_oracle():
    pytest.importorskip("flashinfer")
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(5)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    # Qwen3-4B head config: GQA ratio 4, head_dim 128, which routes the
    # FlashInfer decode to the fa2 tensor-core split-KV kernel.
    head_dim, num_kv_heads, num_heads = 128, 8, 32
    block_size = get_kv_block_size(head_dim)
    cache = PagedAttentionCache(_config(num_kv_heads, num_heads, head_dim),
                                batch_size=2,
                                max_cache_len=8 * block_size,
                                device=device,
                                dtype=dtype,
                                block_size=block_size,
                                use_flashinfer_decode=True)

    prompt_len = 5 * block_size + 3
    keys = torch.randn(2, num_kv_heads, prompt_len, head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    cache.copy_from_dense(0, keys[0], values[0], 0)
    cache.copy_from_dense(0, keys[1], values[1], 1)

    positions = torch.tensor([prompt_len, prompt_len], device=device, dtype=torch.long)
    cache.set_write_position(positions)
    cache.set_active_rows([0, 1])
    new_keys = torch.randn(2, num_kv_heads, 1, head_dim, device=device, dtype=dtype)
    new_values = torch.randn_like(new_keys)
    cache.write(0, new_keys, new_values)

    query = torch.randn(2, num_heads, 1, head_dim, device=device, dtype=dtype)
    actual = cache.attention(0, query, 1.0 / (head_dim**0.5))
    for row in range(2):
        row_keys, row_values = cache.materialize(0, row, prompt_len + 1)
        expected = _dense_oracle(query[row:row + 1], row_keys, row_values, num_heads, num_kv_heads)
        actual_row = actual[row:row + 1].view(1, num_heads, 1, head_dim)
        torch.testing.assert_close(actual_row, expected, atol=0.25, rtol=0.25)


def test_prefix_reuse_shares_blocks_and_releases_after_last_row():
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(4)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim, num_kv_heads = 64, 2
    block_size = get_kv_block_size(head_dim)
    cache = PagedAttentionCache(_config(num_kv_heads, 4, head_dim),
                                batch_size=2,
                                max_cache_len=256,
                                device=device,
                                dtype=dtype,
                                block_size=block_size)
    keys = torch.randn(1, num_kv_heads, 3, head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    cache.copy_from_dense(0, keys[0], values[0], 0)
    prefix_hash = b"same-prefix"
    cache.register_prefix([prefix_hash], 0, 1)

    assert cache.prepare_prefix([prefix_hash], 1) == 1
    assert cache.block_ids[1, 0].item() == cache.block_ids[0, 0].item()
    row_keys, row_values = cache.materialize(0, 1, 3)
    torch.testing.assert_close(row_keys, keys[0])
    torch.testing.assert_close(row_values, values[0])

    cache.clear_row(0)
    assert cache.blocks_in_use == 1
    row_keys, _ = cache.materialize(0, 1, 3)
    torch.testing.assert_close(row_keys, keys[0])
    cache.clear_row(1)
    assert cache.blocks_in_use == 0

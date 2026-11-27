# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import pytest
import torch

from types import SimpleNamespace

import deepspeed
from deepspeed.inference.v2.kernels.ragged_ops.blocked_flash.blocked_flash import BlockedFlashAttn, get_kv_block_size
from deepspeed.ops.op_builder import RaggedOpsBuilder
from deepspeed.runtime.rollout.paged_cache import PagedAttentionCache


def _skip_without_blocked_flash():
    """Skip unless the blocked-attention backend this cache needs can be built.

    The builder needs the optional dskernels package and compute capability >= 8.0, so a
    CUDA installation without them must skip rather than fail: paged caching is opt-in.
    """
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    if not deepspeed.ops.__compatible_ops__[RaggedOpsBuilder.NAME]:
        pytest.skip("ragged device ops are not available on this system")


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
    _skip_without_blocked_flash()
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
    _skip_without_blocked_flash()
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
    _skip_without_blocked_flash()
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
    _skip_without_blocked_flash()
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
    _skip_without_blocked_flash()
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
    _skip_without_blocked_flash()
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
    _skip_without_blocked_flash()
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
    _skip_without_blocked_flash()
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


# ---------------------------------------------------------------------------
# Kernel-free coverage.
#
# The block table, the per-row write positions and the atom table are plain tensor
# logic; the blocked-attention op only needs them to describe valid work. Stubbing
# the CUDA-only pieces (block sizes, block pool, attention kernel) runs all of that
# real code on any device, which is where the contracts below are pinned. The
# kernels themselves are covered by the tests above.
# ---------------------------------------------------------------------------

BLOCK_SIZE = 8


class _FakePool:
    """BlockedKVCache stand-in with the same reserve/free/get_cache contract."""

    def __init__(self, configs, memory_config, *args, **kwargs):
        config = configs[0]
        layers, heads, head_dim = config.cache_shape
        self.cache = torch.zeros((layers, memory_config.size, config.block_size, 2, heads, head_dim))
        self._free = list(range(memory_config.size - 1, -1, -1))

    @property
    def free_blocks(self):
        """Free-block counts per allocator, as BlockedKVCache reports them."""
        return [len(self._free)]

    def get_cache(self, cache_id):
        return self.cache[cache_id]

    def reserve(self, num_blocks):
        if len(self._free) < num_blocks:
            raise RuntimeError("block pool exhausted")
        return torch.tensor([self._free.pop() for _ in range(num_blocks)], dtype=torch.int32)

    def free(self, blocks):
        self._free.extend(int(block) for block in blocks)


class _RecordingKernel:
    """BlockedFlashAttn stand-in: records the atoms it is handed and writes zeros."""

    # The cache validates the K/V dtype against the real kernel's supported set.
    supported_dtypes = BlockedFlashAttn.supported_dtypes
    calls = []

    def __init__(self, head_size, dtype):
        self.head_size = head_size

    def __call__(self, out, q, k, v, atoms, scale):
        _RecordingKernel.calls.append({"atoms": atoms.clone(), "query_rows": q.shape[0]})
        out.zero_()
        return out


@pytest.fixture
def stubbed_ops(monkeypatch):
    """Real cache logic on this device, with the CUDA-only ops stubbed out."""
    from deepspeed.runtime.rollout import paged_cache

    monkeypatch.setattr(paged_cache, "get_kv_block_size", lambda head_dim: BLOCK_SIZE)
    monkeypatch.setattr(paged_cache, "get_q_block_size", lambda head_dim: BLOCK_SIZE // 2)
    monkeypatch.setattr(paged_cache, "BlockedKVCache", _FakePool)
    monkeypatch.setattr(paged_cache, "BlockedFlashAttn", _RecordingKernel)
    _RecordingKernel.calls = []
    return _RecordingKernel.calls


def _make_cache(batch_size, max_cache_len, num_kv_heads=2, num_heads=4, head_dim=64, dtype=torch.bfloat16):
    return PagedAttentionCache(_config(num_kv_heads, num_heads, head_dim),
                               batch_size=batch_size,
                               max_cache_len=max_cache_len,
                               device=torch.device("cpu"),
                               dtype=dtype,
                               block_size=BLOCK_SIZE)


def test_layered_prefill_keeps_one_block_table_per_row(stubbed_ops):
    """Every layer of one prompt must be copied through the same block table.

    Sizing the row once per layer releases and re-reserves its chain, and the allocator
    hands the same blocks back in a different order: the earlier layers' K/V then sits
    in blocks the final table no longer names, so reading a row back permutes it.
    """
    cache = _make_cache(batch_size=1, max_cache_len=6 * BLOCK_SIZE)
    cache.set_write_position(torch.zeros(1, dtype=torch.long))
    prompt_len = 3 * BLOCK_SIZE
    keys_by_layer = [torch.randn(cache.num_kv_heads, prompt_len, cache.head_dim) for _ in range(cache.num_layers)]

    cache.prepare_row(0, prompt_len)
    for layer_idx, keys in enumerate(keys_by_layer):
        cache.copy_from_dense(layer_idx, keys, keys.clone(), 0, prepared=True)

    for layer_idx, keys in enumerate(keys_by_layer):
        stored_keys, stored_values = cache.materialize(layer_idx, 0, prompt_len)
        torch.testing.assert_close(stored_keys, keys)
        torch.testing.assert_close(stored_values, keys)


def test_prepare_row_reuses_the_table_across_layers(stubbed_ops):
    """The row's block table must not change between layers of the same prompt."""
    cache = _make_cache(batch_size=1, max_cache_len=6 * BLOCK_SIZE)
    cache.set_write_position(torch.zeros(1, dtype=torch.long))
    prompt_len = 3 * BLOCK_SIZE
    keys = torch.randn(cache.num_kv_heads, prompt_len, cache.head_dim)

    cache.prepare_row(0, prompt_len)
    table_after_prepare = cache.block_ids[0].clone()
    for layer_idx in range(cache.num_layers):
        cache.copy_from_dense(layer_idx, keys, keys.clone(), 0, prepared=True)
        assert torch.equal(cache.block_ids[0], table_after_prepare)


def test_paged_cache_rejects_unsupported_kv_dtype(stubbed_ops):
    """The blocked-attention kernel has no fp32 path, so fp32 must fail at construction."""
    with pytest.raises(ValueError, match="no fp32 path"):
        _make_cache(batch_size=1, max_cache_len=4 * BLOCK_SIZE, dtype=torch.float32)


def test_paged_cache_rejects_tensor_parallel_shards(stubbed_ops):
    """AutoTP shards the projections, so the config's head counts are not the local ones."""
    from deepspeed.runtime.rollout.paged_cache import install_paged_attention

    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(vocab_size=32,
                                      hidden_size=32,
                                      intermediate_size=64,
                                      num_hidden_layers=1,
                                      num_attention_heads=4,
                                      num_key_value_heads=2,
                                      head_dim=8,
                                      pad_token_id=0)
    model = transformers.Qwen3ForCausalLM(config)
    install_paged_attention(model)

    model.model.layers[0].self_attn.q_proj.is_tp_parallel = True
    with pytest.raises(ValueError, match="tensor-parallel"):
        install_paged_attention(model)


def test_blocks_in_use_counts_physical_blocks(stubbed_ops):
    """A shared prefix block is one physical block, however many rows reference it."""
    cache = _make_cache(batch_size=2, max_cache_len=4 * BLOCK_SIZE)
    cache.set_write_position(torch.zeros(2, dtype=torch.long))
    keys = torch.randn(1, cache.num_kv_heads, BLOCK_SIZE, cache.head_dim)
    cache.set_active_rows([0])
    cache.write(0, keys, keys.clone())
    cache.register_prefix([b"prefix"], 0, 1)
    assert cache.blocks_in_use == 1

    assert cache.prepare_prefix([b"prefix"], 1) == 1
    assert cache.blocks_in_use == 1

    cache.clear_row(0)
    assert cache.blocks_in_use == 1
    cache.clear_row(1)
    assert cache.blocks_in_use == 0

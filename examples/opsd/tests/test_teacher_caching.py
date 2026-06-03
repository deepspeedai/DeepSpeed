# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""CPU-only tests for TeacherLogitCache.

The ``TeacherWrapper`` itself (which wraps deepspeed+transformers) is not
exercised here because it requires a real model and a DeepSpeed launcher; the
caching/streaming pieces are isolated into ``TeacherLogitCache`` so they can
be tested in isolation.
"""

import pytest
import torch

from opsd.teacher import TeacherLogitCache


def test_round_trip_preserves_values_within_dtype():
    torch.manual_seed(0)
    gpu_like = torch.randn(2, 16, 32, dtype=torch.float32)
    cache = TeacherLogitCache.from_gpu_logits(gpu_like, store_dtype=torch.bfloat16)
    assert cache.shape == (2, 16, 32)
    assert cache.dtype == torch.bfloat16
    chunk = cache.chunk_to_device(0, 16, torch.device("cpu"), dtype=torch.float32)
    # bf16 round-trip loses precision; check it stays within bf16's worst-case
    # relative error rather than asserting exact equality.
    assert torch.allclose(chunk, gpu_like, atol=1e-1, rtol=1e-1)


def test_chunk_slicing_is_correct():
    torch.manual_seed(0)
    src = torch.randn(3, 100, 8)
    cache = TeacherLogitCache.from_gpu_logits(src, store_dtype=torch.float32)
    for start, end in [(0, 10), (10, 50), (50, 100), (33, 77)]:
        got = cache.chunk_to_device(start, end, torch.device("cpu"))
        assert got.shape == (3, end - start, 8)
        assert torch.allclose(got, src[:, start:end])


def test_invalid_chunk_bounds_raise():
    cache = TeacherLogitCache.from_gpu_logits(torch.zeros(1, 8, 4), store_dtype=torch.float32)
    with pytest.raises(ValueError, match="invalid"):
        cache.chunk_to_device(0, 9, torch.device("cpu"))
    with pytest.raises(ValueError, match="invalid"):
        cache.chunk_to_device(5, 3, torch.device("cpu"))
    with pytest.raises(ValueError, match="invalid"):
        cache.chunk_to_device(-1, 4, torch.device("cpu"))


def test_rejects_non_3d_logits():
    with pytest.raises(ValueError, match="must be 3-D"):
        TeacherLogitCache(cpu_logits=torch.zeros(8, 32))


def test_rejects_gpu_resident_logits():
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("no CUDA available to construct GPU tensor")
    with pytest.raises(ValueError, match="must live on CPU"):
        TeacherLogitCache(cpu_logits=torch.zeros(1, 8, 4, device="cuda"))


def test_dtype_override_in_chunk_to_device():
    src = torch.randn(2, 8, 16, dtype=torch.float32)
    cache = TeacherLogitCache.from_gpu_logits(src, store_dtype=torch.float32)
    chunk = cache.chunk_to_device(0, 8, torch.device("cpu"), dtype=torch.bfloat16)
    assert chunk.dtype == torch.bfloat16


def test_free_releases_buffer():
    src = torch.randn(2, 32, 16)
    cache = TeacherLogitCache.from_gpu_logits(src, store_dtype=torch.float32)
    assert cache.cpu_logits.numel() == 2 * 32 * 16
    cache.free()
    assert cache.cpu_logits.numel() == 0


def test_default_store_dtype_is_bf16():
    src = torch.randn(1, 4, 8)
    cache = TeacherLogitCache.from_gpu_logits(src)
    assert cache.dtype == torch.bfloat16


def test_streamed_chunked_loss_matches_full_loss():
    """End-to-end check: pulling teacher logits chunk-by-chunk through the
    cache yields the same distillation loss as passing the full teacher tensor
    to ``chunked_distillation_loss`` directly."""
    from deepspeed.runtime.rlhf.losses import chunked_distillation_loss

    torch.manual_seed(0)
    s = torch.randn(2, 64, 32)
    t = torch.randn(2, 64, 32)
    mask = torch.ones(2, 64)

    direct = chunked_distillation_loss(s, t, mask, loss_type="reverse_kl", chunk_size=8)

    cache = TeacherLogitCache.from_gpu_logits(t, store_dtype=torch.float32)
    staged_full = cache.chunk_to_device(0, 64, torch.device("cpu"), dtype=torch.float32)
    via_cache = chunked_distillation_loss(s, staged_full, mask, loss_type="reverse_kl", chunk_size=8)

    assert torch.allclose(direct, via_cache, atol=1e-6)

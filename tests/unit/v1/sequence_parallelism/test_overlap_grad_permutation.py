# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.sequence.layer import DistributedAttention


class _LocalAttention(torch.nn.Module):

    def forward(self, query, key, value):
        return query + 2.0 * key + 3.0 * value


def test_overlap_backward_preserves_shape_changing_producers(monkeypatch):

    class FakeWork:

        def wait(self):
            pass

    class FakeStream:

        def wait_stream(self, stream):
            pass

    class FakeAccelerator:

        def __init__(self):
            self.stream = FakeStream()

        def current_stream(self):
            return self.stream

        def default_stream(self):
            return self.stream

    def all_to_all_single(output, input_tensor, group=None, async_op=False):
        output.copy_(input_tensor)
        return FakeWork() if async_op else None

    monkeypatch.setattr("deepspeed.sequence.layer.dist.get_world_size", lambda group: 2)
    monkeypatch.setattr("deepspeed.sequence.layer.dist.all_to_all_single", all_to_all_single)
    monkeypatch.setattr("deepspeed.sequence.layer.get_accelerator", FakeAccelerator)

    base_sources = tuple(
        torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2) / 16.0 + offset for offset in (0.0, 0.25, 0.5))

    def run(sp_stream):
        sources = tuple(base.detach().clone().requires_grad_(True) for base in base_sources)
        query, key, value = (source.transpose(1, 2) for source in sources)
        attention = DistributedAttention(
            local_attention=_LocalAttention(),
            sequence_process_group=object(),
            scatter_idx=2,
            gather_idx=1,
            sp_stream=sp_stream,
        )
        output = attention(query, key, value, batch_dim_idx=0)
        upstream_grad = torch.arange(1, output.numel() + 1, dtype=output.dtype).reshape_as(output)
        output.backward(upstream_grad)
        return tuple(source.grad for source in sources)

    try:
        synchronous_grads = run(sp_stream=None)
        overlap_grads = run(sp_stream=FakeStream())
    except Exception as exc:
        pytest.fail(f"shape-changing producer backward raised {type(exc).__name__}: {exc}")

    for synchronous, overlap, source in zip(synchronous_grads, overlap_grads, base_sources):
        assert synchronous is not None
        assert overlap is not None
        assert synchronous.shape == source.shape
        assert overlap.shape == source.shape
        assert torch.isfinite(synchronous).all()
        assert torch.isfinite(overlap).all()
        torch.testing.assert_close(overlap, synchronous)

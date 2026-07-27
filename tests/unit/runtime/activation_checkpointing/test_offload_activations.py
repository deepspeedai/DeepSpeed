# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.activation_checkpointing.offload_activations import (
    CheckpointHiddenStatesOffload,
    get_checkpoint_hidden_states_offloading_ctx_manager,
)


def test_unmarked_saved_tensors_pass_through():
    hidden_states = torch.randn(4, 8, requires_grad=True)
    linear = nn.Linear(8, 8)

    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_size=0)
    with offload:
        loss = linear(hidden_states).square().sum()
        loss.backward()

    assert hidden_states.grad is not None
    assert offload.stats.saved_tensors_seen > 0
    assert offload.stats.marked_tensors == 0
    assert offload.stats.offloaded_tensors == 0


def test_marked_cpu_tensor_is_skipped():

    def fn(x):
        return x.sin().square()

    x_base = torch.randn(4, 8, requires_grad=True)
    fn(checkpoint(fn, x_base, use_reentrant=False)).sum().backward()

    x = x_base.detach().clone().requires_grad_(True)
    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_size=0)
    with offload:
        offload.mark(x)
        loss = fn(checkpoint(fn, x, use_reentrant=False)).sum()
        loss.backward()

    # CPU tensors are marked but never offloaded.
    assert offload.stats.marked_tensors == 1
    assert offload.stats.skipped_marked_tensors >= 1
    assert offload.stats.offloaded_tensors == 0
    assert torch.allclose(x.grad, x_base.grad)


def test_marker_offloads_checkpoint_input():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "GradientCheckpointingLayer"):
        pytest.skip("transformers version does not provide GradientCheckpointingLayer")
    from functools import partial

    from transformers import GradientCheckpointingLayer

    class TinyCheckpointLayer(GradientCheckpointingLayer):

        def forward(self, hidden_states):
            hidden_states = hidden_states.sin()
            return hidden_states * hidden_states

    on_accelerator = get_accelerator().is_available() and not get_accelerator().is_synchronized_device()
    device = get_accelerator().device_name() if on_accelerator else "cpu"
    layer = TinyCheckpointLayer()
    layer.gradient_checkpointing = True
    layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    layer.to(device)
    layer.train()
    hidden_states = torch.randn(4, 8, device=device, requires_grad=True)

    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_size=0)
    with offload:
        loss = layer(hidden_states).sum()
        loss.backward()

    assert hidden_states.grad is not None
    assert offload.stats.marked_tensors == 1
    assert offload.stats.saved_tensors_seen >= offload.stats.marked_tensors
    if on_accelerator:
        assert offload.stats.offloaded_tensors == 1
        assert offload.stats.restored_tensors == 1
    else:
        assert offload.stats.skipped_marked_tensors == 1


@pytest.mark.skipif(not get_accelerator().is_available() or get_accelerator().is_synchronized_device(),
                    reason="requires a stream-capable accelerator")
def test_streams_offload_restore_matches_baseline():
    device = get_accelerator().device_name()
    torch.manual_seed(17)
    layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(4)]).to(device)

    def run_step(x):
        h = x
        manager = _current_ctx[0]
        for layer in layers:
            if manager is not None:
                manager.mark(h)
            h = checkpoint(layer, h, use_reentrant=False)
        return h.square().sum()

    def grads():
        return [p.grad.detach().clone() for p in layers.parameters()]

    x = torch.randn(8, 16, device=device, requires_grad=True)

    # Baseline without offloading.
    _current_ctx = [None]
    baseline_loss = run_step(x)
    baseline_loss.backward()
    baseline_grads = grads()
    baseline_x_grad = x.grad.detach().clone()

    manager = get_checkpoint_hidden_states_offloading_ctx_manager(use_streams=True,
                                                                  max_fwd_stash_size=2,
                                                                  min_offload_size=0)
    for step in range(2):
        layers.zero_grad(set_to_none=True)
        x.grad = None
        _current_ctx = [manager]
        with manager:
            loss = run_step(x)
            loss.backward()
        assert torch.allclose(loss, baseline_loss)
        assert torch.allclose(x.grad, baseline_x_grad)
        for g, bg in zip(grads(), baseline_grads):
            assert torch.allclose(g, bg)
        assert manager.stats.offloaded_tensors == manager.stats.restored_tensors
        assert manager.stats.offloaded_tensors > 0
        if step == 1:
            # Second step with the same manager reuses pooled CPU buffers.
            assert manager._cpu_buffer_pool_size > 0


@pytest.mark.skipif(not get_accelerator().is_available() or get_accelerator().is_synchronized_device(),
                    reason="requires a stream-capable accelerator")
def test_marked_view_with_storage_offset():

    def fn(x):
        return x.sin().square()

    device = get_accelerator().device_name()
    base = torch.randn(6, 8, device=device)
    x_base = base[2:].detach().clone().requires_grad_(True)
    fn(checkpoint(fn, x_base, use_reentrant=False)).sum().backward()

    x = base[2:].detach().requires_grad_(True)
    assert x.storage_offset() > 0
    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_size=0)
    with offload:
        offload.mark(x)
        loss = fn(checkpoint(fn, x, use_reentrant=False)).sum()
        loss.backward()

    assert offload.stats.offloaded_tensors == 1
    assert offload.stats.restored_tensors == 1
    assert torch.allclose(x.grad, x_base.grad)

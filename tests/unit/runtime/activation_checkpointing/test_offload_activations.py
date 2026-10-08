# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import inspect
from copy import deepcopy
from functools import partial

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.activation_checkpointing.offload_activations import (
    CheckpointHiddenStatesOffload,
    _ActivationOffloadEngine,
    _current_manager,
    get_checkpoint_hidden_states_offloading_ctx_manager,
)
from unit.common import DistributedTest

_ACCEL = get_accelerator().is_available() and not get_accelerator().is_synchronized_device()


def test_unmarked_saved_tensors_pass_through():
    hidden_states = torch.randn(4, 8, requires_grad=True)
    linear = nn.Linear(8, 8)

    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_bytes=0)
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
    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_bytes=0)
    with offload:
        offload.mark(x)
        loss = fn(checkpoint(fn, x, use_reentrant=False)).sum()
        loss.backward()

    # CPU tensors are marked but never offloaded.
    assert offload.stats.marked_tensors == 1
    assert offload.stats.skipped_marked_tensors >= 1
    assert offload.stats.offloaded_tensors == 0
    assert torch.allclose(x.grad, x_base.grad)


def test_keep_last_count_rejects_negative():
    with pytest.raises(ValueError, match="keep_last_count"):
        CheckpointHiddenStatesOffload(keep_last_count=-1)


def test_gradient_checkpointing_layer_signature_contract():
    """Pin the HF signature _checkpoint_offload_call depends on.

    GradientCheckpointingLayer itself does not define forward(); the contract
    lives on concrete decoder layers and on subclasses used with the marker.
    """
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "GradientCheckpointingLayer"):
        pytest.skip("transformers version does not provide GradientCheckpointingLayer")
    from transformers import GradientCheckpointingLayer

    class TinyCheckpointLayer(GradientCheckpointingLayer):

        def forward(self, hidden_states):
            return hidden_states

    for layer_cls in (TinyCheckpointLayer, ):
        params = [p for p in inspect.signature(layer_cls.forward).parameters.values() if p.name != "self"]
        assert params, f"{layer_cls.__name__}.forward has no params"
        assert params[0].name == "hidden_states", (
            f"_checkpoint_offload_call assumes args[0] is hidden_states, but "
            f"{layer_cls.__name__}.forward now starts with '{params[0].name}'. "
            "Update the marker patch in patch_gradient_checkpointing_layer_marker.")
        assert params[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    except ImportError:
        return
    if not issubclass(LlamaDecoderLayer, GradientCheckpointingLayer):
        pytest.skip("LlamaDecoderLayer is not a GradientCheckpointingLayer")
    params = [p for p in inspect.signature(LlamaDecoderLayer.forward).parameters.values() if p.name != "self"]
    assert params[0].name == "hidden_states"
    assert params[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_marker_patch_restored_after_context_exit():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "GradientCheckpointingLayer"):
        pytest.skip("transformers version does not provide GradientCheckpointingLayer")
    from functools import partial
    from transformers import GradientCheckpointingLayer

    class TinyCheckpointLayer(GradientCheckpointingLayer):

        def forward(self, hidden_states):
            return hidden_states.sin()

    orig_call = GradientCheckpointingLayer.__call__
    layer = TinyCheckpointLayer()
    layer.gradient_checkpointing = True
    layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    layer.train()
    hidden_states = torch.randn(4, 8, requires_grad=True)

    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_bytes=0, keep_last_count=0)
    with offload:
        assert GradientCheckpointingLayer.__call__ is not orig_call
        layer(hidden_states)
    assert GradientCheckpointingLayer.__call__ is orig_call
    assert offload.stats.marked_tensors == 1

    layer(hidden_states)
    assert offload.stats.marked_tensors == 1


def test_marker_patch_restored_on_exception():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "GradientCheckpointingLayer"):
        pytest.skip("transformers version does not provide GradientCheckpointingLayer")
    from transformers import GradientCheckpointingLayer

    orig_call = GradientCheckpointingLayer.__call__
    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_bytes=0)
    with pytest.raises(ValueError, match="boom"):
        with offload:
            raise ValueError("boom")
    assert GradientCheckpointingLayer.__call__ is orig_call


def test_nested_managers_unpatch_on_outer_exit():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "GradientCheckpointingLayer"):
        pytest.skip("transformers version does not provide GradientCheckpointingLayer")
    from transformers import GradientCheckpointingLayer

    orig_call = GradientCheckpointingLayer.__call__
    outer = CheckpointHiddenStatesOffload(use_streams=False, min_offload_bytes=0)
    inner = CheckpointHiddenStatesOffload(use_streams=False, min_offload_bytes=0)
    with outer:
        patched = GradientCheckpointingLayer.__call__
        assert patched is not orig_call
        with inner:
            assert GradientCheckpointingLayer.__call__ is patched
        assert GradientCheckpointingLayer.__call__ is patched
    assert GradientCheckpointingLayer.__call__ is orig_call


def test_same_manager_is_not_reentrant():
    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_bytes=0)
    with offload:
        with pytest.raises(RuntimeError, match="not re-entrant"):
            offload.__enter__()


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

    on_accelerator = _ACCEL
    device = get_accelerator().device_name() if on_accelerator else "cpu"
    layer = TinyCheckpointLayer()
    layer.gradient_checkpointing = True
    layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    layer.to(device)
    layer.train()
    hidden_states = torch.randn(4, 8, device=device, requires_grad=True)

    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_bytes=0, keep_last_count=0)
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


@pytest.mark.skipif(not _ACCEL, reason="requires a stream-capable accelerator")
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
                                                                  max_fwd_stash_count=2,
                                                                  min_offload_bytes=0,
                                                                  keep_last_count=0)
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
            assert manager._cpu_buffer_pool_count > 0


@pytest.mark.skipif(not _ACCEL, reason="requires a stream-capable accelerator")
@pytest.mark.parametrize("use_streams", [False, True])
@pytest.mark.parametrize("keep_last_count", [0, 1, 2])
def test_keep_last_count_matches_baseline(use_streams, keep_last_count):
    device = get_accelerator().device_name()
    torch.manual_seed(3)
    layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(4)]).to(device)

    def run(x, manager=None):
        h = x
        last_marked = None
        for layer in layers:
            if manager is not None:
                manager.mark(h)
                last_marked = h
            h = checkpoint(layer, h, use_reentrant=False)
        return h.square().sum(), last_marked

    x = torch.randn(8, 16, device=device, requires_grad=True)
    baseline_loss, _ = run(x)
    baseline_loss.backward()
    baseline_x_grad = x.grad.detach().clone()
    baseline_grads = [p.grad.detach().clone() for p in layers.parameters()]

    layers.zero_grad(set_to_none=True)
    x.grad = None
    offload = CheckpointHiddenStatesOffload(use_streams=use_streams,
                                            min_offload_bytes=0,
                                            keep_last_count=keep_last_count)
    with offload:
        loss, last_marked = run(x, manager=offload)
        if keep_last_count:
            expected_ids = list(range(offload._next_id - keep_last_count + 1, offload._next_id + 1))
        else:
            expected_ids = []
        assert list(offload._keep_last) == expected_ids
        if keep_last_count == 1:
            assert next(iter(offload._keep_last.values())) is last_marked
        loss.backward()

    assert torch.allclose(loss, baseline_loss)
    assert torch.allclose(x.grad, baseline_x_grad)
    for g, bg in zip([p.grad for p in layers.parameters()], baseline_grads):
        assert torch.allclose(g, bg)
    assert offload.stats.marked_tensors == 4
    assert offload.stats.offloaded_tensors == 4 - keep_last_count
    assert offload.stats.restored_tensors == 4 - keep_last_count
    assert offload.stats.kept_last_tensors == keep_last_count


@pytest.mark.skipif(not _ACCEL, reason="requires a stream-capable accelerator")
def test_marked_view_with_storage_offset():

    def fn(x):
        return x.sin().square()

    device = get_accelerator().device_name()
    base = torch.randn(6, 8, device=device)
    x_base = base[2:].detach().clone().requires_grad_(True)
    fn(checkpoint(fn, x_base, use_reentrant=False)).sum().backward()

    x = base[2:].detach().requires_grad_(True)
    assert x.storage_offset() > 0
    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_bytes=0, keep_last_count=0)
    with offload:
        offload.mark(x)
        loss = fn(checkpoint(fn, x, use_reentrant=False)).sum()
        loss.backward()

    assert offload.stats.offloaded_tensors == 1
    assert offload.stats.restored_tensors == 1
    assert torch.allclose(x.grad, x_base.grad)


@pytest.mark.skipif(not _ACCEL, reason="requires a stream-capable accelerator")
@pytest.mark.parametrize("use_pin_memory", [True, False])
def test_contiguous_offload_buffer_pin_flag(use_pin_memory):

    def fn(x):
        return x.sin().square()

    device = get_accelerator().device_name()
    x = torch.randn(4, 8, device=device, requires_grad=True)
    offload = CheckpointHiddenStatesOffload(use_streams=False,
                                            min_offload_bytes=0,
                                            keep_last_count=0,
                                            use_pin_memory=use_pin_memory)
    with offload:
        offload.mark(x)
        loss = fn(checkpoint(fn, x, use_reentrant=False)).sum()
        cpu_bufs = [tracked[0] for tracked in offload._tracker.values()]
        assert cpu_bufs
        for buf in cpu_bufs:
            # Ask the accelerator, so the assertion holds for either pin backend.
            assert get_accelerator().is_pinned(buf) is use_pin_memory
        loss.backward()
    assert offload.stats.offloaded_tensors == 1


@pytest.mark.skipif(not _ACCEL, reason="requires a stream-capable accelerator")
def test_strided_view_offload_matches_baseline():

    def fn(x):
        return x.sin().square()

    device = get_accelerator().device_name()
    base = torch.randn(6, 8, device=device)
    x = base[:, ::2].detach().requires_grad_(True)
    assert not x.is_contiguous()
    x_base = torch.empty_strided(x.size(), x.stride(), dtype=x.dtype,
                                 device=device).copy_(x).detach().requires_grad_(True)
    fn(checkpoint(fn, x_base, use_reentrant=False)).sum().backward()
    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_bytes=0, keep_last_count=0)
    with offload:
        offload.mark(x)
        loss = fn(checkpoint(fn, x, use_reentrant=False)).sum()
        cpu_bufs = [tracked[0] for tracked in offload._tracker.values()]
        assert cpu_bufs
        assert cpu_bufs[0].is_contiguous()
        loss.backward()

    assert offload.stats.offloaded_tensors == 1
    assert torch.allclose(x.grad, x_base.grad)


@pytest.mark.parametrize("use_pin_memory", [True, False])
def test_empty_cpu_like_pins_via_accelerator(monkeypatch, use_pin_memory):
    accel = get_accelerator()
    calls = []
    orig = accel.pin_memory

    def wrapped(*args, **kwargs):
        calls.append(kwargs)
        return orig(*args, **kwargs)

    monkeypatch.setattr(accel, "pin_memory", wrapped)
    offload = CheckpointHiddenStatesOffload(use_streams=False,
                                            min_offload_bytes=0,
                                            keep_last_count=0,
                                            use_pin_memory=use_pin_memory)
    src = torch.randn(6, 8)[:, ::2]
    assert not src.is_contiguous()
    buf, _ = offload._empty_cpu_like(src)
    assert len(calls) == (1 if use_pin_memory else 0)
    assert tuple(buf.size()) == tuple(src.size())
    assert buf.is_contiguous()


def test_buffer_key_shared_across_strides():
    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_bytes=0, keep_last_count=0)
    dense = torch.randn(6, 4)
    strided = torch.randn(6, 8)[:, ::2]
    assert offload._buffer_key(dense) == offload._buffer_key(strided)


@pytest.mark.skipif(not _ACCEL, reason="requires a stream-capable accelerator")
def test_overlapping_view_is_skipped():

    def fn(x):
        return x.sin().square()

    device = get_accelerator().device_name()
    base = torch.randn(4, 8, device=device)
    x = base.expand(2, 4, 8).detach().requires_grad_(True)
    assert 0 in x.stride()
    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_bytes=0, keep_last_count=0)
    with offload:
        offload.mark(x)
        loss = fn(checkpoint(fn, x, use_reentrant=False)).sum()
        loss.backward()
    assert offload.stats.offloaded_tensors == 0
    assert offload.stats.skipped_marked_tensors >= 1
    assert x.grad is not None


@pytest.mark.skipif(not _ACCEL, reason="requires a stream-capable accelerator")
def test_retain_graph_second_backward():

    def fn(x):
        return x.sin().square()

    device = get_accelerator().device_name()
    x = torch.randn(4, 8, device=device, requires_grad=True)
    offload = CheckpointHiddenStatesOffload(use_streams=False, min_offload_bytes=0, keep_last_count=0)
    with offload:
        offload.mark(x)
        loss = fn(checkpoint(fn, x, use_reentrant=False)).sum()
        loss.backward(retain_graph=True)
        assert offload.stats.restored_tensors == 1
        x.grad = None
        loss.backward()
    assert offload.stats.restored_tensors == 1
    assert x.grad is not None


@pytest.mark.skipif(not _ACCEL, reason="requires a stream-capable accelerator")
def test_zero_fwd_stash_retains_no_gpu_activation():
    device = get_accelerator().device_name()
    torch.manual_seed(11)
    layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(4)]).to(device)

    def run(x, manager=None):
        h = x
        for layer in layers:
            if manager is not None:
                manager.mark(h)
            h = checkpoint(layer, h, use_reentrant=False)
        return h.square().sum()

    x = torch.randn(8, 16, device=device, requires_grad=True)
    baseline_loss = run(x)
    baseline_loss.backward()
    baseline_x_grad = x.grad.detach().clone()
    baseline_grads = [p.grad.detach().clone() for p in layers.parameters()]

    layers.zero_grad(set_to_none=True)
    x.grad = None
    offload = CheckpointHiddenStatesOffload(use_streams=True,
                                            min_offload_bytes=0,
                                            max_fwd_stash_count=0,
                                            keep_last_count=0)
    with offload:
        loss = run(x, manager=offload)
        assert not offload._fwd_stash
        loss.backward()

    assert torch.allclose(x.grad, baseline_x_grad)
    for g, bg in zip([p.grad for p in layers.parameters()], baseline_grads):
        assert torch.allclose(g, bg)
    assert offload.stats.offloaded_tensors == 4
    assert offload.stats.restored_tensors == 4


@pytest.mark.skipif(not _ACCEL, reason="requires a stream-capable accelerator")
def test_prefetch_starts_reload_of_previous_id():
    device = get_accelerator().device_name()
    engine = _ActivationOffloadEngine(keep_last_count=0, min_offload_bytes=0, max_fwd_stash_count=0, prefetch=True)
    tensors = [torch.randn(8, 16, device=device) for _ in range(3)]
    ids = [engine.offload_input(t) for t in tensors]
    restored = engine.restore_input(ids[-1])
    assert torch.allclose(restored, tensors[-1])
    assert ids[-2] in engine._reloads
    assert engine.stats.prefetched_tensors >= 1
    restored_prev = engine.restore_input(ids[-2])
    assert torch.allclose(restored_prev, tensors[-2])


@pytest.mark.skipif(not _ACCEL, reason="requires a stream-capable accelerator")
def test_prefetch_restore_matches_baseline():
    device = get_accelerator().device_name()
    torch.manual_seed(23)
    layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(4)]).to(device)

    def run_step(x, manager=None):
        h = x
        for layer in layers:
            if manager is not None:
                manager.mark(h)
            h = checkpoint(layer, h, use_reentrant=False)
        return h.square().sum()

    x = torch.randn(8, 16, device=device, requires_grad=True)
    baseline_loss = run_step(x)
    baseline_loss.backward()
    baseline_x_grad = x.grad.detach().clone()
    baseline_grads = [p.grad.detach().clone() for p in layers.parameters()]

    layers.zero_grad(set_to_none=True)
    x.grad = None
    offload = get_checkpoint_hidden_states_offloading_ctx_manager(use_streams=True,
                                                                  min_offload_bytes=0,
                                                                  keep_last_count=1,
                                                                  prefetch=True)
    with offload:
        loss = run_step(x, manager=offload)
        loss.backward()
    assert torch.allclose(loss, baseline_loss)
    assert torch.allclose(x.grad, baseline_x_grad)
    for g, bg in zip([p.grad for p in layers.parameters()], baseline_grads):
        assert torch.allclose(g, bg)
    assert offload.stats.prefetched_tensors >= 1
    assert offload.stats.offloaded_tensors == offload.stats.restored_tensors


@pytest.mark.skipif(not _ACCEL, reason="requires a stream-capable accelerator")
def test_prefetch_second_step_matches_baseline():
    """Allocator reuse across steps is where a missing compute-stream wait corrupts."""
    device = get_accelerator().device_name()
    torch.manual_seed(29)
    layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(8)]).to(device)
    x = torch.randn(32, 64, device=device, requires_grad=True)

    def run_step(h_in, module, manager=None):
        h = h_in
        for layer in module:
            if manager is not None:
                manager.mark(h)
            h = checkpoint(layer, h, use_reentrant=False)
        return h.square().sum()

    opt = torch.optim.SGD(layers.parameters(), lr=0.01)
    baseline = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        x.grad = None
        loss = run_step(x, layers)
        loss.backward()
        opt.step()
        baseline.append(float(loss.detach()))

    torch.manual_seed(29)
    layers_p = nn.ModuleList([nn.Linear(64, 64) for _ in range(8)]).to(device)
    x_p = torch.randn(32, 64, device=device, requires_grad=True)
    opt_p = torch.optim.SGD(layers_p.parameters(), lr=0.01)
    offload = get_checkpoint_hidden_states_offloading_ctx_manager(use_streams=True,
                                                                  min_offload_bytes=0,
                                                                  keep_last_count=1,
                                                                  prefetch=True)
    prefetch_losses = []
    for _ in range(3):
        opt_p.zero_grad(set_to_none=True)
        x_p.grad = None
        with offload:
            loss = run_step(x_p, layers_p, manager=offload)
            loss.backward()
        opt_p.step()
        prefetch_losses.append(float(loss.detach()))

    for b, p in zip(baseline, prefetch_losses):
        assert abs(b - p) / max(abs(b), 1e-6) < 1e-4
    assert offload.stats.prefetched_tensors >= 1


def _cpu_offload_ds_config(enabled):
    return {
        "train_batch_size": 8,
        "train_micro_batch_size_per_gpu": 8,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "zero_optimization": {
            "stage": 0
        },
        "activation_checkpointing": {
            "cpu_checkpointing": enabled
        },
    }


@pytest.mark.skipif(not _ACCEL, reason="requires a stream-capable accelerator")
class TestEngineCpuCheckpointingConfig(DistributedTest):
    world_size = 1

    def test_cpu_checkpointing_config_configures_native_and_hf_offload(self):
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "GradientCheckpointingLayer"):
            pytest.skip("transformers version does not provide GradientCheckpointingLayer")
        from transformers import GradientCheckpointingLayer

        class TinyCheckpointLayer(GradientCheckpointingLayer):

            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, dim)

            def forward(self, hidden_states):
                return self.linear(hidden_states).sin()

        class TinyGCModel(nn.Module):

            def __init__(self, dim, n_layers):
                super().__init__()
                self.layers = nn.ModuleList([TinyCheckpointLayer(dim) for _ in range(n_layers)])
                for layer in self.layers:
                    layer.gradient_checkpointing = True
                    layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
                self.head = nn.Linear(dim, dim)

            def forward(self, hidden_states):
                for layer in self.layers:
                    hidden_states = layer(hidden_states)
                return self.head(hidden_states).square().sum()

        import deepspeed.runtime.activation_checkpointing.checkpointing as ds_ckpt
        saved = {
            "PARTITION_ACTIVATIONS": ds_ckpt.PARTITION_ACTIVATIONS,
            "CONTIGUOUS_CHECKPOINTING": ds_ckpt.CONTIGUOUS_CHECKPOINTING,
            "num_layers": ds_ckpt.num_layers,
            "CPU_CHECKPOINT": ds_ckpt.CPU_CHECKPOINT,
            "SYNCHRONIZE": ds_ckpt.SYNCHRONIZE,
            "PROFILE_TIME": ds_ckpt.PROFILE_TIME,
            "mpu": ds_ckpt.mpu,
            "deepspeed_checkpointing_enabled": ds_ckpt.deepspeed_checkpointing_enabled,
        }
        device = get_accelerator().device_name()
        dim = 128
        n_layers = 4
        torch.manual_seed(7)
        base = TinyGCModel(dim, n_layers).to(device)
        x = torch.randn(8, dim, device=device)

        try:
            ref_model = deepcopy(base)
            ref_engine, _, _, _ = deepspeed.initialize(model=ref_model,
                                                       model_parameters=ref_model.parameters(),
                                                       config=_cpu_offload_ds_config(False))
            assert ref_engine._activation_cpu_offload_ctx is None
            x_ref = x.clone()
            ref_loss = ref_engine(x_ref)
            ref_engine.backward(ref_loss)
            ref_grads = [p.grad.detach().clone() for p in ref_engine.module.parameters() if p.grad is not None]
            ref_engine.destroy()

            test_model = deepcopy(base)
            test_engine, _, _, _ = deepspeed.initialize(model=test_model,
                                                        model_parameters=test_model.parameters(),
                                                        config=_cpu_offload_ds_config(True))
            assert ds_ckpt.is_configured()
            assert ds_ckpt.CPU_CHECKPOINT
            assert test_engine._activation_cpu_offload_ctx is not None
            x_test = x.clone()
            test_loss = test_engine(x_test)
            test_engine.backward(test_loss)
            stats = test_engine._activation_cpu_offload_ctx.stats
            assert stats.marked_tensors == n_layers
            assert stats.offloaded_tensors > 0
            assert stats.offloaded_tensors == stats.restored_tensors
            assert torch.allclose(test_loss, ref_loss)
            test_grads = [p.grad.detach().clone() for p in test_engine.module.parameters() if p.grad is not None]
            for g, rg in zip(test_grads, ref_grads):
                assert torch.allclose(g, rg)
            assert not test_engine._activation_cpu_offload_entered
            test_engine.eval()
            with torch.no_grad():
                test_engine(x.clone())
            assert not test_engine._activation_cpu_offload_entered
            test_engine.destroy()
        finally:
            for name, value in saved.items():
                setattr(ds_ckpt, name, value)

    def test_cpu_checkpointing_exits_offload_on_backward_exception(self):
        """A raised engine.backward() must still leave the CPU-offload context."""
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "GradientCheckpointingLayer"):
            pytest.skip("transformers version does not provide GradientCheckpointingLayer")
        from transformers import GradientCheckpointingLayer

        import deepspeed.runtime.activation_checkpointing.checkpointing as ds_ckpt

        class TinyCheckpointLayer(GradientCheckpointingLayer):

            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, dim)

            def forward(self, hidden_states):
                return self.linear(hidden_states)

        class TinyGCModel(nn.Module):

            def __init__(self, dim):
                super().__init__()
                self.layer = TinyCheckpointLayer(dim)
                self.layer.gradient_checkpointing = True
                self.layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
                self.head = nn.Linear(dim, 1)

            def forward(self, hidden_states):
                return self.head(self.layer(hidden_states)).sum()

        class _RaiseInBackward(torch.autograd.Function):

            @staticmethod
            def forward(ctx, value):
                return value

            @staticmethod
            def backward(ctx, grad_output):
                raise RuntimeError("injected incomplete checkpoint backward")

        saved = {
            "PARTITION_ACTIVATIONS": ds_ckpt.PARTITION_ACTIVATIONS,
            "CONTIGUOUS_CHECKPOINTING": ds_ckpt.CONTIGUOUS_CHECKPOINTING,
            "num_layers": ds_ckpt.num_layers,
            "CPU_CHECKPOINT": ds_ckpt.CPU_CHECKPOINT,
            "SYNCHRONIZE": ds_ckpt.SYNCHRONIZE,
            "PROFILE_TIME": ds_ckpt.PROFILE_TIME,
            "mpu": ds_ckpt.mpu,
            "deepspeed_checkpointing_enabled": ds_ckpt.deepspeed_checkpointing_enabled,
        }
        device = get_accelerator().device_name()
        engine = None
        try:
            model = TinyGCModel(16).to(device)
            engine, _, _, _ = deepspeed.initialize(model=model,
                                                   model_parameters=model.parameters(),
                                                   config=_cpu_offload_ds_config(True))
            assert engine._activation_cpu_offload_ctx is not None
            x = torch.randn(8, 16, device=device)
            loss = _RaiseInBackward.apply(engine(x))
            assert engine._activation_cpu_offload_entered
            with pytest.raises(RuntimeError, match="injected incomplete checkpoint backward"):
                engine.backward(loss)
            assert not engine._activation_cpu_offload_entered
            assert not engine._running_engine_backward
            assert _current_manager() is None
            retry = engine(x.clone())
            engine.backward(retry)
            assert not engine._activation_cpu_offload_entered
            assert _current_manager() is None
        finally:
            if engine is not None:
                engine.destroy()
            for name, value in saved.items():
                setattr(ds_ckpt, name, value)

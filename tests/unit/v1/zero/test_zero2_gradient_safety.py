# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Compatibility defaults, CPU bookkeeping, and CUDA gradient ordering tests."""

from contextlib import nullcontext
import copy
import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.ops import __compatible_ops__
from deepspeed.runtime import engine as engine_module
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.zero import stage_1_and_2 as zero
from deepspeed.runtime.zero.config import DeepSpeedZeroConfig, offload_gradient_safety_enabled
from deepspeed.runtime.zero.offload_config import DeepSpeedZeroOffloadOptimizerConfig
from deepspeed.utils.timer import NoopTimer

OPTIONS = ("check_offload_gradients", "accumulate_offload_gradients")
REMOVED_OPTIONS = ("copy_oversized_gradients", "track_gradient_streams")


def make_optimizer(dtype=torch.bfloat16, device="cpu", low_precision=False, cpu_offload=True, **options):
    opt = zero.DeepSpeedZeroOptimizer.__new__(zero.DeepSpeedZeroOptimizer)
    for name in OPTIONS:
        setattr(opt, name, options.get(name, False))
    opt.cpu_offload = cpu_offload
    opt._offload_gradient_safety_enabled = offload_gradient_safety_enabled(stage=2, cpu_offload=cpu_offload)
    opt.cpu_offload_pin_memory = False
    opt.device = "cpu"
    opt.dtype = dtype
    opt.master_weights_and_grads_dtype = dtype if low_precision else torch.float32
    opt.low_precision_master_weights_and_grads = low_precision
    opt.use_grad_accum_attribute = False
    opt.partition_gradients = True
    opt.contiguous_gradients = True
    opt.overlap_comm = False
    opt.zenflow = False
    opt.has_moe_layers = False
    opt.autoep_folding_tp_group = None
    opt.gradient_accumulation_steps = 1
    opt.set_gradient_accumulation_boundary(True)
    opt.compute_grad_norm = True
    opt.averaged_gradients = {}
    opt._muon_pending_momentum = {}
    opt.micro_step_id = 0
    opt._pending_offload_events = {}
    opt._offload_accumulated_param_ids = set()
    opt.accumulated_grads_in_cpu = {}
    opt.norm_for_param_grads = {}
    opt.local_overflow = False
    opt.overflow = False
    opt.dp_process_group = None
    opt.model_parallel_rank = 0
    opt.model_parallel_group = None
    opt.ignore_unused_parameters = True
    opt.temp_grad_buffer_for_gpu_offload = torch.zeros(8, dtype=dtype, device=device)
    param = torch.nn.Parameter(torch.zeros(8, dtype=dtype, device=device))
    param.param_idx_in_group = 0
    opt.param_id = {id(param): 0}
    opt.grad_position = {0: [0, 0, 0, 8]}
    opt.bit16_groups = [[param]]
    opt.params_in_partition = [[param]]
    opt.params_already_reduced = [False]
    opt.extra_large_param_to_reduce = {}
    opt.reduce_bucket_size = 4
    opt.ipg_buckets = {torch.float32: zero.IPGBucket(buffer=[torch.empty(16, dtype=dtype, device=device)])}
    master = torch.nn.Parameter(torch.zeros(8, dtype=opt.master_weights_and_grads_dtype))
    master.grad = torch.zeros_like(master)
    opt.single_partition_of_fp32_groups = [master]
    opt.get_param_comm_dtype = lambda p: torch.float32
    opt.report_ipg_memory_usage = lambda *args: None
    return opt, param, master


def backward_values(opt, param, values, boundaries):
    for index, (value, boundary) in enumerate(zip(values, boundaries, strict=True)):
        opt.micro_step_id = index
        opt.set_gradient_accumulation_boundary(boundary)
        param.grad = torch.full_like(param, value)
        opt.copy_grads_in_partition(param)
        param.grad = None
    opt._wait_for_offload_copies()


def test_config_defaults():
    config = DeepSpeedZeroConfig()
    assert all(name not in config.model_dump() for name in REMOVED_OPTIONS)
    assert not config.check_offload_gradients
    assert not config.accumulate_offload_gradients


@pytest.mark.parametrize("requested", [None, False, True])
@pytest.mark.parametrize("name", REMOVED_OPTIONS)
def test_config_rejects_removed_options(name, requested):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        DeepSpeedZeroConfig(stage=2, offload_optimizer={"device": "cpu"}, **{name: requested})


@pytest.mark.parametrize("context,expected", [
    ({}, True),
    ({
        "stage": 0
    }, False),
    ({
        "stage": 1
    }, False),
    ({
        "stage": 3
    }, False),
    ({
        "cpu_offload": False
    }, False),
    ({
        "zenflow": True
    }, False),
    ({
        "pipeline_parallel": True
    }, False),
    ({
        "deepcompile": True
    }, False),
])
def test_gradient_safety_compatibility(context, expected):
    kwargs = dict(stage=2, cpu_offload=True)
    kwargs.update(context)
    assert offload_gradient_safety_enabled(**kwargs) is expected


@pytest.mark.parametrize("name", OPTIONS)
@pytest.mark.parametrize("stage", [0, 1, 3])
def test_config_rejects_wrong_stage(name, stage):
    with pytest.raises(ValueError, match="ZeRO-2"):
        DeepSpeedZeroConfig(stage=stage, offload_optimizer={"device": "cpu"}, **{name: True})


@pytest.mark.parametrize("name", OPTIONS)
def test_config_accepts_independent_options(name):
    config = DeepSpeedZeroConfig(stage=2, offload_optimizer={"device": "cpu"}, **{name: True})
    assert getattr(config, name)
    assert all(not getattr(config, other) for other in OPTIONS if other != name)


@pytest.mark.parametrize("name", ["check_offload_gradients", "accumulate_offload_gradients"])
def test_config_requires_cpu_offload(name):
    with pytest.raises(ValueError, match="CPU optimizer offload"):
        DeepSpeedZeroConfig(stage=2, **{name: True})


@pytest.mark.parametrize("name", OPTIONS)
def test_config_rejects_zenflow(name):
    with pytest.raises(ValueError, match="without ZenFlow"):
        DeepSpeedZeroConfig(stage=2, zenflow={}, offload_optimizer={"device": "cpu"}, **{name: True})


def make_engine(zero_config, deepcompile=False, pipeline=False):
    engine = engine_module.DeepSpeedEngine.__new__(engine_module.DeepSpeedEngine)
    torch.nn.Module.__init__(engine)
    engine.destroy = lambda: None
    engine._config = DeepSpeedConfig({
        "train_batch_size": 1,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": zero_config,
        "compile": {
            "deepcompile": deepcompile
        },
    })
    module = torch.nn.Linear(2, 2)
    if pipeline:
        module = engine_module.PipelineModule.__new__(engine_module.PipelineModule)
        torch.nn.Module.__init__(module)
        module.add_module("linear", torch.nn.Linear(2, 2))
    engine._set_client_model(module)
    engine.param_names = {param: name for name, param in engine.module.named_parameters()}
    engine.mpu = None
    engine.seq_data_parallel_group = None
    engine.has_moe_layers = False
    engine.zenflow = zero_config.get("zenflow") is not None
    engine.gradient_average = True
    engine._is_compiled = False
    engine._deepcompile_active = False
    return engine


@pytest.mark.parametrize("enabled", [(), *[(name, ) for name in OPTIONS], OPTIONS])
def test_engine_forwards_independent_gradient_safety_options(monkeypatch, enabled):
    engine = make_engine({
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        },
        **{
            name: name in enabled
            for name in OPTIONS
        },
    })
    base_optimizer = torch.optim.Adam(engine.module.parameters())
    captured = {}

    def capture_optimizer(optimizer, param_names, **kwargs):
        captured.update(kwargs)
        return optimizer

    monkeypatch.setattr(engine_module, "DeepSpeedZeroOptimizer", capture_optimizer)
    assert engine._configure_zero_optimizer(base_optimizer) is base_optimizer
    for name in OPTIONS:
        assert captured[name] is (name in enabled)
    assert captured["offload_optimizer_config"].device == "cpu"
    assert captured["partition_grads"]


@pytest.mark.parametrize("zero_config,deepcompile,pipeline,expected", [
    ({
        "stage": 2
    }, False, False, False),
    ({
        "stage": 2,
        "overlap_comm": True
    }, False, False, False),
    ({
        "stage": 1,
        "offload_optimizer": {
            "device": "cpu"
        }
    }, False, False, False),
    ({
        "stage": 2,
        "contiguous_gradients": False
    }, False, False, False),
    ({
        "stage": 2,
        "contiguous_gradients": False,
        "offload_optimizer": {
            "device": "cpu"
        }
    }, False, False, True),
    ({
        "stage": 2,
        "offload_optimizer": {
            "device": "none"
        }
    }, False, False, False),
    ({
        "stage": 2,
        "offload_optimizer": {
            "device": "nvme"
        }
    }, False, False, False),
    ({
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    }, False, False, True),
    ({
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        },
        "overlap_comm": True
    }, False, False, True),
    ({
        "stage": 2,
        "zenflow": {},
        "offload_optimizer": {
            "device": "cpu"
        }
    }, False, False, False),
    ({
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    }, True, False, False),
    ({
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    }, False, True, False),
])
def test_engine_automatic_defaults(monkeypatch, zero_config, deepcompile, pipeline, expected):
    engine = make_engine(zero_config, deepcompile, pipeline)
    captured = {}

    def capture_optimizer(optimizer, param_names, **kwargs):
        captured.update(kwargs)
        return optimizer

    monkeypatch.setattr(engine_module, "DeepSpeedZeroOptimizer", capture_optimizer)
    monkeypatch.setattr(engine_module.ZenFlowZeroOptimizer, "create", lambda **kwargs: capture_optimizer)
    engine._configure_zero_optimizer(torch.optim.Adam(engine.module.parameters()))
    assert captured["pipeline_parallel"] is pipeline
    assert captured["deepcompile"] is deepcompile
    assert offload_gradient_safety_enabled(stage=2 if captured["partition_grads"] else 1,
                                           cpu_offload=(captured["offload_optimizer_config"] is not None
                                                        and captured["offload_optimizer_config"].device == "cpu"),
                                           zenflow=captured["zenflow_config"] is not None,
                                           pipeline_parallel=captured["pipeline_parallel"],
                                           deepcompile=captured["deepcompile"]) is expected
    assert not captured["check_offload_gradients"]
    assert not captured["accumulate_offload_gradients"]


@pytest.mark.parametrize("name", OPTIONS)
@pytest.mark.parametrize("feature", ["deepcompile", "pipeline"])
def test_engine_rejects_explicit_unsupported_options_before_construction(monkeypatch, name, feature):
    engine = make_engine({"stage": 2, "offload_optimizer": {"device": "cpu"}, name: True}, **{feature: True})
    monkeypatch.setattr(engine_module, "DeepSpeedZeroOptimizer",
                        lambda *args, **kwargs: pytest.fail("unsupported optimizer was constructed"))
    with pytest.raises(ValueError, match="DeepCompile|pipeline parallelism"):
        engine._configure_zero_optimizer(torch.optim.Adam(engine.module.parameters()))


@pytest.mark.parametrize("outcome", ["success", "fallback", "failure"])
def test_deepcompile_does_not_change_resolved_optimizer_policy(monkeypatch, outcome):
    engine = make_engine({"stage": 2, "offload_optimizer": {"device": "cpu"}}, deepcompile=True)
    captured = {}

    def capture_optimizer(optimizer, param_names, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(engine_module, "DeepSpeedZeroOptimizer", capture_optimizer)
    engine.optimizer = engine._configure_zero_optimizer(torch.optim.Adam(engine.module.parameters()))
    engine.module_forward_pre_hook = None
    engine.module_forward_post_hook = None
    # Isolate the policy regression from native compiler availability. The backend
    # may succeed, fall back, or fail without toggling a live optimizer's policy.
    engine.get_deepspeed_compile_backend = lambda *args: (None if outcome == "fallback" else "eager", None)

    def compile_module(**kwargs):
        if outcome == "failure":
            raise RuntimeError("compile failed")

    monkeypatch.setattr(engine.module, "compile", compile_module)
    engine._create_module_forward_pre_hook = lambda: None
    engine._create_module_forward_post_hook = lambda: None
    if outcome == "failure":
        with pytest.raises(RuntimeError, match="compile failed"):
            engine.compile(backend="eager")
        assert not engine.is_compiled
        assert not engine.is_deepcompile_active()
    else:
        engine.compile(backend="eager")
        assert engine.is_compiled
        assert engine.is_deepcompile_active() is (outcome == "success")
    assert engine.optimizer.deepcompile


def test_late_deepcompile_rejects_without_disabling_live_protections():
    engine = make_engine({"stage": 2, "offload_optimizer": {"device": "cpu"}})
    engine.optimizer = SimpleNamespace(_offload_gradient_safety_enabled=True)
    engine._config.compile_config.deepcompile = True
    with pytest.raises(ValueError, match="before initializing"):
        engine.compile(backend="eager")
    assert engine.optimizer._offload_gradient_safety_enabled
    assert not engine.is_compiled


def test_ordinary_compile_preserves_protections():
    engine = make_engine({"stage": 2, "offload_optimizer": {"device": "cpu"}})
    engine.optimizer = SimpleNamespace(_offload_gradient_safety_enabled=True)
    engine.compile(backend="eager")
    x = torch.ones(1, 2)
    torch.testing.assert_close(engine.module(x), torch.nn.functional.linear(x, engine.module.weight,
                                                                            engine.module.bias))
    assert engine.optimizer._offload_gradient_safety_enabled
    assert engine.is_compiled


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("cpu_offload", [False, True])
@pytest.mark.parametrize("bucket_size", [4, 8, 16])
def test_oversized_copy_is_independent_and_preserves_routing(dtype, cpu_offload, bucket_size):
    opt, param, _ = make_optimizer(dtype, cpu_offload=cpu_offload)
    opt.reduce_bucket_size = bucket_size
    opt.reduce_ipg_grads = lambda **kwargs: None
    original = torch.arange(8, dtype=dtype)
    param.grad = original
    original_alias = original.view_as(original)
    opt.reduce_independent_p_g_buckets_and_remove_grads(param, 0)
    oversized = param.numel() > bucket_size
    assert (torch.float32 in opt.extra_large_param_to_reduce) == oversized
    assert torch.equal(param.grad, original_alias)
    assert param.grad.dtype == dtype
    independent = not oversized or cpu_offload
    assert (param.grad.data_ptr() != original_alias.data_ptr()) == independent
    original_alias.fill_(42)
    assert torch.equal(param.grad, torch.full_like(param, 42)) != independent
    assert opt.ipg_buckets[torch.float32].elements == 8
    assert opt.ipg_buckets[torch.float32].params == [(0, 0, 0)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("low_precision", [False, True])
@pytest.mark.parametrize("boundaries", [[False, False, True], [True, True, True], [False, False, False]])
def test_accumulation_preserves_nonzero_first_contribution(dtype, low_precision, boundaries):
    opt, param, master = make_optimizer(dtype, low_precision=low_precision, accumulate_offload_gradients=True)
    backward_values(opt, param, [1, 2, 4], boundaries)
    assert torch.equal(master.grad, torch.full_like(master, 7))
    assert not opt.local_overflow
    opt.reset_cpu_buffers()
    assert not opt._offload_accumulated_param_ids
    assert torch.count_nonzero(master.grad) == 0
    backward_values(opt, param, [3], [True])
    assert torch.equal(master.grad, torch.full_like(master, 3))


def test_default_offload_accumulates_across_nonboundary_backwards():
    opt, param, master = make_optimizer()
    backward_values(opt, param, [1, 2, 4], [False, False, True])
    assert torch.equal(master.grad, torch.full_like(master, 7))


def test_reduction_and_offload_consume_owned_oversized_buffer():
    opt, param, master = make_optimizer()
    opt.is_param_in_current_partition = {0: True}
    opt.average_tensor = lambda tensor, dtype: tensor.add_(2)
    param.grad = torch.ones_like(param)
    alias = param.grad.view_as(param)
    opt.reduce_independent_p_g_buckets_and_remove_grads(param, 0)
    alias.fill_(float("nan"))
    opt.reduce_ipg_grads()
    assert torch.equal(master.grad, torch.full_like(master, 3))
    assert param.grad is None
    assert not opt.extra_large_param_to_reduce
    assert not opt.ipg_buckets[torch.float32].params
    assert not opt.local_overflow


def test_accumulation_includes_zero_reads_and_gas_greater_than_one():
    opt, param, master = make_optimizer(accumulate_offload_gradients=True)
    opt.gradient_accumulation_steps = 4
    backward_values(opt, param, [1, 0, 2, 4], [False, False, False, True])
    assert torch.equal(master.grad, torch.full_like(master, 7))


def test_late_first_use_does_not_restore_stale_accumulator():
    opt, param, master = make_optimizer(accumulate_offload_gradients=True)
    opt.accumulated_grads_in_cpu[0] = torch.full_like(param, 99)
    opt.micro_step_id = 7
    param.grad = torch.full_like(param, 3)
    opt.copy_grads_in_partition(param)
    assert torch.equal(master.grad, torch.full_like(master, 3))


def test_partial_owned_fragment():
    opt, param, master = make_optimizer(accumulate_offload_gradients=True, low_precision=True)
    opt.grad_position[0] = [0, 2, 1, 3]
    backward_values(opt, param, [1, 2, 4], [False, True, True])
    expected = torch.zeros_like(master)
    expected[1:4] = 7
    assert torch.equal(master.grad, expected)


class FakeEvent:

    def __init__(self, on_sync=lambda: None):
        self.producer = None
        self.synced = False
        self.on_sync = on_sync

    def record(self, stream):
        self.producer = stream

    def synchronize(self):
        self.on_sync()
        self.synced = True


class FakeStream:

    def __init__(self):
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)


def fake_accelerator(current):
    return SimpleNamespace(resolves_data_dependency=lambda: False,
                           current_stream=lambda: current,
                           Event=FakeEvent,
                           stream=lambda stream: nullcontext())


@pytest.mark.parametrize("overlap", [False, True])
def test_average_waits_all_producers_even_without_overlap(monkeypatch, overlap):
    opt, _, _ = make_optimizer()
    current, other, consumer = FakeStream(), FakeStream(), FakeStream()
    bucket = opt.ipg_buckets[torch.float32]
    monkeypatch.setattr(zero, "get_accelerator", lambda: fake_accelerator(current))
    opt._record_bucket_producer(bucket)
    monkeypatch.setattr(zero, "get_accelerator", lambda: fake_accelerator(other))
    opt._record_bucket_producer(bucket)
    monkeypatch.setattr(zero, "get_accelerator", lambda: fake_accelerator(current))
    opt.overlap_comm = overlap
    opt.reduction_stream = consumer
    opt.reduce_scatter = False
    opt._record_gradient_stream = lambda tensor, stream: None
    used = consumer if overlap else current
    opt.gradient_reduction_w_predivide = lambda *args: pytest.fail("missing producer waits") if len(used.waited
                                                                                                    ) != 2 else None
    opt.average_tensor(torch.zeros(8), torch.float32)
    assert set(used.waited) == set(bucket.ready_events.values())
    bucket.reuse_events[0] = FakeEvent()
    bucket.clear()
    assert not bucket.ready_events
    assert bucket.reuse_events


@pytest.mark.parametrize("producers", [1, 2])
def test_bucket_reuse_waits_for_previous_consumer(monkeypatch, producers):
    opt, param, _ = make_optimizer()
    opt.reduce_bucket_size = 16
    bucket = opt.ipg_buckets[torch.float32]
    event = FakeEvent()
    bucket.reuse_events[0] = event
    opt._record_gradient_stream = lambda tensor, stream: None
    for index in range(producers):
        if index:
            param = torch.nn.Parameter(torch.zeros_like(param))
            param.param_idx_in_group = index
            opt.param_id[id(param)] = index
            opt.params_already_reduced.append(False)
        stream = FakeStream()
        monkeypatch.setattr(zero, "get_accelerator", lambda: fake_accelerator(stream))
        param.grad = torch.ones_like(param)
        opt.reduce_independent_p_g_buckets_and_remove_grads(param, 0)
        assert stream.waited == [event]
        assert bucket.reuse_events[0] is event
        assert bucket.ready_events[stream].producer is stream
    assert len(bucket.ready_events) == producers


@pytest.mark.parametrize("bucket_size,elements", [(16, 0), (16, 8), (4, 8)])
def test_only_buffer_consumers_replace_reuse_event(monkeypatch, bucket_size, elements):
    opt, param, _ = make_optimizer()
    opt.reduce_bucket_size = bucket_size
    opt.is_param_in_current_partition = {0: True}
    opt.average_tensor = lambda *args: None
    opt.copy_grads_in_partition = lambda param: None
    bucket = opt.ipg_buckets[torch.float32]
    bucket.elements = elements
    previous, consumer = FakeEvent(), FakeStream()
    bucket.reuse_events[0] = previous
    if elements:
        param.grad = torch.ones_like(param)
        bucket.params.append((0, 0, 0))
    if elements > bucket_size:
        opt.extra_large_param_to_reduce[torch.float32] = param
    monkeypatch.setattr(zero, "get_accelerator", lambda: fake_accelerator(consumer))
    opt.reduce_ipg_grads()
    if 0 < elements <= bucket_size:
        assert bucket.reuse_events[0] is not previous
        assert bucket.reuse_events[0].producer is consumer
    else:
        assert bucket.reuse_events[0] is previous


def test_cpu_offload_orders_successive_writes_across_streams(monkeypatch):
    opt, param, master = make_optimizer()
    first, second = FakeStream(), FakeStream()
    opt._record_gradient_stream = lambda tensor, stream: None
    monkeypatch.setattr(zero, "get_accelerator", lambda: fake_accelerator(first))
    param.grad = torch.ones_like(param)
    opt.async_inplace_copy_grad_to_fp32_buffer_from_gpu(param)
    assert not first.waited
    event = opt._pending_offload_events.get(first)

    original_copy = torch.Tensor.copy_

    def copy_after_wait(destination, source, **kwargs):
        assert second.waited == [event]
        return original_copy(destination, source, **kwargs)

    monkeypatch.setattr(zero, "get_accelerator", lambda: fake_accelerator(second))
    monkeypatch.setattr(torch.Tensor, "copy_", copy_after_wait)
    param.grad = torch.full_like(param, 3)
    opt.async_inplace_copy_grad_to_fp32_buffer_from_gpu(param)
    assert torch.equal(master.grad, torch.full_like(master, 3))
    assert opt._pending_offload_events[first] is event
    assert opt._pending_offload_events[second].producer is second


def test_record_stream_is_separate_from_readiness(monkeypatch):
    opt, _, _ = make_optimizer()
    stream = FakeStream()
    recorded = []
    monkeypatch.setattr(zero, "get_accelerator", lambda: fake_accelerator(stream))
    opt._record_gradient_stream(SimpleNamespace(record_stream=recorded.append), stream)
    assert recorded == [stream]
    assert not stream.waited


def test_cpu_accumulator_waits_before_pageable_source_is_read():
    opt, param, master = make_optimizer(accumulate_offload_gradients=True)
    backward_values(opt, param, [1], [True])
    event = FakeEvent(lambda: opt.accumulated_grads_in_cpu[0].fill_(3))
    opt._pending_offload_events["producer"] = event
    backward_values(opt, param, [4], [True])
    assert event.synced
    assert torch.equal(master.grad, torch.full_like(master, 7))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_offload_guard_scans_after_copy_completion(bad):
    opt, _, master = make_optimizer(check_offload_gradients=True)
    event = FakeEvent(lambda: master.grad.fill_(bad))
    opt._pending_offload_events["producer"] = event
    opt._all_reduce_overflow = bool
    assert opt.has_overflow()
    assert event.synced
    assert not opt._pending_offload_events
    assert not opt.local_overflow  # The pre-copy tracker never saw this corruption.


def test_offload_guard_allows_finite_zero_gradients():
    opt, _, _ = make_optimizer(check_offload_gradients=True)
    opt._all_reduce_overflow = bool
    assert not opt.has_overflow()


def test_global_overflow_includes_remote_rank(monkeypatch):
    opt, _, _ = make_optimizer(check_offload_gradients=True)
    calls = []
    monkeypatch.setattr(zero.dist, "all_reduce", lambda tensor, **kwargs: (calls.append(kwargs), tensor.fill_(1)))
    opt._model_parallel_all_reduce = lambda **kwargs: calls.append(kwargs)
    assert opt.has_overflow()
    assert len(calls) == 2


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_group_norm_is_not_hidden_by_outer_norm(bad):
    opt, _, _ = make_optimizer(check_offload_gradients=True)
    opt.complete_grad_norm_calculation_for_cpu_offload = lambda params: torch.tensor(bad)
    assert not torch.isfinite(opt.scaled_global_norm())


class NoopTimers:

    def __call__(self, name):
        return SimpleNamespace(start=lambda: None, stop=lambda: None)

    def log(self, names):
        pass


def prepare_step(opt, param, master, monkeypatch, mock_collectives=True):
    if mock_collectives:
        monkeypatch.setattr(zero.dist, "get_rank", lambda **kwargs: 0)
        monkeypatch.setattr(zero.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(zero, "all_gather_dp_groups", lambda **kwargs: None)
    monkeypatch.setattr(zero, "see_memory_usage", lambda *args, **kwargs: None)
    opt.custom_loss_scaler = False
    opt.loss_scaler = SimpleNamespace(cur_scale=1.0, update_scale=lambda overflow: None)
    opt.check_grad_overflow = True
    opt.clip_grad = 0.0
    opt.timers = NoopTimers()
    opt.optimizer = torch.optim.Adam([master], lr=0.01)
    opt.torch_autocast_gradscaler = None
    opt.real_dp_process_group = [None]
    opt.parallel_partitioned_bit16_groups = [[param.detach()]]
    opt.param_buffer_of_bit16_for_cpu_offload_groups = [torch.zeros_like(param)]
    opt.bit16_groups_flat = [param.detach()]
    opt.nccl_start_alignment_factor = 2
    opt.allgather_bucket_size = 16
    opt._lazy_init_hp_params_optimizer_state = lambda: None
    opt._update_model_bit16_weights = lambda index: None


@pytest.mark.parametrize("fault", ["cpu_gradient", "norm", "remote"])
def test_step_rejects_before_any_weight_or_optimizer_mutation(monkeypatch, fault):
    opt, param, master = make_optimizer(check_offload_gradients=True, accumulate_offload_gradients=True)
    prepare_step(opt, param, master, monkeypatch)
    master.grad.fill_(1)
    opt.optimizer.step()  # Seed real Adam moments and step counter.
    before_param = master.detach().clone()
    before_state = copy.deepcopy(opt.optimizer.state[master])
    backward_values(opt, param, [2, 3], [True, True])
    if fault == "cpu_gradient":
        opt._pending_offload_events["copy"] = FakeEvent(lambda: master.grad.fill_(float("nan")))
    elif fault == "norm":
        opt.complete_grad_norm_calculation_for_cpu_offload = lambda params: torch.tensor(-1.0)
    else:
        monkeypatch.setattr(zero.dist, "all_reduce", lambda tensor, **kwargs: tensor.fill_(1))
    opt.step()
    assert opt.overflow
    assert torch.equal(master, before_param)
    for key, value in before_state.items():
        assert torch.equal(opt.optimizer.state[master][key], value)
    assert not opt._offload_accumulated_param_ids
    assert not opt._pending_offload_events
    assert not torch.count_nonzero(master.grad)


def test_valid_steps_match_summed_gradient_reference_and_reset(monkeypatch):
    opt, param, master = make_optimizer(check_offload_gradients=True, accumulate_offload_gradients=True)
    prepare_step(opt, param, master, monkeypatch)
    ref = torch.nn.Parameter(master.detach().clone())
    ref_optimizer = torch.optim.Adam([ref], lr=0.01)
    for values in ([1, 2, 4], [3, 0, 1]):
        backward_values(opt, param, values, [True] * len(values))
        opt.step()
        ref.grad = torch.full_like(ref, sum(values))
        ref_optimizer.step()
        assert not opt.overflow
        assert torch.equal(master, ref)
        for key in ref_optimizer.state[ref]:
            assert torch.equal(opt.optimizer.state[master][key], ref_optimizer.state[ref][key])
        assert not opt._offload_accumulated_param_ids
        assert not torch.count_nonzero(master.grad)


def _distributed_rejection(rank, rendezvous):
    # Exercise real CPU collectives without building DeepSpeed's CPU comm extensions.
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(__compatible_ops__, "deepspeed_shm_comm", False)
        dist.init_distributed("gloo",
                              auto_mpi_discovery=False,
                              init_method=rendezvous,
                              rank=rank,
                              world_size=2,
                              timeout=timedelta(seconds=60))
    try:
        with pytest.MonkeyPatch.context() as patch:
            for fault in ("cpu_gradient", "norm"):
                opt, param, master = make_optimizer(check_offload_gradients=True, accumulate_offload_gradients=True)
                prepare_step(opt, param, master, patch, mock_collectives=False)
                master.grad.fill_(1)
                opt.optimizer.step()
                before_param = master.detach().clone()
                before_state = copy.deepcopy(opt.optimizer.state[master])
                backward_values(opt, param, [2, 3], [True, True])
                if fault == "cpu_gradient" and rank == 1:
                    master.grad[0] = float("nan")
                if fault == "norm":
                    opt.complete_grad_norm_calculation_for_cpu_offload = lambda params: torch.tensor(-1.0 if rank == 1
                                                                                                     else 1.0)
                opt.step()
                assert opt.overflow
                assert torch.equal(master, before_param)
                for key, value in before_state.items():
                    assert torch.equal(opt.optimizer.state[master][key], value)
                assert not opt._offload_accumulated_param_ids
    finally:
        dist.destroy_process_group()


def test_two_rank_cpu_overflow_consensus(tmp_path):
    torch.multiprocessing.spawn(_distributed_rejection,
                                args=(f"file://{tmp_path / 'rendezvous'}", ),
                                nprocs=2,
                                join=True)


def _distributed_default_training(rank, rendezvous):
    os.environ["LOCAL_RANK"] = str(rank)
    accelerator = get_accelerator()
    accelerator.set_device(rank)
    device = accelerator.device_name()
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(__compatible_ops__, "deepspeed_shm_comm", False)
        dist.init_distributed(accelerator.communication_backend_name(),
                              auto_mpi_discovery=False,
                              init_method=rendezvous,
                              rank=rank,
                              world_size=2,
                              timeout=timedelta(seconds=60))
    try:
        # Direct construction must honor the same offload and context restrictions.
        constructor_cases = [
            (True, True, None, False, False, False),
            (True, False, None, False, False, False),
            (False, True, "cpu", False, False, False),
            (True, True, "none", False, False, False),
            (True, True, "cpu", False, False, True),
            (True, False, "cpu", False, False, True),
            (True, True, "cpu", True, False, False),
            (True, True, "cpu", False, True, False),
        ]
        for partition_grads, contiguous, offload, pipeline, deepcompile, expected in constructor_cases:
            model = torch.nn.Linear(4, 4).to(device)
            offload_config = DeepSpeedZeroOffloadOptimizerConfig(device=offload) if offload else None
            opt = zero.DeepSpeedZeroOptimizer(torch.optim.SGD(model.parameters(), lr=0.01), {
                param: name
                for name, param in model.named_parameters()
            },
                                              NoopTimer(), {},
                                              partition_grads=partition_grads,
                                              contiguous_gradients=contiguous,
                                              offload_optimizer_config=offload_config,
                                              pipeline_parallel=pipeline,
                                              deepcompile=deepcompile,
                                              reduce_bucket_size=8)
            assert opt._offload_gradient_safety_enabled is expected
            opt.destroy()

        # Use actual two-rank training and an independent full-batch reference.
        cases = [
            ({
                "stage": 2
            }, False),
            ({
                "stage": 2,
                "overlap_comm": True
            }, False),
            ({
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu"
                }
            }, True),
            ({
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu"
                },
                "overlap_comm": True
            }, True),
            ({
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu"
                },
                "contiguous_gradients": False
            }, True),
            ({
                "stage": 1
            }, False),
            ({
                "stage": 1,
                "offload_optimizer": {
                    "device": "cpu"
                }
            }, False),
        ]
        for zero_config, expected in cases:
            torch.manual_seed(42)
            model = torch.nn.Linear(4, 4)
            reference = copy.deepcopy(model)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
            model.to(device)
            config = {
                "train_micro_batch_size_per_gpu": 2,
                "gradient_accumulation_steps": 2,
                "zero_allow_untested_optimizer": True,
                "zero_force_ds_cpu_optimizer": False,
                "zero_optimization": {
                    "reduce_bucket_size": 8,
                    **zero_config
                },
            }
            engine, opt, _, _ = deepspeed.initialize(model=model,
                                                     optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
                                                     config=config,
                                                     dist_init_required=False)
            try:
                assert opt._offload_gradient_safety_enabled is expected
                for step in range(2):
                    reference_optimizer.zero_grad()
                    for microstep in range(2):
                        inputs = (torch.arange(16, dtype=torch.float32).reshape(4, 4) + step + microstep) / 16
                        target = inputs.flip(-1) / 2
                        (torch.nn.functional.mse_loss(reference(inputs), target) / 2).backward()
                        local_inputs = inputs.chunk(2)[rank].to(device)
                        local_target = target.chunk(2)[rank].to(device)
                        engine.backward(torch.nn.functional.mse_loss(engine(local_inputs), local_target))
                        if microstep == 1:
                            for param, ref in zip(model.parameters(), reference.parameters()):
                                torch.testing.assert_close(param.get_full_hp_grad().cpu(),
                                                           ref.grad,
                                                           rtol=1e-5,
                                                           atol=1e-6,
                                                           msg=f"gradient mismatch: {zero_config}, step={step}")
                        engine.step()
                    reference_optimizer.step()
                    assert engine.global_steps == step + 1
                    for param, ref in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(param.cpu(), ref, rtol=1e-5, atol=1e-6)
            finally:
                engine.destroy()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(get_accelerator().device_name() not in ("cpu", "cuda")
                    or (get_accelerator().device_name() == "cuda" and get_accelerator().device_count() < 2),
                    reason="Requires CPU or two CUDA devices")
def test_two_rank_offload_and_nonoffload_training(tmp_path):
    torch.multiprocessing.spawn(_distributed_default_training,
                                args=(f"file://{tmp_path / 'training-rendezvous'}", ),
                                nprocs=2,
                                join=True)


@pytest.mark.parametrize("checkpoint_folder", [None, "checkpoint"])
def test_restore_discards_pending_accumulation(checkpoint_folder):
    opt, param, master = make_optimizer(accumulate_offload_gradients=True)
    backward_values(opt, param, [1, 2], [True, True])
    loaded = []
    opt._load_legacy_checkpoint = lambda *args: loaded.append("legacy")
    opt._load_universal_checkpoint = lambda *args: loaded.append("universal")
    opt.load_state_dict([], checkpoint_folder=checkpoint_folder)
    assert loaded == ["universal" if checkpoint_folder else "legacy"]
    assert opt.micro_step_id == -1
    assert not opt._offload_accumulated_param_ids
    assert not torch.count_nonzero(master.grad)
    backward_values(opt, param, [4], [True])
    assert torch.equal(master.grad, torch.full_like(master, 4))


@pytest.mark.skipif(get_accelerator().device_name() != "cuda" or not get_accelerator().is_available(),
                    reason="CUDA required for actual stream ordering")
@pytest.mark.parametrize("overlap", [False, True])
def test_cuda_oversized_producer_handoff(overlap):
    opt, param, _ = make_optimizer(device="cuda")
    producer, consumer = get_accelerator().Stream(), get_accelerator().Stream()
    opt.reduction_stream = consumer
    opt.overlap_comm = overlap
    opt.reduce_ipg_grads = lambda **kwargs: None
    producer.wait_stream(get_accelerator().current_stream())
    with get_accelerator().stream(producer):
        torch.cuda._sleep(2_000_000)  #ignore-cuda
        param.grad = torch.full_like(param, 7)
        opt.reduce_independent_p_g_buckets_and_remove_grads(param, 0)
    opt.reduce_scatter = False
    result = []
    opt.gradient_reduction_w_predivide = lambda tensor, dtype: result.append(tensor.clone())
    with get_accelerator().stream(consumer):
        opt.average_tensor(param.grad.view(-1), torch.float32)
    consumer.synchronize()
    assert torch.equal(result[0].cpu(), torch.full((8, ), 7, dtype=torch.bfloat16))

# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""ZeRO-2 offload gradient storage ownership and stream ordering tests."""

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
from deepspeed.runtime.zero import stage_1_and_2 as zero
from deepspeed.runtime.zero.offload_config import DeepSpeedZeroOffloadOptimizerConfig
from deepspeed.utils.timer import NoopTimer


def make_optimizer(dtype=torch.bfloat16, device="cpu", low_precision=False, cpu_offload=True):
    opt = zero.DeepSpeedZeroOptimizer.__new__(zero.DeepSpeedZeroOptimizer)
    opt.cpu_offload = cpu_offload
    opt._offload_gradient_safety_enabled = cpu_offload
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
    opt, param, master = make_optimizer()
    backward_values(opt, param, [1], [False])
    # The earlier copy into the CPU accumulator only lands once its event completes.
    event = FakeEvent(lambda: opt.accumulated_grads_in_cpu[0].fill_(3))
    opt._pending_offload_events["producer"] = event
    opt.micro_step_id = 1
    opt.set_gradient_accumulation_boundary(True)
    param.grad = torch.full_like(param, 4)
    opt.copy_grads_in_partition(param)
    assert event.synced
    assert torch.equal(master.grad, torch.full_like(master, 7))


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


def test_step_consumes_gradients_after_offload_copies_complete(monkeypatch):
    opt, param, master = make_optimizer()
    prepare_step(opt, param, master, monkeypatch)
    ref = torch.nn.Parameter(master.detach().clone())
    ref_optimizer = torch.optim.Adam([ref], lr=0.01)
    backward_values(opt, param, [2], [True])
    # Only the completed copy holds the gradient the CPU optimizer must consume.
    event = FakeEvent(lambda: master.grad.fill_(5))
    opt._pending_offload_events["producer"] = event
    opt.step()
    ref.grad = torch.full_like(ref, 5)
    ref_optimizer.step()
    assert event.synced
    assert not opt.overflow
    assert torch.equal(master, ref)


@pytest.mark.parametrize("operation", ["reset", "legacy_load", "universal_load"])
def test_cpu_buffer_reuse_waits_for_offload_copies(operation):
    opt, _, _ = make_optimizer()
    event = FakeEvent()
    opt._pending_offload_events["producer"] = event
    opt._load_legacy_checkpoint = lambda *args: None
    opt._load_universal_checkpoint = lambda *args: None
    if operation == "reset":
        opt.reset_cpu_buffers()
    else:
        opt.load_state_dict([], checkpoint_folder="checkpoint" if operation == "universal_load" else None)
    assert event.synced
    assert not opt._pending_offload_events


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
        # NVMe offload shares the CPU gradient path, so it must be protected too.
        constructor_cases = [
            (True, True, None, False),
            (True, False, None, False),
            (False, True, "cpu", False),
            (True, True, "none", False),
            (True, True, "cpu", True),
            (True, False, "cpu", True),
            (True, True, "nvme", True),
        ]
        for partition_grads, contiguous, offload, expected in constructor_cases:
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

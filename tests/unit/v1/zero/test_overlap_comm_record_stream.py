# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from contextlib import nullcontext
import os

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import deepspeed
import deepspeed.comm as dist
import deepspeed.runtime.zero.stage_1_and_2 as zero_stage12
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.utils import safe_get_full_grad
from unit.common import DistributedTest
from unit.util import bf16_required_version_check


class _FakeTensor:

    def __init__(self):
        self.recorded_streams = []
        self.copied_from = None

    def copy_(self, other):
        self.copied_from = other
        return self

    def record_stream(self, stream):
        self.recorded_streams.append(stream)


class _FakeAccelerator:

    def __init__(self, resolves_data_dependency, current_device_name="cpu"):
        self._resolves_data_dependency = resolves_data_dependency
        self._current_device_name = current_device_name

    def resolves_data_dependency(self):
        return self._resolves_data_dependency

    def stream(self, stream):
        return nullcontext()

    def current_stream(self):
        return object()

    def current_device_name(self):
        return self._current_device_name

    def synchronize(self):
        return None


def _build_overlap_optimizer(monkeypatch, *, resolves_data_dependency):
    optimizer = DeepSpeedZeroOptimizer.__new__(DeepSpeedZeroOptimizer)
    optimizer.overlap_comm = True
    optimizer.reduction_stream = object()
    optimizer.dp_process_group = object()
    optimizer.previous_reduced_grads = {}

    allreduced = _FakeTensor()
    synced = [_FakeTensor(), _FakeTensor()]

    optimizer.allreduce_bucket = lambda *args, **kwargs: allreduced
    optimizer.unflatten = lambda allreduced_tensor, small_bucket: synced

    monkeypatch.setattr(
        zero_stage12,
        "get_accelerator",
        lambda: _FakeAccelerator(resolves_data_dependency),
    )
    monkeypatch.setattr(zero_stage12.dist, "get_rank", lambda group=None: 0)
    return optimizer, allreduced, synced


def test_allreduce_and_copy_records_stream_for_overlap_comm(monkeypatch):
    optimizer, allreduced, synced = _build_overlap_optimizer(monkeypatch, resolves_data_dependency=False)
    bucket = [_FakeTensor(), _FakeTensor()]

    optimizer.allreduce_and_copy(bucket, torch.float16)

    assert allreduced.recorded_streams == [optimizer.reduction_stream]
    for buf, expected_synced in zip(bucket, synced):
        assert buf.copied_from is expected_synced
        assert buf.recorded_streams == [optimizer.reduction_stream]


def test_allreduce_and_copy_with_multiple_ranks_records_only_local_buffers(monkeypatch):
    optimizer, allreduced, synced = _build_overlap_optimizer(monkeypatch, resolves_data_dependency=False)
    bucket = [_FakeTensor(), _FakeTensor()]

    optimizer.allreduce_and_copy_with_multiple_ranks(
        bucket,
        torch.float16,
        bucket_ranks=[0, 1],
    )

    assert allreduced.recorded_streams == [optimizer.reduction_stream]
    assert bucket[0].copied_from is synced[0]
    assert bucket[0].recorded_streams == [optimizer.reduction_stream]
    assert bucket[1].copied_from is None
    assert bucket[1].recorded_streams == []


def test_allreduce_and_copy_with_multiple_ranks_records_consumer_buffers(monkeypatch):
    optimizer, allreduced, synced = _build_overlap_optimizer(monkeypatch, resolves_data_dependency=False)
    bucket = [_FakeTensor(), _FakeTensor()]

    optimizer.allreduce_and_copy_with_multiple_ranks(
        bucket,
        torch.float16,
        bucket_ranks=[frozenset((0, 1)), frozenset((1, ))],
    )

    assert allreduced.recorded_streams == [optimizer.reduction_stream]
    assert bucket[0].copied_from is synced[0]
    assert bucket[0].recorded_streams == [optimizer.reduction_stream]
    assert bucket[1].copied_from is None
    assert bucket[1].recorded_streams == []


class _FakeWaitStream:
    """A stream stand-in that records which streams it was told to wait on."""

    def __init__(self, operations=None):
        self.waited_on = []
        self.waited_on_events = []
        self.recorded_events = []
        self.operations = operations

    def wait_stream(self, other):
        self.waited_on.append(other)

    def wait_event(self, event):
        self.waited_on_events.append(event)

    def record_event(self):
        event = object()
        self.recorded_events.append(event)
        if self.operations is not None:
            self.operations.append("record")
        return event


class _FakeAcceleratorWithCurrentStream(_FakeAccelerator):

    def __init__(self, resolves_data_dependency, current_stream):
        super().__init__(resolves_data_dependency)
        self._current_stream = current_stream

    def current_stream(self):
        return self._current_stream


class _CopyOrderMode(TorchDispatchMode):

    def __init__(self, operations):
        self.operations = operations

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.copy_.default:
            self.operations.append("copy")
        return func(*args, **(kwargs or {}))


def _build_average_tensor_optimizer(monkeypatch, *, copy_streams):
    optimizer = DeepSpeedZeroOptimizer.__new__(DeepSpeedZeroOptimizer)
    optimizer.overlap_comm = True
    optimizer.reduce_scatter = False  # take the early-return reduce path, isolating the wait logic
    optimizer.reduction_stream = _FakeWaitStream()
    comm_dtype = torch.float16
    bucket = zero_stage12.IPGBucket()
    bucket.copy_streams = set(copy_streams)
    optimizer.ipg_buckets = {comm_dtype: bucket}
    reduced = []
    optimizer.gradient_reduction_w_predivide = lambda tensor, dt: reduced.append(dt)
    current = _FakeWaitStream()
    monkeypatch.setattr(
        zero_stage12,
        "get_accelerator",
        lambda: _FakeAcceleratorWithCurrentStream(False, current),
    )
    return optimizer, comm_dtype, current, reduced


def test_average_tensor_waits_on_all_ipg_bucket_producer_streams(monkeypatch):
    # #8061: the reduction stream must wait on every stream that produced a copy into
    # the contiguous IPG bucket, not just the current stream, because under
    # torch.compile those copies can be issued on multiple autograd streams.
    s1, s2 = object(), object()
    optimizer, comm_dtype, current, reduced = _build_average_tensor_optimizer(monkeypatch, copy_streams=[s1, s2])

    optimizer.average_tensor(torch.zeros(4), comm_dtype)

    assert set(optimizer.reduction_stream.waited_on) == {s1, s2}
    assert current.waited_on == []
    assert reduced == [comm_dtype]


def test_average_tensor_falls_back_to_current_stream_without_producers(monkeypatch):
    # The extra-large-param path reduces without copying into the bucket, so
    # copy_streams is empty: preserve the original behavior of waiting on the
    # current stream.
    optimizer, comm_dtype, current, _ = _build_average_tensor_optimizer(monkeypatch, copy_streams=[])

    optimizer.average_tensor(torch.zeros(4), comm_dtype)

    assert optimizer.reduction_stream.waited_on == [current]


def test_ipg_bucket_clear_resets_copy_streams():
    bucket = zero_stage12.IPGBucket()
    assert bucket.copy_streams == set()
    bucket.copy_streams.add(object())
    bucket.clear()
    assert bucket.copy_streams == set()


def test_ipg_buffer_reuse_waits_on_matching_completion_only():
    optimizer = DeepSpeedZeroOptimizer.__new__(DeepSpeedZeroOptimizer)
    optimizer.overlap_comm = True
    bucket = zero_stage12.IPGBucket()
    producer = _FakeWaitStream()

    bucket.index = 0
    optimizer._wait_for_ipg_buffer_reuse(bucket, producer)
    assert producer.waited_on_events == []

    buffer_0_complete = object()
    buffer_1_complete = object()
    bucket.reduction_complete_events = [buffer_0_complete, buffer_1_complete]

    bucket.index = 1
    optimizer._wait_for_ipg_buffer_reuse(bucket, producer)
    assert producer.waited_on_events == [buffer_1_complete]

    bucket.index = 0
    optimizer._wait_for_ipg_buffer_reuse(bucket, producer)
    assert producer.waited_on_events == [buffer_1_complete, buffer_0_complete]


def test_records_reduction_completion_for_physical_buffer_index():
    optimizer = DeepSpeedZeroOptimizer.__new__(DeepSpeedZeroOptimizer)
    optimizer.overlap_comm = True
    optimizer.reduction_stream = _FakeWaitStream()
    bucket = zero_stage12.IPGBucket()
    bucket.reduction_complete_events = [None, None]

    bucket.index = 0
    optimizer._record_ipg_buffer_reduction_complete(bucket)
    buffer_0_complete = optimizer.reduction_stream.recorded_events[-1]
    assert bucket.reduction_complete_events == [buffer_0_complete, None]

    bucket.index = 1
    optimizer._record_ipg_buffer_reduction_complete(bucket)
    buffer_1_complete = optimizer.reduction_stream.recorded_events[-1]
    assert bucket.reduction_complete_events == [buffer_0_complete, buffer_1_complete]


def test_ipg_bucket_clear_preserves_buffer_completion_state():
    bucket = zero_stage12.IPGBucket()
    completion_events = [object(), object()]
    bucket.reduction_complete_events = completion_events

    bucket.clear()

    # clear() starts the next logical fill of the same physical buffer. Its prior
    # reduction completion must remain visible until that buffer is reused.
    assert bucket.reduction_complete_events == completion_events


def test_setup_buckets_resets_completion_for_new_physical_buffers(monkeypatch):
    optimizer = DeepSpeedZeroOptimizer.__new__(DeepSpeedZeroOptimizer)
    optimizer.ready_for_gradients = False
    optimizer.micro_step_id = 0
    optimizer.contiguous_gradients = True
    optimizer.overlap_comm = True
    optimizer.reduce_bucket_size = 4
    optimizer.dtype = torch.float32
    bucket = zero_stage12.IPGBucket()
    bucket.reduction_complete_events = [object(), object()]
    optimizer.ipg_buckets = {torch.float32: bucket}
    monkeypatch.setattr(zero_stage12, "get_accelerator", lambda: _FakeAccelerator(False))

    optimizer.setup_buckets()

    assert len(bucket.buffer) == 2
    assert bucket.reduction_complete_events == [None, None]


def test_reduction_completion_is_isolated_per_dtype_bucket():
    optimizer = DeepSpeedZeroOptimizer.__new__(DeepSpeedZeroOptimizer)
    optimizer.overlap_comm = True
    optimizer.reduction_stream = _FakeWaitStream()
    fp16_bucket = zero_stage12.IPGBucket()
    bf16_bucket = zero_stage12.IPGBucket()
    fp16_bucket.reduction_complete_events = [None, None]
    bf16_bucket.reduction_complete_events = [None, None]

    optimizer._record_ipg_buffer_reduction_complete(fp16_bucket)

    assert fp16_bucket.reduction_complete_events[0] is optimizer.reduction_stream.recorded_events[-1]
    assert bf16_bucket.reduction_complete_events == [None, None]


def test_ipg_buffer_completion_state_machine_across_dtypes_and_producers():
    optimizer = DeepSpeedZeroOptimizer.__new__(DeepSpeedZeroOptimizer)
    optimizer.reduction_stream = _FakeWaitStream()
    buckets = {
        torch.float16: zero_stage12.IPGBucket(reduction_complete_events=[None, None]),
        torch.bfloat16: zero_stage12.IPGBucket(reduction_complete_events=[None, None]),
    }
    sequence = [
        (torch.float16, 0, 1),
        (torch.bfloat16, 0, 2),
        (torch.float16, 1, 2),
        (torch.float16, 0, 2),
        (torch.bfloat16, 1, 1),
        (torch.bfloat16, 0, 2),
        (torch.float16, 1, 1),
    ]
    last_completion = {}

    for dtype, index, producer_count in sequence:
        bucket = buckets[dtype]
        bucket.index = index
        expected_event = last_completion.get((dtype, index))
        for _ in range(producer_count):
            producer = _FakeWaitStream()
            optimizer._wait_for_ipg_buffer_reuse(bucket, producer)
            assert producer.waited_on_events == ([] if expected_event is None else [expected_event])

        optimizer._record_ipg_buffer_reduction_complete(bucket)
        completion_event = optimizer.reduction_stream.recorded_events[-1]
        last_completion[(dtype, index)] = completion_event
        bucket.clear()
        assert bucket.reduction_complete_events[index] is completion_event


def test_reduce_ipg_grads_records_completion_after_buffer_consumers(monkeypatch):
    operations = []
    optimizer = DeepSpeedZeroOptimizer.__new__(DeepSpeedZeroOptimizer)
    optimizer.contiguous_gradients = True
    optimizer.overlap_comm = True
    optimizer.cpu_offload = False
    optimizer.partition_gradients = True
    optimizer.reduction_stream = _FakeWaitStream(operations)
    optimizer.extra_large_param_to_reduce = {}
    optimizer.params_already_reduced = {7: False}
    optimizer.is_param_in_current_partition = {7: True}
    optimizer.bit16_groups = [[object()]]
    optimizer.copy_grads_in_partition = lambda param: operations.append("copy")
    optimizer.average_tensor = lambda tensor, dtype: operations.append("average")
    bucket = zero_stage12.IPGBucket(buffer=[torch.zeros(4)], reduction_complete_events=[None], elements=4)
    bucket.params = [(0, 0, 7)]
    optimizer.ipg_buckets = {torch.float32: bucket}
    monkeypatch.setattr(
        zero_stage12,
        "get_accelerator",
        lambda: _FakeAcceleratorWithCurrentStream(False, _FakeWaitStream()),
    )

    optimizer.reduce_ipg_grads(torch.float32)

    assert operations == ["average", "copy", "record"]


def test_ipg_buffer_reuse_wait_precedes_gradient_copy(monkeypatch):
    operations = []
    optimizer = DeepSpeedZeroOptimizer.__new__(DeepSpeedZeroOptimizer)
    optimizer.reduce_bucket_size = 4
    optimizer.contiguous_gradients = True
    optimizer.overlap_comm = True
    optimizer.zenflow = False
    optimizer.params_already_reduced = {7: False}
    optimizer._maybe_reduce_autoep_folding_tp_gradient = lambda param, grad: None
    optimizer.get_param_id = lambda param: 7
    optimizer.get_param_comm_dtype = lambda param: torch.float32
    optimizer.get_gradient_for_reduction = lambda param: param.grad
    optimizer.report_ipg_memory_usage = lambda *args: None
    optimizer._wait_for_ipg_buffer_reuse = lambda bucket, stream: operations.append("wait")
    bucket = zero_stage12.IPGBucket(buffer=[torch.zeros(4), torch.zeros(4)])
    bucket.reduction_complete_events = [object(), None]
    optimizer.ipg_buckets = {torch.float32: bucket}
    current = _FakeWaitStream()
    monkeypatch.setattr(
        zero_stage12,
        "get_accelerator",
        lambda: _FakeAcceleratorWithCurrentStream(False, current),
    )
    param = torch.nn.Parameter(torch.zeros(2))
    param.grad = torch.ones(2)
    param.param_idx_in_group = 0

    with _CopyOrderMode(operations):
        optimizer.reduce_independent_p_g_buckets_and_remove_grads(param, 0)

    assert operations[:2] == ["wait", "copy"]


class _MultiBucketModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False) for _ in range(8)])
        with torch.no_grad():
            for index, layer in enumerate(self.layers):
                values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
                layer.weight.copy_((values + index + 1) / 32)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = torch.tanh(layer(inputs))
        return inputs


def _run_zero2_buffer_reuse(overlap_comm, reduce_bucket_size, delay_cycles=0):
    model = _MultiBucketModel().to(dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = {
        "train_micro_batch_size_per_gpu": 2,
        "zero_allow_untested_optimizer": True,
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": overlap_comm,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "reduce_bucket_size": reduce_bucket_size,
        },
    }
    engine, *_ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
    reuse_waits = 0
    if overlap_comm:
        original_wait = engine.optimizer._wait_for_ipg_buffer_reuse

        def count_reuse_waits(bucket, producer_stream):
            nonlocal reuse_waits
            if bucket.reduction_complete_events[bucket.index] is not None:
                reuse_waits += 1
            original_wait(bucket, producer_stream)

        engine.optimizer._wait_for_ipg_buffer_reuse = count_reuse_waits
        original_average = engine.optimizer.average_tensor

        def delayed_average(tensor, communication_data_type):
            with get_accelerator().stream(engine.optimizer.reduction_stream):
                torch.cuda._sleep(delay_cycles)  #ignore-cuda
            return original_average(tensor, communication_data_type)

        engine.optimizer.average_tensor = delayed_average

    losses = []
    gradients = []
    try:
        device = get_accelerator().current_device_name()
        rank = dist.get_rank()
        for step in range(4):
            values = torch.arange(8, device=device, dtype=torch.float32).reshape(2, 4)
            inputs = ((values + rank * 3 + step) / 8).to(torch.bfloat16)
            loss = engine(inputs).float().square().mean()
            engine.backward(loss)
            losses.append(loss.detach().cpu())
            gradients.append(
                [safe_get_full_grad(param).detach().float().cpu() for param in engine.module.parameters()])
            engine.step()
        parameters = [param.detach().float().cpu() for param in engine.module.parameters()]
    finally:
        engine.destroy()
    return losses, gradients, parameters, reuse_waits


class TestZero2IPGBufferReuse(DistributedTest):
    world_size = 2

    def test_overlap_matches_non_overlap_through_repeated_buffer_reuse(self):
        if not bf16_required_version_check():
            pytest.skip("BF16 ZeRO-2 test requires BF16 accelerator support.")
        if not hasattr(torch.cuda, "_sleep"):  #ignore-cuda
            pytest.skip("CUDA sleep helper is unavailable.")

        stress_cases = [(20, int(2e7)), (32, int(5e7)), (48, int(1e8))]
        stress_repeats = int(os.environ.get("DS_ZERO_BUFFER_REUSE_STRESS_REPEATS", "1"))
        for repeat in range(stress_repeats):
            reduce_bucket_size, delay_cycles = stress_cases[repeat % len(stress_cases)]
            reference = _run_zero2_buffer_reuse(overlap_comm=False, reduce_bucket_size=reduce_bucket_size)
            overlap = _run_zero2_buffer_reuse(overlap_comm=True,
                                              reduce_bucket_size=reduce_bucket_size,
                                              delay_cycles=delay_cycles)

            assert overlap[3] >= 2
            for actual, expected in zip(overlap[0], reference[0]):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            for actual_step, expected_step in zip(overlap[1], reference[1]):
                for actual, expected in zip(actual_step, expected_step):
                    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
            for actual, expected in zip(overlap[2], reference[2]):
                torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)

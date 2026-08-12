# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
from torch.fx import Graph, GraphModule

import deepspeed.compile.profilers.graph_profile as graph_profile


class FakeRandom:

    def fork_rng(self, devices):
        return self

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeAccelerator:

    def __init__(self):
        self.event_count = 0

    def current_device(self):
        return "cpu"

    def memory_allocated(self):
        return 0

    def max_memory_allocated(self):
        return 0

    def reset_peak_memory_stats(self):
        return None

    def Event(self, enable_timing=True):
        event = FakeEvent(f"event-{self.event_count}")
        self.event_count += 1
        return event

    def synchronize(self):
        return None

    def random(self):
        return FakeRandom()


class FakeDeepCompileHandle:

    def __init__(self):
        self.events = []

    def enable_profiling(self, enabled):
        self.events.append(("enable", enabled))

    def clear_all_gathered_params(self):
        self.events.append(("clear", None))


class FakeEvent:

    def __init__(self, name):
        self.name = name
        self.records = []

    def record(self):
        self.records.append(self.name)

    def elapsed_time(self, end_event):
        return 1.0


def _make_empty_graph_module():
    graph = Graph()
    graph.output(None)
    return GraphModule(torch.nn.Module(), graph)


def test_profile_helpers_drop_warmup_and_intermediate_outputs():
    deleted = []

    class Output:

        def __init__(self, index):
            self.index = index

        def __del__(self):
            deleted.append(self.index)

    outputs_created = []

    def call_fn():
        output = Output(len(outputs_created))
        outputs_created.append(output.index)
        return output

    start_events = [FakeEvent(f"start-{i}") for i in range(3)]
    end_events = [FakeEvent(f"end-{i}") for i in range(3)]

    graph_profile._run_warmup_for_profile(call_fn, warmup=2)
    out = graph_profile._run_repeatedly_for_profile(call_fn,
                                                    iteration=3,
                                                    start_events=start_events,
                                                    end_events=end_events)

    assert out.index == 4
    assert outputs_created == [0, 1, 2, 3, 4]
    assert deleted == [0, 1, 2, 3]
    assert [event.records for event in start_events] == [["start-0"], ["start-1"], ["start-2"]]
    assert [event.records for event in end_events] == [["end-0"], ["end-1"], ["end-2"]]


def test_profiling_interpreter_wall_time_excludes_warmup(monkeypatch):
    fake_handle = FakeDeepCompileHandle()
    fake_accelerator = FakeAccelerator()

    monkeypatch.setattr(graph_profile, "get_deepcompile_handle", lambda: fake_handle)
    monkeypatch.setattr(graph_profile, "get_accelerator", lambda: fake_accelerator)
    monkeypatch.setattr(graph_profile, "_get_mem_usage_out_of_torch", lambda: 0)
    monkeypatch.setattr(graph_profile, "is_comm_op", lambda node: False)
    monkeypatch.setattr(graph_profile, "is_release_node", lambda node: False)
    monkeypatch.setattr(graph_profile.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(graph_profile.dist, "get_rank", lambda: 0)

    timestamps = iter(range(20))
    monkeypatch.setattr(graph_profile.time, "time", lambda: next(timestamps))

    def timed_identity(x):
        graph_profile.time.time()
        return x

    graph = Graph()
    x = graph.placeholder("x")
    y = graph.call_function(timed_identity, (x, ))
    graph.output(y)
    gm = GraphModule(torch.nn.Module(), graph)

    interpreter = graph_profile.ProfilingInterpreter(gm, iteration=3, warmup=2)
    interpreter.run(torch.ones(1))

    call_node = next(node for node in gm.graph.nodes if node.op == "call_function")
    assert call_node.meta["wall_time"] == pytest.approx((4 / 3) * 1000)


def test_memory_profiling_interpreter_clears_gathered_params_after_failure(monkeypatch):
    fake_handle = FakeDeepCompileHandle()

    monkeypatch.setattr(graph_profile, "get_deepcompile_handle", lambda: fake_handle)
    monkeypatch.setattr(graph_profile, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(graph_profile, "_all_real_if_tensor", lambda args: True)
    monkeypatch.setattr(graph_profile, "_get_mem_usage_out_of_torch", lambda: 0)

    def raise_from_run(self, *args):
        raise RuntimeError("synthetic memory profile failure")

    monkeypatch.setattr(graph_profile.Interpreter, "run", raise_from_run)

    interpreter = graph_profile.MemoryProfilingInterpreter(_make_empty_graph_module())
    interpreter.mem_record.append(("partial", 1, 1, 1))

    assert interpreter.run() is None
    assert not interpreter.profile_complete
    assert interpreter.mem_record == []
    assert fake_handle.events == [("enable", True), ("clear", None), ("enable", False)]


def test_memory_profiling_interpreter_disables_profiling_if_cleanup_fails(monkeypatch):
    fake_handle = FakeDeepCompileHandle()

    def fail_clear():
        fake_handle.events.append(("clear", None))
        raise RuntimeError("cleanup failed")

    fake_handle.clear_all_gathered_params = fail_clear

    monkeypatch.setattr(graph_profile, "get_deepcompile_handle", lambda: fake_handle)
    monkeypatch.setattr(graph_profile, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(graph_profile, "_all_real_if_tensor", lambda args: True)
    monkeypatch.setattr(graph_profile, "_get_mem_usage_out_of_torch", lambda: 0)
    monkeypatch.setattr(graph_profile.Interpreter, "run", lambda self, *args: None)

    interpreter = graph_profile.MemoryProfilingInterpreter(_make_empty_graph_module())

    with pytest.raises(RuntimeError, match="cleanup failed"):
        interpreter.run()

    assert fake_handle.events == [("enable", True), ("clear", None), ("enable", False)]


class FakePinnedTensor(torch.Tensor):
    """A CPU tensor that reports itself as pinned without needing an accelerator."""

    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data)

    def is_pinned(self, device=None):
        return True

    def to(self, *args, **kwargs):
        raise AssertionError("a pinned host tensor must be left where it is")


def test_to_leaves_pinned_host_tensors_alone():
    # Pinned host tensors inside a graph are offloaded values. Copying one to the device would undo
    # the offload the compile pass just made and hide the memory it gave back from the profile.
    pinned = FakePinnedTensor(torch.zeros(4))
    assert graph_profile._to(pinned, torch.device("cpu")) is pinned

    plain = torch.zeros(4)
    assert graph_profile._to(plain, torch.device("cpu")) is not pinned


class RecordingAllReduce:
    """Stands in for the collective, recording shapes so a test can see who took part.

    Set peer_failed to make the vote come back as "someone else failed", which is what a healthy
    rank sees when another rank hit an error.
    """

    def __init__(self, peer_failed=False):
        self.shapes = []
        self.peer_failed = peer_failed

    def __call__(self, tensor, op=None):
        self.shapes.append(tuple(tensor.shape))
        if self.peer_failed and tensor.numel() == 1:
            tensor.fill_(1)


def _install_fake_distributed(monkeypatch, all_reduce):
    monkeypatch.setattr(graph_profile.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(graph_profile.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(graph_profile.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(graph_profile.dist, "barrier", lambda: None)


def test_abort_helper_passes_when_every_rank_is_healthy(monkeypatch):
    all_reduce = RecordingAllReduce()
    _install_fake_distributed(monkeypatch, all_reduce)

    # One vote per call, and no exception when nobody reported a failure.
    graph_profile._abort_if_any_rank_failed(None, torch.device("cpu"))
    assert all_reduce.shapes == [(1, )]


def test_abort_helper_raises_locally_without_distributed():
    # Single-process runs have nobody to agree with, so the error is raised as it stands.
    with pytest.raises(graph_profile.ProfileAborted, match="local failure"):
        graph_profile._abort_if_any_rank_failed(RuntimeError("local failure"), torch.device("cpu"), distributed=False)


def test_healthy_rank_aborts_when_another_rank_failed(monkeypatch):
    # The point of the vote: this rank is fine, but it must stop rather than run on into
    # collectives the failed rank will never reach.
    _install_fake_distributed(monkeypatch, RecordingAllReduce(peer_failed=True))

    with pytest.raises(graph_profile.ProfileAborted, match="another rank"):
        graph_profile._abort_if_any_rank_failed(None, torch.device("cpu"))


def test_memory_profiler_finishes_the_node_collectives_before_reporting_a_failure(monkeypatch):
    # A node that raises used to unwind straight out of the loop, skipping this node's memory
    # reduce and every collective after it, leaving the other ranks waiting until the watchdog
    # killed the job half an hour later.
    all_reduce = RecordingAllReduce()
    _install_fake_distributed(monkeypatch, all_reduce)
    monkeypatch.setattr(graph_profile, "get_deepcompile_handle", lambda: FakeDeepCompileHandle())
    monkeypatch.setattr(graph_profile, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(graph_profile, "_all_real_if_tensor", lambda args: True)
    monkeypatch.setattr(graph_profile, "_get_mem_usage_out_of_torch", lambda: 0)

    def out_of_memory(x):
        raise RuntimeError("CUDA out of memory. Tried to allocate 6.96 GiB")

    graph = Graph()
    x = graph.placeholder("x")
    y = graph.call_function(out_of_memory, (x, ))
    graph.output(y)
    gm = GraphModule(torch.nn.Module(), graph)

    interpreter = graph_profile.MemoryProfilingInterpreter(gm)

    # run() swallows it the way it always has, so the caller still gets "profile incomplete".
    assert interpreter.run(torch.ones(1)) is None
    assert not interpreter.profile_complete
    assert interpreter.mem_record == []

    # The failing node took part in its memory reduce (2 values) and then in the vote (1 value):
    # the same pair of collectives every healthy rank issues for that node.
    assert all_reduce.shapes[-2:] == [(2, ), (1, )]


def test_time_profiler_finishes_the_node_collectives_before_reporting_a_failure(monkeypatch):
    all_reduce = RecordingAllReduce()
    _install_fake_distributed(monkeypatch, all_reduce)
    monkeypatch.setattr(graph_profile, "get_deepcompile_handle", lambda: FakeDeepCompileHandle())
    monkeypatch.setattr(graph_profile, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(graph_profile, "_all_real_if_tensor", lambda args: True)
    monkeypatch.setattr(graph_profile, "_get_mem_usage_out_of_torch", lambda: 0)
    monkeypatch.setattr(graph_profile, "is_comm_op", lambda node: False)
    monkeypatch.setattr(graph_profile, "is_release_node", lambda node: False)

    def out_of_memory(x):
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.06 GiB")

    graph = Graph()
    x = graph.placeholder("x")
    y = graph.call_function(out_of_memory, (x, ))
    graph.output(y)
    gm = GraphModule(torch.nn.Module(), graph)

    interpreter = graph_profile.ProfilingInterpreter(gm, iteration=1, warmup=0)

    assert interpreter.run(torch.ones(1)) is None

    # Cache vote (1), the timing reduce this rank has nothing to contribute to (5), then the
    # failure vote (1). Skipping the middle one is what stranded the other ranks.
    assert all_reduce.shapes[-3:] == [(1, ), (5, ), (1, )]
    assert graph_profile.is_profile_incomplete(gm.graph)

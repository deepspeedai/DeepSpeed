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

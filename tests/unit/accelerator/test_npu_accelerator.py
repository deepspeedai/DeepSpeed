# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.accelerator.npu_accelerator import NPU_Accelerator


def _stub_npu(monkeypatch, graph_runtime):
    """Replace torch.npu with a minimal stub so the tests run without torch_npu."""
    monkeypatch.setattr(torch, "npu", graph_runtime, raising=False)
    return NPU_Accelerator.__new__(NPU_Accelerator)


def test_npu_create_graph_returns_runtime_graph(monkeypatch):
    # The runtime's own graph type is the contract: graph_process and the
    # training/inference graph paths capture into whatever object this returns.
    created = []

    class _FakeGraph:

        def __init__(self):
            created.append(self)

    class _StubNpu:
        NPUGraph = _FakeGraph

    accelerator = _stub_npu(monkeypatch, _StubNpu())

    graph = accelerator.create_graph()
    assert graph in created


def test_npu_capture_to_graph_delegates_to_runtime(monkeypatch):
    # capture_to_graph must hand the graph (plus pool/stream) to the runtime
    # context manager and return its result; graph_process captures inside it.
    calls = []

    class _StubNpu:

        @staticmethod
        def graph(graph, pool=None, stream=None):
            calls.append((graph, pool, stream))
            return "capture-context"

    accelerator = _stub_npu(monkeypatch, _StubNpu())

    graph = object()
    stream = object()
    assert accelerator.capture_to_graph(graph, pool="pool", stream=stream) == "capture-context"
    assert calls == [(graph, "pool", stream)]

    # Callers rely on pool/stream defaulting to None (e.g. graph_process).
    assert accelerator.capture_to_graph(graph) == "capture-context"
    assert calls[-1] == (graph, None, None)


def test_npu_replay_graph_invokes_graph_replay(monkeypatch):
    # replay_graph must replay the captured work, matching the accelerator
    # contract so graph_process re-executes the recorded function.

    class _FakeGraph:

        def __init__(self):
            self.replays = 0

        def replay(self):
            self.replays += 1

    class _StubNpu:
        NPUGraph = _FakeGraph

    accelerator = _stub_npu(monkeypatch, _StubNpu())

    graph = accelerator.create_graph()
    accelerator.replay_graph(graph)
    assert graph.replays == 1

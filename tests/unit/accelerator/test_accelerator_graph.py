# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.accelerator import get_accelerator


def test_graph_capture_replay_roundtrip():
    # Exercise the three graph interfaces end-to-end on whatever device the
    # accelerator exposes, so one test covers every graph-capable backend.
    accel = get_accelerator()
    device = accel.device_name()
    graph = accel.create_graph()
    if graph is None:
        pytest.skip(f"{device} accelerator does not support graph capture")

    # Static input/output buffers; the capture records work on these addresses.
    src = torch.zeros(1, device=device)
    dst = torch.zeros(1, device=device)

    # Warm up on a side stream first (the same pattern graph_process uses) so
    # the capture starts from a populated memory pool.
    stream = accel.Stream()
    stream.wait_stream(accel.current_stream())
    with accel.stream(stream):
        dst.copy_(src)
    accel.current_stream().wait_stream(stream)

    with accel.capture_to_graph(graph):
        dst.copy_(src)
    # Capture runs the body eagerly, so dst already holds the copied value.
    assert dst.item() == 0.0

    # Feed a new value into the static input and replay: only a real replay
    # re-executes the recorded copy, otherwise dst keeps its pre-replay value.
    src.fill_(1.0)
    accel.replay_graph(graph)
    assert dst.item() == 1.0


def test_graph_ops_run_eagerly_without_graph_support():
    # Backends without graph support must run capture bodies eagerly and treat
    # replay as a no-op, keeping graph_process correct when capture is off.
    # This is the contract that runs on CPU-only CI.
    accel = get_accelerator()
    device = accel.device_name()
    graph = accel.create_graph()
    if graph is not None:
        pytest.skip(f"{device} accelerator supports graph capture")

    assert graph is None

    value = torch.zeros(1, device=device)
    with accel.capture_to_graph(graph):
        value.fill_(1.0)
    assert value.item() == 1.0

    accel.replay_graph(graph)  # no-op: must not raise nor touch the value
    assert value.item() == 1.0

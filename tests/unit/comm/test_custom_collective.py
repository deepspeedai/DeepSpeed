# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Contract tests for the custom_collective inference all-reduce route.

The routing contract: inference_all_reduce must never engage the custom
P2P kernel unless initialize() succeeded on this topology. Everything
else — CPU tensors, non-P2P fabrics, world != 2, oversized messages —
falls back unchanged to the regular collectives.
"""

import deepspeed.comm as dist

from deepspeed.comm import custom_collective


def test_not_ready_before_initialize():
    """A fresh process must report not-ready: routing decisions all see
    False and stay on NCCL until initialize() has proven the fast path."""
    assert not custom_collective.custom_collective_ready(2)


def test_initialize_noops_without_dist():
    """Without the process group initialized, initialize() must return
    silently (no raise) — single-process users never see the fast path."""
    if dist.is_initialized():
        import pytest
        pytest.skip("distributed already initialized")
    custom_collective.initialize()
    assert not custom_collective.custom_collective_ready(2)


def test_payload_gate_blocks_when_not_ready():
    """The size half of the dispatch gate: without a successful initialize
    the advertised max payload is zero, so no message can ever qualify."""
    if custom_collective.custom_collective_ready(2):
        import pytest
        pytest.skip("custom AR initialized on this topology")
    assert custom_collective.custom_collective_max_bytes() == 0


def test_oversized_message_fails_the_gate():
    """custom_collective_max_bytes is the payload ceiling; contract: an 8MB message
    must exceed it (the kernel's IPC buffers are 2x4MB)."""
    if not custom_collective.custom_collective_ready(2):
        import pytest
        pytest.skip("custom AR not initialized on this topology")
    assert 8 * 1024 * 1024 > custom_collective.custom_collective_max_bytes()


def test_unimplemented_collective_never_ready():
    """Per-op readiness: all-gather has no fast path today, so it must
    report not-ready and a zero ceiling even where all-reduce is active."""
    assert not custom_collective.custom_collective_ready(2, op="all_gather")
    assert custom_collective.custom_collective_max_bytes(op="all_gather") == 0

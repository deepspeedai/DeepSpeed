# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Contract tests for the custom_collective inference all-reduce route.

The routing contract: inference_all_reduce must never engage the custom
P2P kernel unless initialize() succeeded on this topology. Everything
else — CPU tensors, non-P2P fabrics, world != 2, oversized messages —
falls back unchanged to the regular collectives.
"""

import pytest
import torch

import deepspeed.comm as dist

from unit.common import DistributedTest
from deepspeed.accelerator import get_accelerator
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


class TestSubgroupReduceStaysGroupScoped(DistributedTest):
    """inference_all_reduce must reduce within the caller's process group.

    The CPU shm fast kernel is bound to the WORLD topology and takes no group
    argument; letting it serve a proper TP subgroup would combine sibling
    subgroups' partial sums -- the TestTpGradNorm[2-*] CI failure, where tp=2
    subgroups in a 4-rank world received 4-way sums instead of 2-way.
    """
    world_size = 4

    def test(self):
        rank = dist.get_rank()
        # new_group is collective: every rank must create both groups.
        low_pair = dist.new_group([0, 1])
        high_pair = dist.new_group([2, 3])
        subgroup = low_pair if rank < 2 else high_pair

        # No explicit device index: on CUDA the current device is the one the
        # test rank owns; on CPU this is simply 'cpu'.
        device = torch.device(get_accelerator().device_name())
        payload = torch.full((8, ), float(rank), dtype=torch.float32, device=device)
        dist.inference_all_reduce(payload, op=dist.ReduceOp.SUM, group=subgroup)

        # rank 0+1 -> 1, rank 2+3 -> 5; a world-scoped reduction would give 6.
        expected = 1.0 if rank < 2 else 5.0
        assert torch.all(payload == expected), f"rank {rank}: {payload[0].item()} != {expected}"

        # Sanity: a world-wide call through the same entry stays correct too
        # (and for world-sized TP this is the configuration that keeps the
        # fast path eligible).
        world_payload = torch.full((8, ), float(rank), dtype=torch.float32, device=device)
        dist.inference_all_reduce(world_payload, op=dist.ReduceOp.SUM)
        assert torch.all(world_payload == 6.0), f"rank {rank}: {world_payload[0].item()} != 6.0"


class TestCustomARTwoDeviceParity(DistributedTest):
    """End-to-end parity over the production inference_all_reduce entry.

    Drives a real two-rank process group through the same dispatch the
    inference engine uses. On a P2P-capable pair this exercises the IPC
    exchange, handshake, and reduction kernel at every supported dtype and
    payload boundary; elsewhere initialize() must detect the fabric cannot
    serve the fast path and the hook must still produce correct results
    via the fallback. A plain collective after the initialize attempt
    guards against poisoning the standard backend on failed setups.
    """
    world_size = 2

    def test(self):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("custom AR is an accelerator-only fast path")
        device = torch.device(get_accelerator().device_name(), get_accelerator().current_device())

        # Reference sanity before touching the custom path: proves the pair
        # can run collectives at all, so a later failure is attributable.
        probe = torch.ones(1, device=device)
        dist.all_reduce(probe)
        assert probe.item() == 2

        # Route through the production hook so the test validates dispatch
        # decisions, not just the kernel in isolation.
        payload = torch.ones(1024, device=device)
        dist.inference_all_reduce(payload, op=dist.ReduceOp.SUM)

        ready = custom_collective.custom_collective_ready(2)
        max_bytes = custom_collective.custom_collective_max_bytes() if ready else 0

        # Same sanity after initialize(): a failed or unused fast path must
        # leave the standard backend fully functional.
        probe = torch.ones(1, device=device)
        dist.all_reduce(probe)
        assert probe.item() == 2

        element_sizes = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}
        for dtype, elem_bytes in element_sizes.items():
            if ready:
                # Sweep the payload boundaries of the kernel's IPC buffers:
                # odd unaligned counts, half the ceiling, and exactly full.
                max_elems = max_bytes // elem_bytes
                counts = sorted({1, 3, 7, max_elems // 2, max_elems})
            else:
                counts = [1, 3, 7, 1024]
            for count in counts:
                torch.manual_seed(count)
                tensor = torch.randn(count, dtype=dtype, device=device)
                reference = tensor.clone()
                dist.all_reduce(reference)
                dist.inference_all_reduce(tensor, op=dist.ReduceOp.SUM)
                assert torch.allclose(tensor.float(), reference.float(), rtol=2e-2, atol=2e-2), \
                    f"mismatch at dtype={dtype} count={count} ready={ready}"

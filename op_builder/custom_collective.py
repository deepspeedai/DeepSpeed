# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# No top-level torch/deepspeed imports: op_builder eagerly imports every
# top-level module, and torch-less environments (sdist build) must survive it.
from __future__ import annotations

from .builder import CUDAOpBuilder


class CustomCollectiveBuilder(CUDAOpBuilder):
    """One-shot P2P all-reduce for latency-bound TP=2 inference decode."""

    BUILD_VAR = "DS_BUILD_CUSTOM_COLLECTIVE"
    NAME = "custom_collective"

    def __init__(self, name=None):
        super().__init__(name=self.NAME if name is None else name)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        return ['csrc/comm/custom_collective.cu']

    def load(self, verbose=False):
        ext = super().load(verbose)
        return CUDACustomCollective(ext)


class CUDACustomCollective:
    """CUDA IPC backend for the custom-collective fast path.

    Wraps the compiled primitives (buffer minting, P2P connect, handshake,
    the one-shot kernel) with the CUDA-specific topology checks and handle
    exchange, so the comm layer stays accelerator-agnostic.

    This class is the reference implementation of the backend contract that
    deepspeed.comm.custom_collective.initialize() drives. Any accelerator
    backend (e.g. a future XPU port) only needs its builder's load() to
    return an object with the same five methods and semantics:

    - ``create_buffers``-backed ``initialize(group) -> bool``:
      allocate the shared buffers, return a small opaque handle the caller
      exchanges out-of-band (here: dist.all_gather_object), map the peer's
      handle, then probe the link. Pure plumbing lives here; it must be
      idempotent-safe to call once and must never engage the fast path on
      a topology it cannot serve (return False, the caller falls back to
      the standard comm library).
    - ``ready(world_size) -> bool``:
      fast path usable for this process-group size; capability gating
      (ISA level, dtype support) belongs here so callers need no
      backend-specific checks.
    - ``max_bytes() -> int``:
      payload ceiling of the pre-registered buffers; 0 means nothing
      qualifies. The kernel must reject or the gate must block anything
      larger.
    - ``all_reduce(tensor) -> tensor``:
      in-place one-shot reduce; the tensor is contiguous and within
      max_bytes by contract of the caller.
    - ``handshake``-backed probing inside initialize:
      exercise the actual flag protocol over the mapped link with a
      bounded spin, and force an all-ranks-agree verdict, so a fabric
      that maps but cannot carry the protocol degrades to the standard
      comm library instead of deadlocking the unbounded data-path spin.
    """

    def __init__(self, ext):
        self._ext = ext

    def initialize(self, group) -> bool:
        """Allocate IPC buffers and exchange handles with the peer.

        Must run before graph capture (buffer addresses are static from
        here on). Returns False when the topology cannot use the fast path,
        leaving the standard comm library as the answer.
        """
        import socket

        import torch
        from deepspeed.accelerator import get_accelerator
        import deepspeed.comm as dist  # lazy: avoid import cycle with the comm layer
        from deepspeed.comm.custom_collective import _agree

        world = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        if world != 2:
            return False  # v1 targets the TP=2 pair; bigger worlds fall back to NCCL
        # Local device ids of a TP pair are not always {0, 1} (e.g. {2, 3} in
        # TP2 x DP4) and the peer may sit on another host, so ask the peer
        # where it lives instead of assuming.
        dev = get_accelerator().current_device()
        host = socket.gethostname()
        where = [None, None]
        dist.all_gather_object(where, (host, dev), group=group)
        peer_host, peer_dev = where[1 - rank]
        local_ok = (peer_host == host and peer_dev != dev
                    and torch.cuda.can_device_access_peer(dev, peer_dev))  #ignore-cuda
        # Agree before any IPC step: a rank that cannot proceed must never
        # leave its peer blocked inside the exchange below.
        if not _agree(local_ok, group):
            return False

        # Pure CUDA IPC: the extension cudaMallocs its buffers and mints the
        # handle; only the 64 raw bytes cross the wire.
        handle = self._ext.create_buffers()
        gathered = [None, None]
        dist.all_gather_object(gathered, handle, group=group)
        # peer_dev assumes both ranks enumerate GPUs identically (same
        # CUDA_VISIBLE_DEVICES job); a mismatched ordinal fails the CE probe
        # below and falls back rather than engaging the fast path.
        try:
            self._ext.connect(gathered[1 - rank], peer_dev)
            connected = True
        except RuntimeError:
            connected = False
        if not _agree(connected, group):
            return False
        # Copy-engine probe before any kernel touches the mapping: a fabric
        # whose IPC mapping is silently broken kills the CUDA context on the
        # first in-kernel dereference, so only SM-safe links proceed to the
        # flag-protocol handshake below.
        try:
            probed = self._ext.api_probe()
        except RuntimeError:
            probed = False
        if not _agree(probed, group):
            return False
        get_accelerator().synchronize()
        # Publish-and-wait probe with a bounded spin: PCIe P2P fabrics may not
        # propagate the flag protocol NVLink honors — hang-proof the fast path
        # by falling back to NCCL where the probe fails.
        ok = self._ext.handshake()
        get_accelerator().synchronize()
        ok = self._ext.handshake() and ok  # second pass exercises the other parity
        # A unidirectional P2P fault fails the probe on one rank only; enabling
        # the fast path there while the peer falls back to NCCL would spin on
        # flags that never arrive. Force both ranks to the same verdict.
        ok = _agree(ok, group)
        if not ok:
            import os
            if os.environ.get("CUSTOM_COLLECTIVE_DEBUG"):
                print(f"[custom_collective] handshake FAILED on rank {rank} — NCCL fallback", flush=True)
        return ok

    def ready(self, world_size: int) -> bool:
        return self._ext.ready(world_size)

    def max_bytes(self) -> int:
        return self._ext.max_bytes()

    def all_reduce(self, tensor):
        return self._ext.all_reduce(tensor)

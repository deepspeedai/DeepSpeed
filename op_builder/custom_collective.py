# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import torch

from deepspeed.accelerator import get_accelerator

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
        import deepspeed.comm as dist  # lazy: avoid import cycle with the comm layer

        world = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        if world != 2:
            return False  # v1 targets the TP=2 pair; bigger worlds fall back to NCCL
        dev = get_accelerator().current_device()
        if not torch.cuda.can_device_access_peer(dev, 1 - dev):  #ignore-cuda
            return False  # no P2P (e.g. cross-CPU PCIe) — NCCL stays the right answer

        # Pure CUDA IPC: the extension cudaMallocs its buffers and mints the
        # handle; only the 64 raw bytes cross the wire.
        handle = self._ext.create_buffers()
        gathered = [None, None]
        dist.all_gather_object(gathered, handle, group=group)
        self._ext.connect(gathered[1 - rank])
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
        ok_t = torch.tensor([int(ok)], dtype=torch.int)
        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN, group=group)
        ok = bool(ok_t.item())
        if not ok:
            import os
            if os.environ.get("CUSTOM_COLLECTIVE_DEBUG"):
                print(f"[custom_collective] handshake FAILED on rank {rank} — NCCL fallback", flush=True)
        return ok

    def ready(self, world_size: int) -> bool:
        return self._ext.ready(world_size)

    def max_bytes(self) -> int:
        return self._ext.max_bytes()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._ext.all_reduce(tensor)

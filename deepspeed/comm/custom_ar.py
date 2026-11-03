# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Latency-optimized inference all-reduce for P2P-capable CUDA pairs.

Fills the deepspeed.comm.inference_all_reduce hook (the same slot the CPU
backend fills with shm+AVX): the comm layer asks ``custom_ar_ready()`` and
routes small CUDA tensors through the one-shot P2P kernel, everything else
falls back unchanged. Other accelerators can implement the same route with
their native fast path.
"""

import torch
import torch.distributed as tdist  # noqa: F401  (comm-layer internal, mirrors torch.py)

from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger

_state = {"op": None, "inited": False, "failed": False}


def _build():
    builder = get_accelerator().create_op_builder("CustomARBuilder")
    return None if builder is None else builder.load()


def custom_ar_ready(world_size: int = 2) -> bool:
    return _state["inited"] and _state["op"].ready(world_size)


def custom_ar_max_bytes() -> int:
    return int(_state["op"].max_bytes()) if _state["inited"] else 0


def initialize(group=None):
    """Allocate IPC buffers and exchange handles with the peer.

    Must run before CUDA-graph capture (buffer addresses are static from
    here on). Silently no-ops when the topology cannot use the fast path,
    leaving NCCL as the answer.
    """
    if _state["inited"] or _state["failed"] or not get_accelerator().is_available():
        return
    if not tdist.is_initialized():
        return
    world = tdist.get_world_size(group=group)
    rank = tdist.get_rank(group=group)
    if world != 2:
        return  # v1 targets the TP=2 pair; bigger worlds fall back to NCCL
    dev = get_accelerator().current_device()
    if not torch.cuda.can_device_access_peer(dev, 1 - dev):  #ignore-cuda
        return  # no P2P (e.g. cross-CPU PCIe) — NCCL stays the right answer

    try:
        op = _build()
    except Exception as e:
        _state["failed"] = True
        logger.warning(f"custom_ar JIT build failed ({e}); inference all-reduce stays on NCCL")
        return
    if op is None:
        _state["failed"] = True
        return
    # Pure CUDA IPC: the extension cudaMallocs its buffers and mints the
    # handle; only the 64 raw bytes cross the wire.
    handle = op.create_buffers()
    gathered = [None, None]
    tdist.all_gather_object(gathered, handle, group=group)
    op.connect(gathered[1 - rank])
    get_accelerator().synchronize()
    # Publish-and-wait probe with a bounded spin: PCIe P2P fabrics may not
    # propagate the flag protocol NVLink honors — hang-proof the fast path
    # by falling back to NCCL where the probe fails.
    ok = op.handshake()
    get_accelerator().synchronize()
    ok = op.handshake() and ok  # second pass exercises the other parity
    # A unidirectional P2P fault fails the probe on one rank only; enabling
    # the fast path there while the peer falls back to NCCL would spin on
    # flags that never arrive. Force both ranks to the same verdict.
    ok_t = torch.tensor([int(ok)], dtype=torch.int)
    tdist.all_reduce(ok_t, op=tdist.ReduceOp.MIN, group=group)
    ok = bool(ok_t.item())
    if not ok:
        import os as _os
        if _os.environ.get("CUSTOM_AR_DEBUG"):
            print(f"[custom_ar] handshake FAILED on rank {rank} — NCCL fallback", flush=True)
        return
    _state["op"] = op
    _state["inited"] = True


def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    return _state["op"].all_reduce(tensor)

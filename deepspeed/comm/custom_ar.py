# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Latency-optimized inference all-reduce for P2P-capable CUDA pairs.

Fills the deepspeed.comm.inference_all_reduce hook (the same slot the CPU
backend fills with shm+AVX): the comm layer asks ``custom_ar_ready()`` and
routes small CUDA tensors through the one-shot P2P kernel, everything else
falls back unchanged. Other accelerators can implement the same route with
their native fast path.
"""

import os

import torch
import torch.distributed as tdist  # noqa: F401  (comm-layer internal, mirrors torch.py)

_state = {"op": None, "inited": False}
_TOTAL_BYTES = 256 + 2 * 4 * 1024 * 1024  # flags + double 4MB buffers (matches csrc)


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build():
    from torch.utils.cpp_extension import load

    return load(name="ds_custom_ar",
                sources=[os.path.join(_repo_root(), "csrc", "inference", "custom_ar.cu")],
                extra_cuda_cflags=["-O3"],
                verbose=False)


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
    if _state["inited"] or not torch.cuda.is_available():
        return
    if not tdist.is_initialized():
        return
    world = tdist.get_world_size(group=group)
    rank = tdist.get_rank(group=group)
    if world != 2:
        return  # v1 targets the TP=2 pair; bigger worlds fall back to NCCL
    dev = torch.cuda.current_device()
    if not torch.cuda.can_device_access_peer(dev, 1 - dev):
        return  # no P2P (e.g. cross-CPU PCIe) — NCCL stays the right answer

    op = _build()
    # Python owns the buffers so they can ride torch's CUDA-IPC sharing.
    buf = torch.zeros(_TOTAL_BYTES, dtype=torch.uint8, device="cuda")
    seq = torch.zeros(1, dtype=torch.int32, device="cuda")
    op.initialize(buf, seq)

    # Share the storage (not just the view) so the peer maps the allocation.
    # _share_cuda_()/​_new_shared_cuda round-trip the full state tuple.
    info = buf.untyped_storage()._share_cuda_()
    gathered = [None, None]
    tdist.all_gather_object(gathered, info, group=group)
    peer_info = gathered[1 - rank]
    # The raw IPC handle bytes open on OUR device context in the extension
    # (torch's reconstruction would map on the producer's device instead).
    op.connect(peer_info[1])
    torch.cuda.synchronize()
    _state["op"] = op
    _state["inited"] = True


def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    return _state["op"].all_reduce(tensor)

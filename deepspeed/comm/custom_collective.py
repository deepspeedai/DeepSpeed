# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Latency-optimized inference all-reduce for P2P-capable accelerator pairs.

Fills the deepspeed.comm.inference_all_reduce hook (the same slot the CPU
backend fills with shm+AVX): the comm layer asks ``custom_collective_ready()``
and routes small accelerator tensors through the one-shot P2P kernel, everything
else falls back unchanged. Other accelerators can implement the same route
with their native fast path.

Readiness is per-collective: a builder may ship a fast all-reduce while
all-gather/reduce-scatter stay on the standard backend. Accelerator
specifics live behind the builder (see op_builder/custom_collective.py),
whose load() returns a backend object implementing initialize / ready /
max_bytes / all_reduce.
"""

import torch  # noqa: F401  (tensor API surface for callers; verdict tensors are built lazily)
import torch.distributed as tdist  # noqa: F401  (comm-layer internal, mirrors torch.py)

from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger

_state = {"op": None, "inited": False, "failed": False, "ranks": None}

# Collectives this fast path implements; queried per-op so accelerators
# that ship only a subset fall back per-collective, not globally.
SUPPORTED_OPS = {"all_reduce"}


def _build():
    builder = get_accelerator().create_op_builder("CustomCollectiveBuilder")
    return None if builder is None else builder.load()


def _group_ranks(group):
    return tuple(tdist.get_process_group_ranks(group if group is not None else tdist.group.WORLD))


def _agree(flag: bool, group) -> bool:
    """Collective AND: both ranks must pass or both fall back.

    The verdict tensor must live on the accelerator: an NCCL-only group has
    no CPU backend, so a CPU-tensor all_reduce raises and the error path
    leaves the state unlatched.
    """
    t = torch.tensor([int(flag)], dtype=torch.int, device=get_accelerator().current_device_name())
    tdist.all_reduce(t, op=tdist.ReduceOp.MIN, group=group)
    return bool(t.item())


def custom_collective_ready(world_size: int = 2, op: str = "all_reduce", group=None) -> bool:
    # The IPC buffers are mapped to one peer; a different 2-rank group must
    # not reuse them and silently reduce against the stale mapping.
    return (_state["inited"] and op in SUPPORTED_OPS and _state["op"].ready(world_size)
            and _state["ranks"] == _group_ranks(group))


def settled() -> bool:
    """initialize() reached a verdict: engaged, or explicitly declined this
    topology (latched ``failed``). False means it exited before latching
    either state -- i.e. it crashed and the caller swallowed the exception."""
    return _state["inited"] or _state["failed"]


def custom_collective_max_bytes(op: str = "all_reduce") -> int:
    if not _state["inited"] or op not in SUPPORTED_OPS:
        return 0
    return int(_state["op"].max_bytes())


def initialize(group=None):
    """Bring up the accelerator's native fast-collective backend.

    Must run before graph capture (buffer addresses are static from
    here on). Silently no-ops when the backend cannot serve this topology.
    """
    if _state["inited"] or _state["failed"] or not get_accelerator().is_available():
        return
    if not tdist.is_initialized():
        return
    # Cheap eligibility first: never JIT for topologies the fast path cannot serve
    if tdist.get_world_size(group=group) != 2:
        return
    try:
        backend = _build()
    except Exception as e:
        backend = None
        logger.warning(f"custom_collective JIT build failed ({e}); inference all-reduce stays on the comm library")
    # Latch failure up front: the caller swallows exceptions, so an unlatched
    # failure would retry _build() and the failing collective on every call.
    _state["failed"] = True
    # The backend exchange below is collective; a rank that failed to load must
    # not strand its peer inside it. Agree on the verdict before proceeding.
    if not _agree(backend is not None, group) or not backend.initialize(group):
        return
    _state["op"] = backend
    _state["ranks"] = _group_ranks(group)
    _state["inited"] = True
    _state["failed"] = False


def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    return _state["op"].all_reduce(tensor)

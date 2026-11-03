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

import torch  # noqa: F401  (tensor API surface for callers)
import torch.distributed as tdist  # noqa: F401  (comm-layer internal, mirrors torch.py)

from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger

_state = {"op": None, "inited": False, "failed": False}

# Collectives this fast path implements; queried per-op so accelerators
# that ship only a subset fall back per-collective, not globally.
SUPPORTED_OPS = {"all_reduce"}


def _build():
    builder = get_accelerator().create_op_builder("CustomCollectiveBuilder")
    return None if builder is None else builder.load()


def custom_collective_ready(world_size: int = 2, op: str = "all_reduce") -> bool:
    return _state["inited"] and op in SUPPORTED_OPS and _state["op"].ready(world_size)


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
    try:
        backend = _build()
    except Exception as e:
        _state["failed"] = True
        logger.warning(f"custom_collective JIT build failed ({e}); inference all-reduce stays on the comm library")
        return
    if backend is None or not backend.initialize(group):
        _state["failed"] = True
        return
    _state["op"] = backend
    _state["inited"] = True


def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    return _state["op"].all_reduce(tensor)

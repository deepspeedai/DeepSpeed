# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, List, Tuple
from dataclasses import dataclass, field

from torch.fx import Graph


@dataclass
class ProfilingResult:
    fwd_graph: Graph = None
    bwd_graph: Graph = None
    needs_backward: bool = False
    fwd_mem: List[Tuple[str, int, int, int]] = field(default_factory=list)  # name, current_alloc, delta, peak
    bwd_mem: List[Tuple[str, int, int, int]] = field(default_factory=list)
    fwd_mem_complete: bool = True
    bwd_mem_complete: bool = True
    fwd_time: List[Tuple[str, int, int]] = field(default_factory=list)  # name, device_time, wall_time
    bwd_time: List[Tuple[str, int, int]] = field(default_factory=list)
    fwd_tensor_sizes: List[Tuple[str, int]] = field(default_factory=list)  # name, size
    bwd_tensor_sizes: List[Tuple[str, int]] = field(default_factory=list)
    param_indices: List[Tuple[int, int, Tuple[int, ...]]] = field(default_factory=list)  # index, ds_id, ds_shape
    # Keep newly added fields at the end so positional construction of the
    # long-standing profiling fields remains backward compatible.
    process_group: Any = None
    # AOTAutograd invokes the optimized forward before it has compiled the
    # optimized backward.  Retain the accepted forward plan and its original
    # memory profile so the backward compiler can make one session-wide
    # admission decision before the native shared backing is allocated.
    prefetch_arena_forward_graph: Any = None
    prefetch_arena_forward_plan: Any = None
    prefetch_arena_forward_mem: Any = None
    prefetch_arena_forward_reserved_mem: Any = None
    prefetch_arena_forward_pool_reclaimable: Any = None
    prefetch_arena_session_accepted: Any = None
    prefetch_arena_session_capacity_bound: int = 0
    prefetch_arena_session_reason: Any = None

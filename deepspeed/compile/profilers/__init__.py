# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Tuple
from dataclasses import dataclass, field

from torch.fx import Graph


@dataclass
class ProfilingResult:
    fwd_graph: Graph = None
    bwd_graph: Graph = None
    needs_backward: bool = False
    fwd_mem: List[Tuple[str, int, int, int]] = field(default_factory=list)  # name, current_alloc, delta, peak
    bwd_mem: List[Tuple[str, int, int, int]] = field(default_factory=list)
    fwd_time: List[Tuple[str, int, int]] = field(default_factory=list)  # name, device_time, wall_time
    bwd_time: List[Tuple[str, int, int]] = field(default_factory=list)
    fwd_tensor_sizes: List[Tuple[str, int]] = field(default_factory=list)  # name, size
    bwd_tensor_sizes: List[Tuple[str, int]] = field(default_factory=list)
    param_indices: List[Tuple[int, int, Tuple[int, ...]]] = field(default_factory=list)  # index, ds_id, ds_shape

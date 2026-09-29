# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from abc import ABC, abstractmethod


def refresh_static_tensors(static, new):
    """Copy `new` into the tensors `static` holds, following dicts, lists and tuples.

    A replay reuses the tensors that were captured into the graph, so a later call's values
    reach the model only by being copied into them. Matching on `torch.is_tensor` alone stops
    at the top level, which leaves a kwarg that keeps its tensors inside a container holding
    whatever capture saw. SDXL passes `added_cond_kwargs` as
    `{"text_embeds": ..., "time_ids": ...}`, so every image after the first was built from the
    first prompt's conditioning, with no error to show for it.

    The graph can only run the inputs it captured, so a call whose tensors or containers do not
    line up with them raises instead of replaying stale values. Plain values are left alone.
    """
    if torch.is_tensor(static) and torch.is_tensor(new):
        static.copy_(new)
    elif isinstance(static, dict) and isinstance(new, dict):
        for key in static.keys() | new.keys():
            refresh_static_tensors(static.get(key), new.get(key))
    elif isinstance(static, (list, tuple)) and isinstance(new, (list, tuple)) and len(static) == len(new):
        for captured, latest in zip(static, new):
            refresh_static_tensors(captured, latest)
    elif any(torch.is_tensor(x) or isinstance(x, (dict, list, tuple)) for x in (static, new)):
        raise ValueError("CUDA graph replay needs the same tensors and containers as the captured call, "
                         f"got {type(new).__name__} where capture had {type(static).__name__}")


class CUDAGraph(ABC):

    def __init__(self, enable_cuda_graph=False):
        super().__init__()
        self.enable_cuda_graph = enable_cuda_graph

    @abstractmethod
    def _create_cuda_graph(self):
        """
        Create CUDA graph(s)
        """
        raise NotImplementedError

    @abstractmethod
    def _graph_replay(self):
        """
        Replay CUDA graph(s)
        """
        raise NotImplementedError

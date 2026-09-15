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
    """
    if torch.is_tensor(static):
        if torch.is_tensor(new):
            static.copy_(new)
    elif isinstance(static, dict):
        if isinstance(new, dict):
            for key, captured in static.items():
                if key in new:
                    refresh_static_tensors(captured, new[key])
    elif isinstance(static, (list, tuple)):
        if isinstance(new, (list, tuple)) and len(static) == len(new):
            for captured, latest in zip(static, new):
                refresh_static_tensors(captured, latest)


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

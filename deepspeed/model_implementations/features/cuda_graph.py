'''
Copyright 2023 The Microsoft DeepSpeed Team
'''
from abc import ABC, abstractmethod


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

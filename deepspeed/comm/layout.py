
import numpy as np
from dataclasses import dataclass
from enum import Enum
# This class creates the layout for all the devices used for the training system;
# It gets the world-size, together wit all the different parallelism degree (tp, pp, ep, sp, and dp)
# Then it creates a multi-dimensional list based on the order of the parallelism provided
# We use this list later to create the process groups for each of the paralllel ranks
# the order specifies from outer to inner parallel dimension

class DimName(Enum):
    TP = 't'
    PP = 'p'
    EP = 'e'
    DP = 'd'
    SP = 's'

@dataclass
class Dim_p:
    tag: str
    size: int

class Layout:
    def __init__(self, 
                 world_size: int,
                 parallel_dims: tuple(Dim_p)):
        self.world_size_ = world_size
        ranks = np.arange(0, world_size)
        
        layout_order = ''.join([dim.tag for dim in parallel_dims])
        self.layout_ = np.reshape(ranks, [dim.size for dim in parallel_dims])

        shape_list = self.layout_.shape
        self.layout_order_ = layout_order
        self.parallel_keys_ = [order[i] for i in range(len(order))]
        self.parallel_map_ = dict(zip(self.parallel_keys_, shape_list))

    def get_neighbors(self, parallel_key):
        neighbors = np.copy(self.layout_)
        paralle_size = self.parallel_map_[parallel_key]
        all_rest_dim = self.world_size_ // paralle_size
        transposed_layout = ''.join([pk for pk in self.parallel_keys_ if pk != parallel_key])
        transposed_layout = parallel_key + transposed_layout
        neighbors = np.einsum(f'{self.layout_order_}->{transposed_layout}', neighbors)
        neighbors = neighbors.reshape(paralle_size, all_rest_dim)
        return [neighbors[:,i].tolist() for i in range(all_rest_dim)]

    def get_ranks_for_parallel_dim(self, dim: Dim):
        return self.get_neighbors(dim.value)
    
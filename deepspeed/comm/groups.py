from .layout import Layout, ParallelDIMs, ParallelName

def Groups:
    def __init__(self, parallel_dims: ParallelDIMs, comm_order: str):
        Layout _layout(dist.get_world_size(), parallel_dims, comm_order)

    def create_group_for_parallel_dim(dim: str)
import torch
from .utils import *
from deepspeed import utils

supported_torch_version = False

# See more details at: https://github.com/pytorch/pytorch/pull/48767
# The PG API in torch versions lesser than 1.8 are different so it is
# non-trivial to support both in the same API. We will just use the
# DS comm. backend in deepspeed/comm/comm.py if torch version if 1.8+.

if older_torch():
    # Add custom deepspeed torch comm functions here since we can't import deepspeed.comm
    # NOTE: We can't call torch.distributed directly here. Current hack is to import functions before calling them.
    supported_torch_version = False
    from torch.distributed import *

    def get_world_group():
        return group.WORLD

    def get_global_rank(group, group_rank):
        if hasattr(torch.distributed.distributed_c10d, "get_global_rank"):
            from torch.distributed.distributed_c10d import get_global_rank as _get_global_rank
        else:
            from torch.distributed.distributed_c10d import _get_global_rank
        return _get_global_rank(group, group_rank)

    def allgather_fn(output_tensor, input_tensor, group=None, async_op=False):
        from torch.distributed import all_gather, get_world_size
        from torch import chunk
        output_tensors = list(chunk(output_tensor, get_world_size(group)))
        return all_gather(output_tensors, input_tensor, group=group, async_op=async_op)

    def reduce_scatter_fn(output_tensor, input_tensor, group=None, async_op=False):
        from torch.distributed import reduce_scatter, get_world_size
        from torch import chunk
        input_tensor_lst = list(chunk(input_tensor, get_world_size(group)))
        return reduce_scatter(output_tensor, input_tensor_lst, group=group)

    def configure(deepspeed_config=None,
                  enabled=None,
                  prof_all=None,
                  prof_ops=None,
                  verbose=None):
        utils.logger.warn(
            "Communication logging is not supported in torch versions older than 1.8")

else:
    supported_torch_version = True
    from .comm import *

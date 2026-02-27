import torch 
import torch.distributed as dist
from .sp_dp_registry import get_group, is_setup, sp_size, dp_size

@torch.library.custom_op("autosp::all_to_all", mutates_args=())
def all_to_all(
    input: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    name: str,
) -> torch.Tensor:
    """
    All-to-all collective for SDPA tensors [B, N, S, H].
    
    For QKV (scatter_idx=1, gather_idx=2):
        [B, N, S/P, H] -> [B, N/P, S, H]
    For O (scatter_idx=2, gather_idx=1):
        [B, N/P, S, H] -> [B, N, S/P, H]
    """
    assert is_setup(), 'Incorrect initialization of SP/DP mesh.'
    B, dim1, dim2, H = input.shape
    gid = dist.get_rank() // sp_size()
    group = get_group(gid)
    
    if scatter_idx == 1:
        N, local_S = dim1, dim2
        input_t = input.reshape(B, sp_size(), N // sp_size(), local_S, H)
        input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
        
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=group)
        
        output = output.permute(1, 2, 0, 3, 4).contiguous()
        output = output.reshape(B, N // sp_size(), sp_size() * local_S, H)
    else:  # scatter_idx == 2, O: scatter sequence, gather heads
        local_N, S = dim1, dim2
        input_t = input.reshape(B, local_N, sp_size(), S // sp_size(), H)
        input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()
        
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=group)
        
        output = output.permute(1, 0, 2, 3, 4).contiguous()
        output = output.reshape(B, sp_size() * local_N, S // sp_size(), H)
    
    return output


@torch.library.register_fake("autosp::all_to_all")
def all_to_all_fake(input: torch.Tensor, scatter_idx: int, gather_idx: int, name: str):
    B, dim1, dim2, H = input.shape
    if scatter_idx == 1:
        return input.new_empty(B, dim1 // sp_size(), dim2 * sp_size(), H)
    else:
        return input.new_empty(B, dim1 * sp_size(), dim2 // sp_size(), H)


def _all_to_all_backward_setup(ctx, inputs, output):
    _, scatter_idx, gather_idx, name = inputs
    ctx.scatter_idx = gather_idx
    ctx.gather_idx = scatter_idx
    ctx.name = name + "_grad"


def _all_to_all_backward(ctx, grad):
    return (
        all_to_all(grad, ctx.scatter_idx, ctx.gather_idx, ctx.name),
        None, None, None, None
    )


torch.library.register_autograd(
    "autosp::all_to_all", _all_to_all_backward, setup_context=_all_to_all_backward_setup
)

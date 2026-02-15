import torch 
import torch.distributed as dist

@torch.library.custom_op("autosp::all_to_all", mutates_args=())
def all_to_all(
    input: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    world_size: int,
    name: str,
) -> torch.Tensor:
    B, dim1, dim2, H = input.shape
    
    if scatter_idx == 1:
        N, local_S = dim1, dim2
        input_t = input.reshape(B, world_size, N // world_size, local_S, H)
        input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
        
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=dist.group.WORLD)
        
        output = output.permute(1, 2, 0, 3, 4).contiguous()
        output = output.reshape(B, N // world_size, world_size * local_S, H)
    else:
        local_N, S = dim1, dim2
        input_t = input.reshape(B, local_N, world_size, S // world_size, H)
        input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()
        
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=dist.group.WORLD)
        
        output = output.permute(1, 0, 2, 3, 4).contiguous()
        output = output.reshape(B, world_size * local_N, S // world_size, H)
    
    return output


@torch.library.register_fake("autosp::all_to_all")
def all_to_all_fake(input: torch.Tensor, scatter_idx: int, gather_idx: int, world_size: int, name: str):
    B, dim1, dim2, H = input.shape
    if scatter_idx == 1:
        return input.new_empty(B, dim1 // world_size, dim2 * world_size, H)
    else:
        return input.new_empty(B, dim1 * world_size, dim2 // world_size, H)


def _all_to_all_backward_setup(ctx, inputs, output):
    _, scatter_idx, gather_idx, world_size, name = inputs
    ctx.scatter_idx = gather_idx
    ctx.gather_idx = scatter_idx
    ctx.world_size = world_size
    ctx.name = name + "_grad"


def _all_to_all_backward(ctx, grad):
    return (
        all_to_all(grad, ctx.scatter_idx, ctx.gather_idx, ctx.world_size, ctx.name),
        None, None, None, None,
    )


torch.library.register_autograd(
    "autosp::all_to_all", _all_to_all_backward, setup_context=_all_to_all_backward_setup
)

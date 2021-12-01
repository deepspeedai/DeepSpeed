from typing import List, Tuple
import torch


def is_moe_param(param: torch.Tensor) -> bool:
    if hasattr(param, "allreduce") and not param.allreduce:
        return True
    return False


def split_params_into_shared_and_expert_params(
        params: List[torch.nn.Parameter]) -> Tuple[torch.nn.Parameter,
                                                   torch.nn.Parameter]:
    shared_params, expert_params = [], []
    for p in params:
        if is_moe_param(p):
            expert_params.append(p)
        else:
            shared_params.append(p)
    return shared_params, expert_params


def split_params_grads_into_shared_and_expert_params(
        group: List[torch.nn.Parameter]) -> Tuple[torch.nn.Parameter,
                                                  torch.nn.Parameter]:
    """Split grad of parameters into grads of non-expert params
    and grads of expert params. This is useful while computing
    grad-norms for clipping and overflow detection

        group (List[torch.nn.Parameter]):
    Args:
            The group of parameters to split

    Returns:
        Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
        list of gradients for non MoE params, list of gradients of MoE params
    """
    expert_grads = []
    shared_grads = []
    for p in group:
        if p.grad is not None:
            if is_moe_param(p):
                expert_grads.append(p.grad.to(p.dtype))
            else:
                shared_grads.append(p.grad.to(p.dtype))
    return shared_grads, expert_grads

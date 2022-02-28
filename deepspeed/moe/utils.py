from typing import List, Tuple, Dict
import torch
import deepspeed.utils.groups as groups
from .layer import MoE


def has_moe_layers(m):
    has_moe = False
    num_experts = 0
    for _, module in m.named_modules():
        if isinstance(module, MoE):
            has_moe = True
            num_experts = module.num_experts
            break
    return has_moe, num_experts


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


def split_params_into_different_moe_groups_for_optimizer(
        param_groups: Tuple[Dict]) -> Tuple[Dict]:
    """Split parameters into different MoE groups for optimizer

    Args:
        param_groups (Tuple[Dict]):
            The list of parameter groups to split

    Returns:
        Tuple[Dict]:
        list of MoE/non-MoE groups for optimizer
    """
    if isinstance(param_groups, tuple):
        param_groups = list(param_groups)  # Tuple cannot be modified
    elif isinstance(param_groups, dict):
        param_groups = [param_groups]
    elif not isinstance(param_groups, list):
        raise ValueError(f"Unknown param group type of {type(param_groups)}")

    # gather all data parallel group names
    data_parallel_group_names = set()
    for param_group in param_groups:
        for param in param_group["params"]:
            if is_moe_param(param):
                data_parallel_group_names.add(param.group_name)
    data_parallel_group_names = list(data_parallel_group_names)
    group_moe = {}
    # Create the param MoE groups, leave param assign to next step
    for param_group in param_groups:
        group_moe[param_group['name']] = {}
        for key in data_parallel_group_names:
            group_moe[param_group['name']][key] = {}
            group_moe[param_group['name']][key]['name'] = key
            group_moe[param_group['name']][key]['moe'] = True
            for ori_key in param_group.keys():
                if ori_key != 'name':
                    if ori_key == 'params':
                        group_moe[param_group['name']][key][ori_key] = []
                    else:
                        group_moe[
                            param_group['name']][key][ori_key] = param_group[ori_key]
    # Assign param
    for param_group in param_groups:
        new_params = []
        for param in param_group['params']:
            if is_moe_param(param):
                group_moe[param_group['name']][param.group_name]['params'].append(param)
                # param_group['params'].remove(param)
            else:
                new_params.append(param)
        param_group['params'] = new_params

    # Flatten the moe groups
    for k, v in group_moe.items():
        for k1, v1 in v.items():
            param_groups.append(v1)

    return tuple(param_groups)

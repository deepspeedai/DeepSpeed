'''
Copyright 2021 The Microsoft DeepSpeed Team
'''
# The file has been adapted from two fairscale files:
# (1) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/moe_layer.py
# (2) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/top2gate.py
# Git commit hash: 34df606902a240567a0d898037ece55c2f1336cf
# We retain the following license from the original files:

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import time
from time import perf_counter
import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}


def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from swtich transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 - epsilon,
                             device=device),
            high=torch.tensor(1.0 + epsilon,
                              device=device)).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


import torch.distributed as dist

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                group: dist.ProcessGroup,
                input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


from torch import nn
import torch.nn.functional as F

import math


def top1gating(logits: torch.Tensor,
               capacity_factor: float,
               min_capacity: int,
               used_token: torch.Tensor = None,
               noisy_gate_policy: Optional[str] = None) -> Tuple[Tensor,
                                                                 Tensor,
                                                                 Tensor]:
    """Implements Top1Gating on logits."""
    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # round-up
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if capacity < min_capacity:
        capacity = min_capacity

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(
        logits_w_noise if noisy_gate_policy == 'RSample' else gates,
        dim=1)
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # mask only used tokens
    if used_token is not None:
        mask1 = torch.einsum("s,se->se", used_token, mask1)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts

    uniform = exp_selection_uniform_map.get(logits.device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(0.0,
                             device=logits.device),
            high=torch.tensor(1.0,
                              device=logits.device)).rsample
        exp_selection_uniform_map[logits.device] = uniform

    mask1_rand = mask1 * uniform(mask1.shape)

    assert logits.shape[0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or inrease your batch size."

    _, top_idx = torch.topk(mask1_rand, k=capacity, dim=0)

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(new_mask1, dim=0) - 1

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * new_mask1, dim=1)

    # Normalize gate probabilities
    mask1_float = new_mask1.float()
    gates = gates * mask1_float

    locations1_sc = F.one_hot(locations1_s, num_classes=capacity).float()
    combine_weights = torch.einsum("se,sc->sec", gates, locations1_sc)
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts


def top2gating(logits: torch.Tensor,
               capacity_factor: float) -> Tuple[Tensor,
                                                Tensor,
                                                Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    # logits_fp32 = logits.to(torch.float32)
    gates = F.softmax(logits, dim=1)

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # capacity = (2 * num_tokens // num_experts) * capacity_factor
    # round-up
    capacity = math.ceil((2 * num_tokens / num_experts) * capacity_factor)

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = torch.einsum("se,se->s", gates, mask1_float)
    gates2_s = torch.einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = torch.einsum("s,se->se", gates1_s, mask1_float)
    gates2 = torch.einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = F.one_hot(locations1_s, num_classes=capacity).float()
    locations2_sc = F.one_hot(locations2_s, num_classes=capacity).float()
    combine1_sec = torch.einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = torch.einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts


class TopKGate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 eval_capacity_factor: float = 1.0,
                 min_capacity: int = 4,
                 noisy_gate_policy: Optional[str] = None) -> None:
        super().__init__()

        # Only top-1 and top-2 are supported at the moment.
        if k != 1 and k != 2:
            raise ValueError('Only top-1 and top-2 gatings are supported.')
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy

    def forward(
        self,
        input: torch.Tensor,
        used_token: torch.Tensor = None
    ) -> Tuple[Tensor,
               Tensor,
               Tensor]:  # type: ignore
        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        input_fp32 = input.float()
        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        logits = self.wg(input_fp32)
        if self.k == 1:
            return top1gating(
                logits,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity,
                used_token,
                self.noisy_gate_policy if self.training else None)
        else:
            return top2gating(
                logits,
                self.capacity_factor if self.training else self.eval_capacity_factor)


class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """
    def __init__(self,
                 gate: Module,
                 experts: Module,
                 num_local_experts: int,
                 group: Optional[Any] = None) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.num_local_experts = num_local_experts

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        # assert len(input) == 1, "only single input Tensor supported"
        # assert len(input[0].shape) == 3 or len(input[0].shape) == 2, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        # removed wrong assert
        # assert input[0].shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"
        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[-1]
        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = input[0].reshape(-1, d_model)
        self.l_aux, combine_weights, dispatch_mask, self.exp_counts  = self.gate(reshaped_input, input[1])
        dispatched_input = torch.einsum("sec,sm->ecm",
                                        dispatch_mask.type_as(input[0]),
                                        reshaped_input)

        #print(f"alltoall called at rank:{dist.get_rank()} with dispatched_input shape:{dispatched_input.shape}")
        a = time.perf_counter()
        dispatched_input = _AllToAll.apply(self.group, dispatched_input)
        b = time.perf_counter()
        #print(f"alltoall took {b-a} seconds at rank:{dist.get_rank()}")
        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.world_size,
                                                    self.num_local_experts,
                                                    -1,
                                                    d_model)
        #print(f"reshaped input after alltoall called at rank:{dist.get_rank()} is dispatched_input shape:{dispatched_input.shape}")
        expert_output = self.experts(dispatched_input)
        expert_output = _AllToAll.apply(self.group, expert_output)
        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.world_size * self.num_local_experts,
                                              -1,
                                              d_model)
        combined_output = torch.einsum("sec,ecm->sm",
                                       combine_weights.type_as(input[0]),
                                       expert_output)
        return combined_output.reshape(input[0].shape)

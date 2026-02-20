import deepspeed
from deepspeed.tops import MoEGating, MoEGather
import torch

import time

import torch.nn.functional as F
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple
from torch import Tensor


exp_selection_uniform_map: Dict[torch.device, Callable] = {}

@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]

@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()

def top1gating(logits: Tensor,
               capacity: int,
               use_rts: bool = True) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = torch.tensor(capacity).to(torch.int64)

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = me.mul_(ce).sum() * num_experts

    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=logits.device),
                                                          high=torch.tensor(1.0, device=logits.device)).rsample
            exp_selection_uniform_map[logits.device] = uniform

        mask1_rand = mask1 #* uniform(mask1.shape)
    else:
        mask1_rand = mask1

    top_idx = _top_idx(mask1_rand, capacity)

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1

    locations1 = torch.cumsum(mask1, dim=0) - 1

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    gates = gates * mask1_float

    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    combine_weights = torch.einsum("se,sc->sec", gates, locations1_sc)

    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts


n_tokens = 4096*4
n_experts = 64
hidden_size = 3072
capacity = 256

hidden_states = torch.randn(n_tokens, hidden_size, requires_grad=True).bfloat16().cuda()
hidden_states1 = hidden_states.clone() #torch.ones(n_tokens, hidden_size, requires_grad=True).bfloat16().cuda()
logits = torch.randn(n_tokens, n_experts, requires_grad=True).cuda()
logits_f = logits.clone() #torch.ones(n_tokens, n_experts, requires_grad=True).cuda()
# logits = logits.bfloat16()
weight = torch.randn(hidden_size, hidden_size, requires_grad=True).cuda().bfloat16()
weight1 = weight.clone() #torch.ones(hidden_size, hidden_size, requires_grad=True).cuda().bfloat16()
moe_gating = MoEGating(top_k=1)
moe_gather = MoEGather(top_k=1)

def run_baseline(logits, hidden_states):
    gate_out = top1gating(logits, n_tokens // n_experts)
    dispatched_input = torch.einsum("sec,sm->ecm", gate_out[2].type_as(hidden_states), hidden_states)
    out = torch.matmul(dispatched_input, weight1)
    out = torch.einsum("sec,ecm->sm", gate_out[1].type_as(hidden_states), out)
    return gate_out[0], out

def run_deepspeed(hidden_states, logits):
    l_aux, moe_inp, scores, mapped_slots = moe_gating(hidden_states, logits, 1.0)
    out = torch.matmul(moe_inp, weight)
    out = moe_gather(out, scores, mapped_slots,)
    return l_aux, out, scores, mapped_slots


logits_f.retain_grad()
hidden_states1.retain_grad()

logits.retain_grad()
hidden_states.retain_grad()

weight.retain_grad()
weight1.retain_grad()

l_aux, moe_input, scores, mapped_slots = run_deepspeed(hidden_states, logits)
print(l_aux.item(), moe_input.norm().item(), hidden_states.norm().item())
loss = l_aux + moe_input.sum()
loss.backward()
    
l_aux1, moe_input1 = run_baseline(logits_f, hidden_states1)
print(l_aux1.item(), moe_input1.norm().item(), hidden_states1.norm().item())
loss1 = moe_input1.sum() + l_aux1
loss1.backward()

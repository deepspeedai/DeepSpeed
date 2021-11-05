'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import torch.nn.init as init
import torch
import torch.distributed as dist

from deepspeed.utils import logger, log_dist

import deepspeed.utils.groups as groups
from .sharded_moe import MOELayer, TopKGate
from .experts import Experts
import copy
import typing


class MoE(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=1,
                 k=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 noisy_gate_policy: typing.Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts=True,
                 use_tutel: bool = False):
        """Initialize an MoE layer.

        Arguments:
            hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.

            expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).

            num_experts (int, optional): default=1, the total number of experts per layer.

            k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.

            capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.

            eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.

            min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.

            noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.

            drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to infinite capacity).

            use_rts (bool, optional): default=True, whether to use Random Token Selection.

            use_tutel (bool, optional): default=False, whether to use Tutel optimizations (if installed).
        """

        super(MoE, self).__init__()

        assert groups.is_initialized(), \
            'Please call deepspeed.utils.groups.initialize() before using MoE layers'
        assert noisy_gate_policy is None or noisy_gate_policy in ['None', 'Jitter', 'RSample'], \
            'Unsupported noisy_gate_policy: ' + noisy_gate_policy

        num_local_experts = num_experts // groups.get_expert_parallel_world_size()

        log_dist(
            f'num_experts: {num_experts} | num_local_experts: {num_local_experts} | expert_parallel_size: {groups.get_expert_parallel_world_size()}',
            [0])

        self.num_experts = num_experts
        experts = Experts(expert, num_local_experts)
        self.deepspeed_moe = MOELayer(TopKGate(hidden_size,
                                               num_experts,
                                               k,
                                               capacity_factor,
                                               eval_capacity_factor,
                                               min_capacity,
                                               noisy_gate_policy,
                                               drop_tokens,
                                               use_rts),
                                      experts,
                                      num_local_experts,
                                      group=groups.get_expert_parallel_group(),
                                      use_tutel=use_tutel)

    def forward(self, hidden_states, used_token=None):
        """ MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        output = self.deepspeed_moe(hidden_states, used_token)
        return output, self.deepspeed_moe.l_aux, self.deepspeed_moe.exp_counts

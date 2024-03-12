# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import json
import argparse
import torch
from collections import OrderedDict

from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.moe.layer import MoE
from deepspeed.accelerator import get_accelerator

import deepspeed.comm as dist
from .common import preferred_dtype


class SimpleModel(torch.nn.Module):

    def __init__(self, hidden_dim, empty_grad=False, nlayers=1):
        super(SimpleModel, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for i in range(nlayers)])
        if empty_grad:
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.empty_grad = empty_grad

    def forward(self, x, y):
        if len(self.linears) == 1:
            x = self.linears[0](x)
        else:
            for i, l in enumerate(self.linears):
                x = self.linears[i // 2](x) + l(x)
        return self.cross_entropy_loss(x, y)


class SimpleFrozenModel(torch.nn.Module):

    def __init__(self, hidden_dim, empty_grad=False):
        super(SimpleFrozenModel, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for i in range(2)])
        if empty_grad:
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.empty_grad = empty_grad
        # Freeze first layer
        self.linears[0].weight.requires_grad = False
        self.linears[0].bias.requires_grad = False

    def custom_state_dict(self, *args, **kwargs):
        state_dict = super(SimpleFrozenModel, self).state_dict(*args, **kwargs)
        custom = OrderedDict()
        for k, v in state_dict.items():
            if 'linears.0.weight' not in k:
                custom[k] = v
        return custom

    def forward(self, x, y):
        if len(self.linears) == 1:
            x = self.linears[0](x)
        else:
            for i, l in enumerate(self.linears):
                x = self.linears[i // 2](x) + l(x)
        return self.cross_entropy_loss(x, y)


class Curriculum_SimpleModel(SimpleModel):

    def __init__(self, hidden_dim, empty_grad=False):
        super(Curriculum_SimpleModel, self).__init__(hidden_dim, empty_grad)

    def forward(self, x, y, **kwargs):
        seqlen = kwargs.get('curriculum_seqlen', None)
        loss = super(Curriculum_SimpleModel, self).forward(x, y)
        return loss, seqlen


class SimpleMoEModel(torch.nn.Module):

    def __init__(self, hidden_dim, num_experts=4, ep_size=1, use_residual=False):
        super(SimpleMoEModel, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        expert = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Linear(hidden_dim, hidden_dim))
        # using two MoE layers to check implications of sharing a single storage
        self.moe_1 = MoE(hidden_size=hidden_dim,
                         expert=expert,
                         ep_size=ep_size,
                         use_residual=use_residual,
                         num_experts=num_experts,
                         k=1)
        # interleaving MoE modules with dense to create an opportunity
        # for gradients to be merged in ZeRO stage 2 average_tensor reduce bucket
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.moe_2 = MoE(hidden_size=hidden_dim,
                         expert=expert,
                         ep_size=ep_size,
                         use_residual=use_residual,
                         num_experts=num_experts,
                         k=1)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden_dim = self.linear1(x)
        output, _, _ = self.moe_1(hidden_dim)
        output = self.linear2(output)
        output, _, _ = self.moe_2(output)
        output = self.linear3(output)
        hidden_dim = hidden_dim + output
        sentence_embed = hidden_dim.mean(1)
        return self.cross_entropy_loss(sentence_embed, y)


class SimplePRMoEModel(torch.nn.Module):

    def __init__(self, hidden_dim, num_experts=2, ep_size=1, use_residual=False):
        super(SimplePRMoEModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = MoE(hidden_size=hidden_dim,
                           expert=linear2,
                           ep_size=ep_size,
                           use_residual=use_residual,
                           num_experts=num_experts,
                           k=1)
        linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = MoE(hidden_size=hidden_dim,
                           expert=linear3,
                           ep_size=ep_size,
                           use_residual=use_residual,
                           num_experts=int(2 * num_experts),
                           k=1)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden_dim = x
        hidden_dim = self.linear(hidden_dim)
        output, _, _ = self.linear2(hidden_dim)
        output, _, _ = self.linear3(output)
        hidden_dim = hidden_dim + output
        sentence_embed = hidden_dim.mean(1)
        return self.cross_entropy_loss(sentence_embed, y)


class UnusedParametersModel(SimpleModel):

    def __init__(self, hidden_dim, empty_grad=False):
        super().__init__(hidden_dim, empty_grad)

        self.unused_linear = torch.nn.Linear(hidden_dim, hidden_dim)


class LinearStack(torch.nn.Module):

    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.input_layer = torch.nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=False)
            for x in range(num_layers)
        ])
        self.output_layer = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class LinearStackPipe(PipelineModule):

    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128, num_layers=4, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        layers.append(LayerSpec(torch.nn.Linear, self.input_dim, self.hidden_dim))
        for x in range(self.num_layers):
            layers.append(LayerSpec(torch.nn.Linear, self.hidden_dim, self.hidden_dim, bias=False))
            layers.append(lambda x: x)
        layers.append(LayerSpec(torch.nn.Linear, self.hidden_dim, self.output_dim))

        super().__init__(layers=layers, loss_fn=torch.nn.CrossEntropyLoss(), **kwargs)


class SimpleOptimizer(torch.optim.Optimizer):

    def __init__(self, params, lr=0.11072018):
        defaults = dict(lr=lr)
        super(SimpleOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SimpleOptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)

        return loss


class HybridStateOptimizer(torch.optim.Optimizer):

    def __init__(self, params, lr=0.11072018):
        defaults = dict(lr=lr)
        super(HybridStateOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(HybridStateOptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['integer_step'] = 0
                    state['tensor_step'] = torch.zeros(1, device=p.device)

                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)
                state['integer_step'] += 1
                state['tensor_step'] += 1

        return loss


class PLD_SimpleModel(SimpleModel):

    def __init__(self, hidden_dim, empty_grad=False):
        super(PLD_SimpleModel, self).__init__(hidden_dim, empty_grad)

    def forward(self, x, y, **kwargs):
        pld = kwargs.get('progressive_layer_drop', False)
        theta = kwargs.get('pld_theta', 1.0)
        hidden_dim = super(PLD_SimpleModel, self).forward(x, y)
        return hidden_dim


def random_dataset(total_samples, hidden_dim, device, dtype=preferred_dtype()):
    train_data = torch.randn(total_samples, hidden_dim, device=device, dtype=dtype)
    train_label = torch.empty(total_samples, dtype=torch.long, device=device).random_(hidden_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    return train_dataset


def random_dataloader(model, total_samples, hidden_dim, device, dtype=preferred_dtype()):
    batch_size = model.train_micro_batch_size_per_gpu()
    train_dataset = random_dataset(total_samples, hidden_dim, device, dtype=dtype)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    return train_loader


def sequence_dataloader(model, total_samples, hidden_dim, device, seq_len: int = 32, dtype=preferred_dtype()):
    batch_size = model.train_micro_batch_size_per_gpu()
    train_data = torch.randn(total_samples, seq_len, hidden_dim, device=device, dtype=dtype)
    train_label = torch.empty(total_samples, dtype=torch.long, device=device).random_(hidden_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    return train_loader


def create_config_from_dict(tmpdir, config_dict):
    config_path = os.path.join(tmpdir, 'temp_config.json')
    with open(config_path, 'w') as fd:
        json.dump(config_dict, fd)
    return config_path


def create_deepspeed_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args='')
    args.deepspeed = True
    if dist.is_initialized():
        # We assume up to one full node executing unit tests
        assert dist.get_world_size() <= get_accelerator().device_count()
        args.local_rank = dist.get_rank()
    return args


def args_from_dict(tmpdir, config_dict):
    args = create_deepspeed_args()
    config_path = create_config_from_dict(tmpdir, config_dict)
    args.deepspeed_config = config_path
    return args

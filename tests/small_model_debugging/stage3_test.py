# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

import deepspeed

###################################
# Setup
###################################


class VerboseLinear(torch.nn.Linear):

    def __init__(self, **kwargs):
        print(f'Begin VerboseLinear.__init__')
        super().__init__(**kwargs)
        print(f'End VerboseLinear.__init__')


class LinearStack(torch.nn.Module):

    def __init__(self, input_dim=2, hidden_dim=4, output_dim=4, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.input_layer = VerboseLinear(in_features=self.input_dim, out_features=self.hidden_dim)
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=False)
            for x in range(num_layers)
        ])
        self.output_layer = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)
        self.identity = torch.nn.Identity()

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.identity(x)
        return x


###################################
# DRIVER
###################################


def test_driver():
    print()
    print('BUILDING MODEL')
    with deepspeed.zero.Init():
        model = LinearStack()
    print()

    # parted = [name for (name, p) in model.named_parameters() if p._partitioned]
    # not_parted = [name for (name, p) in model.named_parameters() if not p._partitioned]
    # print('partitioned: ', parted)
    # print('full: ', not_parted)
    # print()

    model.train()

    test_input = torch.rand(1, model.input_dim)
    grad_output = torch.rand(1, model.output_dim)

    grad_output.requires_grad = False
    test_input.requires_grad = False

    print()
    print('BEGINNING FORWARD')
    print()

    output = model(test_input)
    output.backward(grad_output)

    # parted = [name for (name, p) in model.named_parameters() if p._partitioned]
    # not_parted = [name for (name, p) in model.named_parameters() if not p._partitioned]
    # print('partitioned: ', parted)
    # print('full:' , not_parted)
    # print()

    #samyamspeed.disable()


test_driver()

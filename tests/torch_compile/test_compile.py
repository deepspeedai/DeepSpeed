# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed import comm

import torch
import intel_extension_for_pytorch  # noqa: F401 # type: ignore
from torch.utils.data import Dataset, DataLoader

torch._dynamo.config.cache_size_limit = 100

import collections


def get_dynamo_stats():
    # TODO: consider deepcopy'ing the entire counters struct and
    # adding a helper to do subtraction on it
    return collections.Counter({
        "calls_captured": torch._dynamo.utils.counters["stats"]["calls_captured"],
        "unique_graphs": torch._dynamo.utils.counters["stats"]["unique_graphs"],
        "graph_breaks": sum(torch._dynamo.utils.counters["graph_break"].values()),
        # NB: The plus removes zero counts
        "unique_graph_breaks": len(+torch._dynamo.utils.counters["graph_break"]),
        "autograd_captures": torch._dynamo.utils.counters["compiled_autograd"]["captures"],
        "autograd_compiles": torch._dynamo.utils.counters["compiled_autograd"]["compiles"],
        "cudagraph_skips": torch._dynamo.utils.counters["inductor"]["cudagraph_skips"],
    })


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size).to(torch.bfloat16)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


data_size = 1024
data_length = 100
rand_loader = DataLoader(dataset=RandomDataset(data_size, data_length), batch_size=1, shuffle=False)


class MyModule(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc0 = torch.nn.Linear(1024, 256, bias=False)
        self.fc1 = torch.nn.Linear(256, 256, bias=False)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data, residual):
        output = residual + self.fc1(self.fc0(self.dropout(data))) * 0.5
        return output


model = MyModule()
params = model.parameters()

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
parser.add_argument('--deepspeed_config',
                    type=str,
                    default='ds_config.json',
                    help='path to DeepSpeed configuration file')
cmd_args = parser.parse_args()

# initialize the DeepSpeed engine
model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=params)
model_engine.compile()

residual = torch.rand(256, 256, dtype=torch.float).to(get_accelerator().current_device_name())

start_stats = get_dynamo_stats()

for step, batch in enumerate(rand_loader):
    if step % 10 == 0 and comm.get_rank() == 0:
        print(f'step={step}')
    # forward() method
    loss = model_engine(batch.to(get_accelerator().current_device_name()), residual).sum()
    # runs backpropagation
    model_engine.backward(loss)
    # weight update
    model_engine.step()

dynamo_stats = get_dynamo_stats()
dynamo_stats.subtract(start_stats)

if comm.get_rank() == 0:
    print(dynamo_stats)

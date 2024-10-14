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


def get_dynamo_stats():
    return torch._dynamo.utils.counters["graph_break"]


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
                    default='ds_config_z3.json',
                    help='path to DeepSpeed configuration file')
cmd_args = parser.parse_args()

# initialize the DeepSpeed engine
model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=params)
model_engine.compile()

residual = torch.rand(256, 256, dtype=torch.float).to(get_accelerator().current_device_name())

start_stats = get_dynamo_stats()

if comm.get_rank() == 0:
    #print(dynamo_stats['graph_breaks'])
    for item in start_stats.items():
        print(item)

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

if comm.get_rank() == 0:
    # print break down of graph break stats with markdown, print in table format, start with reason, then count
    # print a tag 'dynamo_output' before each line to allow post processing
    print("dynamo_output | Reason | Count |")
    print("dynamo_output | ------ | ----- |")
    for item in dynamo_stats.items():
        # replace '|' in item[0] with a literal '|' to avoid mess with table format
        item = (item[0].replace('|', r'\|'), item[1])
        print(f"dynamo_output | {item[0]} | {item[1]} |")
    print(f"dynamo_output | Total | {sum(dynamo_stats.values())} |")

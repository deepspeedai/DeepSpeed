# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Testing on a 8 GPUs node
NDEV_PER_NODE=2 torchrun --nnodes 1 --nproc-per-node 8 test_mics_config.py
"""

import os
import json
import argparse
import torch
import deepspeed
from torch.utils.data.distributed import DistributedSampler
import deepspeed.comm as dist


class SimpleModel(torch.nn.Module):

    def __init__(self, hidden_dim, empty_grad=False):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        if empty_grad:
            self.layers2 = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim)])
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden = x
        hidden = self.linear(hidden)
        return self.cross_entropy_loss(hidden, y)


def create_config_from_dict(tmpdir, config_dict):
    config_path = os.path.join(tmpdir, 'temp_config.json')
    with open(config_path, 'w') as fd:
        json.dump(config_dict, fd)
    return config_path


def get_data_loader(model, total_samples, hidden_dim, device):
    batch_size = model.train_micro_batch_size_per_gpu()
    train_data = torch.randn(total_samples, hidden_dim, device=device, dtype=torch.float)
    train_label = torch.empty(total_samples, dtype=torch.long, device=device).random_(hidden_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    return train_loader


def get_args(tmpdir, config_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument('--zero', type=int, default=3)
    parser.add_argument('--local_rank', type=int)

    parser.add_argument('--mics_shard_size', default=2, type=int)
    parser.add_argument('--mics_hierarchical_params_gather', default=False, action='store_true')
    args = parser.parse_args()  #args=''

    config_dict["zero_optimization"]["stage"] = args.zero
    config_dict["zero_optimization"]["mics_shard_size"] = args.mics_shard_size
    config_dict["zero_optimization"]["mics_hierarchical_params_gather"] = args.mics_hierarchical_params_gather

    # print('config_dict["zero_optimization"]', config_dict["zero_optimization"])
    config_path = create_config_from_dict(tmpdir, config_dict)

    args.deepspeed_config = config_path
    return args


def print0(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


rank = int(os.environ['RANK'])
print('seed:', 2222 + rank)
torch.random.manual_seed(2222 + rank)

config_dict = {
    "train_batch_size": 8,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015,
        }
    },
    "fp16": {
        "enabled": False,
        "initial_scale_power": 15
    },
    "zero_optimization": {
        "stage": 3,
        "reduce_bucket_size": 20,
        "mics_shard_size": 4,
        "mics_hierarchical_params_gather": True,
        "stage3_model_persistence_threshold": 10
    }
}
#        "initial_scale_power": 15
args = get_args('/tmp/', config_dict)
hidden_dim = 32

with deepspeed.zero.MiCS_Init(config_dict_or_path=config_dict):
    model = SimpleModel(hidden_dim, empty_grad=False)
# print('------> init model with deepspeed.zero.Init()')

model, _, _, _ = deepspeed.initialize(args=args,
                                      model=model,
                                      model_parameters=model.parameters(),
                                      dist_init_required=True)


def print_params(tag, model):
    if dist.get_rank() == 0:
        for n, p in model.named_parameters():
            print0("{} {}:{}".format(tag, n, p))


data_loader = get_data_loader(model=model, total_samples=1000, hidden_dim=hidden_dim, device=model.device)
#print_params('pre-train', model)
for n, batch in enumerate(data_loader):
    loss = model(batch[0], batch[1])
    if dist.get_rank() == 0:
        print("LOSS:", loss.item())
    model.backward(loss)
    model.step()
    #print_params('step={}'.format(n), model)
    if n == 5: break

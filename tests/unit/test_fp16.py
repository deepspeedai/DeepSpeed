import torch
import deepspeed
import argparse
import pytest
import json
import os
from common import distributed_test


def create_config_from_dict(tmpdir, config_dict):
    config_path = os.path.join(tmpdir, 'temp_config.json')
    with open(config_path, 'w') as fd:
        json.dump(config_dict, fd)
    return config_path


class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_dim, empty_grad=False):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        if empty_grad:
            self.layers2 = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim)])
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden_dim = x
        hidden_dim = self.linear(hidden_dim)
        return self.cross_entropy_loss(hidden_dim, y)


def test_temp_config_json(tmpdir):
    config_dict = {
        "train_batch_size": 1,
    }
    config_path = create_config_from_dict(tmpdir, config_dict)
    config_json = json.load(open(config_path, 'r'))
    assert 'train_batch_size' in config_json


def prepare_optimizer_parameters(model):
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [{
        'params': [p for n,
                   p in param_optimizer],
        'weight_decay': 0.0
    }]
    return optimizer_grouped_parameters


def get_data_loader(model, total_samples, hidden_dim, device):
    batch_size = model.train_micro_batch_size_per_gpu()
    train_data = torch.randn(total_samples, hidden_dim, device=device, dtype=torch.half)
    train_label = torch.empty(total_samples,
                              dtype=torch.long,
                              device=device).random_(hidden_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    return train_loader


def get_args(tmpdir, config_dict):
    config_path = create_config_from_dict(tmpdir, config_dict)
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args='')
    args.deepspeed = True
    args.deepspeed_config = config_path
    args.local_rank = 0
    return args


def test_lamb_fp16_basic(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Lamb",
            "params": {
                "lr": 0.00015,
                "max_grad_norm": 1.0
            }
        },
        "fp16": {
            "enabled": True
        }
    }
    args = get_args(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[1, 2])
    def _test_lamb_fp16_basic(args, model, hidden_dim):
        model, _, _,_ = deepspeed.initialize(args=args,
                                             model=model,
                                             model_parameters=model.parameters(),
                                             dist_init_required=False)
        data_loader = get_data_loader(model=model,
                                      total_samples=50,
                                      hidden_dim=hidden_dim,
                                      device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_lamb_fp16_basic(args=args, model=model, hidden_dim=hidden_dim)


def test_lamb_fp16_empty_grad(tmpdir):
    config_dict = {
        "train_batch_size": 1,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Lamb",
            "params": {
                "lr": 0.00015,
                "max_grad_norm": 1.0
            }
        },
        "fp16": {
            "enabled": True
        }
    }
    args = get_args(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=True)

    @distributed_test(world_size=[1])
    def _test_lamb_fp16_empty_grad(args, model, hidden_dim):
        model, _, _,_ = deepspeed.initialize(args=args,
                                             model=model,
                                             model_parameters=model.parameters(),
                                             dist_init_required=False)
        data_loader = get_data_loader(model=model,
                                      total_samples=50,
                                      hidden_dim=hidden_dim,
                                      device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_lamb_fp16_empty_grad(args=args, model=model, hidden_dim=hidden_dim)


def test_adamw_fp16_basic(tmpdir):
    config_dict = {
        "train_batch_size": 1,
        "steps_per_print": 1,
        "fp16": {
            "enabled": True
        }
    }
    args = get_args(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[1])
    def _test_adamw_fp16_basic(args, model, hidden_dim):
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _,_ = deepspeed.initialize(args=args,
                                             model=model,
                                             optimizer=optimizer,
                                             dist_init_required=False)
        data_loader = get_data_loader(model=model,
                                      total_samples=50,
                                      hidden_dim=hidden_dim,
                                      device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_adamw_fp16_basic(args=args, model=model, hidden_dim=hidden_dim)


def test_adamw_fp16_empty_grad(tmpdir):
    config_dict = {
        "train_batch_size": 1,
        "steps_per_print": 1,
        "fp16": {
            "enabled": True
        }
    }
    args = get_args(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=True)

    @distributed_test(world_size=[1])
    def _test_adamw_fp16_empty_grad(args, model, hidden_dim):
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _,_ = deepspeed.initialize(args=args,
                                             model=model,
                                             optimizer=optimizer,
                                             dist_init_required=False)
        data_loader = get_data_loader(model=model,
                                      total_samples=50,
                                      hidden_dim=hidden_dim,
                                      device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_adamw_fp16_empty_grad(args=args, model=model, hidden_dim=hidden_dim)

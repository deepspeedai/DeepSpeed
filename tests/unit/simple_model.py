import os
import json
import argparse
import torch

from deepspeed.pipe import PipelineModule, LayerSpec


class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_dim, empty_grad=False, rank=0):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        if empty_grad:
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.rank = rank
        self.empty_grad = empty_grad

    def forward(self, x, y):
        hidden_dim = x
        if self.rank == 0 and self.empty_grad:
            hidden_dim = self.linear(hidden_dim) + self.linear2(hidden_dim)
        else:
            hidden_dim = self.linear(hidden_dim)
        return self.cross_entropy_loss(hidden_dim, y)


class LinearStack(torch.nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.input_layer = torch.nn.Linear(in_features=self.input_dim,
                                           out_features=self.hidden_dim)
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(in_features=self.hidden_dim,
                            out_features=self.hidden_dim,
                            bias=False) for x in range(num_layers)
        ])
        self.output_layer = torch.nn.Linear(in_features=self.hidden_dim,
                                            out_features=self.output_dim)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class LinearStackPipe(PipelineModule):
    def __init__(self,
                 input_dim=128,
                 hidden_dim=128,
                 output_dim=128,
                 num_layers=4,
                 **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        layers.append(LayerSpec(torch.nn.Linear, self.input_dim, self.hidden_dim))
        for x in range(self.num_layers):
            layers.append(
                LayerSpec(torch.nn.Linear,
                          self.hidden_dim,
                          self.hidden_dim,
                          bias=False))
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
                    state['tensor_step'] = torch.zeros(1)

                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)
                state['integer_step'] += 1
                state['tensor_step'] += 1

        return loss


class PLD_SimpleModel(SimpleModel):
    def __init__(self, hidden_dim, empty_grad=False, rank=0):
        super(PLD_SimpleModel, self).__init__(hidden_dim, empty_grad, rank)

    def forward(self, x, y, **kwargs):
        pld = kwargs.get('progressive_layer_drop', False)
        theta = kwargs.get('pld_theta', 1.0)
        hidden_dim = super(PLD_SimpleModel, self).forward(x, y)
        return hidden_dim


def random_dataloader(model, total_samples, hidden_dim, device, dtype=torch.half):
    batch_size = model.train_micro_batch_size_per_gpu()
    train_data = torch.randn(total_samples, hidden_dim, device=device, dtype=dtype)
    train_label = torch.empty(total_samples,
                              dtype=torch.long,
                              device=device).random_(hidden_dim)
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
    if torch.distributed.is_initialized():
        # We assume up to one full node executing unit tests
        assert torch.distributed.get_world_size() <= torch.cuda.device_count()
        args.local_rank = torch.distributed.get_rank()
    else:
        args.local_rank = 0
    return args


def args_from_dict(tmpdir, config_dict):
    args = create_deepspeed_args()
    config_path = create_config_from_dict(tmpdir, config_dict)
    args.deepspeed_config = config_path
    return args

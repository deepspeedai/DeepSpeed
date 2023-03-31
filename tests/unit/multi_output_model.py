# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch


class MultiOutputModel(torch.nn.Module):

    def __init__(self, hidden_dim, weight_value):
        super(MultiOutputModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear.weight.data.fill_(weight_value)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        losses = []
        for x, y in zip(inputs, targets):
            hidden_dim = self.linear(x)
            loss = self.cross_entropy_loss(hidden_dim, y)
            losses.append(loss)
        return tuple(losses)


def multi_output_dataloader(model, total_samples, hidden_dim, device, inputs, targets):
    assert len(inputs) == len(targets)
    batch_size = model.train_micro_batch_size_per_gpu()

    train_data = [
        torch.full(size=(total_samples, hidden_dim), fill_value=x, device=device, dtype=torch.half, requires_grad=True)
        for x in inputs
    ]

    train_label = [torch.empty(total_samples, device=device, dtype=torch.long).fill_(y) for y in targets]

    train_dataset = torch.utils.data.TensorDataset(*train_data, *train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    return train_loader

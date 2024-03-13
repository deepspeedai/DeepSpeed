# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import pytest
import deepspeed
from unit.common import DistributedTest
from unit.util import skip_on_arch
from deepspeed.accelerator import get_accelerator

if get_accelerator().device_name() == 'hpu':
    pytest.skip("sparse_gradients not supported by HPU.", allow_module_level=True)


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(10, 3, mode="sum", sparse=True)
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, x, offsets):
        return self.linear(self.emb(x, offsets))


class Adam(torch.optim.Optimizer):

    def __init__(self, dense_params, sparse_params):
        super().__init__(dense_params + sparse_params, defaults={})
        self.adam = torch.optim.Adam(dense_params)
        self.adam_sparse = torch.optim.SparseAdam(sparse_params)

    @torch.no_grad()
    def step(self, closure=None):
        loss_1 = self.adam.step(closure)
        loss_2 = self.adam_sparse.step(closure)

        if loss_1 is not None and loss_2 is not None:
            return loss_1 + loss_2
        return loss_1 or loss_2


def get_model_optimizer():
    torch.manual_seed(0)
    model = Model()
    optimizer = Adam(list(model.linear.parameters()), list(model.emb.parameters()))
    return model, optimizer


def get_data(device):
    x = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long, device=device)
    offsets = torch.tensor([0, 4], dtype=torch.long, device=device)
    y = torch.tensor([[1.0], [0.0]], device=device)
    return x, offsets, y


class TestSparseAdam(DistributedTest):
    world_size = 2

    def test(self):
        skip_on_arch(min_arch=7)

        config_dict = {"train_batch_size": 2, "steps_per_print": 1, "sparse_gradients": True}
        model, optimizer = get_model_optimizer()
        loss = torch.nn.BCEWithLogitsLoss()
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config_dict)

        x, offsets, y = get_data(engine.device)

        engine.gradient_average = True
        res = engine(x, offsets)
        engine.backward(loss(res, y))

        averaged_grads = {}
        for k, v in engine.named_parameters():
            grad = v.grad.to_dense() if v.grad.is_sparse else v.grad
            averaged_grads[k] = grad
            v.grad = None

        engine.gradient_average = False
        res = engine(x, offsets)
        engine.backward(loss(res, y))

        for k, v in engine.named_parameters():
            grad = v.grad.to_dense() if v.grad.is_sparse else v.grad
            assert torch.allclose(grad, averaged_grads[k] * engine.world_size)

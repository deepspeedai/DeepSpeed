import torch
import torch.distributed as dist
import deepspeed
import pytest
from common import distributed_test


def test_sparse_adam(tmpdir):
    config_dict = {"train_batch_size": 2, "steps_per_print": 1, "sparse_gradients": True}

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

    model = Model()
    optimizer = Adam(list(model.linear.parameters()), list(model.emb.parameters()))

    @distributed_test(world_size=[2])
    def _test(model, optimizer):
        engine, _, _, _ = deepspeed.initialize(model=model,
                                              optimizer=optimizer,
                                              config=config_dict)
        loss = torch.nn.BCEWithLogitsLoss()
        x = torch.tensor([1,
                          2,
                          4,
                          5,
                          4,
                          3,
                          2,
                          9],
                         dtype=torch.long,
                         device=engine.device)
        offsets = torch.tensor([0, 4], dtype=torch.long, device=engine.device)
        y = torch.tensor([[1.0], [0.0]], device=engine.device)
        res = engine(x, offsets)
        with pytest.raises(AssertionError):
            engine.backward(loss(res, y))
        engine.step()

    _test(model, optimizer)

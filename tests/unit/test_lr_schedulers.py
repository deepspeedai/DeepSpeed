import torch
import deepspeed
import argparse
import pytest
import json
import os
from common import distributed_test
from simple_model import SimpleModel, SimpleOptimizer, random_dataloader, args_from_dict


@pytest.mark.parametrize("scheduler_type,params",
                         [("WarmupLR",
                           {}),
                          ("OneCycle",
                           {
                               'cycle_min_lr': 0,
                               'cycle_max_lr': 0
                           }),
                          ("LRRangeTest",
                           {})])
def test_get_lr_before_train(tmpdir, scheduler_type, params):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            },
        },
        "scheduler": {
            "type": scheduler_type,
            "params": params
        },
        "gradient_clipping": 1.0
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[1])
    def _test_get_lr_before_train(args, model, hidden_dim):
        model, _, _, lr_scheduler = deepspeed.initialize(args=args,
                                             model=model,
                                             model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)
        for n, batch in enumerate(data_loader):
            # get lr before training starts
            lr_scheduler.get_lr()
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_get_lr_before_train(args=args, model=model, hidden_dim=hidden_dim)

import pytest
from typing import Callable
import torch
from torch.optim import Optimizer, Adam, AdamW
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

from simple_model import args_from_dict, SimpleModel
from common import distributed_test

import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.lr_schedules import WARMUP_LR, WarmupLR
from deepspeed.runtime.config import ADAM_OPTIMIZER


@pytest.mark.parametrize('optimizer_type', [None, Optimizer, Callable])
def test_client_optimizer(tmpdir, optimizer_type):
    def _optimizer_callable(params) -> Optimizer:
        return AdamW(params=params)

    hidden_dim = 10
    model = SimpleModel(hidden_dim)

    config_dict = {'train_batch_size': 1}
    if optimizer_type is None:
        client_optimizer = None
        config_dict['optimizer'] = {'type': ADAM_OPTIMIZER}
    elif optimizer_type is Optimizer:
        client_optimizer = Adam(model.parameters())
    else:
        client_optimizer = _optimizer_callable

    args = args_from_dict(tmpdir, config_dict)

    @distributed_test(world_size=[1])
    def _test_client_optimizer(args, model, client_optimizer):
        _, ds_optimizer, _, _ = deepspeed.initialize(args=args,
                                                    model=model,
                                                    model_parameters=list(model.parameters()),
                                                    optimizer=client_optimizer)
        if client_optimizer is None:
            assert isinstance(ds_optimizer, FusedAdam)
        elif isinstance(client_optimizer, Optimizer):
            assert ds_optimizer == client_optimizer
        else:
            assert isinstance(ds_optimizer, AdamW)

    _test_client_optimizer(args=args, model=model, client_optimizer=client_optimizer)


@pytest.mark.parametrize('scheduler_type, optimizer_type',
                         [(None,
                           None),
                          (None,
                           Optimizer),
                          (None,
                           Callable),
                          (_LRScheduler,
                           None),
                          (_LRScheduler,
                           Optimizer),
                          (_LRScheduler,
                           Callable),
                          (Callable,
                           None),
                          (Callable,
                           Optimizer),
                          (Callable,
                           Callable)])
def test_client_lr_scheduler(tmpdir, scheduler_type, optimizer_type):
    def _my_lambda(epoch):
        return epoch // 10

    def _optimizer_callable(params) -> Optimizer:
        return torch.optim.AdamW(params=params)

    def _lr_scheduler_callable(optimizer) -> _LRScheduler:
        return LambdaLR(optimizer, _my_lambda)

    hidden_dim = 10
    model = SimpleModel(hidden_dim)

    config_dict = {'train_batch_size': 1}

    client_optimizer = None
    client_scheduler = None

    if optimizer_type is None:
        config_dict['optimizer'] = {'type': ADAM_OPTIMIZER}
    elif optimizer_type is Optimizer:
        client_optimizer = torch.optim.Adam(model.parameters())
    else:
        client_optimizer = _optimizer_callable

    if scheduler_type is None:
        config_dict['scheduler'] = {'type': WARMUP_LR, 'params': {}}
    elif scheduler_type == _LRScheduler:
        if isinstance(client_optimizer, Optimizer):
            client_scheduler = LambdaLR(client_optimizer, _my_lambda)
        else:
            # Verify invalid combination is correctly handled
            client_scheduler = LambdaLR(torch.optim.Adam(model.parameters()), _my_lambda)
    else:
        client_scheduler = _lr_scheduler_callable

    args = args_from_dict(tmpdir, config_dict)

    @distributed_test(world_size=[1])
    def _test_client_lr_scheduler(args, model, optimizer, lr_scheduler):
        if isinstance(lr_scheduler,
                      _LRScheduler) and not isinstance(optimizer,
                                                       Optimizer):
            with pytest.raises(AssertionError):
                _, _, _, _ = deepspeed.initialize(args=args,
                                                  model=model,
                                                  model_parameters=list(model.parameters()),
                                                  optimizer=optimizer,
                                                  lr_scheduler=lr_scheduler)
        else:
            _, _, _, ds_lr_scheduler = deepspeed.initialize(args=args,
                                                            model=model,
                                                            model_parameters=list(model.parameters()),
                                                            optimizer=optimizer,
                                                            lr_scheduler=lr_scheduler)
            if lr_scheduler is None:
                assert isinstance(ds_lr_scheduler, WarmupLR)
            elif isinstance(lr_scheduler, _LRScheduler):
                assert ds_lr_scheduler == lr_scheduler
            else:
                assert isinstance(ds_lr_scheduler, LambdaLR)

    _test_client_lr_scheduler(args=args,
                              model=model,
                              optimizer=client_optimizer,
                              lr_scheduler=client_scheduler)

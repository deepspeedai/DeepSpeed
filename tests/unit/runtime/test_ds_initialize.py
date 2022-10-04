import pytest
from typing import Callable
import torch
from torch.optim import Optimizer, Adam, AdamW
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

from unit.simple_model import SimpleModel, random_dataloader
from unit.common import DistributedTest
from unit.util import required_torch_version

import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.lr_schedules import WARMUP_LR, WarmupLR
from deepspeed.runtime.config import ADAM_OPTIMIZER
from deepspeed.runtime.utils import see_memory_usage


@pytest.mark.parametrize('zero_stage', [0, 3])
class TestNoOptim(DistributedTest):
    world_size = 1

    def test(self, zero_stage):
        if zero_stage == 3 and not required_torch_version():
            pytest.skip("zero-3 param offload requires at least torch 1.8")

        ds_config = {
            'train_batch_size': self.world_size,
            'fp16': {
                'enabled': True
            },
            'zero_optimization': {
                "stage": zero_stage,
                "offload_param": {
                    "device": "cpu"
                }
            }
        }
        # 20B test
        #hidden_dim = 16 * 1024
        hidden_dim = 4

        with deepspeed.zero.Init(enabled=zero_stage == 3, config_dict_or_path=ds_config):
            model = SimpleModel(hidden_dim, nlayers=78)
        see_memory_usage('pre-init', force=True)
        model, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
        see_memory_usage('post-init', force=True)
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.half)
        for batch in data_loader:
            model(batch[0], batch[1])
        see_memory_usage('post-fwds', force=True)


@pytest.mark.parametrize('optimizer_type', [None, Optimizer, Callable])
class TestClientOptimizer(DistributedTest):
    world_size = 1

    def test(self, optimizer_type):
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

        _, ds_optimizer, _, _ = deepspeed.initialize(config=config_dict,
                                                    model=model,
                                                    model_parameters=list(model.parameters()),
                                                    optimizer=client_optimizer)
        if client_optimizer is None:
            assert isinstance(ds_optimizer, FusedAdam)
        elif isinstance(client_optimizer, Optimizer):
            assert ds_optimizer == client_optimizer
        else:
            assert isinstance(ds_optimizer, AdamW)


@pytest.mark.parametrize("scheduler_type", [None, _LRScheduler, Callable])
@pytest.mark.parametrize("optimizer_type", [None, Optimizer, Callable])
class TestClientLrScheduler(DistributedTest):
    world_size = 1

    def test(self, scheduler_type, optimizer_type):
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
                client_scheduler = LambdaLR(torch.optim.Adam(model.parameters()),
                                            _my_lambda)
        else:
            client_scheduler = _lr_scheduler_callable

        if isinstance(client_scheduler,
                      _LRScheduler) and not isinstance(client_optimizer,
                                                       Optimizer):
            with pytest.raises(AssertionError):
                _, _, _, _ = deepspeed.initialize(config=config_dict,
                                                  model=model,
                                                  model_parameters=list(model.parameters()),
                                                  optimizer=client_optimizer,
                                                  lr_scheduler=client_scheduler)
        else:
            _, _, _, ds_lr_scheduler = deepspeed.initialize(config=config_dict,
                                                            model=model,
                                                            model_parameters=list(model.parameters()),
                                                            optimizer=client_optimizer,
                                                            lr_scheduler=client_scheduler)
            if client_scheduler is None:
                assert isinstance(ds_lr_scheduler, WarmupLR)
            elif isinstance(client_scheduler, _LRScheduler):
                assert ds_lr_scheduler == client_scheduler
            else:
                assert isinstance(ds_lr_scheduler, LambdaLR)

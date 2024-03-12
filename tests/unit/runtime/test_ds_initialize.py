# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
from typing import Callable
import torch
from torch.optim import Optimizer, Adam, AdamW
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

from unit.simple_model import SimpleModel, random_dataloader
from unit.common import DistributedTest
from unit.util import bf16_required_version_check, required_amp_check

import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.lr_schedules import WARMUP_LR, WarmupLR
from deepspeed.runtime.config import ADAM_OPTIMIZER
from deepspeed.runtime.utils import see_memory_usage, required_torch_version
from deepspeed.accelerator import get_accelerator


@pytest.mark.parametrize('zero_stage', [0, 3])
class TestNoOptim(DistributedTest):
    world_size = 1

    def test(self, zero_stage):
        if zero_stage == 3 and not required_torch_version(min_version=1.8):
            pytest.skip("zero-3 param offload requires at least torch 1.8")

        ds_config = {
            'train_batch_size': self.world_size,
            'zero_optimization': {
                "stage": zero_stage,
                "offload_param": {
                    "device": "cpu"
                }
            }
        }
        if get_accelerator().is_fp16_supported():
            ds_config["fp16"] = {"enabled": True}
        elif get_accelerator().is_bf16_supported():
            ds_config["bf16"] = {"enabled": True}
        # 20B test
        #hidden_dim = 16 * 1024
        hidden_dim = 4

        with deepspeed.zero.Init(enabled=zero_stage == 3, config_dict_or_path=ds_config):
            model = SimpleModel(hidden_dim, nlayers=78)
        see_memory_usage('pre-init', force=True)
        model, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
        see_memory_usage('post-init', force=True)
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
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


@pytest.mark.parametrize('client_parameters', [True, False])
class TestConfigOptimizer(DistributedTest):
    world_size = 1

    def test(self, client_parameters):
        ds_config = {"train_batch_size": 1, "optimizer": {"type": "Adam", "params": {"lr": 0.001}}}

        hidden_dim = 10
        model = SimpleModel(hidden_dim)

        if client_parameters:
            model_parameters = list(model.parameters())
        else:
            model_parameters = None

        _, ds_optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, model_parameters=model_parameters)

        assert isinstance(ds_optimizer, FusedAdam)


@pytest.mark.parametrize('optimizer_extension', ['zero1', 'zero2', 'zero3', 'amp', None])
@pytest.mark.parametrize('model_dtype', ['fp16', 'bf16', 'fp32'])
@pytest.mark.parametrize('grad_accum_dtype', [None, 'fp16', 'bf16', 'fp32'])
class TestOptimizerImplementation(DistributedTest):
    world_size = 1
    reuse_dist_env = True

    def test(self, optimizer_extension, model_dtype, grad_accum_dtype):
        if not get_accelerator().is_fp16_supported():
            if model_dtype == 'fp16' or grad_accum_dtype == 'fp16':
                pytest.skip("fp16 is not supported")
        if optimizer_extension == 'zero1':
            zero_stage = 1
        elif optimizer_extension == 'zero2':
            zero_stage = 2
        elif optimizer_extension == 'zero3':
            zero_stage = 3
        else:
            zero_stage = 0
        amp = (optimizer_extension == 'amp')
        fp16 = (model_dtype == 'fp16')
        bf16 = (model_dtype == 'bf16')
        # Skip checks
        if bf16 and not bf16_required_version_check():
            pytest.skip(
                "DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )
        if amp and not required_amp_check():
            pytest.skip("Amp is not installed can't run amp check")
        # Config declaration
        ds_config = {
            "train_batch_size": 1,
            'fp16': {
                'enabled': fp16
            },
            'bf16': {
                'enabled': bf16
            },
            'amp': {
                'enabled': amp
            },
            'zero_optimization': {
                "stage": zero_stage
            },
            "data_types": {
                "grad_accum_dtype": grad_accum_dtype
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.001
                }
            }
        }

        key = (optimizer_extension, model_dtype, grad_accum_dtype)

        # Enumerate supported configurations
        is_supported = {}
        # ZeRO 1 Wrapper
        is_supported[('zero1', 'fp16', None)] = True
        is_supported[('zero1', 'fp16', 'fp16')] = True
        is_supported[('zero1', 'fp16', 'bf16')] = True
        is_supported[('zero1', 'fp16', 'fp32')] = True
        is_supported[('zero1', 'bf16', None)] = True
        is_supported[('zero1', 'bf16', 'fp16')] = True
        is_supported[('zero1', 'bf16', 'bf16')] = True
        is_supported[('zero1', 'bf16', 'fp32')] = True
        is_supported[('zero1', 'fp32', None)] = True
        is_supported[('zero1', 'fp32', 'fp16')] = True
        is_supported[('zero1', 'fp32', 'bf16')] = True
        is_supported[('zero1', 'fp32', 'fp32')] = True
        # ZeRO 2 Wrapper
        is_supported[('zero2', 'fp16', None)] = True
        is_supported[('zero2', 'fp16', 'fp16')] = True
        is_supported[('zero2', 'fp16', 'bf16')] = True
        is_supported[('zero2', 'fp16', 'fp32')] = True
        is_supported[('zero2', 'bf16', None)] = True
        is_supported[('zero2', 'bf16', 'fp16')] = True
        is_supported[('zero2', 'bf16', 'bf16')] = True
        is_supported[('zero2', 'bf16', 'fp32')] = True
        is_supported[('zero2', 'fp32', None)] = True
        is_supported[('zero2', 'fp32', 'fp16')] = True
        is_supported[('zero2', 'fp32', 'bf16')] = True
        is_supported[('zero2', 'fp32', 'fp32')] = True
        # ZeRO 3 Wrapper
        is_supported[('zero3', 'fp16', None)] = True
        is_supported[('zero3', 'fp16', 'fp16')] = True
        is_supported[('zero3', 'fp16', 'bf16')] = True
        is_supported[('zero3', 'fp16', 'fp32')] = True
        is_supported[('zero3', 'bf16', None)] = True
        is_supported[('zero3', 'bf16', 'fp16')] = True
        is_supported[('zero3', 'bf16', 'bf16')] = True
        is_supported[('zero3', 'bf16', 'fp32')] = True
        is_supported[('zero3', 'fp32', None)] = True
        is_supported[('zero3', 'fp32', 'fp16')] = True
        is_supported[('zero3', 'fp32', 'bf16')] = True
        is_supported[('zero3', 'fp32', 'fp32')] = True
        # Amp Wrapper
        is_supported[('amp', 'fp32', None)] = True
        is_supported[('amp', 'fp32', 'fp32')] = True
        # FP16 Wrapper
        is_supported[(None, 'fp16', None)] = True
        is_supported[(None, 'fp16', 'fp16')] = True
        # BF16 Wrapper
        is_supported[(None, 'bf16', 'fp32')] = True
        is_supported[(None, 'bf16', None)] = True
        # No Wrapper
        is_supported[(None, 'fp32', None)] = True
        is_supported[(None, 'fp32', 'fp32')] = True

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model_parameters = list(model.parameters())

        if key in is_supported:
            _, ds_optimizer, _, _ = deepspeed.initialize(config=ds_config,
                                                         model=model,
                                                         model_parameters=model_parameters)
            assert True
        else:
            with pytest.raises(NotImplementedError):
                _, ds_optimizer, _, _ = deepspeed.initialize(config=ds_config,
                                                             model=model,
                                                             model_parameters=model_parameters)


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
                client_scheduler = LambdaLR(torch.optim.Adam(model.parameters()), _my_lambda)
        else:
            client_scheduler = _lr_scheduler_callable

        if isinstance(client_scheduler, _LRScheduler) and not isinstance(client_optimizer, Optimizer):
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

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
from collections import namedtuple
from typing import Dict, List, NamedTuple, Set, Tuple
import pytest
import deepspeed.comm as dist
import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.modules.container import ModuleList
from torch.nn.modules.loss import L1Loss
from torch.nn.parameter import Parameter

from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader

import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from deepspeed.runtime.zero.utils import ZeRORuntimeException
from deepspeed.accelerator import get_accelerator


def run_unbalanced_gradients(model, data_loader):

    def drop_some_gradients(model, iter):
        odd_iteration = iter % 2
        for i, p in enumerate(model.parameters()):
            p.requires_grad = (i % 2) == odd_iteration

    def enable_grads(model):
        for p in model.parameters():
            p.requires_grad = True

    for i, batch in enumerate(data_loader):
        drop_some_gradients(model, i + 1)
        loss = model(batch[0], batch[1])
        model.backward(loss)
        model.step()
        enable_grads(model)


def dump_state_dict(model):
    if dist.get_rank() == 0:
        print("state_dict:")
        for name, param in model.named_parameters():
            print(f"{name} {param.data}")


@pytest.mark.parametrize('zero_stage', [1, 2, 3])
class TestZeroUnbalancedGradients(DistributedTest):
    world_size = 1

    def test(self, zero_stage):
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 2,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": zero_stage
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            }
        }
        hidden_dim = 4

        model = SimpleModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model, total_samples=16, hidden_dim=hidden_dim, device=model.device)

        run_unbalanced_gradients(model, data_loader)


# testing the fix https://github.com/microsoft/DeepSpeed/pull/1227
class TestZero3RepeatForwardLoop(DistributedTest):
    world_size = 1

    def test(self, zero_stage=3):
        # force all params to be partitioned by forcing threshold=0
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 2,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": zero_stage,
                "stage3_param_persistence_threshold": 0
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            }
        }
        hidden_dim = 4

        class AlbertLikeModel(torch.nn.Module):

            def __init__(self, hidden_dim):
                super().__init__()
                self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
                self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

            def forward(self, x, y):
                # run the same layer multiple times in a loop - to test a stack of forwards, followed by a stack of backwards
                hidden = x
                for i in range(3):
                    hidden = hidden + self.linear(hidden)
                return self.cross_entropy_loss(hidden, y)

        model = AlbertLikeModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model, total_samples=16, hidden_dim=hidden_dim, device=model.device)

        for i, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


# testing the fix https://github.com/microsoft/DeepSpeed/pull/1227
# also reproduces the https://github.com/microsoft/DeepSpeed/pull/1372
@pytest.mark.parametrize('zero_stage', [2, 3])
@pytest.mark.parametrize('freeze_params', [True, False])
class TestZeroToFP32(DistributedTest):
    world_size = 2

    def test_1_param_group(self, tmpdir, zero_stage, freeze_params):
        # XXX: ideally refactor with the 2_param_group test as 75% is the same
        # force all params to be partitioned by forcing threshold=0
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 2,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": zero_stage,
                "stage3_param_persistence_threshold": 0
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            }
        }

        class MyModel(torch.nn.Module):

            def __init__(self, hidden_dim, n_layers, freeze_params):
                super().__init__()
                # to reproduce https://github.com/microsoft/DeepSpeed/pull/1372 it is important that
                # the number of total elements is uneven:
                # (1) 4 layers of 3*(3+1)=12 elements each, 48 in total
                self.ll = torch.nn.ModuleList(torch.nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers))
                # (2) the following adds 4+1=5 elements
                self.classifier = torch.nn.Linear(4, 1)
                # total 48+5=53 (uneven as desired) elements
                self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
                if freeze_params:
                    self.ll[0].weight.requires_grad = False
                    self.ll[0].bias.requires_grad = False

            def forward(self, x, y):
                hidden = x
                for l in self.ll:
                    hidden = l(hidden)
                return self.cross_entropy_loss(hidden, y)

        hidden_dim = 3  # do not change

        world_size = dist.get_world_size()
        # we want at least 2x layers as there are gpus to trigger round_robin_fp16_groups reshuffle in zero2
        n_layers = world_size * 2
        model = MyModel(hidden_dim=hidden_dim, n_layers=n_layers, freeze_params=freeze_params)

        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        # Flush zero stage 3 cache
        model.empty_partition_cache()

        data_loader = random_dataloader(model=model, total_samples=16, hidden_dim=hidden_dim, device=model.device)

        for i, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.empty_partition_cache()
        model.save_checkpoint(tmpdir)

        # make sure all sides saved it
        dist.barrier()

        orig_state_dict = {}
        for name, param in model.module.named_parameters():
            if zero_stage == 3:
                with deepspeed.zero.GatheredParameters(param, modifier_rank=None):
                    orig_state_dict[name] = param.detach().cpu()
            else:
                orig_state_dict[name] = param.detach().cpu()

        if zero_stage == 3:
            with deepspeed.zero.GatheredParameters(model.parameters(), modifier_rank=None):
                fp32_model = load_state_dict_from_zero_checkpoint(model.module, tmpdir)
                fp32_state_dict = fp32_model.state_dict()
        else:
            fp32_model = load_state_dict_from_zero_checkpoint(model.module, tmpdir)
            fp32_state_dict = fp32_model.state_dict()

        #dump_state_dict(fp32_model)

        if dist.get_rank() == 0:
            for name in orig_state_dict.keys():
                # float() workaround for torch<1.6
                assert torch.allclose(orig_state_dict[name].float(), fp32_state_dict[name].float())

    def test_2_param_groups(self, tmpdir, zero_stage, freeze_params):
        # TODO:
        # - need to test with multiple param groups
        # force all params to be partitioned by forcing threshold=0
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 2,
            "steps_per_print": 1,
            "zero_allow_untested_optimizer": 1,
            "zero_optimization": {
                "stage": zero_stage,
                "stage3_param_persistence_threshold": 0
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            }
        }

        class MyModel(torch.nn.Module):

            def __init__(self, hidden_dim, n_layers, freeze_params):
                super().__init__()
                self.ll = torch.nn.ModuleList(torch.nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers))
                self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
                if freeze_params:
                    self.ll[0].weight.requires_grad = False
                    self.ll[0].bias.requires_grad = False

            def forward(self, x, y):
                hidden = x
                for l in self.ll:
                    hidden = l(hidden)
                return self.cross_entropy_loss(hidden, y)

        hidden_dim = 3

        world_size = dist.get_world_size()
        n_layers = world_size * 2
        model = MyModel(hidden_dim=hidden_dim, n_layers=n_layers, freeze_params=freeze_params)

        optim_groups = [
            {
                "params": [l.weight for l in model.ll],
                "weight_decay": 0.01,
            },
            {
                "params": [l.bias for l in model.ll],
                "weight_decay": 0.0
            },
        ]
        optim = torch.optim.SGD(optim_groups, lr=0.1)

        model, _, _, _ = deepspeed.initialize(model=model,
                                              model_parameters=model.parameters(),
                                              optimizer=optim,
                                              config=config_dict)
        model.empty_partition_cache()

        data_loader = random_dataloader(model=model, total_samples=16, hidden_dim=hidden_dim, device=model.device)

        for i, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.empty_partition_cache()
        model.save_checkpoint(tmpdir)

        # make sure all sides saved it
        dist.barrier()

        #dump_state_dict(model)

        orig_state_dict = {}
        for name, param in model.module.named_parameters():
            if zero_stage == 3:
                with deepspeed.zero.GatheredParameters(param, modifier_rank=None):
                    orig_state_dict[name] = param.detach().cpu()
            else:
                orig_state_dict[name] = param.detach().cpu()

        if zero_stage == 3:
            with deepspeed.zero.GatheredParameters(model.parameters(), modifier_rank=None):
                fp32_model = load_state_dict_from_zero_checkpoint(model.module, tmpdir)
                fp32_state_dict = fp32_model.state_dict()
        else:
            fp32_model = load_state_dict_from_zero_checkpoint(model.module, tmpdir)
            fp32_state_dict = fp32_model.state_dict()

        #dump_state_dict(fp32_model)

        if dist.get_rank() == 0:
            for name in orig_state_dict.keys():
                # float() workaround for torch<1.6
                assert torch.allclose(orig_state_dict[name].float(), fp32_state_dict[name].float())


@pytest.mark.parametrize("allgather_bucket_size", [1000, 1001])
class TestIncorectAllgatherBucketSize(DistributedTest):
    world_size = 1

    def test(self, allgather_bucket_size, zero_stage=2):
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 2,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": zero_stage,
                "allgather_bucket_size": allgather_bucket_size
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            }
        }
        hidden_dim = 4

        model = SimpleModel(hidden_dim=hidden_dim)
        if allgather_bucket_size % 2 == 0:
            model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        else:
            with pytest.raises(AssertionError) as assertinfo:
                model, _, _, _ = deepspeed.initialize(config=config_dict,
                                                      model=model,
                                                      model_parameters=model.parameters())
            assert "allgather_bucket_size must be a multiple of nccl_start_alignment_factor" in str(assertinfo)


class TestPartitionNcclAlignment(DistributedTest):
    world_size = 4

    def test(self, zero_stage=2):
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 2,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": zero_stage
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            }
        }
        hidden_dim = 4

        model = SimpleModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        # get nccl all-gather send buffers alignment factor
        nccl_start_alignment_factor = model.optimizer.nccl_start_alignment_factor

        parallel_partitioned_bit16_groups = model.optimizer.parallel_partitioned_bit16_groups if zero_stage == 2 else model.optimizer.parallel_partitioned_fp16_groups
        for data_parallel_partitions in parallel_partitioned_bit16_groups:
            for partition_id, partitioned_data in enumerate(data_parallel_partitions):
                # verify that data partition start locations are 4-byte aligned
                assert (partitioned_data.data_ptr() % (2 * nccl_start_alignment_factor) == 0)


def _ds_initialize_for_param_partitioning_testing(model: Module, cfg: dict) -> DeepSpeedEngine:
    ds_engine, _, _, _ = deepspeed.initialize(config=cfg, model=model, model_parameters=model.parameters())

    return ds_engine


def _assert_partition_status(model: Module, valid_statuses: Set[ZeroParamStatus]) -> None:
    for _, param in model.named_parameters():
        assert param.ds_status in valid_statuses, param.ds_summary()


def _assert_fully_available(model: Module) -> None:
    for _, param in model.named_parameters():
        assert param.ds_status == ZeroParamStatus.AVAILABLE


class EltwiseMultiplicationModule(Module):

    def __init__(self, weight: Parameter) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: Tensor) -> Tensor:
        _assert_fully_available(self)
        result = self.weight * x

        return result


class EltwiseMultiplicationTestNetwork_Dict(Module):
    """used for testing purposes"""

    def __init__(
        self,
        weight1: Parameter,
        weight2: Parameter,
        weight3: Parameter,
    ) -> None:
        super().__init__()
        self.__layer1 = EltwiseMultiplicationModule(weight1)
        self.__layer2 = EltwiseMultiplicationModule(weight2)
        self.__layer3 = EltwiseMultiplicationModule(weight3)

        self.loss = L1Loss(reduction="none")

    def forward(self, x: Tensor, y: Tensor, use_module_trace: bool, param_prefetching: bool) -> Dict[str, Tensor]:
        _assert_partition_status(self,
                                 {ZeroParamStatus.NOT_AVAILABLE, ZeroParamStatus.INFLIGHT, ZeroParamStatus.AVAILABLE}
                                 if use_module_trace else {ZeroParamStatus.NOT_AVAILABLE})

        pre_layer_expected_states = {
            ZeroParamStatus.INFLIGHT if param_prefetching else ZeroParamStatus.NOT_AVAILABLE,
            ZeroParamStatus.AVAILABLE,
        }

        post_layer_expected_states = {
            ZeroParamStatus.AVAILABLE if param_prefetching else ZeroParamStatus.NOT_AVAILABLE,
        }

        _assert_partition_status(self.__layer1, pre_layer_expected_states)
        hidden1 = self.__layer1(x)
        _assert_partition_status(self.__layer1, post_layer_expected_states)

        _assert_partition_status(self.__layer2, pre_layer_expected_states)
        hidden2 = self.__layer2(hidden1)
        _assert_partition_status(self.__layer2, post_layer_expected_states)

        _assert_partition_status(self.__layer3, pre_layer_expected_states)
        y_hat = self.__layer3(hidden2)
        _assert_partition_status(self.__layer3, post_layer_expected_states)

        loss = self.loss(y_hat, y)

        _assert_partition_status(self,
                                 {ZeroParamStatus.NOT_AVAILABLE, ZeroParamStatus.INFLIGHT, ZeroParamStatus.AVAILABLE}
                                 if use_module_trace else {ZeroParamStatus.NOT_AVAILABLE})

        return {
            "hidden1": hidden1,
            "hidden2": hidden2,
            "y_hat": y_hat,
            "loss": loss,
        }

    @staticmethod
    def to_dict(outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return outputs


class EltwiseMultiplicationNamedTuple(NamedTuple):
    hidden1: Tensor
    hidden2: Tensor
    y_hat: Tensor
    loss: Tensor


class EltwiseMultiplicationTestNetwork_NamedTuple(EltwiseMultiplicationTestNetwork_Dict):

    def forward(self, *args, **kwargs) -> EltwiseMultiplicationNamedTuple:
        outputs_dicts = super().forward(*args, **kwargs)
        return EltwiseMultiplicationNamedTuple(hidden1=outputs_dicts['hidden1'],
                                               hidden2=outputs_dicts['hidden2'],
                                               y_hat=outputs_dicts['y_hat'],
                                               loss=outputs_dicts['loss'])

    @staticmethod
    def to_dict(outputs: EltwiseMultiplicationNamedTuple) -> Dict[str, Tensor]:
        return {
            "hidden1": outputs.hidden1,
            "hidden2": outputs.hidden2,
            "y_hat": outputs.y_hat,
            "loss": outputs.loss,
        }


EltwiseMultiplication_namedtuple = namedtuple('EltwiseMultiplication_namedtuple',
                                              ['hidden1', 'hidden2', 'y_hat', 'loss'])


class EltwiseMultiplicationTestNetwork_namedtuple(EltwiseMultiplicationTestNetwork_Dict):

    def forward(self, *args, **kwargs) -> EltwiseMultiplication_namedtuple:
        outputs_dicts = super().forward(*args, **kwargs)
        return EltwiseMultiplication_namedtuple(hidden1=outputs_dicts['hidden1'],
                                                hidden2=outputs_dicts['hidden2'],
                                                y_hat=outputs_dicts['y_hat'],
                                                loss=outputs_dicts['loss'])

    @staticmethod
    def to_dict(outputs: EltwiseMultiplicationNamedTuple) -> Dict[str, Tensor]:
        return {
            "hidden1": outputs.hidden1,
            "hidden2": outputs.hidden2,
            "y_hat": outputs.y_hat,
            "loss": outputs.loss,
        }


class EltwiseMultiplicationTestNetwork_Tuple(EltwiseMultiplicationTestNetwork_Dict):

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        outputs_dicts = super().forward(*args, **kwargs)
        return (outputs_dicts['hidden1'], outputs_dicts['hidden2'], outputs_dicts['y_hat'], outputs_dicts['loss'])

    @staticmethod
    def to_dict(outputs: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Dict[str, Tensor]:
        return {
            "hidden1": outputs[0],
            "hidden2": outputs[1],
            "y_hat": outputs[2],
            "loss": outputs[3],
        }


class EltwiseMultiplicationTestNetwork_List(EltwiseMultiplicationTestNetwork_Dict):

    def forward(self, *args, **kwargs) -> List[Tensor]:
        outputs_dicts = super().forward(*args, **kwargs)
        return [outputs_dicts['hidden1'], outputs_dicts['hidden2'], outputs_dicts['y_hat'], outputs_dicts['loss']]

    @staticmethod
    def to_dict(outputs: List[Tensor]) -> Dict[str, Tensor]:
        return {
            "hidden1": outputs[0],
            "hidden2": outputs[1],
            "y_hat": outputs[2],
            "loss": outputs[3],
        }


@pytest.mark.parametrize("param_persistence_threshold", [0, 10])
@pytest.mark.parametrize("fp16_enabled", [True, False])
@pytest.mark.parametrize("contiguous_gradients", [True, False])
@pytest.mark.parametrize("offload_optimizer", [True, False])
@pytest.mark.parametrize("zero_grad", [True, False])
@pytest.mark.parametrize("prefetching", [True, False])
@pytest.mark.parametrize("model_class", [
    EltwiseMultiplicationTestNetwork_Dict, EltwiseMultiplicationTestNetwork_NamedTuple,
    EltwiseMultiplicationTestNetwork_namedtuple, EltwiseMultiplicationTestNetwork_Tuple,
    EltwiseMultiplicationTestNetwork_List
])
class TestZero3ParamPartitioningBase(DistributedTest):
    world_size = 2

    def test(
        self,
        param_persistence_threshold: int,
        fp16_enabled: bool,
        contiguous_gradients: bool,
        offload_optimizer: bool,
        zero_grad: bool,
        prefetching: bool,
        model_class: EltwiseMultiplicationTestNetwork_Dict,
    ) -> None:
        if offload_optimizer and not contiguous_gradients:
            return

        m = 3
        n = 5
        weights = [Parameter(torch.zeros((m, n), dtype=torch.float32)) for _ in range(3)]
        model = model_class(*weights)
        prefetch_bucket_size = sum([p.numel() for p in model.parameters(recurse=True)])
        cfg = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "stage3_param_persistence_threshold": param_persistence_threshold,
                "contiguous_gradients": contiguous_gradients,
                "stage3_prefetch_bucket_size": prefetch_bucket_size if prefetching else 0
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1.
                }
            },
            "fp16": {
                "enabled": fp16_enabled,
                "loss_scale": 1.,
            }
        }

        if offload_optimizer:
            cfg["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }

        ds_engine = _ds_initialize_for_param_partitioning_testing(model, cfg)
        for i, weight in enumerate(weights):
            weight.ds_tensor.data = torch.full_like(weight.ds_tensor.data, (i + 1) * (1 + dist.get_rank()))

        def create_tensor(vals, dtype: torch.dtype = None) -> Tensor:
            return torch.as_tensor(vals,
                                   dtype=dtype or (torch.float16 if fp16_enabled else torch.float32),
                                   device=ds_engine.device)

        expected_hidden1 = create_tensor([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 2, 2],
            [2, 2, 2, 2, 2],
        ])
        expected_hidden2 = create_tensor([
            [2, 2, 2, 2, 2],
            [2, 2, 2, 8, 8],
            [8, 8, 8, 8, 8],
        ])
        expected_yhat = create_tensor([[6, 6, 6, 6, 6], [6, 6, 6, 48, 48], [48, 48, 48, 48, 48]])
        expected_loss = create_tensor([
            [5, 5, 5, 5, 5],
            [5, 5, 5, 47, 47],
            [47, 47, 47, 47, 47],
        ])

        for train_iter in range(3):
            activations = ds_engine(
                x=torch.ones((m, n), dtype=torch.float16 if fp16_enabled else torch.float32, device=ds_engine.device),
                y=torch.ones((m, n), dtype=torch.float16 if fp16_enabled else torch.float32, device=ds_engine.device),
                use_module_trace=train_iter > 0,
                param_prefetching=prefetching and train_iter > 0,
            )
            # for ease in testing convert outputs to dict.
            activations = model_class.to_dict(activations)
            assert torch.allclose(activations["hidden1"], expected_hidden1)
            assert torch.allclose(activations["hidden2"], expected_hidden2)
            assert torch.allclose(activations["y_hat"], expected_yhat)
            assert torch.allclose(activations["loss"], expected_loss)

            ds_engine.backward(activations["loss"].sum())

            # check the gradients
            grad_partitions = ds_engine.optimizer.get_fp32_grad_partitions()
            assert set(grad_partitions.keys()) == {0
                                                   }, f"should have one parameter group but got {len(grad_partitions)}"
            assert set(grad_partitions[0].keys()) == {0, 1, 2}
            dloss_wrt_layer1 = grad_partitions[0][0]
            dloss_wrt_layer2 = grad_partitions[0][1]
            dloss_wrt_layer3 = grad_partitions[0][2]

            assert dloss_wrt_layer1.dtype == torch.float
            assert dloss_wrt_layer2.dtype == torch.float
            assert dloss_wrt_layer3.dtype == torch.float

            # layer1 = [..., 1, 2, ...]
            # layer2 = [..., 2, 4, ...]
            # layer3 = [..., 3, 6, ...]
            # dloss_wrt_layer3 = hidden2
            # dloss_wrt_layer2 = layer3 * hidden1
            # dloss_wrt_layer1 = layer3 * layer2 * x

            grad_multiplier = 1 if zero_grad else (train_iter + 1)
            if dist.get_rank() == 0:
                assert torch.allclose(dloss_wrt_layer3.to(get_accelerator().device_name()),
                                      grad_multiplier * create_tensor([2] * 8, torch.float))
                assert torch.allclose(dloss_wrt_layer2.to(get_accelerator().device_name()),
                                      grad_multiplier * create_tensor([3 * 1] * 8, torch.float))
                assert torch.allclose(dloss_wrt_layer1.to(get_accelerator().device_name()),
                                      grad_multiplier * create_tensor([3 * 2 * 1] * 8, torch.float))
            elif dist.get_rank() == 1:
                # parameters dont split evenly across ranks so rank 1 has a zero-padded
                # partition
                assert torch.allclose(dloss_wrt_layer3.to(get_accelerator().device_name()),
                                      grad_multiplier * create_tensor(([8] * 7) + [0], torch.float))
                assert torch.allclose(dloss_wrt_layer2.to(get_accelerator().device_name()),
                                      grad_multiplier * create_tensor(([6 * 2] * 7) + [0], torch.float))
                assert torch.allclose(dloss_wrt_layer1.to(get_accelerator().device_name()),
                                      grad_multiplier * create_tensor(([6 * 4 * 1] * 7) + [0], torch.float))
            else:
                raise RuntimeError("test has world size of two")

            if zero_grad:
                ds_engine.optimizer.zero_grad()

        # TODO. add testing for this - for now we just call it to make sure it
        # doesn't throw
        ds_engine.optimizer.step()
        # taking an optimizer step invalidates all parameters, make sure everything
        # has been partitioned afterwards
        _assert_partition_status(ds_engine, {ZeroParamStatus.NOT_AVAILABLE})
        assert not math.isclose(ds_engine.optimizer._global_grad_norm, 0.0)


@pytest.mark.parametrize("init_context_manager", [True, False])
@pytest.mark.parametrize("reduce_scatter", [True, False])
class TestZero3ParamPartitioningLargeParam(DistributedTest):
    world_size = 4

    def test(self, init_context_manager: bool, reduce_scatter: bool, param_sz: int = 8100) -> None:

        class LargeParamModel(Module):

            def __init__(self):
                super().__init__()
                self.param = Parameter(torch.zeros((param_sz, ), dtype=torch.float32))

                # only do weight initialization on root rank to
                # make sure we are broadcasting correctly from rank 0
                if dist.get_rank() == 0:
                    partition_sz = math.ceil(self.param.numel() / dist.get_world_size())
                    offset = 0
                    for rank in range(dist.get_world_size()):
                        with torch.no_grad():
                            self.param[offset:offset + partition_sz].fill_(rank)
                        offset += partition_sz

            def forward(self, x: Tensor) -> Tensor:
                return x * self.param

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": reduce_scatter,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1.
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 1.,
            }
        }
        with deepspeed.zero.Init(mem_efficient_linear=False, enabled=init_context_manager):
            model = LargeParamModel()
        ds_engine = _ds_initialize_for_param_partitioning_testing(model, ds_config)

        for train_iter in range(3):  # test multiple iterations to cover prefetching
            activation: Tensor = ds_engine(torch.ones(param_sz, dtype=torch.float16, device=ds_engine.device))

            partition_sz = math.ceil(param_sz / self.world_size)
            for rank_idx, start_idx in enumerate(range(0, param_sz, partition_sz)):
                activation_from_partition = activation[start_idx:start_idx + partition_sz]
                assert torch.allclose(activation_from_partition, torch.full_like(activation_from_partition, rank_idx))

            ds_engine.backward(activation.sum())
            ds_engine.allreduce_gradients()

            avgd_gradients = ds_engine.optimizer.averaged_gradients
            assert set(avgd_gradients.keys()) == {0}, "should only have one parameter group"
            weight_gradient, = avgd_gradients[0]
            expected_weight_gradient = (train_iter + 1) * torch.full_like(weight_gradient, 1)

            assert torch.allclose(weight_gradient, expected_weight_gradient)


@pytest.mark.parametrize("param_sz", [100, 1_000, 10_000])
@pytest.mark.parametrize("n_layers", [100, 1_000])
@pytest.mark.parametrize("init_context_manager", [True, False])
class TestZero3ParamPartitioningManyParams(DistributedTest):
    world_size = 4

    def test(self, param_sz: int, n_layers: int, init_context_manager: bool) -> None:

        class ManyParamModel(Module):

            def __init__(self) -> None:
                super().__init__()

                self.modulelist = ModuleList(
                    EltwiseMultiplicationModule(weight=Parameter(torch.empty((param_sz, ), dtype=torch.float32)))
                    for _ in range(n_layers))

                for layer_num, module in enumerate(self.modulelist):
                    with deepspeed.zero.GatheredParameters(module.weight, modifier_rank=0):
                        param: Parameter = module.weight
                        partition_sz = math.ceil(param.numel() / dist.get_world_size())
                        offset = 0
                        for rank in range(dist.get_world_size()):
                            with torch.no_grad():
                                param[offset:offset + partition_sz].fill_(2 * layer_num * rank)
                            offset += partition_sz

            def forward(self, x: Tensor) -> Tensor:
                activations = []

                for module in self.modulelist:
                    x = module(x)
                    activations.append(x)

                return activations

        ds_cfg = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "contiguous_gradients": True,
                "overlap_comm": True,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1.
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 1.,
            }
        }

        with deepspeed.zero.Init(config=ds_cfg, mem_efficient_linear=False, enabled=init_context_manager):
            model = ManyParamModel()

        ds_engine = _ds_initialize_for_param_partitioning_testing(model, ds_cfg)

        for _ in range(3):  # test multiple iterations to cover prefetching
            activations: List[Tensor] = ds_engine(
                torch.ones((param_sz, ), dtype=torch.float16, device=ds_engine.device))
            assert len(activations) == n_layers

            partition_sz = math.ceil(param_sz / self.world_size)
            expected_activations = torch.empty(param_sz, dtype=torch.float16, device=ds_engine.device)
            for start_idx in range(0, param_sz, partition_sz):
                expected_activations[start_idx:start_idx + partition_sz] = dist.get_rank()

            for layer_num, activation in enumerate(activations):
                expected_activations *= 2 * layer_num
                assert torch.allclose(activation, expected_activations)

            # TODO. finish writing this test
            ds_engine.backward(activations[-1].sum())

            avgd_gradients = ds_engine.optimizer.averaged_gradients
            assert set(avgd_gradients.keys()) == {0}, "should only have one parameter group"
            weight_gradients: List[Tensor] = avgd_gradients[0]

            for layer_num, activation in enumerate(weight_gradients):
                pass


class TestZero3InitForParentWeightInitialization(DistributedTest):
    world_size = 4

    def test(self):

        class ModelWhereParentInitializesChildWeights(Module):

            def __init__(self) -> None:
                super().__init__()

                self.linear = Linear(12, 1)

                self.apply(self.__init_weights)

            def __init_weights(self, module):
                if isinstance(module, Linear):
                    with torch.no_grad():
                        module.weight.fill_(1 + dist.get_rank())

        ds_cfg = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "contiguous_gradients": True,
                "overlap_comm": True,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1.
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 1.,
            }
        }

        with deepspeed.zero.Init(config=ds_cfg, mem_efficient_linear=False, enabled=True):
            model = ModelWhereParentInitializesChildWeights()

        assert model.linear.weight.ds_tensor.numel() == math.ceil(12 / self.world_size)
        assert torch.allclose(model.linear.weight.ds_tensor, torch.full_like(model.linear.weight.ds_tensor, 1))


@pytest.mark.skip("not working")
@pytest.mark.parametrize("param_persistence_threshold", [0, 10])
@pytest.mark.parametrize("contiguous_gradients", [True, False])
@pytest.mark.parametrize("offload_optimizer", [True, False])
@pytest.mark.parametrize("zero_grad", [True, False])
@pytest.mark.parametrize("prefetching", [True, False])
@pytest.mark.parametrize("model_class", [
    EltwiseMultiplicationTestNetwork_Dict, EltwiseMultiplicationTestNetwork_NamedTuple,
    EltwiseMultiplicationTestNetwork_namedtuple, EltwiseMultiplicationTestNetwork_Tuple,
    EltwiseMultiplicationTestNetwork_List
])
class TestZero3ParamPartitioningBaseBF16(DistributedTest):
    world_size = 2

    def test(self, param_persistence_threshold: int, contiguous_gradients: bool, offload_optimizer: bool,
             zero_grad: bool, prefetching: bool, model_class: EltwiseMultiplicationTestNetwork_Dict) -> None:
        if offload_optimizer and not contiguous_gradients:
            return

        m = 3
        n = 5
        weights = [Parameter(torch.zeros((m, n), dtype=torch.float32)) for _ in range(3)]
        model = model_class(*weights)
        prefetch_bucket_size = sum([p.numel() for p in model.parameters(recurse=True)])
        cfg = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "stage3_param_persistence_threshold": param_persistence_threshold,
                "contiguous_gradients": contiguous_gradients,
                "stage3_prefetch_bucket_size": prefetch_bucket_size if prefetching else 0
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1.
                }
            },
            "bf16": {
                "enabled": True,
                "loss_scale": 1.,
            }
        }

        if offload_optimizer:
            cfg["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }

        ds_engine = _ds_initialize_for_param_partitioning_testing(model, cfg)
        for i, weight in enumerate(weights):
            weight.ds_tensor.data = torch.full_like(weight.ds_tensor.data, (i + 1) * (1 + dist.get_rank()))

        def create_tensor(vals):
            return torch.as_tensor(vals, dtype=torch.bfloat16, device=ds_engine.device)

        expected_hidden1 = create_tensor([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 2, 2],
            [2, 2, 2, 2, 2],
        ])
        expected_hidden2 = create_tensor([
            [2, 2, 2, 2, 2],
            [2, 2, 2, 8, 8],
            [8, 8, 8, 8, 8],
        ])
        expected_yhat = create_tensor([[6, 6, 6, 6, 6], [6, 6, 6, 48, 48], [48, 48, 48, 48, 48]])
        expected_loss = create_tensor([
            [5, 5, 5, 5, 5],
            [5, 5, 5, 47, 47],
            [47, 47, 47, 47, 47],
        ])

        for train_iter in range(3):
            _assert_partition_status(ds_engine, {ZeroParamStatus.NOT_AVAILABLE})
            activations = ds_engine(
                x=torch.ones((m, n), dtype=torch.bfloat16, device=ds_engine.device),
                y=torch.ones((m, n), dtype=torch.bfloat16, device=ds_engine.device),
                use_module_trace=train_iter > 0,
                param_prefetching=prefetching and train_iter > 0,
            )
            # for ease in testing convert outputs to dict.
            activations = model_class.to_dict(activations)
            assert torch.allclose(activations["hidden1"], expected_hidden1)
            assert torch.allclose(activations["hidden2"], expected_hidden2)
            assert torch.allclose(activations["y_hat"], expected_yhat)
            assert torch.allclose(activations["loss"], expected_loss)

            ds_engine.backward(activations["loss"].sum())
            _assert_partition_status(ds_engine, {ZeroParamStatus.NOT_AVAILABLE})

            # check the gradients
            grad_partitions = ds_engine.optimizer.get_fp32_grad_partitions()
            assert set(grad_partitions.keys()) == {0
                                                   }, f"should have one parameter group but got {len(grad_partitions)}"
            assert set(grad_partitions[0].keys()) == {0, 1, 2}
            dloss_wrt_layer1 = grad_partitions[0][0]
            dloss_wrt_layer2 = grad_partitions[0][1]
            dloss_wrt_layer3 = grad_partitions[0][2]

            # layer1 = [..., 1, 2, ...]
            # layer2 = [..., 2, 4, ...]
            # layer3 = [..., 3, 6, ...]
            # dloss_wrt_layer3 = hidden2
            # dloss_wrt_layer2 = layer3 * hidden1
            # dloss_wrt_layer1 = layer3 * layer2 * x

            expected_grad_dtype = torch.float32 if offload_optimizer else torch.bfloat16

            grad_multiplier = 1 if zero_grad else (train_iter + 1)
            if dist.get_rank() == 0:
                assert torch.allclose(dloss_wrt_layer3.to(get_accelerator().device_name()),
                                      grad_multiplier * create_tensor([2] * 8).to(expected_grad_dtype))
                assert torch.allclose(dloss_wrt_layer2.to(get_accelerator().device_name()),
                                      grad_multiplier * create_tensor([3 * 1] * 8).to(expected_grad_dtype))
                assert torch.allclose(dloss_wrt_layer1.to(get_accelerator().device_name()),
                                      grad_multiplier * create_tensor([3 * 2 * 1] * 8).to(expected_grad_dtype))
            elif dist.get_rank() == 1:
                # parameters dont split evenly across ranks so rank 1 has a zero-padded
                # partition
                assert torch.allclose(dloss_wrt_layer3.to(get_accelerator().device_name()),
                                      grad_multiplier * create_tensor(([8] * 7) + [0]).to(expected_grad_dtype))
                assert torch.allclose(dloss_wrt_layer2.to(get_accelerator().device_name()),
                                      grad_multiplier * create_tensor(([6 * 2] * 7) + [0]).to(expected_grad_dtype))
                assert torch.allclose(dloss_wrt_layer1.to(get_accelerator().device_name()),
                                      grad_multiplier * create_tensor(([6 * 4 * 1] * 7) + [0]).to(expected_grad_dtype))
            else:
                raise RuntimeError("test has world size of two")

            if zero_grad:
                ds_engine.optimizer.zero_grad()

        # TODO. add testing for this - for now we just call it to make sure it
        # doesn't throw
        ds_engine.optimizer.step()
        _assert_partition_status(ds_engine, {ZeroParamStatus.NOT_AVAILABLE})


class TestZeroOffloadStage1(DistributedTest):
    world_size = 2

    def test(self):
        config_dict = {
            "train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {
                    "device": "cpu"
                }
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
        dist.barrier()
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.parametrize('return_type', [tuple, list, dict])
class TestZero3DictFwd(DistributedTest):
    world_size = 1

    def test(self, return_type):
        config_dict = {
            "train_batch_size": 4,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 3
            }
        }
        hidden_dim = 10

        class MyModel(torch.nn.Module):

            def __init__(self, hidden_dim):
                super(MyModel, self).__init__()
                self.l1 = torch.nn.Linear(hidden_dim, hidden_dim)
                self.cel = torch.nn.CrossEntropyLoss()

            def forward(self, x, y):
                x = self.l1(x)
                loss = self.cel(x, y)
                if return_type == dict:
                    val = {'a': x, 'loss': loss, 'b': 1, 'c': None}
                elif return_type == list:
                    val = [x, loss]
                elif return_type == tuple:
                    val = (x, loss)
                else:
                    raise NotImplementedError
                return val

        with deepspeed.zero.Init():
            model = MyModel(hidden_dim)

        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
        dist.barrier()
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            if return_type == dict:
                loss = loss['loss']
            else:
                loss = loss[1]
            model.backward(loss)
            model.step()


@pytest.mark.parametrize('zero_stage', [1, 2, 3])
class TestZeroAdamOptimizerStepCount(DistributedTest):
    world_size = 1

    def test(self, zero_stage):
        # force all params to be partitioned by forcing threshold=0
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 2,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": zero_stage,
                "stage3_param_persistence_threshold": 0,
                "sub_group_size": 4,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            }
        }
        hidden_dim = 4

        model = SimpleModel(hidden_dim=hidden_dim, nlayers=12)
        model, optimizer, _, _ = deepspeed.initialize(config=config_dict,
                                                      model=model,
                                                      model_parameters=model.parameters())
        data_loader = random_dataloader(model=model, total_samples=16, hidden_dim=hidden_dim, device=model.device)

        for i, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

            step_counts = []
            if zero_stage == 3:
                for sub_group_id, _ in enumerate(optimizer.fp16_groups):
                    fp32_param = optimizer.fp32_partitioned_groups_flat[sub_group_id]
                    state = optimizer.optimizer.state[fp32_param]
                    step_counts.append(state['step'])
                assert all(step == step_counts[0] for step in step_counts)
            elif zero_stage == 1 or zero_stage == 2:
                for param_group in optimizer.optimizer.param_groups:
                    for param in param_group['params']:
                        state = optimizer.optimizer.state[param]
                        step_counts.append(state['step'])
                assert all(step == step_counts[0] for step in step_counts)


class TestZeroFrozenWeights(DistributedTest):
    world_size = 1

    def test(self):
        config_dict = {
            "train_batch_size": 4,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 3
            }
        }
        hidden_dim = 10

        class MyModel(torch.nn.Module):

            def __init__(self, hidden_dim):
                super(MyModel, self).__init__()
                self.l1 = torch.nn.Linear(hidden_dim, hidden_dim)
                self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
                self.act = torch.nn.ReLU()
                self.cel = torch.nn.CrossEntropyLoss()

                # freeze one fc
                self.l2.weight.requires_grad = False
                self.l2.bias.requires_grad = False

            def forward(self, x, y):
                x = self.l1(x)
                x = self.act(x)
                x = self.l2(x)
                loss = self.cel(x, y)
                val = (x, loss)
                return val

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            model = MyModel(hidden_dim)

        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
        dist.barrier()
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            loss = loss[1]
            model.backward(loss)
            model.step()


@pytest.mark.parametrize('force_ds_optim', [True, False])
class TestZeroOffloadOptim(DistributedTest):
    world_size = 1

    def test(self, force_ds_optim):
        config_dict = {
            "train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "steps_per_print": 1,
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {
                    "device": "cpu"
                }
            },
            "zero_force_ds_cpu_optimizer": force_ds_optim,
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)

        optimizer = torch.optim.Adam(model.parameters())

        if force_ds_optim:
            with pytest.raises(ZeRORuntimeException):
                model, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config_dict)
        else:
            model, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config_dict)


@pytest.mark.parametrize('training', [True, False])
class TestZeroPartitionCache(DistributedTest):
    world_size = 1

    def test_training_partition_cache(self, training):
        hidden_dim = 10
        config_dict = {
            "train_batch_size": 2,
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "zero_optimization": {
                "stage": 3,
                "stage3_param_persistence_threshold": hidden_dim
            }
        }
        if training:
            config_dict["optimizer"] = {"type": "Adam"}

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            model = SimpleModel(hidden_dim, empty_grad=False)

        model, _, _, _ = deepspeed.initialize(model=model, config=config_dict)

        dtype = torch.half
        data_loader = random_dataloader(model=model,
                                        total_samples=6,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=dtype)

        for _, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            if training:
                model.backward(loss)
                model.step()

        persist_param_size = sum([p.numel() for p in model.parameters() if p.ds_persist])

        assert persist_param_size >= sum([p.numel() for p in model.parameters()])

        model.empty_partition_cache()
        assert sum([p.numel() for p in model.parameters()]) == 0

import torch
import pytest
import json
import argparse
import os
import torch.distributed as dist

from common import distributed_test
from simple_model import SimpleModel, random_dataloader, args_from_dict

import deepspeed
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint


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
def test_zero_unbalanced_gradients(tmpdir, zero_stage):
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

    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 4

    model = SimpleModel(hidden_dim=hidden_dim)

    @distributed_test(world_size=[1])
    def _test_zero_unbalanced_gradients(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=16,
                                        hidden_dim=hidden_dim,
                                        device=model.device)

        run_unbalanced_gradients(model, data_loader)

    _test_zero_unbalanced_gradients(args=args, model=model, hidden_dim=hidden_dim)


# testing the fix https://github.com/microsoft/DeepSpeed/pull/1227
@pytest.mark.parametrize('zero_stage', [3])
def test_zero3_repeat_forward_loop(tmpdir, zero_stage):

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

    args = args_from_dict(tmpdir, config_dict)
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

    @distributed_test(world_size=[1])
    def _test_zero3_repeat_forward_loop(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=16,
                                        hidden_dim=hidden_dim,
                                        device=model.device)

        for i, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_zero3_repeat_forward_loop(args=args, model=model, hidden_dim=hidden_dim)


# testing the fix https://github.com/microsoft/DeepSpeed/pull/1227
# also reproduces the https://github.com/microsoft/DeepSpeed/pull/1372
@pytest.mark.parametrize('zero_stage', [2, 3])
def test_zero_to_fp32_1_param_group(tmpdir, zero_stage):

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

    @distributed_test(world_size=[2])
    def _test_zero_to_fp32():
        class MyModel(torch.nn.Module):
            def __init__(self, hidden_dim, n_layers):
                super().__init__()
                # to reproduce https://github.com/microsoft/DeepSpeed/pull/1372 it is important that
                # the number of total elements is uneven:
                # (1) 4 layers of 3*(3+1)=12 elements each, 48 in total
                self.ll = torch.nn.ModuleList(
                    torch.nn.Linear(hidden_dim,
                                    hidden_dim) for i in range(n_layers))
                # (2) the following adds 4+1=5 elements
                self.classifier = torch.nn.Linear(4, 1)
                # total 48+5=53 (uneven as desired) elements
                self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

            def forward(self, x, y):
                hidden = x
                for l in self.ll:
                    hidden = l(hidden)
                return self.cross_entropy_loss(hidden, y)

        args = args_from_dict(tmpdir, config_dict)
        hidden_dim = 3  # do not change

        world_size = dist.get_world_size()
        # we want at least 2x layers as there are gpus to trigger round_robin_fp16_groups reshuffle in zero2
        n_layers = world_size * 2
        model = MyModel(hidden_dim=hidden_dim, n_layers=n_layers)

        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=16,
                                        hidden_dim=hidden_dim,
                                        device=model.device)

        for i, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.save_checkpoint(tmpdir)

        # make sure all sides saved it
        dist.barrier()

        if zero_stage == 3:
            with deepspeed.zero.GatheredParameters(list(
                    model.module.parameters(recurse=True)),
                                                   modifier_rank=None):
                pass  # this forces gathering the model

        #dump_state_dict(model)

        orig_state_dict = {}
        for name, param in model.module.named_parameters():
            orig_state_dict[name] = param.detach().cpu()

        if dist.get_rank() == 0:
            fp32_model = load_state_dict_from_zero_checkpoint(model.module, tmpdir)
            #dump_state_dict(fp32_model)

            fp32_state_dict = fp32_model.state_dict()
            for name in orig_state_dict.keys():
                # float() workaround for torch<1.6
                assert torch.allclose(orig_state_dict[name].float(),
                                      fp32_state_dict[name].float())

    _test_zero_to_fp32()


@pytest.mark.parametrize('zero_stage', [2, 3])
def test_zero_to_fp32_2_param_groups(tmpdir, zero_stage):

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

    @distributed_test(world_size=[2])
    def _test_zero_to_fp32():
        class MyModel(torch.nn.Module):
            def __init__(self, hidden_dim, n_layers):
                super().__init__()
                self.ll = torch.nn.ModuleList(
                    torch.nn.Linear(hidden_dim,
                                    hidden_dim) for i in range(n_layers))
                self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

            def forward(self, x, y):
                hidden = x
                for l in self.ll:
                    hidden = l(hidden)
                return self.cross_entropy_loss(hidden, y)

        args = args_from_dict(tmpdir, config_dict)
        hidden_dim = 3

        world_size = dist.get_world_size()
        n_layers = world_size * 2
        model = MyModel(hidden_dim=hidden_dim, n_layers=n_layers)

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

        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters(),
                                              optimizer = optim,
        )
        data_loader = random_dataloader(model=model,
                                        total_samples=16,
                                        hidden_dim=hidden_dim,
                                        device=model.device)

        for i, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.save_checkpoint(tmpdir)

        # make sure all sides saved it
        dist.barrier()

        if zero_stage == 3:
            with deepspeed.zero.GatheredParameters(list(
                    model.module.parameters(recurse=True)),
                                                   modifier_rank=None):
                pass  # this forces gathering the model

        #dump_state_dict(model)

        orig_state_dict = {}
        for name, param in model.module.named_parameters():
            orig_state_dict[name] = param.detach().cpu()

        if dist.get_rank() == 0:
            fp32_model = load_state_dict_from_zero_checkpoint(model.module, tmpdir)
            #dump_state_dict(fp32_model)

            fp32_state_dict = fp32_model.state_dict()
            for name in orig_state_dict.keys():
                # float() workaround for torch<1.6
                assert torch.allclose(orig_state_dict[name].float(),
                                      fp32_state_dict[name].float())

    _test_zero_to_fp32()


@pytest.mark.parametrize('zero_stage, allgather_bucket_size', [(2, 1000), (2, 1001)])
def test_incorrect_allgather_bucket_size(tmpdir, zero_stage, allgather_bucket_size):
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

    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 4

    model = SimpleModel(hidden_dim=hidden_dim)

    @distributed_test(world_size=[1])
    def _test_incorrect_allgather_bucket_size(args, model, hidden_dim):
        if allgather_bucket_size % 2 == 0:
            model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        else:
            with pytest.raises(AssertionError) as assertinfo:
                model, _, _, _ = deepspeed.initialize(args=args,
                                                  model=model,
                                                  model_parameters=model.parameters())
            assert "allgather_bucket_size must be a multiple of nccl_start_alignment_factor" in str(
                assertinfo)

    _test_incorrect_allgather_bucket_size(args=args, model=model, hidden_dim=hidden_dim)


@pytest.mark.parametrize('zero_stage, world_size', [(2, 2), (2, 3), (2, 4)])
def test_partition_nccl_alignment(tmpdir, zero_stage, world_size):
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

    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 4

    model = SimpleModel(hidden_dim=hidden_dim)

    @distributed_test(world_size=world_size)
    def _test_partition_nccl_alignment(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())

        # get nccl all-gather send buffers alignment factor
        nccl_start_alignment_factor = model.optimizer.nccl_start_alignment_factor

        for data_parallel_partitions in model.optimizer.parallel_partitioned_fp16_groups:
            for partition_id, partitioned_data in enumerate(data_parallel_partitions):
                # verify that data partition start locations are 4-byte aligned
                assert (partitioned_data.data_ptr() %
                        (2 * nccl_start_alignment_factor) == 0)

    _test_partition_nccl_alignment(args=args, model=model, hidden_dim=hidden_dim)

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
@pytest.mark.parametrize('zero_stage', [2, 3])
def test_zero_to_fp32(tmpdir, zero_stage):

    # TODO:
    # - need to test with multiple param groups

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
        hidden_dim = 2

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

        def dump_state_dict(model):
            if dist.get_rank() != 0:
                return
            for name, param in model.named_parameters():
                print(f"{name} {param}")

        if zero_stage == 3:
            with deepspeed.zero.GatheredParameters(list(
                    model.module.parameters(recurse=True)),
                                                   modifier_rank=None):
                pass  # this forces gathering the model

        #dump_state_dict(model)

        orig_state_dict = {}
        for name, param in model.module.named_parameters():
            orig_state_dict[name] = param.detach().cpu()
        print(orig_state_dict)

        fp32_model = load_state_dict_from_zero_checkpoint(model.module, tmpdir)
        #dump_state_dict(fp32_model)

        fp32_state_dict = fp32_model.state_dict()
        for name in orig_state_dict.keys():
            # float() workaround for torch<1.6
            assert torch.allclose(orig_state_dict[name].float(),
                                  fp32_state_dict[name].float())

    _test_zero_to_fp32()

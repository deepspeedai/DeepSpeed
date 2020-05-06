import torch
import deepspeed
from deepspeed.pt.deepspeed_zero_optimizer import FP16_DeepSpeedZeroOptimizer

from deepspeed.pt.fp16_optimizer import FP16_Optimizer
from deepspeed.pt.fp16_unfused_optimizer import FP16_UnfusedOptimizer

import argparse
import pytest
import json
import os
from common import distributed_test
from simple_model import SimpleModel, random_dataloader, args_from_dict


def compare_deepspeed_states(saved_model, loaded_model):
    # These are compared in more depth in other places
    assert hasattr(loaded_model, 'module')

    assert saved_model.csr_tensor_module_names == loaded_model.csr_tensor_module_names
    assert saved_model.skipped_steps == loaded_model.skipped_steps
    assert saved_model.global_steps == loaded_model.global_steps


def compare_lr_scheduler_states(saved_model, loaded_model):
    if saved_model.lr_scheduler is None:
        assert loaded_model.lr_scheduler is None
        return

    saved = saved_model.lr_scheduler.state_dict()
    loaded = loaded_model.lr_scheduler.state_dict()
    assert sorted(saved.keys()) == sorted(loaded.keys())
    for key in saved.keys():
        if isinstance(saved[key], torch.Tensor):
            assert torch.equal(saved[key], loaded[key])
        else:
            assert saved[key] == loaded[key]


def compare_model_states(saved_model, loaded_model):
    compare_deepspeed_states(saved_model, loaded_model)

    for p0, p1 in zip(saved_model.module.parameters(), loaded_model.module.parameters()):
        assert torch.allclose(p0,p1,atol=1e-07), f"FP16 model state {p0} is not equal to {p1}"

    if isinstance(saved_model.optimizer, FP16_DeepSpeedZeroOptimizer):
        for p0, p1 in zip(saved_model.optimizer.single_partition_of_fp32_groups, loaded_model.optimizer.single_partition_of_fp32_groups):
            assert torch.allclose(p0,p1,atol=1e-07), f"Fp32 model states {p0} is not equal to {p1}"

    elif isinstance(saved_model.optimizer, FP16_Optimizer):
        for p0, p1 in zip(saved_model.optimizer.fp32_groups_flat, loaded_model.optimizer.fp32_groups_flat):
            assert torch.allclose(p0,p1,atol=1e-07), f"FP32 model states {p0} is not equal to {p1}"

    elif isinstance(saved_model.optimizer, FP16_UnfusedOptimizer):
        for params0, params1 in zip(saved_model.optimizer.fp32_groups, loaded_model.optimizer.fp32_groups):
            for p0, p1 in zip(params0, params1):
                assert torch.allclose(p0,p1,atol=1e-07), f"FP32 model states {p0} is not equal to {p1}"

    else:
        assert False, 'Unexpected Optimizer Type'


def compare_optimizer_states(saved_model, loaded_model, hidden_dim):
    compare_model_states(saved_model, loaded_model)

    assert hasattr(loaded_model, 'optimizer')

    for state0, state1 in zip(saved_model.optimizer.optimizer.state.values(),
                              loaded_model.optimizer.optimizer.state.values()):
        for s0, s1 in zip(state0.values(), state1.values()):
            if isinstance(s0, torch.Tensor) and isinstance(s1, torch.Tensor):
                assert torch.equal(s0, s1)
            else:
                assert s0 == s1


def checkpoint_correctness_verification(save_folder,
                                        args,
                                        model,
                                        hidden_dim,
                                        load_optimizer_states=True):

    ds_model, _, _,_ = deepspeed.initialize(args=args,
                                            model=model,
                                            model_parameters=model.parameters())
    data_loader = random_dataloader(model=ds_model,
                                    total_samples=50,
                                    hidden_dim=hidden_dim,
                                    device=ds_model.device)
    for n, batch in enumerate(data_loader):
        loss = ds_model(batch[0], batch[1])
        ds_model.backward(loss)
        ds_model.step()

    trained_model = ds_model

    save_tag = '1'

    trained_model.save_checkpoint(save_folder, save_tag)

    loaded_model, _, _,_ = deepspeed.initialize(args=args,
                                            model=model,
                                            model_parameters=model.parameters())

    loaded_model.load_checkpoint(save_folder,
                                 save_tag,
                                 load_optimizer_states=load_optimizer_states)

    compare_lr_scheduler_states(trained_model, loaded_model)

    if load_optimizer_states:
        compare_optimizer_states(trained_model, loaded_model, hidden_dim)
    else:
        compare_model_states(trained_model, loaded_model)


def test_checkpoint_unfused_optimizer(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Lamb",
            "params": {
                "lr": 0.00015,
                "max_grad_norm": 1.0
            }
        },
        "fp16": {
            "enabled": True
        },
        "scheduler": {
            "type": "OneCycle",
            "params": {
                "cycle_first_step_size": 1000,
                "cycle_first_stair_count": 500,
                "cycle_second_step_size": 1000,
                "cycle_second_stair_count": 500,
                "decay_step_size": 1000,
                "cycle_min_lr": 0.0001,
                "cycle_max_lr": 0.0010,
                "decay_lr_rate": 0.001,
                "cycle_min_mom": 0.85,
                "cycle_max_mom": 0.99,
                "decay_mom_rate": 0.0
            }
        }
    }

    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[2])
    def _test_checkpoint_unfused_optimizer(args,
                                           model,
                                           hidden_dim,
                                           load_optimizer_states):
        checkpoint_correctness_verification(tmpdir,
                                            args,
                                            model,
                                            hidden_dim,
                                            load_optimizer_states=load_optimizer_states)

    _test_checkpoint_unfused_optimizer(args=args,
                                       model=model,
                                       hidden_dim=hidden_dim,
                                       load_optimizer_states=True)
    _test_checkpoint_unfused_optimizer(args=args,
                                       model=model,
                                       hidden_dim=hidden_dim,
                                       load_optimizer_states=False)


def test_checkpoint_fused_optimizer(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015,
                "betas": [0.8,
                          0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },
        "fp16": {
            "enabled": True
        }
    }

    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[2])
    def _test_checkpoint_fused_optimizer(args, model, hidden_dim, load_optimizer_states):
        checkpoint_correctness_verification(tmpdir,
                                            args,
                                            model,
                                            hidden_dim,
                                            load_optimizer_states=load_optimizer_states)

    _test_checkpoint_fused_optimizer(args=args,
                                     model=model,
                                     hidden_dim=hidden_dim,
                                     load_optimizer_states=True)
    _test_checkpoint_fused_optimizer(args=args,
                                     model=model,
                                     hidden_dim=hidden_dim,
                                     load_optimizer_states=False)


def test_checkpoint_zero_optimizer(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015,
                "betas": [0.8,
                          0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": True
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[2])
    def _test_checkpoint_zero_optimizer(args, model, hidden_dim, load_optimizer_states):
        checkpoint_correctness_verification(tmpdir,
                                            args,
                                            model,
                                            hidden_dim,
                                            load_optimizer_states=load_optimizer_states)

    _test_checkpoint_zero_optimizer(args=args,
                                    model=model,
                                    hidden_dim=hidden_dim,
                                    load_optimizer_states=True)
    _test_checkpoint_zero_optimizer(args=args,
                                    model=model,
                                    hidden_dim=hidden_dim,
                                    load_optimizer_states=False)

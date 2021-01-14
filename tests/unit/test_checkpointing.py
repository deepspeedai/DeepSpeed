import torch

import torch.distributed as dist

import deepspeed
from deepspeed.runtime.zero.stage2 import FP16_DeepSpeedZeroOptimizer
from deepspeed.runtime.zero.stage1 import FP16_DeepSpeedZeroOptimizer_Stage1

from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer

from deepspeed.runtime.pipe.topology import *
PipeTopo = PipeDataParallelTopology

from deepspeed.ops.op_builder import FusedLambBuilder, CPUAdamBuilder

import argparse
import pytest
import json
import os
import numbers
from common import distributed_test
from simple_model import *


def compare_deepspeed_states(saved_model, loaded_model):
    # These are compared in more depth in other places
    assert hasattr(loaded_model, 'module')

    assert saved_model.csr_tensor_module_names == loaded_model.csr_tensor_module_names
    assert saved_model.skipped_steps == loaded_model.skipped_steps
    assert saved_model.global_steps == loaded_model.global_steps


def compare_model_states(saved_model, loaded_model, compare_optimizer=True):
    compare_deepspeed_states(saved_model, loaded_model)

    for p0, p1 in zip(saved_model.module.parameters(), loaded_model.module.parameters()):
        assert id(p0) != id(p1), f'Comparing fp16 model state tensor against itself : {id(p0)} <====> {id(p1)}'
        assert torch.allclose(p0, p1, atol=1e-07), f"FP16 model state {p0} is not equal to {p1}"

    if not compare_optimizer:
        return

    if isinstance(saved_model.optimizer, FP16_DeepSpeedZeroOptimizer):
        for p0, p1 in zip(saved_model.optimizer.single_partition_of_fp32_groups, loaded_model.optimizer.single_partition_of_fp32_groups):
            assert id(p0) != id(p1), f'Comparing fp32 model state tensor against itself: {id(p0)} <====> {id(p1)}'
            assert torch.allclose(p0, p1, atol=1e-07), f"Fp32 model states {p0} is not equal to {p1}"

    elif isinstance(saved_model.optimizer, FP16_DeepSpeedZeroOptimizer_Stage1):
        for partition0, partition1 in zip(saved_model.optimizer.local_sub_partitions_of_fp32_groups, loaded_model.optimizer.local_sub_partitions_of_fp32_groups):
            for p0, p1 in zip(partition0, partition1):
                assert id(p0) != id(p1), f'Comparing fp32 model state tensor against itself: {id(p0)} <====> {id(p1)}'
                assert torch.allclose(p0, p1, atol=1e-07), f"Fp32 model states {p0} is not equal to {p1}"

    elif isinstance(saved_model.optimizer, FP16_Optimizer):
        for p0, p1 in zip(saved_model.optimizer.fp32_groups_flat, loaded_model.optimizer.fp32_groups_flat):
            assert id(p0) != id(p1), f'Comparing fp32 model state tensor against itself: {id(p0)} <====> {id(p1)}'
            assert torch.allclose(p0, p1, atol=1e-07), f"FP32 model states {p0} is not equal to {p1}"

    elif isinstance(saved_model.optimizer, FP16_UnfusedOptimizer):
        for params0, params1 in zip(saved_model.optimizer.fp32_groups, loaded_model.optimizer.fp32_groups):
            for p0, p1 in zip(params0, params1):
                assert id(p0) != id(p1), f'Comparing fp32 model state tensor against itself: {id(p0)} <====> {id(p1)}'
                assert torch.allclose(p0, p1, atol=1e-07), f"FP32 model states {p0} is not equal to {p1}"
    elif isinstance(saved_model.optimizer, torch.optim.Optimizer):
        pass
    else:
        assert False, f'Unexpected Optimizer Type: {saved_model.optimizer}'


def compare_optimizer_states(saved_model, loaded_model, hidden_dim, fp16=True):
    saved_optimizer = saved_model.optimizer.optimizer if fp16 else saved_model.optimizer
    loaded_optimizer = loaded_model.optimizer.optimizer if fp16 else loaded_model.optimizer

    for state0, state1 in zip(saved_optimizer.state.values(),
                              loaded_optimizer.state.values()):
        for s0, s1 in zip(state0.values(), state1.values()):
            if isinstance(s0, torch.Tensor) and isinstance(s1, torch.Tensor):
                assert id(s0) != id(s1), f'Comparing optimizer state tensor against itself: {id(s0)} <====> {id(s1)}'
                assert torch.equal(s0, s1)
            else:
                assert s0 == s1


def compare_lr_scheduler_states(saved_model, loaded_model):
    assert hasattr(saved_model, 'lr_scheduler')
    assert hasattr(loaded_model, 'lr_scheduler')

    saved_scheduler = saved_model.lr_scheduler
    loaded_scheduler = loaded_model.lr_scheduler

    assert hasattr(saved_scheduler, 'state_dict')
    assert hasattr(loaded_scheduler, 'state_dict')

    saved_sd = saved_scheduler.state_dict()
    loaded_sd = loaded_scheduler.state_dict()

    print(f"saved_sd = {saved_sd}")
    print(f"loaded_sd = {loaded_sd}")

    assert saved_sd.keys() == loaded_sd.keys()

    for state0, state1 in zip(saved_sd.values(), loaded_sd.values()):
        if isinstance(state0, numbers.Number) and isinstance(state1, numbers.Number):
            assert state0 == state1


def create_deepspeed_model(args, model, base_optimizer):
    if base_optimizer is None:
        ds_model, _, _, _ = deepspeed.initialize(args=args,
                                                 model=model,
                                                 model_parameters=model.parameters())
    else:
        ds_model, _, _, _ = deepspeed.initialize(args=args,
                                                model=model,
                                                optimizer=base_optimizer)

    return ds_model


def checkpoint_correctness_verification(args,
                                        models,
                                        hidden_dim,
                                        tmpdir,
                                        load_optimizer_states=False,
                                        load_lr_scheduler_states=False,
                                        fp16=True,
                                        train_batch=False,
                                        base_optimizers=[None,
                                                         None],
                                        empty_tag=False):
    dtype = torch.half if fp16 else torch.float32
    ds_model = create_deepspeed_model(args=args,
                                      model=models[0],
                                      base_optimizer=base_optimizers[0])

    data_loader = random_dataloader(model=ds_model,
                                    total_samples=50,
                                    hidden_dim=hidden_dim,
                                    device=ds_model.device,
                                    dtype=dtype)

    if train_batch:
        ds_model.set_dataloader(data_loader)
        for n, batch in enumerate(data_loader):
            loss = ds_model.train_batch()
    else:
        for n, batch in enumerate(data_loader):
            loss = ds_model(batch[0], batch[1])
            ds_model.backward(loss)
            ds_model.step()

    trained_model = ds_model

    save_folder = os.path.join(tmpdir, 'saved_checkpoint')
    save_tag = None if empty_tag else '1'

    trained_model.save_checkpoint(save_folder, tag=save_tag)

    loaded_model = create_deepspeed_model(args=args,
                                          model=models[1],
                                          base_optimizer=base_optimizers[1])

    loaded_model.load_checkpoint(save_folder,
                                 tag=save_tag,
                                 load_optimizer_states=load_optimizer_states,
                                 load_lr_scheduler_states=load_lr_scheduler_states)

    compare_model_states(trained_model, loaded_model)

    if load_optimizer_states:
        compare_optimizer_states(trained_model, loaded_model, hidden_dim, fp16)

    if load_lr_scheduler_states:
        compare_lr_scheduler_states(trained_model, loaded_model)


@pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedLambBuilder.NAME],
                    reason="lamb is not compatible")
def test_checkpoint_unfused_optimizer(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Lamb",
            "params": {
                "lr": 0.00015
            }
        },
        "gradient_clipping": 1.0,
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

    models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

    @distributed_test(world_size=[2])
    def _test_checkpoint_unfused_optimizer(args,
                                           models,
                                           hidden_dim,
                                           load_optimizer_states):
        checkpoint_correctness_verification(args,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=load_optimizer_states)

    _test_checkpoint_unfused_optimizer(args=args,
                                       models=models,
                                       hidden_dim=hidden_dim,
                                       load_optimizer_states=True)

    _test_checkpoint_unfused_optimizer(args=args,
                                       models=models,
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

    models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

    @distributed_test(world_size=[2])
    def _test_checkpoint_fused_optimizer(args,
                                         models,
                                         hidden_dim,
                                         load_optimizer_states):
        checkpoint_correctness_verification(args,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=load_optimizer_states)

    _test_checkpoint_fused_optimizer(args=args,
                                     models=models,
                                     hidden_dim=hidden_dim,
                                     load_optimizer_states=True)

    _test_checkpoint_fused_optimizer(args=args,
                                     models=models,
                                     hidden_dim=hidden_dim,
                                     load_optimizer_states=False)


@pytest.mark.parametrize('zero_stage, use_cpu_offload',
                         [
                             (1,
                              False),
                             (2,
                              False),
                             (2,
                              True),
                         ])
def test_checkpoint_zero_optimizer(tmpdir, zero_stage, use_cpu_offload):
    if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
        pytest.skip("cpu-adam is not compatible")

    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": 'Adam',
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
        "zero_optimization": {
            "stage": zero_stage,
            "cpu_offload": use_cpu_offload
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

    @distributed_test(world_size=[2])
    def _test_checkpoint_zero_optimizer(args, models, hidden_dim, load_optimizer_states):
        checkpoint_correctness_verification(args,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=load_optimizer_states)

    _test_checkpoint_zero_optimizer(args=args,
                                    models=models,
                                    hidden_dim=hidden_dim,
                                    load_optimizer_states=True)


@pytest.mark.parametrize('zero_stage, use_cpu_offload',
                         [
                             (1,
                              False),
                             (2,
                              False),
                             (2,
                              True),
                         ])
def test_checkpoint_zero_no_optimizer(tmpdir, zero_stage, use_cpu_offload):
    if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
        pytest.skip("cpu-adam is not compatible")

    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": 'Adam',
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
        "zero_optimization": {
            "stage": zero_stage,
            "cpu_offload": use_cpu_offload
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

    @distributed_test(world_size=[2])
    def _test_checkpoint_zero_no_optimizer(args,
                                           models,
                                           hidden_dim,
                                           load_optimizer_states):
        checkpoint_correctness_verification(args,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=load_optimizer_states)

    _test_checkpoint_zero_no_optimizer(args=args,
                                       models=models,
                                       hidden_dim=hidden_dim,
                                       load_optimizer_states=False)


@pytest.mark.parametrize('zero_stage, use_cpu_offload',
                         [
                             (0,
                              False),
                             (1,
                              False),
                             (2,
                              False),
                             (2,
                              True),
                         ])
def test_checkpoint_lr_scheduler(tmpdir, zero_stage, use_cpu_offload):
    if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
        pytest.skip("cpu-adam is not compatible")

    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": 'Adam',
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
        "zero_optimization": {
            "stage": zero_stage,
            "cpu_offload": use_cpu_offload
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000
            }
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

    @distributed_test(world_size=[2])
    def _test_checkpoint_lr_scheduler(args,
                                      models,
                                      hidden_dim,
                                      load_optimizer_states,
                                      load_lr_scheduler_states):
        checkpoint_correctness_verification(
            args,
            models=models,
            hidden_dim=hidden_dim,
            tmpdir=tmpdir,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states)

    _test_checkpoint_lr_scheduler(args=args,
                                  models=models,
                                  hidden_dim=hidden_dim,
                                  load_optimizer_states=False,
                                  load_lr_scheduler_states=True)


@pytest.mark.parametrize('zero_stage, use_cpu_offload',
                         [
                             (0,
                              False),
                             (1,
                              False),
                             (2,
                              False),
                             (2,
                              True),
                         ])
def test_checkpoint_no_lr_scheduler(tmpdir, zero_stage, use_cpu_offload):
    if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
        pytest.skip("cpu-adam is not compatible")

    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": 'Adam',
            "params": {
                "lr": 1e-5
            }
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": zero_stage,
            "cpu_offload": use_cpu_offload
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000
            }
        },
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

    @distributed_test(world_size=[2])
    def _test_checkpoint_no_lr_scheduler(args,
                                         models,
                                         hidden_dim,
                                         load_optimizer_states,
                                         load_lr_scheduler_states):
        checkpoint_correctness_verification(
            args,
            models=models,
            hidden_dim=hidden_dim,
            tmpdir=tmpdir,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states)

    _test_checkpoint_no_lr_scheduler(args=args,
                                     models=models,
                                     hidden_dim=hidden_dim,
                                     load_optimizer_states=False,
                                     load_lr_scheduler_states=False)


def test_checkpoint_fp32_optimizer(tmpdir):
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
            "enabled": False
        }
    }

    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

    @distributed_test(world_size=[2])
    def _test_checkpoint_fp32_optimizer(args, models, hidden_dim):
        checkpoint_correctness_verification(args,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            fp16=False)

    _test_checkpoint_fp32_optimizer(args=args, models=models, hidden_dim=hidden_dim)


@pytest.mark.parametrize("zero_stage", [0, 1])
def test_checkpoint_pipe_engine(zero_stage, tmpdir, stages=2):
    config_dict = {
        "train_batch_size": 2,
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-5
            }
        },
        "zero_optimization": {
            "stage": zero_stage
        },
        "fp16": {
            "enabled": zero_stage > 0
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

    @distributed_test(world_size=4)
    def _test(save_folder, num_stages):
        args = args_from_dict(tmpdir, config_dict)
        models = [LinearStackPipe(num_stages=num_stages) for _ in range(2)]
        checkpoint_correctness_verification(args=args,
                                            models=models,
                                            hidden_dim=models[0].hidden_dim,
                                            tmpdir=save_folder,
                                            fp16=config_dict['fp16']['enabled'],
                                            load_optimizer_states=True,
                                            load_lr_scheduler_states=True,
                                            train_batch=True)

    _test(tmpdir, num_stages=stages)


@pytest.mark.parametrize("base_topo,test_topo",
                         [
                             (PipeTopo(num_pp=1,
                                       num_dp=4),
                              PipeTopo(num_pp=4,
                                       num_dp=1)),
                             (PipeTopo(num_pp=2,
                                       num_dp=2),
                              PipeTopo(num_pp=2,
                                       num_dp=2)),
                             (PipeTopo(num_pp=4,
                                       num_dp=1),
                              PipeTopo(num_pp=2,
                                       num_dp=2)),
                         ])
def test_checkpoint_pipe_module(base_topo, test_topo, tmpdir):
    @distributed_test(world_size=4)
    def _test(base_topo, test_topo, save_folder):
        base_model = LinearStackPipe(topology=base_topo)
        base_model.save_state_dict(save_folder)

        dist.barrier()

        test_model = LinearStackPipe(topology=test_topo)
        test_model.load_state_dir(save_folder)

        # Base and test can have different lengths, so make sure we map from the
        # smaller to larger model
        if len(base_model.forward_funcs) < len(test_model.forward_funcs):
            A = base_model
            B = test_model
        else:
            A = test_model
            B = base_model

        # Compare layers individually since partitions are different
        for idx, A_layer in enumerate(A.forward_funcs):
            if not hasattr(A_layer, 'parameters'):
                # Skip functionals, etc.
                continue

            # Find the corresponding layer in B
            global_idx = idx + A._local_start
            B_local_idx = global_idx - B._local_start
            B_layer = B.forward_funcs[B_local_idx]

            # Compare layer parameters
            for p0, p1 in zip(A_layer.parameters(), B_layer.parameters()):
                assert torch.allclose(p0, p1, atol=1e-07), f"Model state {p0} is not equal to {p1}"

    _test(base_topo, test_topo, save_folder=tmpdir)


@pytest.mark.parametrize('zero_stage', [1, 2])
def test_checkpoint_zero_hybrid_optimizer_state(tmpdir, zero_stage):
    config_dict = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": zero_stage
        },
        "zero_allow_untested_optimizer": True,
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        }
    }

    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10
    models = [SimpleModel(hidden_dim=hidden_dim) for _ in range(2)]
    optimizers = [HybridStateOptimizer(model.parameters()) for model in models]

    @distributed_test(world_size=[2])
    def _test_checkpoint_zero_hybrid_optimizer_state(args,
                                                     models,
                                                     optimizers,
                                                     hidden_dim):
        checkpoint_correctness_verification(args,
                                            models=models,
                                            base_optimizers=optimizers,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=True)

    _test_checkpoint_zero_hybrid_optimizer_state(args=args,
                                                 models=models,
                                                 optimizers=optimizers,
                                                 hidden_dim=hidden_dim)


def test_checkpoint_latest(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            }
        }
    }
    hidden_dim = 10
    args = args_from_dict(tmpdir, config_dict)
    models = [SimpleModel(hidden_dim=hidden_dim) for _ in range(2)]

    @distributed_test(world_size=[1])
    def _helper(args, models):
        checkpoint_correctness_verification(args,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=True,
                                            load_lr_scheduler_states=False,
                                            fp16=False,
                                            empty_tag=True)

    _helper(args, models)


def test_checkpoint_missing_latest(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            }
        }
    }
    hidden_dim = 10
    args = args_from_dict(tmpdir, config_dict)

    model = SimpleModel(hidden_dim, rank=args.local_rank)

    @distributed_test(world_size=[1])
    def _helper(args, model, hidden_dim):
        model, _, _,_ = deepspeed.initialize(args=args,
                                             model=model,
                                             model_parameters=model.parameters())
        # should be no-op, since latest doesn't exist
        model.load_checkpoint(tmpdir)

    _helper(args=args, model=model, hidden_dim=hidden_dim)


@pytest.mark.parametrize('valid_mode', ["FAIL", "WARN", "IGNORE"])
def test_checkpoint_unique_tag(tmpdir, valid_mode):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            }
        },
        "checkpoint": {
            "tag_validation": valid_mode
        }
    }
    hidden_dim = 10
    args = args_from_dict(tmpdir, config_dict)

    model = SimpleModel(hidden_dim, rank=args.local_rank)

    @distributed_test(world_size=[2])
    def _helper(args, model, hidden_dim):
        model, _, _,_ = deepspeed.initialize(args=args,
                                             model=model,
                                             model_parameters=model.parameters())
        if valid_mode == "FAIL":
            with pytest.raises(AssertionError):
                model.save_checkpoint(save_dir=tmpdir,
                                      tag=f"tag-{torch.distributed.get_rank()}")
        else:
            model.save_checkpoint(save_dir=tmpdir,
                                  tag=f"tag-{torch.distributed.get_rank()}")

    _helper(args=args, model=model, hidden_dim=hidden_dim)


def test_checkpoint_unknown_tag_validation(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            }
        },
        "checkpoint": {
            "tag_validation": "foo"
        }
    }
    hidden_dim = 10
    args = args_from_dict(tmpdir, config_dict)

    model = SimpleModel(hidden_dim, rank=args.local_rank)

    @distributed_test(world_size=[1])
    def _helper(args, model, hidden_dim):
        with pytest.raises(deepspeed.DeepSpeedConfigError):
            model, _, _,_ = deepspeed.initialize(args=args,
                                                 model=model,
                                                 model_parameters=model.parameters())

    _helper(args=args, model=model, hidden_dim=hidden_dim)

import deepspeed
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

from deepspeed.runtime.pipe.topology import *

PipeTopo = PipeDataParallelTopology

from deepspeed.ops.op_builder import FusedLambBuilder, CPUAdamBuilder

from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from .util import required_minimum_torch_version, required_torch_version

import itertools
import pytest
import numbers
from .common import distributed_test
from .simple_model import *


def compare_deepspeed_states(saved_model, loaded_model):
    # These are compared in more depth in other places
    assert hasattr(loaded_model, 'module')

    assert saved_model.sparse_tensor_module_names == loaded_model.sparse_tensor_module_names
    assert saved_model.skipped_steps == loaded_model.skipped_steps
    assert saved_model.global_steps == loaded_model.global_steps


def compare_model_states(saved_model,
                         loaded_model,
                         compare_optimizer=True,
                         load_module_only=False):
    if not load_module_only:
        compare_deepspeed_states(saved_model, loaded_model)

    for p0, p1 in zip(saved_model.module.named_parameters(), loaded_model.module.named_parameters()):
        np0, p0 = p0
        np1, p1 = p1
        if 'deepspeed_moe.gate.wg' in np0:
            # these params are converted to float at runtime, cast to half for comparison
            p1 = p1.half()
            p0 = p0.half()
        assert id(p0) != id(p1), f'Comparing fp16 model state tensor against itself : {id(p0)} <====> {id(p1)}'
        try:
            assert torch.allclose(p0, p1, atol=1e-07), f"FP16 model state {p0} is not equal to {p1}, names:{np0}, {np1}"
        except RuntimeError as err:
            print(f"FP16 model state {p0} is not equal to {p1}, names:{np0}, {np1}")
            raise err

    if not compare_optimizer:
        return

    if DeepSpeedZeroOptimizer_Stage3 is not None and isinstance(
            saved_model.optimizer,
            DeepSpeedZeroOptimizer_Stage3):
        for p0, p1 in zip(saved_model.optimizer.fp32_partitioned_groups_flat, loaded_model.optimizer.fp32_partitioned_groups_flat):
            assert torch.allclose(p0, p1, atol=1e-07), f"Fp32 model states {p0} is not equal to {p1}"

    elif isinstance(saved_model.optimizer, DeepSpeedZeroOptimizer):
        for p0, p1 in zip(saved_model.optimizer.single_partition_of_fp32_groups, loaded_model.optimizer.single_partition_of_fp32_groups):
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


def _compare_state_dicts(state0, state1, expected_mismatch_keys=[]):
    for (k0, s0), (k1, s1) in zip(state0.items(), state1.items()):
        assert k0 == k1, f'failure due to key mismatch {k0} != {k1}'
        if k0 in expected_mismatch_keys:
            continue
        if isinstance(s0, torch.Tensor) and isinstance(s1, torch.Tensor):
            assert id(s0) != id(s1), f'Comparing optimizer state tensor against itself: {id(s0)} <====> {id(s1)}'
            assert torch.equal(s0.to('cpu'), s1.to('cpu'))
        else:
            assert s0 == s1, f'failures with keys = {k0}, {k1}, values = {type(s0[0])} and {type(s1[0])}'


def compare_optimizer_states(saved_model, loaded_model, hidden_dim, fp16=True):
    saved_optimizer = saved_model.optimizer.optimizer if fp16 else saved_model.optimizer
    loaded_optimizer = loaded_model.optimizer.optimizer if fp16 else loaded_model.optimizer

    for state0, state1 in zip(saved_optimizer.state.values(),
                              loaded_optimizer.state.values()):
        _compare_state_dicts(state0, state1)


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
                                        empty_tag=False,
                                        seq_dataloader=False,
                                        load_module_only=False):
    dtype = torch.half if fp16 else torch.float32
    ds_model = create_deepspeed_model(args=args,
                                      model=models[0],
                                      base_optimizer=base_optimizers[0])

    if seq_dataloader:
        data_loader = sequence_dataloader(model=ds_model,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=ds_model.device,
                                          dtype=dtype)
    else:
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

    dist.barrier()

    loaded_model = create_deepspeed_model(args=args,
                                          model=models[1],
                                          base_optimizer=base_optimizers[1])
    assert list(trained_model.parameters())[0].dtype == list(
        loaded_model.parameters())[0].dtype

    loaded_model.load_checkpoint(save_folder,
                                 tag=save_tag,
                                 load_optimizer_states=load_optimizer_states,
                                 load_lr_scheduler_states=load_lr_scheduler_states,
                                 load_module_only=load_module_only)

    compare_model_states(trained_model,
                         loaded_model,
                         compare_optimizer=load_optimizer_states,
                         load_module_only=load_module_only)

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


@pytest.mark.parametrize('zero_stage, use_cpu_offload, adam_optimizer',
                         [(1,
                           False,
                           'Adam'),
                          (2,
                           False,
                           'Adam'),
                          (2,
                           True,
                           'deepspeed_adam'),
                          (3,
                           False,
                           'Adam'),
                          (3,
                           True,
                           'deepspeed_adam')])
def test_checkpoint_zero_optimizer(tmpdir, zero_stage, use_cpu_offload, adam_optimizer):
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
            "enabled": True,
            "initial_scale_power": 8
        },
        "wall_clock_breakdown": True,
        "zero_optimization": {
            "stage": zero_stage,
            "cpu_offload": use_cpu_offload
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    @distributed_test(world_size=[2])
    def _test_checkpoint_zero_optimizer(args,
                                        zero_stage,
                                        hidden_dim,
                                        load_optimizer_states):
        if zero_stage == 3:
            with deepspeed.zero.Init():
                models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]
        else:
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(args,
                                            models,
                                            hidden_dim,
                                            tmpdir,
                                            load_optimizer_states=load_optimizer_states)

    _test_checkpoint_zero_optimizer(args=args,
                                    zero_stage=zero_stage,
                                    hidden_dim=hidden_dim,
                                    load_optimizer_states=True)


@pytest.mark.parametrize('zero_stage, use_cpu_offload, adam_optimizer',
                         [(1,
                           False,
                           "Adam"),
                          (2,
                           False,
                           "Adam"),
                          (2,
                           True,
                           'deepspeed_adam'),
                          (3,
                           False,
                           'Adam'),
                          (3,
                           True,
                           'deepspeed_adam')])
def test_checkpoint_zero_no_optimizer(tmpdir,
                                      zero_stage,
                                      use_cpu_offload,
                                      adam_optimizer):
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

    @distributed_test(world_size=[1])
    def _test_checkpoint_zero_no_optimizer(args,
                                           zero_stage,
                                           hidden_dim,
                                           load_optimizer_states):
        if zero_stage == 3:
            global DeepSpeedZeroOptimizer_Stage3
            from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
            with deepspeed.zero.Init():
                models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]
        else:
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(args,
                                            models,
                                            hidden_dim,
                                            tmpdir,
                                            load_optimizer_states=load_optimizer_states)

    _test_checkpoint_zero_no_optimizer(args=args,
                                       zero_stage=zero_stage,
                                       hidden_dim=hidden_dim,
                                       load_optimizer_states=False)


@pytest.mark.parametrize('zero_stage, use_cpu_offload, adam_optimizer',
                         [(0,
                           False,
                           'Adam'),
                          (1,
                           False,
                           'Adam'),
                          (2,
                           False,
                           'Adam'),
                          (2,
                           True,
                           'deepspeed_adam'),
                          (3,
                           False,
                           'Adam'),
                          (3,
                           True,
                           'deepspeed_adam')])
def test_checkpoint_lr_scheduler(tmpdir, zero_stage, use_cpu_offload, adam_optimizer):
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

    @distributed_test(world_size=[2])
    def _test_checkpoint_lr_scheduler(args,
                                      zero_stage,
                                      hidden_dim,
                                      load_optimizer_states,
                                      load_lr_scheduler_states):
        if zero_stage == 3:
            global DeepSpeedZeroOptimizer_Stage3
            from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
            with deepspeed.zero.Init():
                models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]
        else:
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(
            args,
            models,
            hidden_dim,
            tmpdir,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states)

    _test_checkpoint_lr_scheduler(args=args,
                                  zero_stage=zero_stage,
                                  hidden_dim=hidden_dim,
                                  load_optimizer_states=False,
                                  load_lr_scheduler_states=True)


@pytest.mark.parametrize('zero_stage, use_cpu_offload, adam_optimizer',
                         [(0,
                           False,
                           'Adam'),
                          (1,
                           False,
                           'Adam'),
                          (2,
                           False,
                           'Adam'),
                          (2,
                           True,
                           'deepspeed_adam'),
                          (3,
                           False,
                           'Adam'),
                          (3,
                           True,
                           'deepspeed_adam')])
def test_checkpoint_no_lr_scheduler(tmpdir, zero_stage, use_cpu_offload, adam_optimizer):
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

    @distributed_test(world_size=[2])
    def _test_checkpoint_no_lr_scheduler(args,
                                         zero_stage,
                                         hidden_dim,
                                         load_optimizer_states,
                                         load_lr_scheduler_states):
        if zero_stage == 3:
            with deepspeed.zero.Init():
                models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]
        else:
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(
            args,
            models,
            hidden_dim,
            tmpdir,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states)

    _test_checkpoint_no_lr_scheduler(args=args,
                                     zero_stage=zero_stage,
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


@pytest.mark.parametrize(
    "base_topo,test_topo",
    [
        #(PipeTopo(num_pp=1,
        #          num_dp=4),
        # PipeTopo(num_pp=4,
        #          num_dp=1)),
        #(PipeTopo(num_pp=2,
        #          num_dp=2),
        # PipeTopo(num_pp=2,
        #          num_dp=2)),
        #(PipeTopo(num_pp=4,
        #          num_dp=1),
        # PipeTopo(num_pp=2,
        #          num_dp=2)),
    ])
def test_checkpoint_pipe_module(base_topo, test_topo, tmpdir):
    @distributed_test(world_size=4)
    def _test(base_topo, test_topo, save_folder):
        checkpoint_engine = TorchCheckpointEngine()
        base_model = LinearStackPipe(topology=base_topo)
        base_model.save_state_dict(save_folder, checkpoint_engine=checkpoint_engine)

        dist.barrier()

        test_model = LinearStackPipe(topology=test_topo)
        test_model.load_state_dir(save_folder, checkpoint_engine=checkpoint_engine)

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

    model = SimpleModel(hidden_dim)

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

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[2])
    def _helper(args, model, hidden_dim):
        model, _, _,_ = deepspeed.initialize(args=args,
                                             model=model,
                                             model_parameters=model.parameters())
        if valid_mode == "FAIL":
            with pytest.raises(AssertionError):
                model.save_checkpoint(save_dir=tmpdir, tag=f"tag-{dist.get_rank()}")
        else:
            model.save_checkpoint(save_dir=tmpdir, tag=f"tag-{dist.get_rank()}")

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

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1])
    def _helper(args, model, hidden_dim):
        with pytest.raises(deepspeed.DeepSpeedConfigError):
            model, _, _,_ = deepspeed.initialize(args=args,
                                                 model=model,
                                                 model_parameters=model.parameters())

    _helper(args=args, model=model, hidden_dim=hidden_dim)


@pytest.mark.parametrize("ep_size", [4])
def test_checkpoint_moe(tmpdir, ep_size):
    if not required_torch_version():
        pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

    config_dict = {
        "train_batch_size": 8,
        "steps_per_print": 1,
        "fp16": {
            "enabled": True
        }
    }
    hidden_dim = 16
    args = args_from_dict(tmpdir, config_dict)

    @distributed_test(world_size=[4])
    def _helper(args):
        models = [
            SimpleMoEModel(hidden_dim=hidden_dim,
                           num_experts=ep_size,
                           ep_size=ep_size) for _ in range(2)
        ]
        optimizers = [torch.optim.AdamW(params=model.parameters()) for model in models]
        checkpoint_correctness_verification(args,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=True,
                                            load_lr_scheduler_states=False,
                                            fp16=config_dict["fp16"]["enabled"],
                                            empty_tag=True,
                                            base_optimizers=optimizers,
                                            seq_dataloader=True)

    _helper(args)


@pytest.mark.parametrize("ep_size, load_optim_states",
                         [(4,
                           True),
                          (4,
                           False),
                          (2,
                           True),
                          (2,
                           False)])
def test_checkpoint_moe_and_zero(tmpdir, ep_size, load_optim_states):
    if not required_torch_version():
        pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

    config_dict = {
        "train_batch_size": 8,
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
            "enabled": True,
            "initial_scale_power": 8
        },
        "zero_optimization": {
            "stage": 2,
        }
    }
    hidden_dim = 16
    args = args_from_dict(tmpdir, config_dict)

    def create_param_groups(model):
        # param group must have a random unique name (for now)
        # TODO: clean-up this requirement, the unique name should not be required here
        return {'params': [p for p in model.parameters()], 'name': 'random-unique-name'}

    @distributed_test(world_size=[4])
    def _helper(args):
        models = [
            SimpleMoEModel(hidden_dim=hidden_dim,
                           num_experts=ep_size,
                           ep_size=ep_size) for _ in range(2)
        ]
        params = [
            split_params_into_different_moe_groups_for_optimizer(
                create_param_groups(model)) for model in models
        ]
        optimizers = [torch.optim.AdamW(params=param) for param in params]
        checkpoint_correctness_verification(args,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=load_optim_states,
                                            load_lr_scheduler_states=False,
                                            fp16=config_dict["fp16"]["enabled"],
                                            empty_tag=True,
                                            base_optimizers=optimizers,
                                            seq_dataloader=True)

    _helper(args)


@pytest.mark.parametrize('zero_stage', [0, 1, 2, 3])
def test_checkpoint_load_module_only(tmpdir, zero_stage):
    config_dict = {
        "train_batch_size": 2,
        "optimizer": {
            "type": 'Adam'
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        },
        "zero_optimization": {
            "stage": zero_stage,
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    @distributed_test(world_size=[2])
    def _go(args, zero_stage, hidden_dim):
        if zero_stage == 3:
            with deepspeed.zero.Init():
                models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]
        else:
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(args,
                                            models,
                                            hidden_dim,
                                            tmpdir,
                                            load_module_only=True)

    _go(args, zero_stage, hidden_dim)


@pytest.mark.parametrize(["to_save_model_has_embedding",
                          "to_save_model_sparse"],
                         [
                             [False,
                              False],
                             [True,
                              False],
                             [True,
                              True],
                         ])
@pytest.mark.parametrize(["destination_has_embedding",
                          "destination_sparse"],
                         [
                             [False,
                              False],
                             [True,
                              False],
                             [True,
                              True],
                         ])
def test_non_strict_load_sparse(tmpdir,
                                to_save_model_has_embedding,
                                to_save_model_sparse,
                                destination_has_embedding,
                                destination_sparse):
    config_dict = {"train_batch_size": 2}

    class ModelNoEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 1)

        def forward(self, x):
            return self.linear(x)

    class ModelEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(10, 3)
            self.linear = torch.nn.Linear(3, 1)

        def forward(self, x, offsets):
            return self.linear(self.emb(x, offsets))

    @distributed_test(world_size=[2])
    def _test(model_to_save, model_destination):
        engine_to_save, _, _, _ = deepspeed.initialize(
            model=model_to_save, config={"train_batch_size": 2, "sparse_gradients": to_save_model_sparse}
        )
        engine_destination, _, _, _ = deepspeed.initialize(
            model=model_destination, config={"train_batch_size": 2, "sparse_gradients": destination_sparse}
        )

        save_folder = os.path.join(tmpdir, 'saved_checkpoint')
        save_tag = '1'

        engine_to_save.save_checkpoint(save_folder, tag=save_tag)

        is_sparse_destination = isinstance(model_destination,
                                           ModelEmbedding) and destination_sparse
        if isinstance(model_destination,
                      ModelEmbedding) and model_destination.emb.sparse:
            assert "emb.weight" in engine_destination.sparse_tensor_module_names
        engine_destination.load_checkpoint(save_folder,
                                           tag=save_tag,
                                           load_module_strict=False,
                                           load_optimizer_states=False,
                                           load_lr_scheduler_states=False,
                                           load_module_only=False)
        if isinstance(model_destination,
                      ModelEmbedding) and isinstance(model_to_save,
                                                     ModelEmbedding):
            assert engine_destination.sparse_tensor_module_names == engine_to_save.sparse_tensor_module_names
        elif isinstance(model_destination, ModelEmbedding):
            assert not is_sparse_destination or "emb.weight" in engine_destination.sparse_tensor_module_names
        else:
            assert len(engine_destination.sparse_tensor_module_names) == 0

    if to_save_model_has_embedding:
        model_to_save = ModelEmbedding()
    else:
        model_to_save = ModelNoEmbedding()
    if destination_has_embedding:
        model_destination = ModelEmbedding()
    else:
        model_destination = ModelNoEmbedding()
    _test(model_to_save, model_destination)


@pytest.mark.parametrize(["elastic_save",
                          "elastic_load",
                          "load_optim"],
                         itertools.product(*[[True,
                                              False],
                                             [True,
                                              False],
                                             [True,
                                              False]]))
def test_checkpoint_zero_elastic(tmpdir, elastic_save, elastic_load, load_optim):
    ds_config = {
        "train_batch_size": 2,
        "optimizer": {
            "type": 'Adam'
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        },
        "zero_optimization": {
            "stage": 2,
            "elastic_checkpoint": elastic_save
        }
    }
    hidden_dim = 10

    @distributed_test(world_size=[2])
    def _go():
        # torch 1.2.* stores raw tensor id numbers in checkpoint state which leads to
        # false positive mismatches in checkpoint state comparisons.
        # Newer torch versions store tensor ids as 0, 1, 2, ...
        expected_mismatch_keys = [] if required_minimum_torch_version(1,
                                                                      4) else ['params']
        models = [SimpleModel(hidden_dim) for _ in range(2)]
        model, _, _, _ = deepspeed.initialize(config=ds_config,
                                              model=models[0],
                                              model_parameters=models[0].parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
        if load_optim:
            torch.save(model.optimizer.optimizer.state_dict(),
                       os.path.join(tmpdir,
                                    'opt-state-dict'))
        model.save_checkpoint(tmpdir)

        ds_config["zero_optimization"]["elastic_checkpoint"] = elastic_load
        model, _, _, _ = deepspeed.initialize(config=ds_config,
                                              model=models[1],
                                              model_parameters=models[1].parameters())
        model.load_checkpoint(tmpdir, load_optimizer_states=load_optim)

        if load_optim:
            saved_sd = torch.load(os.path.join(tmpdir, 'opt-state-dict'))
            curr_sd = model.optimizer.optimizer.state_dict()
            for curr_param_group, saved_param_group in zip(curr_sd['param_groups'], saved_sd['param_groups']):
                _compare_state_dicts(curr_param_group,
                                     saved_param_group,
                                     expected_mismatch_keys)

        data_loader = random_dataloader(model=model,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _go()


@pytest.mark.parametrize(["elastic_save",
                          "elastic_load",
                          "load_optim"],
                         itertools.product(*[[True,
                                              False],
                                             [True,
                                              False],
                                             [True,
                                              False]]))
def test_checkpoint_zero_elastic_dp_change(tmpdir,
                                           elastic_save,
                                           elastic_load,
                                           load_optim):
    ds_config = {
        "train_batch_size": 4,
        "optimizer": {
            "type": 'Adam'
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        },
        "zero_optimization": {
            "stage": 2,
            "elastic_checkpoint": elastic_save
        }
    }
    hidden_dim = 10
    models = [SimpleModel(hidden_dim) for _ in range(2)]

    @distributed_test(world_size=[4])
    def _go2(models):
        model, _, _, _ = deepspeed.initialize(config=ds_config,
                                              model=models[0],
                                              model_parameters=models[0].parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        if load_optim:
            torch.save(model.optimizer.optimizer.state_dict(),
                       os.path.join(tmpdir,
                                    'opt-state-dict'))
        model.save_checkpoint(tmpdir)

    _go2(models)

    @distributed_test(world_size=[2])
    def _go1(models):
        ds_config["zero_optimization"]["elastic_checkpoint"] = elastic_load
        model, _, _, _ = deepspeed.initialize(config=ds_config,
                                                  model=models[1],
                                                  model_parameters=models[1].parameters())
        if load_optim:
            with pytest.raises(deepspeed.runtime.zero.utils.ZeRORuntimeException):
                model.load_checkpoint(tmpdir, load_optimizer_states=load_optim)
        else:
            model.load_checkpoint(tmpdir, load_optimizer_states=load_optim)

    _go1(models)


@pytest.mark.parametrize('zero_stage', [0, 1, 2, 3])
def test_immediate_save_load(tmpdir, zero_stage):
    config_dict = {
        "train_batch_size": 4,
        "optimizer": {
            "type": 'Adam'
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        },
        "zero_optimization": {
            "stage": zero_stage,
        }
    }
    hidden_dim = 10
    model = SimpleModel(hidden_dim)
    args = args_from_dict(tmpdir, config_dict)

    @distributed_test(world_size=[1])
    def _test_immediate_save_load(args, model, tmpdir):

        ds_model = create_deepspeed_model(args=args, model=model, base_optimizer=None)
        ds_model.save_checkpoint(tmpdir)
        ds_model.load_checkpoint(tmpdir,
                                 load_optimizer_states=False,
                                 load_lr_scheduler_states=False,
                                 load_module_only=False)

    _test_immediate_save_load(args, model, tmpdir)


@pytest.mark.parametrize('zero_stage', [0, 1, 2, 3])
def test_load_immediate_save(tmpdir, zero_stage):
    config_dict = {
        "train_batch_size": 4,
        "optimizer": {
            "type": 'Adam'
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        },
        "zero_optimization": {
            "stage": zero_stage,
        }
    }
    hidden_dim = 10
    model = SimpleModel(hidden_dim)
    args = args_from_dict(tmpdir, config_dict)

    @distributed_test(world_size=[1])
    def _test_load_immediate_save(args, model, tmpdir):

        # 1. pretrain a model and save it
        dtype = torch.half
        ds_model = create_deepspeed_model(args=args, model=model, base_optimizer=None)
        data_loader = random_dataloader(model=ds_model,
                                        total_samples=1,
                                        hidden_dim=hidden_dim,
                                        device=ds_model.device,
                                        dtype=dtype)
        for n, batch in enumerate(data_loader):
            loss = ds_model(batch[0], batch[1])
            ds_model.backward(loss)
            ds_model.step()
        ds_model.save_checkpoint(tmpdir)

        # 2. load and immediately save a model with a fresh ds engine
        ds_model = create_deepspeed_model(args=args, model=model, base_optimizer=None)
        ds_model.load_checkpoint(tmpdir,
                                 load_optimizer_states=False,
                                 load_lr_scheduler_states=False,
                                 load_module_only=False)
        ds_model.save_checkpoint(tmpdir)

    _test_load_immediate_save(args, model, tmpdir)


@pytest.mark.parametrize('zero_stage', [0, 1, 2, 3])
def test_save_before_accum_grad_is_done(tmpdir, zero_stage):
    config_dict = {
        "optimizer": {
            "type": 'Adam'
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        },
        "zero_optimization": {
            "stage": zero_stage,
            "stage3_gather_fp16_weights_on_model_save": True,
        },
        "gradient_accumulation_steps": 2,
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": 2,
    }
    hidden_dim = 10
    model = SimpleModel(hidden_dim)
    args = args_from_dict(tmpdir, config_dict)

    @distributed_test(world_size=[1])
    def _test_save_before_accum_grad_is_done(args, model, tmpdir):

        # This test reproduces a bug where one tries to retrieve a 16bit model before grad_accum
        # cycle was completed.
        # So we config grad_accum=2 and step only once and save_16bit_model
        ds_model = create_deepspeed_model(args=args, model=model, base_optimizer=None)

        data_loader = random_dataloader(model=ds_model,
                                        total_samples=2,
                                        hidden_dim=hidden_dim,
                                        device=ds_model.device,
                                        dtype=torch.half)

        batch = next(iter(data_loader))
        loss = ds_model(batch[0], batch[1])
        ds_model.backward(loss)
        ds_model.step()

        # we stepped only once, and now save 16bit model before gradient_accumulation_steps=2 is complete
        ds_model.save_16bit_model(tmpdir, "model.pt")

        # let's test just as well that we can save the checkpoint too
        ds_model.save_checkpoint(tmpdir)

    _test_save_before_accum_grad_is_done(args, model, tmpdir)

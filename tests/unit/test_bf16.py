import math
import torch
import deepspeed
import pytest
from deepspeed.ops.adam import FusedAdam
from common import distributed_test
from deepspeed.ops.op_builder import CPUAdamBuilder
from simple_model import SimpleModel, SimpleOptimizer, random_dataloader, args_from_dict
from util import bf16_required_version_check


@pytest.mark.parametrize('zero_stage, use_cpu_offload', [(2, False)])
def test_adam_bf16_zero_onecycle_compatibility(tmpdir, zero_stage, use_cpu_offload):
    if not bf16_required_version_check():
        pytest.skip(
            " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
        )

    if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
        pytest.skip("cpu-adam is not compatible")

    config_dict = {
        "train_batch_size": 1,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            }
        },
        "scheduler": {
            "type": "OneCycle",
            "params": {
                "cycle_first_step_size": 16000,
                "cycle_first_stair_count": 8000,
                "decay_step_size": 16000,
                "cycle_min_lr": 1e-06,
                "cycle_max_lr": 3e-05,
                "decay_lr_rate": 1e-07,
                "cycle_min_mom": 0.85,
                "cycle_max_mom": 0.99,
                "decay_mom_rate": 0.0
            }
        },
        "fp16": {
            "enabled": False
        },
        "bfloat16": {
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
    def _test_adam_bf16_zero_onecycle_compatibility(args, zero_stage, hidden_dim):
        model = SimpleModel(hidden_dim)

        model, _, _, _ = deepspeed.initialize(args=args,
                                             model=model,
                                             model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_adam_bf16_zero_onecycle_compatibility(args=args,
                                                zero_stage=zero_stage,
                                                hidden_dim=hidden_dim)


@pytest.mark.parametrize('zero_stage, use_cpu_offload', [(2, False)])
def test_zero_allow_untested_optimizer(tmpdir, zero_stage, use_cpu_offload):
    if not bf16_required_version_check():
        pytest.skip(
            " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
        )

    if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
        pytest.skip("cpu-adam is not compatible")

    config_dict = {
        "train_batch_size": 4,
        "steps_per_print": 1,
        "fp16": {
            "enabled": False,
        },
        "bfloat16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": zero_stage,
            "cpu_offload": use_cpu_offload
        },
        "zero_allow_untested_optimizer": False
    }
    args = args_from_dict(tmpdir, config_dict)

    @distributed_test(world_size=[1])
    def _test_zero_allow_untested_optimizer(args, zero_stage):
        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        optimizer = SimpleOptimizer(model.parameters())
        with pytest.raises(AssertionError):
            model, optim, _, _ = deepspeed.initialize(args=args,
                                                      model=model,
                                                      optimizer=optimizer,
                                                      model_parameters=model.parameters())

    _test_zero_allow_untested_optimizer(args, zero_stage)


@pytest.mark.parametrize('zero_stage, use_cpu_offload', [(2, False)])
def test_zero_empty_partition(tmpdir, zero_stage, use_cpu_offload):
    if not bf16_required_version_check():
        pytest.skip(
            " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
        )

    if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
        pytest.skip("cpu-adam is not compatible")

    if zero_stage == 3:
        pytest.skip("skip for now")

    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "fp16": {
            "enabled": False
        },
        "bfloat16": {
            "enabled": True
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            }
        },
        "zero_optimization": {
            "stage": zero_stage,
            "cpu_offload": use_cpu_offload,
            "reduce_bucket_size": 100,
            "allgather_bucket_size": 100
        }
    }
    args = args_from_dict(tmpdir, config_dict)

    @distributed_test(world_size=[3])
    def _test_zero_empty_partition(args, zero_stage):
        hidden_dim = 1
        model = SimpleModel(hidden_dim)

        # Ensure model has 2 parameters, to cause empty partition with DP=3
        assert len(list(model.parameters())) == 2
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())

        # Now make sure things work..
        data_loader = random_dataloader(model=model,
                                        total_samples=1,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_zero_empty_partition(args=args, zero_stage=zero_stage)


@pytest.mark.parametrize('zero_stage, optimizer_constructor',
                         [(2,
                           torch.optim.Adam),
                          (2,
                           FusedAdam)])
def test_zero_supported_client_optimizer(tmpdir, zero_stage, optimizer_constructor):
    if not bf16_required_version_check():
        pytest.skip(
            " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
        )

    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "fp16": {
            "enabled": False
        },
        "bfloat16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": zero_stage
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    @distributed_test(world_size=[1])
    def _test_zero_supported_client_optimizer(args, zero_stage, optimizer_constructor):
        model = SimpleModel(hidden_dim)

        client_optimizer = optimizer_constructor(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              optimizer=client_optimizer)

    _test_zero_supported_client_optimizer(args=args,
                                          zero_stage=zero_stage,
                                          optimizer_constructor=optimizer_constructor)


def test_zero2_reduce_scatter_off(tmpdir):
    if not bf16_required_version_check():
        pytest.skip(
            " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
        )

    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            }
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "allgather_bucket_size": 2000000000,
            "reduce_bucket_size": 200000000,
            "overlap_comm": False,
            "reduce_scatter": False
        },
        "fp16": {
            "enabled": False
        },
        "bfloat16": {
            "enabled": True
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[2])
    def _helper(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _helper(args=args, model=model, hidden_dim=hidden_dim)


@pytest.mark.parametrize('stage', [2])
def test_zero_empty_grad(tmpdir, stage):
    if not bf16_required_version_check():
        pytest.skip(
            " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
        )

    config_dict = {
        "train_batch_size": 1,
        "steps_per_print": 1,
        "fp16": {
            "enabled": False
        },
        "bfloat16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": stage
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1])
    def _go(args, model, hidden_dim):
        optimizer = torch.optim.Adam(model.parameters())
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              optimizer=optimizer)
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _go(args=args, model=model, hidden_dim=hidden_dim)

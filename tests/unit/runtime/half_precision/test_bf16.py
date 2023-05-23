# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
import pytest
from deepspeed.ops.adam import FusedAdam
from unit.common import DistributedTest
from deepspeed.ops.op_builder import CPUAdamBuilder
from unit.simple_model import SimpleModel, SimpleOptimizer, random_dataloader
from unit.util import bf16_required_version_check
from deepspeed import comm as dist


class TestAdamBF16ZeroOneCycleCompatibility(DistributedTest):
    world_size = 1

    def test(self, zero_stage=2, use_cpu_offload=False):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
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
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            }
        }

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestZeroAllowUntestedOptimizer(DistributedTest):
    world_size = 1

    def test(self, zero_stage=2, use_cpu_offload=False):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_micro_batch_size_per_gpu": 4,
            "steps_per_print": 1,
            "fp16": {
                "enabled": False,
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            },
            "zero_allow_untested_optimizer": False
        }

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        optimizer = SimpleOptimizer(model.parameters())
        with pytest.raises(AssertionError):
            model, optim, _, _ = deepspeed.initialize(config=config_dict,
                                                      model=model,
                                                      optimizer=optimizer,
                                                      model_parameters=model.parameters())


class TestZeroEmptyPartition(DistributedTest):
    world_size = 3

    def test(self, zero_stage=2, use_cpu_offload=False):
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
            "bf16": {
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

        hidden_dim = 1
        model = SimpleModel(hidden_dim)

        # Ensure model has 2 parameters, to cause empty partition with DP=3
        assert len(list(model.parameters())) == 2
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

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


@pytest.mark.parametrize("optimizer_constructor", [torch.optim.Adam, FusedAdam])
class TestZeroSupportedClientOptimizer(DistributedTest):
    world_size = 1

    def test(self, optimizer_constructor, zero_stage=2):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "steps_per_print": 1,
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        client_optimizer = optimizer_constructor(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, optimizer=client_optimizer)


class TestZero2ReduceScatterOff(DistributedTest):
    world_size = 2

    def test(self):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
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
            "bf16": {
                "enabled": True
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestZeroEmptyGrad(DistributedTest):
    world_size = 1

    def test(self, stage=2):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": stage
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.Adam(model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, optimizer=optimizer)
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.parametrize("comp_type", [torch.float16, torch.bfloat16, torch.float], ids=["fp16", "bfp16", "fp32"])
@pytest.mark.parametrize("comm_type", [torch.float16, torch.bfloat16, None], ids=["fp16", "bfp16", "default"])
class TestZeroDtypeCocktail(DistributedTest):
    world_size = 2

    def test(self, comp_type, comm_type):
        if comp_type == torch.bfloat16 or comm_type == torch.bfloat16:
            if not bf16_required_version_check():
                pytest.skip(
                    " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
                )

        type_str = {torch.float16: "fp16", torch.bfloat16: "bfp16"}

        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "steps_per_print": 1,
            "fp16": {
                "enabled": comp_type == torch.float16
            },
            "bf16": {
                "enabled": comp_type == torch.bfloat16
            },
            "zero_optimization": {
                "stage": 2
            },
        }
        if comm_type is not None:
            config_dict["communication_data_type"] = type_str[comm_type]
        else:
            comm_type = comp_type
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.Adam(model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, optimizer=optimizer)
        data_loader = random_dataloader(model=model,
                                        total_samples=2,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=comp_type)

        def custom_reduce(tensor, dst, op=dist.ReduceOp.SUM, group=None, async_op=False):
            assert tensor.dtype == comm_type
            return orig_torch_reduce(tensor, dst, op, group, async_op)

        orig_torch_reduce = dist.reduce
        dist.reduce = custom_reduce
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
        dist.reduce = orig_torch_reduce

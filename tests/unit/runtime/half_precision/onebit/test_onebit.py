# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn as nn
import deepspeed.comm as dist
import deepspeed
import pytest
import os
import numpy as np

from deepspeed.runtime.pipe.topology import PipeDataParallelTopology
from deepspeed.ops.op_builder import OpBuilder
from deepspeed.runtime.pipe.module import PipelineModule
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader
from unit.alexnet_model import AlexNetPipe, train_cifar
from deepspeed.runtime.utils import required_torch_version
from deepspeed.accelerator import get_accelerator

PipeTopo = PipeDataParallelTopology

if not required_torch_version(min_version=1.8):
    pytest.skip(
        "NCCL-based 1-bit compression requires torch 1.8 or higher",
        allow_module_level=True,
    )

rocm_version = OpBuilder.installed_rocm_version()
if rocm_version[0] > 4:
    pytest.skip("NCCL-based 1-bit compression is not yet supported w. ROCm 5 until cupy supports ROCm 5",
                allow_module_level=True)

if get_accelerator().device_name() == 'hpu':
    pytest.skip("1-bit compression is not supported by HPU.", allow_module_level=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
class TestOneBitAdamBasic(DistributedTest):
    world_size = 2

    def test(self, dtype):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "OneBitAdam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "freeze_step": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": (dtype == torch.float16),
                "loss_scale": 0,
                "initial_scale_power": 16,
            },
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(
            model=model,
            total_samples=50,
            hidden_dim=hidden_dim,
            device=model.device,
            dtype=dtype,
        )
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestOneBitAdamExpAvgMask(DistributedTest):
    world_size = 2

    def test(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "OneBitAdam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "freeze_step": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16
            },
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        param_optimizer = list(model.named_parameters())
        mask1 = torch.zeros_like(param_optimizer[0][1].data)
        for col in range(mask1.size()[1]):
            mask1[0][col] += 1
        mask1 = torch.flatten(mask1)
        optimizer_grouped_parameters = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01,
                "exp_avg_mask": mask1,
            },
            {
                "params": [param_optimizer[1][1]],
                "weight_decay": 0.01
            },
        ]

        model, optimizer, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters,
        )
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
        # Test whether the momentum mask works
        for v in optimizer.state.values():
            if v["exp_avg"].size() == mask1.size():
                assert torch.allclose(
                    v["exp_avg"],
                    v["exp_avg"].mul_(mask1.to(device=v["exp_avg"].device)),
                    atol=1e-07,
                ), f"Momentum mask is not working properly"


class TestOneBitAdamCheckpointing(DistributedTest):
    world_size = 2

    def test(self, tmpdir):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "OneBitAdam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "freeze_step": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16
            },
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        param_optimizer = list(model.named_parameters())
        mask1 = torch.zeros_like(param_optimizer[0][1].data)
        mask2 = torch.zeros_like(param_optimizer[0][1].data)
        for col in range(mask1.size()[1]):
            mask1[0][col] += 1
            mask2[1][col] += 1
        mask1 = torch.flatten(mask1)
        mask2 = torch.flatten(mask2)

        optimizer_grouped_parameters_1 = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01,
                "exp_avg_mask": mask1,
            },
            {
                "params": [param_optimizer[1][1]],
                "weight_decay": 0.01
            },
        ]

        optimizer_grouped_parameters_2 = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01,
                "exp_avg_mask": mask2,
            },
            {
                "params": [param_optimizer[1][1]],
                "weight_decay": 0.01
            },
        ]

        optimizer_grouped_parameters_3 = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01
            },
            {
                "params": [param_optimizer[1][1]],
                "weight_decay": 0.01
            },
        ]

        model_1, optimizer_1, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters_1,
        )
        data_loader = random_dataloader(
            model=model_1,
            total_samples=10,
            hidden_dim=hidden_dim,
            device=model_1.device,
        )
        for n, batch in enumerate(data_loader):
            loss = model_1(batch[0], batch[1])
            model_1.backward(loss)
            model_1.step()
        # Test whether momentum mask still exist after saving checkpoint
        assert optimizer_1.optimizer.adam_freeze_key is True
        mask1 = mask1.to(device=optimizer_1.param_groups[0]["exp_avg_mask"].device)
        assert torch.allclose(optimizer_1.param_groups[0]["exp_avg_mask"], mask1,
                              atol=1e-07), f"Incorrect momentum mask"
        save_folder = os.path.join(tmpdir, "saved_checkpoint")
        model_1.save_checkpoint(save_folder, tag=None)
        assert torch.allclose(optimizer_1.param_groups[0]["exp_avg_mask"], mask1,
                              atol=1e-07), f"Momentum mask should not change after saving checkpoint"

        model_2, optimizer_2, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters_2,
        )
        # Test whether momentum mask stays the same after loading checkpoint
        mask2 = mask2.to(device=optimizer_2.param_groups[0]["exp_avg_mask"].device)
        assert torch.allclose(optimizer_2.param_groups[0]["exp_avg_mask"], mask2,
                              atol=1e-07), f"Incorrect momentum mask"
        model_2.load_checkpoint(
            save_folder,
            tag=None,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        assert torch.allclose(optimizer_2.param_groups[0]["exp_avg_mask"], mask2,
                              atol=1e-07), f"Momentum mask should not change after loading checkpoint"
        # Test whether worker&server error is reset
        for v in optimizer_2.state.values():
            assert "worker_error" not in v, f"Incorrect worker error"
            assert "server_error" not in v, f"Incorrect server error"
        assert optimizer_2.optimizer.adam_freeze_key is True

        model_3, optimizer_3, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters_3,
        )
        optimizer_3.optimizer.freeze_step = 20
        data_loader = random_dataloader(
            model=model_3,
            total_samples=50,
            hidden_dim=hidden_dim,
            device=model_3.device,
        )
        for n, batch in enumerate(data_loader):
            loss = model_3(batch[0], batch[1])
            model_3.backward(loss)
            model_3.step()
        assert optimizer_3.optimizer.adam_freeze_key is True
        # Test whether momentum mask stays the same after loading checkpoint
        assert ("exp_avg_mask" not in optimizer_3.param_groups[0]), f"Incorrect momentum mask"
        model_3.load_checkpoint(
            save_folder,
            tag=None,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        assert ("exp_avg_mask"
                not in optimizer_3.param_groups[0]), f"Momentum mask should not change after loading checkpoint"
        # Test whether worker&server error is reset
        for v in optimizer_3.state.values():
            assert "worker_error" not in v, f"Incorrect worker error"
            assert "server_error" not in v, f"Incorrect server error"
        assert optimizer_3.optimizer.adam_freeze_key is False

    def test_overflow(self, tmpdir):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "OneBitAdam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "freeze_step": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16
            },
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model, total_samples=100, hidden_dim=hidden_dim, device=model.device)
        save_folder = os.path.join(tmpdir, "saved_checkpoint")
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            if dist.get_rank() == 0 and n >= 10:
                loss = loss * 1000000.0
            model.backward(loss)
            dist.barrier()
            model.step()
            dist.barrier()
            model.save_checkpoint(save_folder, tag=None)


@pytest.mark.parametrize(
    "topo_config",
    [
        {
            "num_pp": 2,
            "num_dp": 2
        },
    ],
)
class TestOneBitAdamFP16Pipeline(DistributedTest):
    world_size = 4

    def test(self, topo_config):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 4,
            "grandient_accumulation_steps": 1,
            "steps_per_print": 20,
            "optimizer": {
                "type": "OneBitAdam",
                "params": {
                    "lr": 0.00001,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7,
                    "freeze_step": 200,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 0
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16
            },
            "pipeline": {
                "seed_layers": True,
                "activation_checkpoint_interval": 1
            },
        }

        topo = PipeTopo(**topo_config)
        steps = 100

        # TODO: Add correctness tests/asserts comparing with baseline?
        test_net = AlexNetPipe()
        test_model = PipelineModule(layers=test_net.to_layers(), topology=topo, loss_fn=nn.CrossEntropyLoss())
        test_losses = train_cifar(test_model, config=config_dict, num_steps=steps, fp16=config_dict['fp16']['enabled'])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
class TestZeroOneAdamBasic(DistributedTest):
    world_size = 2

    def test(self, dtype):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "ZeroOneAdam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "var_freeze_step": 4,
                    "var_update_scaler": 1,
                    "local_step_scaler": 1,
                    "local_step_clipper": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": (dtype == torch.float16),
                "loss_scale": 0,
                "initial_scale_power": 16,
            },
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(
            model=model,
            total_samples=50,
            hidden_dim=hidden_dim,
            device=model.device,
            dtype=dtype,
        )
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestZeroOneAdamExpAvgMask(DistributedTest):
    world_size = 2

    def test(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "ZeroOneAdam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "var_freeze_step": 4,
                    "var_update_scaler": 1,
                    "local_step_scaler": 1,
                    "local_step_clipper": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16
            },
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        param_optimizer = list(model.named_parameters())
        mask1 = torch.zeros_like(param_optimizer[0][1].data)
        for col in range(mask1.size()[1]):
            mask1[0][col] += 1
        mask1 = torch.flatten(mask1)
        optimizer_grouped_parameters = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01,
                "exp_avg_mask": mask1,
            },
            {
                "params": [param_optimizer[1][1]],
                "weight_decay": 0.01
            },
        ]

        model, optimizer, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters,
        )
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
        # Test whether the momentum mask works
        for v in optimizer.state.values():
            if v["exp_avg"].size() == mask1.size():
                assert torch.allclose(
                    v["exp_avg"],
                    v["exp_avg"].mul_(mask1.to(device=v["exp_avg"].device)),
                    atol=1e-07,
                ), f"Momentum mask is not working properly"


class TestZeroOneAdamCheckpointing(DistributedTest):
    world_size = 2

    def test(self, tmpdir):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "ZeroOneAdam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "var_freeze_step": 4,
                    "var_update_scaler": 1,
                    "local_step_scaler": 1,
                    "local_step_clipper": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16
            },
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        param_optimizer = list(model.named_parameters())
        mask1 = torch.zeros_like(param_optimizer[0][1].data)
        mask2 = torch.zeros_like(param_optimizer[0][1].data)
        for col in range(mask1.size()[1]):
            mask1[0][col] += 1
            mask2[1][col] += 1
        mask1 = torch.flatten(mask1)
        mask2 = torch.flatten(mask2)

        optimizer_grouped_parameters_1 = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01,
                "exp_avg_mask": mask1,
            },
            {
                "params": [param_optimizer[1][1]],
                "weight_decay": 0.01
            },
        ]

        optimizer_grouped_parameters_2 = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01,
                "exp_avg_mask": mask2,
            },
            {
                "params": [param_optimizer[1][1]],
                "weight_decay": 0.01
            },
        ]

        optimizer_grouped_parameters_3 = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01
            },
            {
                "params": [param_optimizer[1][1]],
                "weight_decay": 0.01
            },
        ]

        model_1, optimizer_1, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters_1,
        )
        data_loader = random_dataloader(
            model=model_1,
            total_samples=10,
            hidden_dim=hidden_dim,
            device=model_1.device,
        )
        for n, batch in enumerate(data_loader):
            loss = model_1(batch[0], batch[1])
            model_1.backward(loss)
            model_1.step()
        # Test whether momentum mask still exist after saving checkpoint
        mask1 = mask1.to(device=optimizer_1.param_groups[0]["exp_avg_mask"].device)
        assert torch.allclose(optimizer_1.param_groups[0]["exp_avg_mask"], mask1,
                              atol=1e-07), f"Incorrect momentum mask"
        save_folder = os.path.join(tmpdir, "saved_checkpoint")
        model_1.save_checkpoint(save_folder, tag=None)
        assert torch.allclose(optimizer_1.param_groups[0]["exp_avg_mask"], mask1,
                              atol=1e-07), f"Momentum mask should not change after saving checkpoint"

        model_2, optimizer_2, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters_2,
        )
        # Test whether momentum mask stays the same after loading checkpoint
        mask2 = mask2.to(device=optimizer_2.param_groups[0]["exp_avg_mask"].device)
        assert torch.allclose(optimizer_2.param_groups[0]["exp_avg_mask"], mask2,
                              atol=1e-07), f"Incorrect momentum mask"
        model_2.load_checkpoint(
            save_folder,
            tag=None,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        assert torch.allclose(optimizer_2.param_groups[0]["exp_avg_mask"], mask2,
                              atol=1e-07), f"Momentum mask should not change after loading checkpoint"
        # Test whether worker&server error is reset
        for v in optimizer_2.state.values():
            assert "worker_error" not in v, f"Incorrect worker error"
            assert "server_error" not in v, f"Incorrect server error"

        model_3, optimizer_3, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters_3,
        )
        optimizer_3.optimizer.freeze_step = 20
        data_loader = random_dataloader(
            model=model_3,
            total_samples=50,
            hidden_dim=hidden_dim,
            device=model_3.device,
        )
        for n, batch in enumerate(data_loader):
            loss = model_3(batch[0], batch[1])
            model_3.backward(loss)
            model_3.step()
        # Test whether momentum mask stays the same after loading checkpoint
        assert ("exp_avg_mask" not in optimizer_3.param_groups[0]), f"Incorrect momentum mask"
        model_3.load_checkpoint(
            save_folder,
            tag=None,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        assert ("exp_avg_mask"
                not in optimizer_3.param_groups[0]), f"Momentum mask should not change after loading checkpoint"
        # Test whether worker&server error is reset
        for v in optimizer_3.state.values():
            assert "worker_error" not in v, f"Incorrect worker error"
            assert "server_error" not in v, f"Incorrect server error"

    def test_overflow(self, tmpdir):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "ZeroOneAdam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "var_freeze_step": 4,
                    "var_update_scaler": 1,
                    "local_step_scaler": 1,
                    "local_step_clipper": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16
            },
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model, total_samples=100, hidden_dim=hidden_dim, device=model.device)
        save_folder = os.path.join(tmpdir, "saved_checkpoint")
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            if dist.get_rank() == 0 and n >= 10:
                loss = loss * 1000000.0
            model.backward(loss)
            dist.barrier()
            model.step()
            dist.barrier()
            model.save_checkpoint(save_folder, tag=None)


@pytest.mark.parametrize(
    "topo_config",
    [
        {
            "num_pp": 2,
            "num_dp": 2
        },
    ],
)
class TestZeroOneAdamFP16Pipeline(DistributedTest):
    world_size = 4

    def test(self, topo_config):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 4,
            "grandient_accumulation_steps": 1,
            "steps_per_print": 20,
            "optimizer": {
                "type": "ZeroOneAdam",
                "params": {
                    "lr": 0.00001,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7,
                    "var_freeze_step": 4,
                    "var_update_scaler": 1,
                    "local_step_scaler": 1,
                    "local_step_clipper": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 0
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16
            },
            "pipeline": {
                "seed_layers": True,
                "activation_checkpoint_interval": 1
            },
        }

        topo = PipeTopo(**topo_config)
        steps = 100

        # TODO: Add correctness tests/asserts comparing with baseline?
        test_net = AlexNetPipe()
        test_model = PipelineModule(layers=test_net.to_layers(), topology=topo, loss_fn=nn.CrossEntropyLoss())
        test_losses = train_cifar(test_model, config=config_dict, num_steps=steps, fp16=config_dict['fp16']['enabled'])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
class TestOneBitLambBasic(DistributedTest):
    world_size = 2

    def test(self, dtype):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "OneBitLamb",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "max_coeff": 0.3,
                    "min_coeff": 0.01,
                    "freeze_step": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                    "coeff_beta": 0.9,
                    "factor_max": 1.0,
                    "factor_min": 0.5,
                    "factor_threshold": 0.1,
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": (dtype == torch.float16),
                "loss_scale": 0,
                "initial_scale_power": 16,
            },
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(
            model=model,
            total_samples=50,
            hidden_dim=hidden_dim,
            device=model.device,
            dtype=dtype,
        )
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestOneBitLampExpAvgMask(DistributedTest):
    world_size = 2

    def test(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "OneBitLamb",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "max_coeff": 0.3,
                    "min_coeff": 0.01,
                    "freeze_step": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                    "coeff_beta": 0.9,
                    "factor_max": 1.0,
                    "factor_min": 0.5,
                    "factor_threshold": 0.1,
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16
            },
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        param_optimizer = list(model.named_parameters())
        mask1 = torch.zeros_like(param_optimizer[0][1].data)
        for col in range(mask1.size()[1]):
            mask1[0][col] += 1
        optimizer_grouped_parameters = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01,
                "exp_avg_mask": mask1,
            },
            {
                "params": [param_optimizer[1][1]],
                "weight_decay": 0.01
            },
        ]

        model, optimizer, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters,
        )
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
        # Test whether the momentum mask works
        for v in optimizer.state.values():
            if v["exp_avg"].size() == mask1.size():
                assert torch.allclose(
                    v["exp_avg"],
                    v["exp_avg"].mul_(mask1.to(device=v["exp_avg"].device)),
                    atol=1e-07,
                ), f"Momentum mask is not working properly"


class TestOneBitLambCheckpointing(DistributedTest):
    world_size = 2

    def test(self, tmpdir):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "OneBitLamb",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "max_coeff": 0.3,
                    "min_coeff": 0.01,
                    "freeze_step": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                    "coeff_beta": 0.9,
                    "factor_max": 1.0,
                    "factor_min": 0.5,
                    "factor_threshold": 0.1,
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16
            },
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        param_optimizer = list(model.named_parameters())
        mask1 = torch.zeros_like(param_optimizer[0][1].data)
        mask2 = torch.zeros_like(param_optimizer[0][1].data)
        for col in range(mask1.size()[1]):
            mask1[0][col] += 1
            mask2[1][col] += 1

        optimizer_grouped_parameters_1 = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01,
                "exp_avg_mask": mask1,
            },
            {
                "params": [param_optimizer[1][1]],
                "weight_decay": 0.01
            },
        ]

        optimizer_grouped_parameters_2 = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01,
                "exp_avg_mask": mask2,
            },
            {
                "params": [param_optimizer[1][1]],
                "weight_decay": 0.01
            },
        ]

        optimizer_grouped_parameters_3 = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01
            },
            {
                "params": [param_optimizer[1][1]],
                "weight_decay": 0.01
            },
        ]

        model_1, optimizer_1, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters_1,
        )
        data_loader = random_dataloader(
            model=model_1,
            total_samples=10,
            hidden_dim=hidden_dim,
            device=model_1.device,
        )
        for n, batch in enumerate(data_loader):
            loss = model_1(batch[0], batch[1])
            model_1.backward(loss)
            model_1.step()
        # Test whether momentum mask still exist after saving checkpoint
        assert optimizer_1.optimizer.lamb_freeze_key is True
        mask1 = mask1.to(device=optimizer_1.param_groups[0]["exp_avg_mask"].device)
        assert torch.allclose(optimizer_1.param_groups[0]["exp_avg_mask"], mask1,
                              atol=1e-07), f"Incorrect momentum mask"
        scaling_coeff_1 = []
        for v in optimizer_1.state.values():
            assert "scaling_coeff" in v, f"Incorrect scaling_coeff"
            scaling_coeff_1.append(v["scaling_coeff"])
        save_folder = os.path.join(tmpdir, "saved_checkpoint")
        model_1.save_checkpoint(save_folder, tag=None)
        assert torch.allclose(optimizer_1.param_groups[0]["exp_avg_mask"], mask1,
                              atol=1e-07), f"Momentum mask should not change after saving checkpoint"

        model_2, optimizer_2, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters_2,
        )
        # Test whether momentum mask stays the same after loading checkpoint
        mask2 = mask2.to(device=optimizer_2.param_groups[0]["exp_avg_mask"].device)
        assert torch.allclose(optimizer_2.param_groups[0]["exp_avg_mask"], mask2,
                              atol=1e-07), f"Incorrect momentum mask"
        model_2.load_checkpoint(
            save_folder,
            tag=None,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        assert torch.allclose(optimizer_2.param_groups[0]["exp_avg_mask"], mask2,
                              atol=1e-07), f"Momentum mask should not change after loading checkpoint"
        # Test whether worker&server error is reset
        assert len(optimizer_2.optimizer.worker_errors) == 0, f"Incorrect worker error"
        assert len(optimizer_2.optimizer.server_errors) == 0, f"Incorrect server error"
        # Test whether scaling_coeffs is loaded correctly
        scaling_coeff_2 = []
        for v in optimizer_2.state.values():
            assert "scaling_coeff" in v, f"Incorrect scaling_coeff"
            scaling_coeff_2.append(v["scaling_coeff"])
        assert list(sorted(scaling_coeff_2)) == list(sorted(scaling_coeff_1)), f"Incorrect scaling_coeffs"
        assert optimizer_2.optimizer.lamb_freeze_key is True

        model_3, optimizer_3, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters_3,
        )
        optimizer_3.optimizer.freeze_step = 20
        data_loader = random_dataloader(
            model=model_3,
            total_samples=50,
            hidden_dim=hidden_dim,
            device=model_3.device,
        )
        for n, batch in enumerate(data_loader):
            loss = model_3(batch[0], batch[1])
            model_3.backward(loss)
            model_3.step()
        assert optimizer_3.optimizer.lamb_freeze_key is True
        # Test whether momentum mask stays the same after loading checkpoint
        assert ("exp_avg_mask" not in optimizer_3.param_groups[0]), f"Incorrect momentum mask"
        model_3.load_checkpoint(
            save_folder,
            tag=None,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        assert ("exp_avg_mask"
                not in optimizer_3.param_groups[0]), f"Momentum mask should not change after loading checkpoint"
        # Test whether worker&server error is reset
        assert len(optimizer_3.optimizer.worker_errors) == 0, f"Incorrect worker error"
        assert len(optimizer_3.optimizer.server_errors) == 0, f"Incorrect server error"
        # Test whether scaling_coeffs, lamb_coeff_freeze, last_factor are reset
        for v in optimizer_3.state.values():
            assert v["lamb_coeff_freeze"] == 0.0, f"Incorrect lamb_coeff_freeze"
            assert v["last_factor"] == 1.0, f"Incorrect last_factor"
            assert "scaling_coeff" not in v, f"Incorrect scaling_coeff"
        assert optimizer_3.optimizer.lamb_freeze_key is False

    def test_overflow(self, tmpdir):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "OneBitLamb",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "max_coeff": 0.3,
                    "min_coeff": 0.01,
                    "freeze_step": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                    "coeff_beta": 0.9,
                    "factor_max": 1.0,
                    "factor_min": 0.5,
                    "factor_threshold": 0.1,
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16
            },
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model, total_samples=100, hidden_dim=hidden_dim, device=model.device)
        save_folder = os.path.join(tmpdir, "saved_checkpoint")
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            if dist.get_rank() == 0 and n >= 10:
                loss = loss * 1000000.0
            model.backward(loss)
            dist.barrier()
            model.step()
            dist.barrier()
            model.save_checkpoint(save_folder, tag=None)


@pytest.mark.parametrize(
    "topo_config",
    [
        {
            "num_pp": 2,
            "num_dp": 2
        },
    ],
)
class TestOneBitLambFP16Pipeline(DistributedTest):
    world_size = 4

    def test(self, topo_config):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 4,
            "grandient_accumulation_steps": 1,
            "steps_per_print": 20,
            "optimizer": {
                "type": "OneBitLamb",
                "params": {
                    "lr": 0.00001,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7,
                    "freeze_step": 200,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 0
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16
            },
            "pipeline": {
                "seed_layers": True,
                "activation_checkpoint_interval": 1
            },
        }

        topo = PipeTopo(**topo_config)
        steps = 100

        # TODO: Add correctness tests/asserts comparing with baseline?
        test_net = AlexNetPipe()
        test_model = PipelineModule(layers=test_net.to_layers(), topology=topo, loss_fn=nn.CrossEntropyLoss())
        test_losses = train_cifar(test_model, config=config_dict, num_steps=steps, fp16=config_dict['fp16']['enabled'])


@pytest.mark.sequential
class TestCompressedAllReduceBasic(DistributedTest):
    world_size = 2

    def test(self, tmpdir):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        from deepspeed.runtime.comm.nccl import NcclBackend

        size = dist.get_world_size()
        rank = dist.get_rank()
        backend = NcclBackend()
        local_rank = dist.get_rank()
        device = torch.device(get_accelerator().device_name(), dist.get_rank())

        # A simulated compression function using deepspeed.comm
        def torch_sim(a):
            a_sign = a.sign().add_(1).bool().float().add_(-0.5).mul_(2.0)
            scale = a.norm() / np.sqrt(a.numel())
            a_compressed = scale * a_sign
            a_sign = None
            worker_error = a - a_compressed
            dist.all_reduce(a_compressed)
            a_compressed.mul_(1 / dist.get_world_size())
            a_server_sign = (a_compressed.sign().add_(1).bool().float().add_(-0.5).mul_(2.0))
            a_list = torch.chunk(a_compressed, chunks=dist.get_world_size())
            server_scale = [chunk_a.norm() / np.sqrt(chunk_a.numel()) for chunk_a in a_list]
            a_sign_list = torch.chunk(a_server_sign, dist.get_world_size())
            a_server_compressed = torch.cat([server_scale[i] * a_sign_list[i] for i in range(dist.get_world_size())])
            rank = dist.get_rank()
            server_error = a_list[rank] - server_scale[rank] * a_sign_list[rank]
            get_accelerator().synchronize()
            dist.barrier()
            return a_server_compressed, worker_error, server_error

        tensor_size = 300 * 2**20
        server_size = int(tensor_size / size)
        if tensor_size % (8 * size) != 0:
            right_tensor_size = tensor_size + (8 * size - (tensor_size % (8 * size)))
        else:
            right_tensor_size = tensor_size
        right_server_size = right_tensor_size // size

        # Adding bias to the initialization of the gradient we are communicating
        # In order to get rid of the case where some elements in the gradient are too small
        a = (torch.rand(tensor_size, device=device) - 0.5) + 0.01 * rank

        worker_error = torch.zeros(right_tensor_size, device=device)
        server_error = torch.zeros(right_server_size, device=device)

        a_torch, worker_error_torch, server_error_torch = torch_sim(a)
        get_accelerator().empty_cache()

        a_after = backend.compressed_allreduce(a, worker_error, server_error, local_rank)

        threshold = 1e-6
        magnitude_threshold = 1e-6
        diff_mask = (a_after - a_torch) > threshold
        diff_server_mask = torch.chunk(diff_mask, size)[rank]
        mpi_server = torch.chunk(a_after, size)[rank] + server_error
        torch_server = torch.chunk(a_torch, size)[rank] + server_error_torch

        # If the number in the compensated_server_m is too small (e.g 1e-8), then calling sign() might be problematic
        # The test would skip those numbers that are too small in compensated_server_m
        check_mag_mask = mpi_server[diff_server_mask] > magnitude_threshold
        if torch.sum(check_mag_mask) != 0:
            print("Fails at {} of positions".format(torch.sum(check_mag_mask)))
        assert torch.sum(diff_server_mask) == 0 or torch.sum(check_mag_mask) == 0

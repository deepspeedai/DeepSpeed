# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed.comm as dist
import deepspeed
import pytest
from deepspeed.ops.adam import FusedAdam
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, SimpleOptimizer, random_dataloader, SimpleMoEModel, sequence_dataloader
from deepspeed.utils.torch import required_torch_version
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import CPUAdamBuilder, FusedLambBuilder
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

if torch.half not in get_accelerator().supported_dtypes():
    pytest.skip(f"fp16 not supported, valid dtype: {get_accelerator().supported_dtypes()}", allow_module_level=True)


class TestLambFP32GradClip(DistributedTest):
    world_size = 2

    @pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedLambBuilder.NAME],
                        reason="FusedLambBuilder has not been implemented on this system.")
    def test(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Lamb",
                "params": {
                    "lr": 0.00015
                }
            },
            "gradient_clipping": 1.0
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestLambFP16(DistributedTest):
    world_size = 2

    @pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedLambBuilder.NAME],
                        reason="FusedLambBuilder has not been implemented on this system.")
    def test__basic(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
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
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    @pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedLambBuilder.NAME],
                        reason="FusedLambBuilder has not been implemented on this system.")
    def test_empty_grad(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
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
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim, empty_grad=True)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestAdamFP32EmptyGrad(DistributedTest):
    world_size = 2

    def test(self):
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
            "fp16": {
                "enabled": False
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim, empty_grad=True)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestAdamwFP16Basic(DistributedTest):
    world_size = 1

    def test(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {"train_batch_size": 1, "steps_per_print": 1, "fp16": {"enabled": True}}
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, optimizer=optimizer)
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestFP16OptimizerForMoE(DistributedTest):
    world_size = 2

    def test_unfused_gradnorm(self, monkeypatch):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        if not required_torch_version(min_version=1.8):
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {"train_batch_size": 2, "steps_per_print": 1, "fp16": {"enabled": True}}
        hidden_dim = 10

        def mock_unscale_and_clip_grads(total_norm, apply_scale=True):
            torch_norm_tensor = get_accelerator().FloatTensor([total_norm])
            all_gather_results = [torch.zeros_like(torch_norm_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(all_gather_results, torch_norm_tensor)
            assert len(set([x.item() for x in all_gather_results])) == 1
            return 1.0

        # initialize MoE
        model = SimpleMoEModel(hidden_dim, ep_size=2)
        optimizer = torch.optim.AdamW(params=model.parameters())
        engine, optimizer, _, _ = deepspeed.initialize(config=config_dict,
                                                       model=model,
                                                       optimizer=optimizer,
                                                       dist_init_required=False)
        monkeypatch.setattr(optimizer, 'unscale_and_clip_grads', mock_unscale_and_clip_grads)
        data_loader = sequence_dataloader(model=engine,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=engine.device,
                                          dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()

    def test_fused_gradnorm(self, monkeypatch):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        if not required_torch_version(min_version=1.8):
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {"train_batch_size": 2, "steps_per_print": 1, "fp16": {"enabled": True}}
        hidden_dim = 10

        def mock_unscale_and_clip_grads(grads_groups_flat, total_norm, apply_scale=True):
            torch_norm_tensor = get_accelerator().FloatTensor([total_norm])
            all_gather_results = [torch.zeros_like(torch_norm_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(all_gather_results, torch_norm_tensor)
            assert len(set([x.item() for x in all_gather_results])) == 1
            return 1.0

        # initialize MoE
        model = SimpleMoEModel(hidden_dim, ep_size=2)
        param_group = {'params': [p for p in model.parameters()], 'name': 'random-unique-name'}
        params = split_params_into_different_moe_groups_for_optimizer(param_group)
        # optimizer = torch.optim.AdamW(params=model.parameters())
        optimizer = FusedAdam(params=params)
        engine, optimizer, _, _ = deepspeed.initialize(config=config_dict,
                                                       model=model,
                                                       optimizer=optimizer,
                                                       dist_init_required=False)
        monkeypatch.setattr(optimizer, 'unscale_and_clip_grads', mock_unscale_and_clip_grads)
        data_loader = sequence_dataloader(model=engine,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=engine.device,
                                          dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()

    @pytest.mark.parametrize("fused_lamb_legacy", [(False), (True)])
    @pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedLambBuilder.NAME],
                        reason="FusedLambBuilder has not been implemented on this system.")
    def test_lamb_gradnorm(self, monkeypatch, fused_lamb_legacy: bool):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        if not required_torch_version(min_version=1.8):
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "fp16": {
                "enabled": True
            },
            "optimizer": {
                "type": "Lamb",
                "params": {
                    "lr": 0.00015
                }
            }
        }
        hidden_dim = 10

        def mock_unscale_and_clip_grads(total_norm, apply_scale=True):
            torch_norm_tensor = get_accelerator().FloatTensor([total_norm])
            all_gather_results = [torch.zeros_like(torch_norm_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(all_gather_results, torch_norm_tensor)
            assert len(set([x.item() for x in all_gather_results])) == 1
            return 1.0

        # initialize MoE
        model = SimpleMoEModel(hidden_dim, ep_size=2)
        engine, optimizer, _, _ = deepspeed.initialize(config=config_dict,
                                                       model=model,
                                                       model_parameters=model.parameters(),
                                                       dist_init_required=False)
        monkeypatch.setattr(optimizer, 'unscale_and_clip_grads', mock_unscale_and_clip_grads)
        optimizer.fused_lamb_legacy = fused_lamb_legacy
        data_loader = sequence_dataloader(model=engine,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=engine.device,
                                          dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()


class TestAdamwFP16EmptyGrad(DistributedTest):
    world_size = 1

    def test(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {"train_batch_size": 1, "steps_per_print": 1, "fp16": {"enabled": True}}
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, optimizer=optimizer)
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("use_cpu_offload", [True, False])
class TestAdamFP16ZeroOneCycleCompatibility(DistributedTest):
    world_size = 1

    def test(self, zero_stage, use_cpu_offload):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
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
                                        total_samples=10,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("use_cpu_offload", [True, False])
class TestZeroStaticScale(DistributedTest):
    world_size = 1

    def test(self, zero_stage, use_cpu_offload, hidden_dim=4):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_batch_size": 4,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 138.
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            }
        }

        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        # Ensure the static scaler is configured.
        assert optim.dynamic_loss_scale == False
        assert optim.loss_scaler.loss_scale == 138.

        # Now make sure things work..
        data_loader = random_dataloader(model=model,
                                        total_samples=10,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("use_cpu_offload", [True, False])
class TestZeroAllowUntestedOptimizer(DistributedTest):
    world_size = 1

    def test(self, zero_stage, use_cpu_offload):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_batch_size": 4,
            "steps_per_print": 1,
            "fp16": {
                "enabled": True,
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            },
            "zero_allow_untested_optimizer": False,
            "zero_force_ds_cpu_optimizer": False
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = SimpleOptimizer(model.parameters())
        with pytest.raises(AssertionError):
            model, optim, _, _ = deepspeed.initialize(config=config_dict,
                                                      model=model,
                                                      optimizer=optimizer,
                                                      model_parameters=model.parameters())
            model.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("use_cpu_offload", [True, False])
class TestZeroEmptyPartition(DistributedTest):
    world_size = 3

    def test(self, zero_stage, use_cpu_offload):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        if zero_stage == 3:
            pytest.skip("skip for now")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
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
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("optimizer_constructor", [FusedAdam, torch.optim.Adam])
class TestZeroSupportedClientOptimizer(DistributedTest):
    world_size = 1

    def test(self, zero_stage, optimizer_constructor):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "fp16": {
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
        model.destroy()


class TestZero2ReduceScatterOff(DistributedTest):
    world_size = 2

    def test(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
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
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.parametrize("adam_type", ["Adam", "AdamW"])
@pytest.mark.parametrize("torch_impl", [True, False])
class TestFP16AdamTypes(DistributedTest):
    world_size = 1

    def test(self, adam_type, torch_impl):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 1,
            "steps_per_print": 1,
            "fp16": {
                "enabled": True,
                "initial_scale_power": 10
            },
            "optimizer": {
                "type": adam_type,
                "torch_adam": torch_impl,
                "params": {
                    "lr": 0.00015
                }
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        data_loader = random_dataloader(model=model,
                                        total_samples=10,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)

        for _, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestZero3LazyScatter(DistributedTest):
    world_size = 1

    def test(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 1,
            "steps_per_print": 1,
            "fp16": {
                "enabled": True,
                "initial_scale_power": 10
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 0.00015
                }
            },
            "zero_optimization": {
                "stage": 3
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
        )

        data_loader = random_dataloader(model=model,
                                        total_samples=10,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)

        for _, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.destroy()


@pytest.mark.parametrize('stage', [1, 2, 3])
class TestZeroEmptyGrad(DistributedTest):
    world_size = 1

    def test(self, stage):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 1,
            "steps_per_print": 1,
            "fp16": {
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
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.destroy()


class TestOptimizerInitialization(DistributedTest):
    world_size = 1

    def _make_engine(self, optimizer_name, dtype, weight_decay, legacy_lamb=False):
        model = SimpleModel(4).to(device=get_accelerator().current_device_name(), dtype=dtype)
        initial_weights = [parameter.detach().float().clone() for parameter in model.parameters()]
        kwargs = {"lr": 0.01, "weight_decay": weight_decay}
        parameters = list(model.parameters())
        if optimizer_name == "Adagrad":
            # Eager state must survive replacing the model parameters with master weights.
            parameters = [{"params": parameters[:1], "lr": 0.02}, {"params": parameters[1:]}]
            kwargs["initial_accumulator_value"] = 0.125
            optimizer = torch.optim.Adagrad(parameters, **kwargs)
        elif optimizer_name == "FusedAdam":
            optimizer = FusedAdam(parameters, **kwargs)
        elif optimizer_name == "Lamb":
            from deepspeed.ops.lamb import FusedLamb
            optimizer = FusedLamb(parameters, **kwargs)
        else:
            optimizer = getattr(torch.optim, optimizer_name)(parameters, **kwargs)
        config = {"train_micro_batch_size_per_gpu": 1, "steps_per_print": 1000, "gradient_clipping": 0.0}
        config["fp16" if dtype == torch.float16 else "bf16"] = {"enabled": True}
        if dtype == torch.float16:
            config["fp16"]["loss_scale"] = 1.0
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
        if optimizer_name == "FusedAdam":
            from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
            assert isinstance(engine.optimizer, FP16_Optimizer)
        else:
            from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
            assert isinstance(engine.optimizer, FP16_UnfusedOptimizer)
        if optimizer_name == "Lamb":
            engine.optimizer.fused_lamb_legacy = legacy_lamb
        if optimizer_name == "Adagrad":
            assert engine.optimizer.optimizer is optimizer
        return engine, initial_weights

    def _master_weights(self, engine):
        return torch.cat(
            [parameter.detach().flatten() for group in engine.optimizer.param_groups for parameter in group["params"]])

    def _step(self, engine, step, reference=None):
        dtype = next(engine.module.parameters()).dtype
        inputs = torch.tensor([[0.25, -0.5, 0.75, 1.0]], device=engine.device, dtype=dtype)
        labels = torch.tensor([step % 4], device=engine.device)
        engine.backward(engine(inputs, labels))
        if reference is not None:
            reference_parameters, reference_optimizer = reference
            # Use the actual low-precision gradients so the oracle isolates optimizer updates.
            for parameter, reference_parameter in zip(engine.module.parameters(), reference_parameters):
                reference_parameter.grad = parameter.grad.detach().float().clone()
            reference_optimizer.step()
            reference_optimizer.zero_grad()
        engine.step()

    def _exercise(self, tmpdir, optimizer_name, dtype, weight_decay, legacy_lamb=False):
        engine, initial_weights = self._make_engine(optimizer_name, dtype, weight_decay, legacy_lamb)
        try:
            initial_flat = torch.cat([weight.flatten() for weight in initial_weights])
            torch.testing.assert_close(self._master_weights(engine), initial_flat, rtol=0, atol=0)
            for parameter, initial in zip(engine.module.parameters(), initial_weights):
                torch.testing.assert_close(parameter.float(), initial, rtol=0, atol=0)
            # Construction must not count as a training update, including with zero decay.
            initial_state = engine.optimizer.state_dict()["optimizer_state_dict"]["state"]
            if optimizer_name == "Adagrad":
                assert all(state["step"] == 0 for state in initial_state.values())
            else:
                assert initial_state == {}

            reference = None
            if optimizer_name != "Lamb":
                reference_parameters = [torch.nn.Parameter(weight.clone()) for weight in initial_weights]
                reference_type = "AdamW" if optimizer_name == "FusedAdam" else optimizer_name
                kwargs = {"lr": 0.01, "weight_decay": weight_decay}
                parameters = reference_parameters
                if optimizer_name == "Adagrad":
                    parameters = [{"params": parameters[:1], "lr": 0.02}, {"params": parameters[1:]}]
                    kwargs["initial_accumulator_value"] = 0.125
                reference_optimizer = getattr(torch.optim, reference_type)(parameters, **kwargs)
                reference = reference_parameters, reference_optimizer

            for step in range(21):
                restored = None
                try:
                    if step in (0, 20):
                        tag = f"{optimizer_name}-{dtype}-{weight_decay}-{step}"
                        engine.save_checkpoint(str(tmpdir), tag=tag)
                        restored, _ = self._make_engine(optimizer_name, dtype, weight_decay, legacy_lamb)
                        load_path, _ = restored.load_checkpoint(str(tmpdir), tag=tag)
                        assert load_path is not None
                        torch.testing.assert_close(restored.optimizer.state_dict(), engine.optimizer.state_dict())
                    self._step(engine, step, reference)
                    context = f"{optimizer_name}, {dtype}, weight_decay={weight_decay}, update={step + 1}"
                    optimizer_state = engine.optimizer.state_dict()["optimizer_state_dict"]["state"]
                    assert all(state["step"] == step + 1 for state in optimizer_state.values()), context
                    if reference is not None:
                        expected = torch.cat([parameter.detach().flatten() for parameter in reference[0]])
                        torch.testing.assert_close(self._master_weights(engine),
                                                   expected,
                                                   rtol=2e-6,
                                                   atol=2e-6,
                                                   msg=lambda message: f"{context}\n{message}")
                    if restored is not None:
                        self._step(restored, step)
                        torch.testing.assert_close(restored.optimizer.state_dict(), engine.optimizer.state_dict())
                        torch.testing.assert_close(restored.module.state_dict(), engine.module.state_dict())
                finally:
                    if restored is not None:
                        restored.destroy()
        finally:
            engine.destroy()

    @pytest.mark.parametrize("optimizer_name", ["Adam", "AdamW", "FusedAdam"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_adam_first_update_and_checkpoint(self, tmpdir, optimizer_name, dtype):
        if dtype not in get_accelerator().supported_dtypes():
            pytest.skip(f"{dtype} is not supported")
        for weight_decay in (0.0, 0.1):
            self._exercise(tmpdir, optimizer_name, dtype, weight_decay)

    def test_adagrad_eager_state_and_checkpoint(self, tmpdir):
        self._exercise(tmpdir, "Adagrad", torch.float16, 0.1)

    @pytest.mark.parametrize("legacy_lamb", [False, True])
    @pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedLambBuilder.NAME],
                        reason="FusedLambBuilder has not been implemented on this system.")
    def test_lamb_first_update_and_checkpoint(self, tmpdir, legacy_lamb):
        self._exercise(tmpdir, "Lamb", torch.float16, 0.1, legacy_lamb)

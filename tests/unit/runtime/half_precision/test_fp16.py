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
from deepspeed.runtime.utils import required_torch_version
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import CPUAdamBuilder

try:
    from apex import amp  # noqa: F401 # type: ignore
    _amp_available = True
except ImportError:
    _amp_available = False
amp_available = pytest.mark.skipif(not _amp_available, reason="apex/amp is not installed")


class TestLambFP32GradClip(DistributedTest):
    world_size = 2

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
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

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
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
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
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
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
        data_loader = sequence_dataloader(model=engine, total_samples=50, hidden_dim=hidden_dim, device=engine.device)
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
        # optimizer = torch.optim.AdamW(params=model.parameters())
        optimizer = FusedAdam(params=model.parameters())
        engine, optimizer, _, _ = deepspeed.initialize(config=config_dict,
                                                       model=model,
                                                       optimizer=optimizer,
                                                       dist_init_required=False)
        monkeypatch.setattr(optimizer, 'unscale_and_clip_grads', mock_unscale_and_clip_grads)
        data_loader = sequence_dataloader(model=engine, total_samples=50, hidden_dim=hidden_dim, device=engine.device)
        for n, batch in enumerate(data_loader):
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()

    @pytest.mark.parametrize("fused_lamb_legacy", [(False), (True)])
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
        data_loader = sequence_dataloader(model=engine, total_samples=50, hidden_dim=hidden_dim, device=engine.device)
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
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
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
        data_loader = random_dataloader(model=model, total_samples=10, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


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
        data_loader = random_dataloader(model=model, total_samples=10, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


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
        data_loader = random_dataloader(model=model, total_samples=1, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@amp_available
class TestAmp(DistributedTest):
    world_size = 2

    def test_adam_basic(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {"train_batch_size": 2, "steps_per_print": 1, "amp": {"enabled": True}}
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.Adam(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, optimizer=optimizer)
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    def test_lamb_basic(self):
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
            "amp": {
                "enabled": True,
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    def test_adam_O2(self):
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
            "amp": {
                "enabled": True,
                "opt_level": "O2"
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    def test_adam_O2_empty_grad(self):
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
            "amp": {
                "enabled": True,
                "opt_level": "O2"
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


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
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
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

        data_loader = random_dataloader(model=model, total_samples=10, hidden_dim=hidden_dim, device=model.device)

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
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        data_loader = random_dataloader(model=model, total_samples=10, hidden_dim=hidden_dim, device=model.device)

        for _, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


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
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

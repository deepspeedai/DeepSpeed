import torch
import deepspeed.comm as dist
import deepspeed
import pytest
from deepspeed.ops.adam import FusedAdam
from .common import distributed_test
from deepspeed.ops.op_builder import CPUAdamBuilder
from .simple_model import SimpleModel, SimpleOptimizer, random_dataloader, args_from_dict, create_deepspeed_args, SimpleMoEModel, sequence_dataloader
from .util import required_torch_version

try:
    from apex import amp  # noqa: F401
    _amp_available = True
except ImportError:
    _amp_available = False
amp_available = pytest.mark.skipif(not _amp_available,
                                   reason="apex/amp is not installed")


def test_lamb_fp32_grad_clip(tmpdir):
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
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1, 2])
    def _test_lamb_fp32_grad_clip(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_lamb_fp32_grad_clip(args=args, model=model, hidden_dim=hidden_dim)


def test_lamb_fp16_basic(tmpdir):
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
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1, 2])
    def _test_lamb_fp16_basic(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_lamb_fp16_basic(args=args, model=model, hidden_dim=hidden_dim)


def test_lamb_fp16_empty_grad(tmpdir):
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
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=True)

    @distributed_test(world_size=[2])
    def _test_lamb_fp16_empty_grad(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_lamb_fp16_empty_grad(args=args, model=model, hidden_dim=hidden_dim)


def test_adam_fp32_empty_grad(tmpdir):
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
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=True)

    @distributed_test(world_size=[2])
    def _test_adam_fp32_empty_grad(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_adam_fp32_empty_grad(args=args, model=model, hidden_dim=hidden_dim)


def test_adamw_fp16_basic(tmpdir):
    config_dict = {
        "train_batch_size": 1,
        "steps_per_print": 1,
        "fp16": {
            "enabled": True
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1])
    def _test_adamw_fp16_basic(args, model, hidden_dim):
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              optimizer=optimizer)
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_adamw_fp16_basic(args=args, model=model, hidden_dim=hidden_dim)


def test_unfused_fp16_optimizer_gradnorm_for_moe(tmpdir, monkeypatch):
    if not required_torch_version():
        pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "fp16": {
            "enabled": True
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    def mock_unscale_and_clip_grads(total_norm, apply_scale=True):
        torch_norm_tensor = torch.cuda.FloatTensor([total_norm])
        all_gather_results = [
            torch.zeros_like(torch_norm_tensor) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(all_gather_results, torch_norm_tensor)
        assert len(set([x.item() for x in all_gather_results])) == 1
        return 1.0

    @distributed_test(world_size=[2])
    def _test_unfused_fp16_optimizer(args, hidden_dim):
        # initialize MoE
        model = SimpleMoEModel(hidden_dim, ep_size=2)
        optimizer = torch.optim.AdamW(params=model.parameters())
        engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              optimizer=optimizer,
                                              dist_init_required=False)
        monkeypatch.setattr(optimizer,
                            'unscale_and_clip_grads',
                            mock_unscale_and_clip_grads)
        data_loader = sequence_dataloader(model=engine,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=engine.device)
        for n, batch in enumerate(data_loader):
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()

    _test_unfused_fp16_optimizer(args=args, hidden_dim=hidden_dim)


def test_fused_fp16_optimizer_gradnorm_for_moe(tmpdir, monkeypatch):
    if not required_torch_version():
        pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "fp16": {
            "enabled": True
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    def mock_unscale_and_clip_grads(grads_groups_flat, total_norm, apply_scale=True):
        torch_norm_tensor = torch.cuda.FloatTensor([total_norm])
        all_gather_results = [
            torch.zeros_like(torch_norm_tensor) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(all_gather_results, torch_norm_tensor)
        assert len(set([x.item() for x in all_gather_results])) == 1
        return 1.0

    @distributed_test(world_size=[2])
    def _test_fused_fp16_optimizer(args, hidden_dim):
        # initialize MoE
        model = SimpleMoEModel(hidden_dim, ep_size=2)
        # optimizer = torch.optim.AdamW(params=model.parameters())
        optimizer = FusedAdam(params=model.parameters())
        engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              optimizer=optimizer,
                                              dist_init_required=False)
        monkeypatch.setattr(optimizer,
                            'unscale_and_clip_grads',
                            mock_unscale_and_clip_grads)
        data_loader = sequence_dataloader(model=engine,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=engine.device)
        for n, batch in enumerate(data_loader):
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()

    _test_fused_fp16_optimizer(args=args, hidden_dim=hidden_dim)


@pytest.mark.parametrize("fused_lamb_legacy", [(False), (True)])
def test_lamb_optimizer_gradnorm_for_moe(tmpdir, monkeypatch, fused_lamb_legacy: bool):
    if not required_torch_version():
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
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    def mock_unscale_and_clip_grads(total_norm, apply_scale=True):
        torch_norm_tensor = torch.cuda.FloatTensor([total_norm])
        all_gather_results = [
            torch.zeros_like(torch_norm_tensor) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(all_gather_results, torch_norm_tensor)
        assert len(set([x.item() for x in all_gather_results])) == 1
        return 1.0

    @distributed_test(world_size=[2])
    def _test_lamb_legacy_optimizer_step(args, hidden_dim, fused_lamb_legacy):
        # initialize MoE
        model = SimpleMoEModel(hidden_dim, ep_size=2)
        engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                               model=model,
                                               model_parameters=model.parameters(),
                                               dist_init_required=False)
        monkeypatch.setattr(optimizer,
                            'unscale_and_clip_grads',
                            mock_unscale_and_clip_grads)
        optimizer.fused_lamb_legacy = fused_lamb_legacy
        data_loader = sequence_dataloader(model=engine,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=engine.device)
        for n, batch in enumerate(data_loader):
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()

    _test_lamb_legacy_optimizer_step(args=args,
                                     hidden_dim=hidden_dim,
                                     fused_lamb_legacy=fused_lamb_legacy)


def test_dict_config_adamw_fp16_basic():
    config = {"train_batch_size": 1, "steps_per_print": 1, "fp16": {"enabled": True}}
    args = create_deepspeed_args()
    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1])
    def _test_adamw_fp16_basic(args, model, hidden_dim, config):
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              optimizer=optimizer,
                                              config=config)
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_adamw_fp16_basic(args=args, model=model, hidden_dim=hidden_dim, config=config)


def test_adamw_fp16_empty_grad(tmpdir):
    config_dict = {
        "train_batch_size": 1,
        "steps_per_print": 1,
        "fp16": {
            "enabled": True
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1])
    def _test_adamw_fp16_empty_grad(args, model, hidden_dim):
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              optimizer=optimizer)
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_adamw_fp16_empty_grad(args=args, model=model, hidden_dim=hidden_dim)


@pytest.mark.parametrize('zero_stage, use_cpu_offload',
                         [(1,
                           False),
                          (2,
                           False),
                          (2,
                           True),
                          (3,
                           False),
                          (3,
                           True)])
def test_adam_fp16_zero_onecycle_compatibility(tmpdir, zero_stage, use_cpu_offload):
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

    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    @distributed_test(world_size=[1])
    def _test_adam_fp16_zero_onecycle_compatibility(args, zero_stage, hidden_dim):
        model = SimpleModel(hidden_dim)

        model, _, _,_ = deepspeed.initialize(args=args,
                                             model=model,
                                             model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_adam_fp16_zero_onecycle_compatibility(args=args,
                                                zero_stage=zero_stage,
                                                hidden_dim=hidden_dim)


@pytest.mark.parametrize('zero_stage, use_cpu_offload',
                         [(1,
                           False),
                          (2,
                           False),
                          (2,
                           True),
                          (3,
                           False),
                          (3,
                           True)])
def test_zero_static_scale(tmpdir, zero_stage, use_cpu_offload):
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
    args = args_from_dict(tmpdir, config_dict)

    @distributed_test(world_size=2)
    def _test_zero_static_scale(args, zero_stage, hidden_dim):
        #making hidden size not divisible by DP for covering this scenario
        hidden_dim = hidden_dim
        model = SimpleModel(hidden_dim)

        model, optim, _, _ = deepspeed.initialize(args=args,
                                            model=model,
                                            model_parameters=model.parameters())

        # Ensure the static scaler is configured.
        assert optim.dynamic_loss_scale == False
        assert optim.loss_scaler.loss_scale == 138.

        # Now make sure things work..
        data_loader = random_dataloader(model=model,
                                        total_samples=10,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    #test when hidden_dim is not aligned with world size
    _test_zero_static_scale(args=args, zero_stage=zero_stage, hidden_dim=9)
    #test when hidden_dim is aligned with world size
    _test_zero_static_scale(args=args, zero_stage=zero_stage, hidden_dim=10)


def test_zero_static_scale_deprecated_format(tmpdir):
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
            "stage": 1
        }
    }
    args = args_from_dict(tmpdir, config_dict)

    @distributed_test(world_size=2)
    def _test_zero_static_scale(args):
        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(args=args,
                                                  model=model,
                                                  model_parameters=model.parameters())

        # Ensure the static scaler is configured.
        assert optim.dynamic_loss_scale == False
        assert optim.loss_scaler.loss_scale == 138.

        # Now make sure things work..
        data_loader = random_dataloader(model=model,
                                        total_samples=10,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_zero_static_scale(args)


@pytest.mark.parametrize('zero_stage, use_cpu_offload',
                         [(1,
                           False),
                          (2,
                           False),
                          (2,
                           True),
                          (3,
                           False),
                          (3,
                           True)])
def test_zero_allow_untested_optimizer(tmpdir, zero_stage, use_cpu_offload):
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


@pytest.mark.parametrize('zero_stage, use_cpu_offload',
                         [(1,
                           False),
                          (2,
                           False),
                          (2,
                           True),
                          (3,
                           False),
                          (3,
                           True)])
def test_zero_empty_partition(tmpdir, zero_stage, use_cpu_offload):
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
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_zero_empty_partition(args=args, zero_stage=zero_stage)


@amp_available
def test_adam_amp_basic(tmpdir):
    config_dict = {"train_batch_size": 1, "steps_per_print": 1, "amp": {"enabled": True}}
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1])
    def _test_adam_amp_basic(args, model, hidden_dim):
        optimizer = torch.optim.Adam(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              optimizer=optimizer)
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_adam_amp_basic(args=args, model=model, hidden_dim=hidden_dim)


@amp_available
def test_lamb_amp_basic(tmpdir):
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
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1, 2])
    def _test_lamb_amp_basic(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_lamb_amp_basic(args=args, model=model, hidden_dim=hidden_dim)


@amp_available
def test_adam_amp_o2(tmpdir):
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
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1, 2])
    def _test_adam_amp_o2(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_adam_amp_o2(args=args, model=model, hidden_dim=hidden_dim)


@amp_available
def test_adam_amp_o2_empty_grad(tmpdir):
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
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[2])
    def _test_adam_amp_o2_empty_grad(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_adam_amp_o2_empty_grad(args=args, model=model, hidden_dim=hidden_dim)


@pytest.mark.parametrize('zero_stage, optimizer_constructor',
                         [(1,
                           FusedAdam),
                          (2,
                           torch.optim.Adam),
                          (2,
                           FusedAdam),
                          (3,
                           torch.optim.Adam),
                          (3,
                           FusedAdam)])
def test_zero_supported_client_optimizer(tmpdir, zero_stage, optimizer_constructor):
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
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _helper(args=args, model=model, hidden_dim=hidden_dim)


@pytest.mark.parametrize('adam_type, torch_impl',
                         [('Adam',
                           True),
                          ('Adam',
                           False),
                          ('AdamW',
                           True),
                          ('AdamW',
                           False)])
def test_fp16_adam_types(tmpdir, adam_type, torch_impl):
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
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1])
    def _test_fp16_adam_types(args, model, hidden_dim):

        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())

        data_loader = random_dataloader(model=model,
                                        total_samples=10,
                                        hidden_dim=hidden_dim,
                                        device=model.device)

        for _, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_fp16_adam_types(args=args, model=model, hidden_dim=hidden_dim)


def test_zero3_lazyscatter(tmpdir):
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
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    @distributed_test(world_size=[1])
    def _go(args):
        model = SimpleModel(hidden_dim)

        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())

        data_loader = random_dataloader(model=model,
                                        total_samples=10,
                                        hidden_dim=hidden_dim,
                                        device=model.device)

        for _, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _go(args=args)


@pytest.mark.parametrize('stage', [1, 2, 3])
def test_zero_empty_grad(tmpdir, stage):
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
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _go(args=args, model=model, hidden_dim=hidden_dim)

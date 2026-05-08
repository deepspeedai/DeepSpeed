# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
from types import SimpleNamespace
from deepspeed.ops.op_builder import CPUAdamBuilder
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save, get_model_ckpt_name_for_rank
from deepspeed.checkpoint.constants import (BASE_OPTIMIZER_STATE, FP32_FLAT_GROUPS, OPTIMIZER_STATE_DICT)
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero import ZeroParamStatus
from deepspeed.utils.torch import required_torch_version

from unit.common import DistributedTest, DistributedFixture
from unit.simple_model import *

from unit.checkpoint.common import *

import pytest


class TestZeROCheckpoint(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('zero_stage', [3])
    def test_pipeline_checkpoint_loading(self, tmpdir, zero_stage):
        config_dict = {
            "train_batch_size": 2,
            "optimizer": {
                "type": 'Adam'
            },
            "zero_optimization": {
                "stage": zero_stage,
                "pipeline_loading_checkpoint": True,
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_module_only=True)

    @pytest.mark.parametrize('zero_stage, use_cpu_offload, adam_optimizer', [(0, False, 'Adam'), (1, False, 'Adam'),
                                                                             (2, False, 'Adam'),
                                                                             (2, True, 'deepspeed_adam'),
                                                                             (3, False, 'Adam'),
                                                                             (3, True, 'deepspeed_adam')])
    def test_load_optimizer_state(self, tmpdir, zero_stage, use_cpu_offload, adam_optimizer):
        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": 'Adam',
                "params": {
                    "lr": 0.00015,
                    "betas": [0.8, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "wall_clock_breakdown": True,
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        if zero_stage == 3:
            with deepspeed.zero.Init(config_dict_or_path=config_dict):
                models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]
        else:
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_optimizer_states=True)

    @pytest.mark.parametrize('zero_stage, use_cpu_offload, adam_optimizer', [(1, False, "Adam"), (2, False, "Adam"),
                                                                             (2, True, 'deepspeed_adam'),
                                                                             (3, False, 'Adam'),
                                                                             (3, True, 'deepspeed_adam')])
    def test_not_load_optimizer_state(self, tmpdir, zero_stage, use_cpu_offload, adam_optimizer):
        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": 'Adam',
                "params": {
                    "lr": 0.00015,
                    "betas": [0.8, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        if zero_stage == 3:
            global DeepSpeedZeroOptimizer_Stage3
            from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
            with deepspeed.zero.Init(config_dict_or_path=config_dict):
                models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]
        else:
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_optimizer_states=False)

    @pytest.mark.parametrize('zero_stage', [1, 2])
    def test_hybrid_optimizer_state(self, tmpdir, zero_stage):
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 2,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": zero_stage
            },
            "zero_allow_untested_optimizer": True,
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10
        models = [SimpleModel(hidden_dim=hidden_dim) for _ in range(2)]
        optimizers = [HybridStateOptimizer(model.parameters()) for model in models]

        checkpoint_correctness_verification(config_dict,
                                            models=models,
                                            base_optimizers=optimizers,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=True)

    @pytest.mark.parametrize('zero_stage', [0, 1, 2, 3])
    def test_load_module_only(self, tmpdir, zero_stage):
        if zero_stage == 0 and get_accelerator().device_name() == "cpu":
            pytest.skip("CPU Accelerator does not support this test")
        config_dict = {
            "train_batch_size": 2,
            "optimizer": {
                "type": 'Adam'
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        if zero_stage == 3:
            with deepspeed.zero.Init(config_dict_or_path=config_dict):
                models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]
        else:
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_module_only=True)


class ws4_model_checkpoint(DistributedFixture):
    world_size = 4

    def run(self, class_tmpdir, elastic_save, load_optim):
        config_dict = {
            "train_batch_size": 4,
            "optimizer": {
                "type": 'Adam'
            },
            "zero_optimization": {
                "stage": 2,
                "elastic_checkpoint": elastic_save
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10
        model = SimpleModel(hidden_dim)

        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model, total_samples=8, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        if load_optim:
            torch.save(model.optimizer.optimizer.state_dict(), os.path.join(class_tmpdir, 'opt-state-dict'))
        model.save_checkpoint(class_tmpdir)


class ws4_model_checkpoint_zeropp(DistributedFixture):

    world_size = 4

    def run(self, class_tmpdir):
        config_dict = {
            "train_batch_size": 4,
            "optimizer": {
                "type": 'Adam'
            },
            "zero_optimization": {
                "stage": 3,
                "zero_hpz_partition_size": 2,
            }
        }

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        for param in model.parameters():
            param.data = torch.ones_like(param.data, device=param.data.device, requires_grad=False)

        # save model and zero checkpoint
        torch.save(model.state_dict(), os.path.join(class_tmpdir, "model.pt"))
        ds_model = create_deepspeed_model(config_dict=config_dict, model=model, base_optimizer=None)
        ds_model.save_checkpoint(class_tmpdir)


# DistributedFixture that saves a ZeRO-3 elastic checkpoint from 4 GPUs so that
# TestZeROStage3ElasticCheckpoint (world_size=2) can test cross-world-size loading.
class ws4_zero3_elastic_checkpoint(DistributedFixture):
    world_size = 4

    def run(self, class_tmpdir):
        config_dict = {
            "train_batch_size": 4,
            "optimizer": {
                "type": "Adam"
            },
            "zero_optimization": {
                "stage": 3,
                "elastic_checkpoint": True
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            model = SimpleModel(hidden_dim)
        ds_model = create_deepspeed_model(config_dict=config_dict, model=model, base_optimizer=None)
        data_loader = random_dataloader(model=ds_model, total_samples=8, hidden_dim=hidden_dim, device=ds_model.device)
        for _, batch in enumerate(data_loader):
            loss = ds_model(batch[0], batch[1])
            ds_model.backward(loss)
            ds_model.step()
        ds_model.empty_partition_cache()
        ds_model.save_checkpoint(class_tmpdir)

        # Gather full fp32 parameters on rank 0 and save as a reference for the
        # cross-world-size numerical correctness test.
        ref_params = {}
        params = list(ds_model.module.parameters())
        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            if dist.get_rank() == 0:
                for name, p in ds_model.module.named_parameters():
                    ref_params[name] = p.detach().cpu().float().clone()
        if dist.get_rank() == 0:
            torch.save(ref_params, os.path.join(class_tmpdir, "reference_params.pt"))
        dist.barrier()


@pytest.mark.parametrize("elastic_save", [True, False])
@pytest.mark.parametrize("elastic_load", [True, False])
@pytest.mark.parametrize("load_optim", [True, False])
class TestZeROElasticCheckpoint(DistributedTest):
    world_size = 2

    def test_elastic_checkpoint_fixed_dp(self, tmpdir, elastic_save, elastic_load, load_optim):
        config_dict = {
            "train_batch_size": 2,
            "optimizer": {
                "type": 'Adam'
            },
            "zero_optimization": {
                "stage": 2,
                "elastic_checkpoint": elastic_save
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        # torch 1.2.* stores raw tensor id numbers in checkpoint state which leads to
        # false positive mismatches in checkpoint state comparisons.
        # Newer torch versions store tensor ids as 0, 1, 2, ...
        expected_mismatch_keys = [] if required_torch_version(min_version=1.4) else ['params']
        models = [SimpleModel(hidden_dim) for _ in range(2)]
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=models[0],
                                              model_parameters=models[0].parameters())
        run_steps = 8
        data_loader = random_dataloader(model=model,
                                        total_samples=run_steps,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
        if load_optim:
            opt_state_dict_file = f'opt-state-dict_rank{dist.get_rank()}'
            torch.save(model.optimizer.optimizer.state_dict(), os.path.join(tmpdir, opt_state_dict_file))
        model.save_checkpoint(tmpdir)

        config_dict["zero_optimization"]["elastic_checkpoint"] = elastic_load
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=models[1],
                                              model_parameters=models[1].parameters())
        model.load_checkpoint(tmpdir, load_optimizer_states=load_optim)

        if load_optim:
            saved_sd = torch.load(os.path.join(tmpdir, opt_state_dict_file), weights_only=False)
            curr_sd = model.optimizer.optimizer.state_dict()
            compare_opt_state_dicts(curr_sd, saved_sd, expected_mismatch_keys)

        data_loader = random_dataloader(model=model, total_samples=8, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    def test_elastic_checkpoint_change_dp(self, ws4_model_checkpoint, class_tmpdir, elastic_save, elastic_load,
                                          load_optim):
        config_dict = {
            "train_batch_size": 4,
            "optimizer": {
                "type": 'Adam'
            },
            "zero_optimization": {
                "stage": 2,
                "elastic_checkpoint": elastic_load
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10
        model = SimpleModel(hidden_dim)

        # Load checkpoint with dp world size = 2
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        if load_optim:
            with pytest.raises(deepspeed.runtime.zero.utils.ZeRORuntimeException):
                model.load_checkpoint(class_tmpdir, load_optimizer_states=load_optim)
        else:
            model.load_checkpoint(class_tmpdir, load_optimizer_states=load_optim)


class TestZeROSaveLoadEdgeCase(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('zero_stage', [0, 1, 2, 3])
    def test_immediate_save_load(self, tmpdir, zero_stage):
        config_dict = {
            "train_batch_size": 4,
            "optimizer": {
                "type": 'Adam'
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10
        model = SimpleModel(hidden_dim)

        ds_model = create_deepspeed_model(config_dict=config_dict, model=model, base_optimizer=None)
        ds_model.save_checkpoint(tmpdir)
        ds_model.load_checkpoint(tmpdir,
                                 load_optimizer_states=False,
                                 load_lr_scheduler_states=False,
                                 load_module_only=False)

    @pytest.mark.parametrize('zero_stage', [0, 1, 2, 3])
    def test_load_immediate_save(self, tmpdir, zero_stage):
        if zero_stage == 0 and get_accelerator().device_name() == "cpu":
            pytest.skip("CPU Accelerator does not support this test")
        config_dict = {
            "train_batch_size": 4,
            "optimizer": {
                "type": 'Adam'
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10
        model = SimpleModel(hidden_dim)

        # 1. pretrain a model and save it
        ds_model = create_deepspeed_model(config_dict=config_dict, model=model, base_optimizer=None)
        data_loader = random_dataloader(model=ds_model, total_samples=1, hidden_dim=hidden_dim, device=ds_model.device)
        for _, batch in enumerate(data_loader):
            loss = ds_model(batch[0], batch[1])
            ds_model.backward(loss)
            ds_model.step()

        ds_model.empty_partition_cache()
        ds_model.save_checkpoint(tmpdir)

        # 2. load and immediately save a model with a fresh ds engine
        ds_model = create_deepspeed_model(config_dict=config_dict, model=model, base_optimizer=None)
        ds_model.load_checkpoint(tmpdir,
                                 load_optimizer_states=False,
                                 load_lr_scheduler_states=False,
                                 load_module_only=False)
        ds_model.save_checkpoint(tmpdir)

    @pytest.mark.parametrize('zero_stage', [0, 1, 2, 3])
    def test_save_before_accum_grad_is_done(self, tmpdir, zero_stage):
        config_dict = {
            "optimizer": {
                "type": 'Adam'
            },
            "zero_optimization": {
                "stage": zero_stage,
                "stage3_gather_fp16_weights_on_model_save": True,
            },
            "gradient_accumulation_steps": 2,
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": 4,
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10
        model = SimpleModel(hidden_dim)

        # This test reproduces a bug where one tries to retrieve a 16bit model before grad_accum
        # cycle was completed.
        # So we config grad_accum=2 and step only once and save_16bit_model
        ds_model = create_deepspeed_model(config_dict=config_dict, model=model, base_optimizer=None)

        data_loader = random_dataloader(model=ds_model, total_samples=2, hidden_dim=hidden_dim, device=ds_model.device)

        batch = next(iter(data_loader))
        loss = ds_model(batch[0], batch[1])
        ds_model.backward(loss)
        ds_model.step()

        ds_model.empty_partition_cache()

        # we stepped only once, and now save 16bit model before gradient_accumulation_steps=2 is complete
        ds_model.save_16bit_model(tmpdir, "model.pt")

        # let's test just as well that we can save the checkpoint too
        ds_model.save_checkpoint(tmpdir)


class TestZeROCheckpointFrozenWeights(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    def test_load_optimizer_state(self, tmpdir, zero_stage):

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": 'Adam',
                "params": {
                    "lr": 0.00015,
                    "betas": [0.8, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "wall_clock_breakdown": True,
            "zero_optimization": {
                "stage": zero_stage
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        with deepspeed.zero.Init(enabled=zero_stage == 3, config_dict_or_path=config_dict):
            models = [SimpleFrozenModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_optimizer_states=True)

    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    def test_not_load_optimizer_state(self, tmpdir, zero_stage):

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": 'Adam',
                "params": {
                    "lr": 0.00015,
                    "betas": [0.8, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "zero_optimization": {
                "stage": zero_stage
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True}
        hidden_dim = 10

        with deepspeed.zero.Init(enabled=zero_stage == 3, config_dict_or_path=config_dict):
            models = [SimpleFrozenModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_optimizer_states=False)

    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    def test_load_module_only(self, tmpdir, zero_stage):
        config_dict = {
            "train_batch_size": 2,
            "optimizer": {
                "type": 'Adam'
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        with deepspeed.zero.Init(enabled=zero_stage == 3, config_dict_or_path=config_dict):
            models = [SimpleFrozenModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_module_only=True)

    @pytest.mark.parametrize('zero_stage', [1, 2])
    def test_save_exclude_frozen_weights(self, tmpdir, zero_stage):
        world_size = 1
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": 'Adam'
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        model = SimpleFrozenModel(hidden_dim, empty_grad=False)

        ds_engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

        # Validate backwards-compatibility of including frozen parameters in checkpoint
        all_ckpt_folder = os.path.join(tmpdir, 'all_params')
        ds_engine.save_checkpoint(all_ckpt_folder)
        all_params_ckpt_file = get_model_ckpt_name_for_rank(os.path.join(all_ckpt_folder, 'global_step0'), '00')
        loaded_all_param_model = torch.load(all_params_ckpt_file, weights_only=False)['module']
        all_param_names = set([n for n, p in model.named_parameters()])
        assert set(loaded_all_param_model.keys()) == all_param_names

        # Validate exclusion of frozen parameters
        trainable_ckpt_folder = os.path.join(tmpdir, 'no_frozen_params')
        ds_engine.save_checkpoint(trainable_ckpt_folder, exclude_frozen_parameters=True)

        trainable_ckpt_file = get_model_ckpt_name_for_rank(os.path.join(trainable_ckpt_folder, 'global_step0'), '00')

        # Excluding frozen parameters should reduce checkpoint size
        assert os.path.getsize(all_params_ckpt_file) > os.path.getsize(trainable_ckpt_file)

        loaded_trainable_param_model = torch.load(trainable_ckpt_file, weights_only=False)['module']
        frozen_param_names = set([n for n, p in model.named_parameters() if not p.requires_grad])
        loaded_trainable_param_names = set(loaded_trainable_param_model.keys())
        overlap_names = set.intersection(loaded_trainable_param_names, frozen_param_names)
        assert len(overlap_names) == 0

        trainable_param_names = set([n for n, p in model.named_parameters() if p.requires_grad])
        assert loaded_trainable_param_names == trainable_param_names

    @pytest.mark.parametrize('zero_stage', [1, 2])
    def test_save_exclude_custom_frozen_weights(self, tmpdir, zero_stage):
        world_size = 1
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": 'Adam'
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        model = SimpleFrozenModel(hidden_dim, empty_grad=False)

        ds_engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

        # Validate custom state_dict model
        state_dict_bk = model.state_dict
        model.state_dict = model.custom_state_dict
        custom_state_dict_ckpt_folder = os.path.join(tmpdir, 'custom_state_dict')
        ds_engine.save_checkpoint(custom_state_dict_ckpt_folder, exclude_frozen_parameters=True)

        custom_state_dict_ckpt_file = get_model_ckpt_name_for_rank(
            os.path.join(custom_state_dict_ckpt_folder, 'global_step0'), '00')
        loaded_custom_state_dict_param_model = torch.load(custom_state_dict_ckpt_file, weights_only=False)['module']
        loaded_custom_state_dict_param_names = set(loaded_custom_state_dict_param_model.keys())

        custom_state_dict_param_names = set([k for k, v in model.state_dict().items()])
        trainable_param_names = set([n for n, p in model.named_parameters() if p.requires_grad])
        overlap_names = set.intersection(custom_state_dict_param_names, trainable_param_names)

        assert loaded_custom_state_dict_param_names == overlap_names

        model.state_dict = state_dict_bk


class TestSaveTensorClone(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize('zero_stage', [1, 2])
    @pytest.mark.parametrize('use_cpu_device', [True, False])
    def test_save_tensor_clone(self, tmpdir, zero_stage, use_cpu_device):

        config_dict = {
            "optimizer": {
                "type": "AdamW",
            },
            "zero_optimization": {
                "stage": zero_stage
            },
            "train_batch_size": 1,
            "train_micro_batch_size_per_gpu": 1
        }
        hidden_dim = 1024
        model = SimpleModel(hidden_dim, nlayers=4).half()
        ref_model_state_dict = model.state_dict()

        ds_engine, _, _, _ = deepspeed.initialize(model=model, config_params=config_dict)
        clone_device = torch.device('cpu') if use_cpu_device else get_accelerator().current_device()
        clone_state_dict = clone_tensors_for_torch_save(ds_engine.module.state_dict())
        compare_state_dicts(ref_model_state_dict, clone_state_dict)

        ref_ckpt_file = os.path.join(tmpdir, 'ref_ckpt.pt')
        torch.save(ref_model_state_dict, ref_ckpt_file)
        clone_ckpt_file = os.path.join(tmpdir, 'clone_ckpt.pt')
        torch.save(clone_state_dict, clone_ckpt_file)

        compare_state_dicts(torch.load(ref_ckpt_file, weights_only=False),
                            torch.load(clone_ckpt_file, weights_only=False))


class TestZeRONonDistributed(DistributedTest):
    world_size = 1
    init_distributed = False

    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    def test_chmod_exception_handling(self, monkeypatch, zero_stage):

        config_dict = {
            "optimizer": {
                "type": "AdamW"
            },
            "train_batch_size": 1,
            "zero_optimization": {
                "stage": zero_stage
            }
        }
        args = SimpleNamespace(local_rank=0)
        net = SimpleModel(hidden_dim=4)
        engine, _, _, _ = deepspeed.initialize(args=args,
                                               config=config_dict,
                                               model=net,
                                               model_parameters=net.parameters())

        log_called = False

        def mock_logger_info(message, *args, **kwargs):
            nonlocal log_called
            log_called = True

        monkeypatch.setattr("deepspeed.utils.logger.info", mock_logger_info)
        """
            This is presented for use-cases like Azure Storage File Share (where permissions are not allowed)
            We use a fake file for this test (file not existing would present a similar issue as not being able to chmod)
        """
        fake_recovery_script_dst = os.path.join("tmp", "zero_to_fp32.py")
        engine._change_recovery_script_permissions(fake_recovery_script_dst)

        assert log_called, "Expected deepspeed.utils.logger.info to be called."


class TestZeROPPLoadCheckpoint(DistributedTest):

    world_size = 4

    def test_load_zeropp_model(self, ws4_model_checkpoint_zeropp, class_tmpdir):
        config_dict = {
            "train_batch_size": 4,
            "optimizer": {
                "type": 'Adam'
            },
            "zero_optimization": {
                "stage": 3,
                "zero_hpz_partition_size": 2,
                "stage3_param_persistence_threshold": 1
            }
        }

        # Init model and load saved model
        hidden_dim = 10
        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            model = SimpleModel(hidden_dim)
        ds_model = create_deepspeed_model(config_dict=config_dict, model=model, base_optimizer=None)

        with deepspeed.zero.GatheredParameters(ds_model.module.parameters(), modifier_rank=0):
            if dist.get_rank() == 0:
                state_dict = torch.load(os.path.join(class_tmpdir, "model.pt"))
                ds_model.module.load_state_dict(state_dict)

        # Check the parameters after gather
        params_to_gather = [p for p in ds_model.module.parameters() if p.ds_status == ZeroParamStatus.NOT_AVAILABLE]
        if len(params_to_gather) > 0:
            handle = params_to_gather[0].all_gather_coalesced(params_to_gather)
            handle.wait()
        for ds_param in params_to_gather:
            for v in ds_param.data.cpu().flatten().numpy():
                assert v == 1.0


class TestZeROStage3ElasticCheckpoint(DistributedTest):
    """Unit tests for ZeRO Stage 3 elastic checkpoint save/load."""

    world_size = 2

    def test_elastic_checkpoint_same_world_size(self, tmpdir):
        # Round-trip: save elastic checkpoint with world_size=2, load with world_size=2.
        # Verifies that both model weights and optimizer states are faithfully restored.
        config_dict = {
            "train_batch_size": 2,
            "optimizer": {
                "type": "Adam"
            },
            "zero_optimization": {
                "stage": 3,
                "elastic_checkpoint": True
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            models = [SimpleModel(hidden_dim) for _ in range(2)]
        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_optimizer_states=True)

    def test_elastic_checkpoint_no_optimizer_states(self, tmpdir):
        # Round-trip: save elastic checkpoint, load without optimizer states.
        # Only model weights should be compared; optimizer state is intentionally skipped.
        config_dict = {
            "train_batch_size": 2,
            "optimizer": {
                "type": "Adam"
            },
            "zero_optimization": {
                "stage": 3,
                "elastic_checkpoint": True
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            models = [SimpleModel(hidden_dim) for _ in range(2)]
        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_optimizer_states=False)

    def test_elastic_state_dict_format(self, tmpdir):
        # Verify that elastic_checkpoint=True produces a state dict with BASE_OPTIMIZER_STATE
        # (list of per-param lean tensors per sub-group) and not the rigid OPTIMIZER_STATE_DICT.
        config_dict = {
            "train_batch_size": 2,
            "optimizer": {
                "type": "Adam"
            },
            "zero_optimization": {
                "stage": 3,
                "elastic_checkpoint": True
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            model = SimpleModel(hidden_dim)
        ds_model = create_deepspeed_model(config_dict=config_dict, model=model, base_optimizer=None)
        data_loader = random_dataloader(model=ds_model, total_samples=4, hidden_dim=hidden_dim, device=ds_model.device)
        for _, batch in enumerate(data_loader):
            loss = ds_model(batch[0], batch[1])
            ds_model.backward(loss)
            ds_model.step()

        sd = ds_model.optimizer.state_dict()
        assert BASE_OPTIMIZER_STATE in sd
        assert OPTIMIZER_STATE_DICT not in sd
        assert FP32_FLAT_GROUPS in sd
        # Each sub-group entry in FP32_FLAT_GROUPS should be a flat fp32 tensor (same layout as
        # the rigid format so that zero_to_fp32.py can reconstruct model weights unchanged).
        import torch
        for sub_group in sd[FP32_FLAT_GROUPS]:
            assert torch.is_tensor(sub_group)

    def test_rigid_state_dict_format(self, tmpdir):
        # Verify that elastic_checkpoint=False (default) produces a state dict with
        # OPTIMIZER_STATE_DICT and not BASE_OPTIMIZER_STATE.
        config_dict = {
            "train_batch_size": 2,
            "optimizer": {
                "type": "Adam"
            },
            "zero_optimization": {
                "stage": 3,
                "elastic_checkpoint": False
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            model = SimpleModel(hidden_dim)
        ds_model = create_deepspeed_model(config_dict=config_dict, model=model, base_optimizer=None)
        data_loader = random_dataloader(model=ds_model, total_samples=4, hidden_dim=hidden_dim, device=ds_model.device)
        for _, batch in enumerate(data_loader):
            loss = ds_model(batch[0], batch[1])
            ds_model.backward(loss)
            ds_model.step()

        sd = ds_model.optimizer.state_dict()
        assert OPTIMIZER_STATE_DICT in sd
        assert BASE_OPTIMIZER_STATE not in sd

    def test_elastic_format_autodetected_on_load(self, tmpdir):
        # A checkpoint saved with elastic_checkpoint=True must load correctly even when the
        # loading engine is configured with elastic_checkpoint=False, because load_state_dict()
        # auto-detects the format via the BASE_OPTIMIZER_STATE key.
        save_config = {
            "train_batch_size": 2,
            "optimizer": {
                "type": "Adam"
            },
            "zero_optimization": {
                "stage": 3,
                "elastic_checkpoint": True
            }
        }
        load_config = {
            "train_batch_size": 2,
            "optimizer": {
                "type": "Adam"
            },
            "zero_optimization": {
                "stage": 3,
                "elastic_checkpoint": False
            }
        }
        if get_accelerator().is_bf16_supported():
            save_config["bf16"] = {"enabled": True}
            load_config["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            save_config["fp16"] = {"enabled": True, "initial_scale_power": 8}
            load_config["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        with deepspeed.zero.Init(config_dict_or_path=save_config):
            model_save = SimpleModel(hidden_dim)
        with deepspeed.zero.Init(config_dict_or_path=load_config):
            model_load = SimpleModel(hidden_dim)

        ds_save = create_deepspeed_model(config_dict=save_config, model=model_save, base_optimizer=None)
        data_loader = random_dataloader(model=ds_save, total_samples=4, hidden_dim=hidden_dim, device=ds_save.device)
        for _, batch in enumerate(data_loader):
            loss = ds_save(batch[0], batch[1])
            ds_save.backward(loss)
            ds_save.step()
        ds_save.empty_partition_cache()
        ds_save.save_checkpoint(tmpdir)

        dist.barrier()

        ds_load = create_deepspeed_model(config_dict=load_config, model=model_load, base_optimizer=None)
        ds_load.load_checkpoint(tmpdir, load_optimizer_states=True)
        compare_model_states(ds_save, ds_load, compare_optimizer=True)

    def test_elastic_checkpoint_change_world_size(self, ws4_zero3_elastic_checkpoint, class_tmpdir):
        # Load a ZeRO-3 elastic checkpoint saved with 4 GPUs onto a 2-GPU engine.
        # Verifies both that loading succeeds and that the repartitioned parameters
        # are numerically identical to the reference weights saved by the fixture.
        config_dict = {
            "train_batch_size": 4,
            "optimizer": {
                "type": "Adam"
            },
            "zero_optimization": {
                "stage": 3,
                "elastic_checkpoint": True
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            model = SimpleModel(hidden_dim)
        ds_model = create_deepspeed_model(config_dict=config_dict, model=model, base_optimizer=None)
        ds_model.load_checkpoint(class_tmpdir, load_optimizer_states=True)

        # Compare gathered parameters against the fp32 reference saved by the fixture.
        ref_params = torch.load(os.path.join(class_tmpdir, "reference_params.pt"), weights_only=False)
        params = list(ds_model.module.parameters())
        with deepspeed.zero.GatheredParameters(params):
            if dist.get_rank() == 0:
                for name, p in ds_model.module.named_parameters():
                    assert torch.allclose(p.data.float(), ref_params[name], rtol=1e-3, atol=1e-3), \
                        f"Parameter '{name}' mismatch after cross-world-size elastic load"

        # Confirm training can proceed normally after cross-world-size checkpoint load.
        data_loader = random_dataloader(model=ds_model, total_samples=4, hidden_dim=hidden_dim, device=ds_model.device)
        for _, batch in enumerate(data_loader):
            loss = ds_model(batch[0], batch[1])
            ds_model.backward(loss)
            ds_model.step()

    def test_elastic_checkpoint_load_from_fp32_weights(self, tmpdir):
        # Verify the load_from_fp32_weights=True path: _restore_from_elastic_fp32_partitions()
        # is exercised instead of _restore_from_bit16_weights().
        config_dict = {
            "train_batch_size": 2,
            "optimizer": {
                "type": "Adam"
            },
            "zero_optimization": {
                "stage": 3,
                "elastic_checkpoint": True,
                "load_from_fp32_weights": True,
            }
        }
        if get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        elif get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        hidden_dim = 10

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            models = [SimpleModel(hidden_dim) for _ in range(2)]
        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_optimizer_states=True)

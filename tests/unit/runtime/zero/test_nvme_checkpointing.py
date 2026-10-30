# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import pytest
import deepspeed.comm as dist
import torch

from unit.common import DistributedTest
from unit.simple_model import random_dataloader, SimpleModel

import deepspeed
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.partition_parameters import Init
from deepspeed.runtime.zero.utils import ZeRORuntimeException
from deepspeed.ops.aio import AsyncIOBuilder
from deepspeed.accelerator import get_accelerator


@pytest.mark.sequential
class TestNVMeCheckpointing(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize('param_offload_device, optim_offload_device',
                             [(OffloadDeviceEnum.none, OffloadDeviceEnum.nvme),
                              (OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme),
                              (OffloadDeviceEnum.nvme, OffloadDeviceEnum.none),
                              (OffloadDeviceEnum.nvme, OffloadDeviceEnum.cpu),
                              (OffloadDeviceEnum.nvme, OffloadDeviceEnum.nvme)])
    def test_nvme_checkpointing(self, tmpdir, param_offload_device, optim_offload_device):
        zero_dir, ckpt_dir = os.path.join(tmpdir, "zero"), os.path.join(tmpdir, "checkpoint")

        first_stage_steps, second_stage_steps = 2, 2

        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")

        if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
            pytest.skip('Skip tests since async-io is not compatible')

        torch.manual_seed(123)

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015,
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": param_offload_device,
                    "nvme_path": str(zero_dir)
                },
                "offload_optimizer": {
                    "device": optim_offload_device,
                    "nvme_path": str(zero_dir)
                },
                "sub_group_size": 100,
                "stage3_max_live_parameters": 100,
                "stage3_param_persistence_threshold": 0,
            },
            "aio": {
                "block_size": 1048576  # Minimum AIO bytes, anything smaller than this will not be offloaded
            }
        }

        hidden_dim, nlayers = 2048, 2
        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            model = SimpleModel(hidden_dim, nlayers=nlayers, empty_grad=False)

        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        model.empty_partition_cache()

        assert first_stage_steps > 0

        data_loader = random_dataloader(model=model,
                                        total_samples=first_stage_steps,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        dist.barrier()
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        dist.barrier()
        model.save_checkpoint(ckpt_dir)

        if second_stage_steps > 0:
            second_stage_batches = list(
                random_dataloader(model=model,
                                  total_samples=second_stage_steps,
                                  hidden_dim=hidden_dim,
                                  device=model.device,
                                  dtype=torch.float16))
            dist.barrier()
            for n, batch in enumerate(second_stage_batches):
                loss = model(batch[0], batch[1])
                model.backward(loss)
                model.step()
            dist.barrier()

        final_batch = next(
            iter(
                random_dataloader(model=model,
                                  total_samples=1,
                                  hidden_dim=hidden_dim,
                                  device=model.device,
                                  dtype=torch.float16)))
        dist.barrier()
        loss_before = float(model(final_batch[0], final_batch[1]))

        # Needed in ZeRO 3. Not doing so can give memory leak
        model.destroy()

        # TODO: This should be on the engine? There needs to be a better way.
        Init.param_id = 0

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            model = SimpleModel(hidden_dim, nlayers=nlayers, empty_grad=False)

        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

        model.load_checkpoint(ckpt_dir)

        if second_stage_steps > 0:
            dist.barrier()
            for n, batch in enumerate(second_stage_batches):
                loss = model(batch[0], batch[1])
                model.backward(loss)
                model.step()
            dist.barrier()

        dist.barrier()
        loss_after = float(model(final_batch[0], final_batch[1]))

        assert loss_before == loss_after

    @pytest.mark.parametrize('param_offload_device', [OffloadDeviceEnum.none, OffloadDeviceEnum.nvme])
    @pytest.mark.parametrize('pipeline', [False, True])
    @pytest.mark.parametrize('precision', ['bf16', 'fp16'])
    @pytest.mark.parametrize('load_optimizer_states', [False, True])
    def test_load_module_only(self, tmpdir, param_offload_device, pipeline, precision, load_optimizer_states):
        # Catch both empty swapped masters and checkpoint files overwriting refreshed masters (#8890).
        from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_optimizer_state

        if not getattr(get_accelerator(), f'is_{precision}_supported')():
            pytest.skip(f'{precision} is not supported')
        if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
            pytest.skip('async-io is not compatible')

        hidden_dim = 1024
        dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.001,
                    "torch_adam": True
                }
            },
            "zero_force_ds_cpu_optimizer": False,
            precision: {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": param_offload_device,
                    "max_in_cpu": 0,
                },
                "offload_optimizer": {
                    "device": OffloadDeviceEnum.nvme,
                    "pipeline_read": pipeline,
                    "pipeline_write": pipeline,
                },
                "sub_group_size": 100,
                "stage3_param_persistence_threshold": 0,
            },
            "aio": {
                "block_size": 1048576
            },
        }
        if precision == 'fp16':
            config[precision]['loss_scale'] = 1.0

        def initialize(name):
            for key in ('offload_param', 'offload_optimizer'):
                config['zero_optimization'][key]['nvme_path'] = os.path.join(tmpdir, name)
            with deepspeed.zero.Init(config_dict_or_path=config):
                module = SimpleModel(hidden_dim, empty_grad=False)
            engine, _, _, _ = deepspeed.initialize(model=module, model_parameters=module.parameters(), config=config)
            return engine

        def train_step(engine, batch):
            loss = engine(*batch)
            assert torch.isfinite(loss)
            engine.backward(loss)
            engine.step()

        torch.manual_seed(17)
        source = initialize('source')
        batches = list(
            random_dataloader(source, total_samples=3, hidden_dim=hidden_dim, device=source.device, dtype=dtype))
        for batch in batches[:2]:
            train_step(source, batch)
        with deepspeed.zero.GatheredParameters(list(source.module.parameters())):
            expected = [param.detach().clone() for param in source.module.parameters()]
        checkpoint_dir = os.path.join(tmpdir, 'checkpoint')
        source.save_checkpoint(checkpoint_dir, tag='module-only')
        source.destroy()
        Init.param_id = 0

        target = initialize('target')
        try:
            train_step(target, batches[2])
            parameters = list(target.module.parameters())
            swappable = [
                target.optimizer._swappable_optimizer_subgroup(i)
                for i in range(len(target.optimizer.fp32_partitioned_groups_flat))
            ]
            assert any(swappable) and not all(swappable)
            moments = [safe_get_full_optimizer_state(param, 'exp_avg').clone() for param in parameters]
            # Reading the states releases the buffers, including for the pipelined swapper.
            assert any(master.numel() == 0 for master in target.optimizer.fp32_partitioned_groups_flat)
            for _ in range(2):
                target.load_checkpoint(checkpoint_dir,
                                       tag='module-only',
                                       load_module_only=True,
                                       load_optimizer_states=load_optimizer_states)
                with deepspeed.zero.GatheredParameters(parameters):
                    for param, saved in zip(parameters, expected):
                        torch.testing.assert_close(param, saved, rtol=0, atol=0)
                for param, saved, moment in zip(parameters, expected, moments):
                    torch.testing.assert_close(safe_get_full_fp32_param(param), saved.float(), rtol=0, atol=0)
                    torch.testing.assert_close(safe_get_full_optimizer_state(param, 'exp_avg'), moment, rtol=0, atol=0)
            before_step = [safe_get_full_fp32_param(param).clone() for param in parameters]
            if not load_optimizer_states and precision == 'bf16' and not pipeline:
                with pytest.raises(ZeRORuntimeException, match='custom_load_fn'):
                    target.load_checkpoint(checkpoint_dir,
                                           tag='module-only',
                                           load_module_only=True,
                                           custom_load_fn=lambda **kwargs: pytest.fail('callback must not run'))
                model_path = target._get_ckpt_name(checkpoint_dir, 'module-only')
                checkpoint = torch.load(model_path, weights_only=False)
                shapes = checkpoint['param_shapes']
                checkpoint['param_shapes'] = [{
                    name + '.renamed': shape
                    for name, shape in group.items()
                } for group in shapes]
                torch.save(checkpoint, model_path)
                try:
                    with pytest.raises(ZeRORuntimeException, match='parameter names'):
                        target.load_checkpoint(checkpoint_dir, tag='module-only', load_module_only=True)
                finally:
                    checkpoint['param_shapes'] = shapes
                    torch.save(checkpoint, model_path)
                for param, before in zip(parameters, before_step):
                    torch.testing.assert_close(safe_get_full_fp32_param(param), before, rtol=0, atol=0)
            train_step(target, batches[0])
            after_step = [safe_get_full_fp32_param(param) for param in parameters]
            assert all(torch.isfinite(value).all() for value in after_step)
            assert any(not torch.equal(before, after) for before, after in zip(before_step, after_step))
        finally:
            target.destroy()

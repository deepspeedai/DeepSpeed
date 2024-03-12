# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import deepspeed.comm as dist
import torch

from unit.common import DistributedTest, preferred_dtype
from unit.simple_model import random_dataloader, SimpleModel
from unit.util import bf16_required_version_check

import deepspeed
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad, safe_get_full_optimizer_state
from deepspeed.utils import safe_set_full_fp32_param, safe_set_full_optimizer_state
from deepspeed.utils import safe_get_local_fp32_param, safe_get_local_grad, safe_get_local_optimizer_state
from deepspeed.utils import safe_set_local_fp32_param, safe_set_local_optimizer_state
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.ops.aio import AsyncIOBuilder
from deepspeed.accelerator import get_accelerator

WEIGHT_KEY = 'weight'
FIRST_ORDER_KEY = 'exp_avg'
SECOND_ORDER_KEY = 'exp_avg_sq'


def validate_tensor(model, api_type, opt_states):
    assert api_type in ["full", "local"]
    for _, lp in model.named_parameters():
        param_list = []
        if opt_states:
            param_list.append(
                safe_get_full_optimizer_state(lp, 'exp_avg') if api_type ==
                "full" else safe_get_local_optimizer_state(lp, 'exp_avg'))
            param_list.append(
                safe_get_full_optimizer_state(lp, 'exp_avg_sq') if api_type ==
                "full" else safe_get_local_optimizer_state(lp, 'exp_avg_sq'))
        else:
            param_list.append(safe_get_full_fp32_param(lp) if api_type == "full" else safe_get_local_fp32_param(lp))
            param_list.append(safe_get_full_grad(lp) if api_type == "full" else safe_get_local_grad(lp))
        if lp.requires_grad:
            assert all([p is not None for p in param_list])
        else:
            assert all([p is None for p in param_list])


class MyModel(torch.nn.Module):

    def __init__(self, hidden_dim, frozen_weights):
        super(MyModel, self).__init__()
        self.act = torch.nn.ReLU()
        self.cel = torch.nn.CrossEntropyLoss()
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, 1),
             torch.nn.Linear(1, 1),
             torch.nn.Linear(1, hidden_dim)])
        if frozen_weights:
            self.linears[0].weight.requires_grad = False
            self.linears[0].bias.requires_grad = False

    def forward(self, x, y):
        for l in self.linears:
            x = l(x)
            x = self.act(x)
        return self.cel(x, y)


def run_fragmented_model(model, config_dict, hidden_dim, dtype, validate_after_bwd, validate_after_step):
    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
    data_loader = random_dataloader(model=model,
                                    total_samples=10,
                                    hidden_dim=hidden_dim,
                                    device=model.device,
                                    dtype=dtype)
    dist.barrier()
    for n, batch in enumerate(data_loader):
        loss = model(batch[0], batch[1])
        model.backward(loss)
        validate_after_bwd(model)
        model.step()
        validate_after_step(model)

    # Needed in ZeRO 3. Not doing so can give memory leak
    model.destroy()


@pytest.mark.parametrize('frozen_weights', [True, False])
class TestTensorFragmentGet(DistributedTest):
    # Need multiple gpus to test possible hanging
    world_size = 2
    reuse_dist_env = True

    @pytest.mark.parametrize('api_type', ['local', 'full'])
    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    @pytest.mark.parametrize('offload_device', [OffloadDeviceEnum.none, OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme])
    def test_zero_fragments(self, tmpdir, api_type, zero_stage, offload_device, frozen_weights):
        if offload_device == OffloadDeviceEnum.nvme:
            if zero_stage != 3:
                pytest.skip(f"Nvme offload not supported for zero stage {zero_stage}")
            if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
                pytest.skip('Skip tests since async-io is not compatible')

        if api_type == "local" and zero_stage != 3:
            pytest.skip(f"Local APIs only for zero stage 3 but current stage is {zero_stage}")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }
        if get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 2}
        elif get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}

        if offload_device == OffloadDeviceEnum.cpu:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": offload_device}
        elif offload_device == OffloadDeviceEnum.nvme:
            config_dict["zero_optimization"]["offload_optimizer"] = {
                "device": offload_device,
                "nvme_path": str(tmpdir)
            }

        hidden_dim = 128
        if zero_stage == 3:
            with deepspeed.zero.Init(config_dict_or_path=config_dict):
                model = MyModel(hidden_dim, frozen_weights)
        else:
            model = MyModel(hidden_dim, frozen_weights)

        validate_after_bwd = lambda model: validate_tensor(model, api_type, opt_states=False)
        validate_after_step = lambda model: validate_tensor(model, api_type, opt_states=True)

        run_fragmented_model(model, config_dict, hidden_dim, preferred_dtype(), validate_after_bwd,
                             validate_after_step)

    def test_bf16_fragments(self, frozen_weights):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU accelerator does not support this test yet.")
        if frozen_weights:
            pytest.skip("TODO: Frozen weights not currently supported by BF16 Optimizer")

        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 0,
            }
        }

        hidden_dim = 128
        model = MyModel(hidden_dim, frozen_weights)

        api_type = "full"
        validate_after_bwd = lambda model: validate_tensor(model, api_type, opt_states=False)
        validate_after_step = lambda model: validate_tensor(model, api_type, opt_states=True)

        run_fragmented_model(model, config_dict, hidden_dim, torch.bfloat16, validate_after_bwd, validate_after_step)


def create_random_values(model, key_list, group, use_cuda=True):
    param_values = {}
    for n, lp in model.named_parameters():
        param_shape = lp.ds_shape if hasattr(lp, 'ds_id') else lp.shape
        param_values[n] = {}
        for key in key_list:
            rand_value = torch.rand(param_shape, dtype=torch.float32, device=model.device)
            dist.broadcast(rand_value, src=0, group=group)
            param_values[n][key] = rand_value
    return param_values


def set_param_values_with_dict(model, value_dict):
    for n, lp in model.named_parameters():
        for key, value_tensor in value_dict[n].items():
            if key == WEIGHT_KEY:
                safe_set_full_fp32_param(lp, value_tensor)
            else:
                safe_set_full_optimizer_state(lp, value_tensor, key)


def validate_param_values_with_dict(model, value_dict):
    for n, lp in model.named_parameters():
        for key, expected_tensor in value_dict[n].items():
            if key == WEIGHT_KEY:
                actual_tensor = safe_get_full_fp32_param(lp)
            else:
                actual_tensor = safe_get_full_optimizer_state(lp, key)
            assert torch.equal(expected_tensor, actual_tensor)


def create_random_values_for_local(model, key_list, group, use_cuda=True):
    param_values = {}
    for n, lp in model.named_parameters():
        param_shape = lp.ds_tensor.shape
        param_values[n] = {}
        for key in key_list:
            device = model.device if use_cuda else "cpu"
            rand_value = torch.rand(param_shape, dtype=torch.float32, device=device)
            # dist.broadcast(rand_value, src=0, group=group)
            param_values[n][key] = rand_value
    return param_values


def set_local_param_values_with_dict(model, value_dict):
    for n, lp in model.named_parameters():

        for key, value_tensor in value_dict[n].items():
            if key == WEIGHT_KEY:
                safe_set_local_fp32_param(lp, value_tensor)
            else:
                safe_set_local_optimizer_state(lp, value_tensor, key)


def validate_local_param_values_with_dict(model, value_dict):
    for n, lp in model.named_parameters():
        for key, expected_tensor in value_dict[n].items():
            if key == WEIGHT_KEY:
                actual_tensor = safe_get_local_fp32_param(lp)
            else:
                actual_tensor = safe_get_local_optimizer_state(lp, key)
            assert torch.equal(expected_tensor, actual_tensor)


helper_funcs_mapping = {
    "full": {
        "create_random_values": create_random_values,
        "set_param_values_with_dict": set_param_values_with_dict,
        "validate_param_values_with_dict": validate_param_values_with_dict
    },
    "local": {
        "create_random_values": create_random_values_for_local,
        "set_param_values_with_dict": set_local_param_values_with_dict,
        "validate_param_values_with_dict": validate_local_param_values_with_dict
    }
}


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
class TestTensorFragmentUpdate(DistributedTest):
    # Need multiple gpus to test possible hanging
    world_size = 2
    reuse_dist_env = True

    @pytest.mark.parametrize('api_type', ['local', 'full'])
    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    @pytest.mark.parametrize('offload_device', [OffloadDeviceEnum.none, OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme])
    def test_zero_fragments(self, tmpdir, api_type, zero_stage, offload_device, dtype):

        if dtype == torch.bfloat16 and not bf16_required_version_check(accelerator_check=False):
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        if api_type == "local" and zero_stage != 3:
            pytest.skip(f"Local APIs only for zero stage 3 but current stage is {zero_stage}")

        if offload_device == OffloadDeviceEnum.nvme:
            if zero_stage != 3:
                pytest.skip(f"Nvme offload not supported for zero stage {zero_stage}")
            if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
                pytest.skip('Skip tests since async-io is not compatible')

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }

        if offload_device == OffloadDeviceEnum.cpu:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": offload_device}
        elif offload_device == OffloadDeviceEnum.nvme:
            config_dict["zero_optimization"]["offload_optimizer"] = {
                "device": offload_device,
                "nvme_path": str(tmpdir)
            }

        if dtype == torch.float16:
            if not get_accelerator().is_fp16_supported():
                pytest.skip("fp16 is not supported")
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        elif dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        hidden_dim = 128
        if zero_stage == 3:
            config_dict["zero_optimization"]["param_persistence_threshold"] = hidden_dim
            with deepspeed.zero.Init(config_dict_or_path=config_dict):
                model = SimpleModel(hidden_dim)
        else:
            model = SimpleModel(hidden_dim)

        world = dist.get_world_size()
        group = dist.new_group(ranks=list(range(world)))

        dist.barrier()

        def validate_func(model):
            optim_keys = [WEIGHT_KEY, FIRST_ORDER_KEY, SECOND_ORDER_KEY]
            helper_funcs = helper_funcs_mapping[api_type]
            optim_state_values = helper_funcs["create_random_values"](
                model, optim_keys, group, use_cuda=offload_device == OffloadDeviceEnum.none)
            helper_funcs["set_param_values_with_dict"](model, optim_state_values)
            helper_funcs["validate_param_values_with_dict"](model, optim_state_values)

        run_fragmented_model(model, config_dict, hidden_dim, dtype, lambda _: None, validate_func)

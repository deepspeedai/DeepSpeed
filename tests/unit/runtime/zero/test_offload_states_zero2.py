# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest
from unit.simple_model import random_dataloader, SimpleModel
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum, OffloadStateTypeEnum

def validate_hp_params_device(model, device: torch.device):
    """Validates that the sharded FP32 parameters are on the specified device."""
    for p in model.optimizer.single_partition_of_fp32_groups:
        assert p.device == device, f"FP32 param partition is on {p.device}, expected {device}"

# Validation function to check the device of LP (FP16/BF16) model parameters.
def validate_lp_params_device(model, device: torch.device):
    """Validates that the sharded LP parameters are on the specified device."""
    for p in model.parameters():
        assert p.device == device, f"LP param partition is on {p.device}, expected {device}"

def validate_adam_states_device(model, device: torch.device):
    """Validates that the sharded Adam optimizer states are on the specified device."""
    for p in model.optimizer.single_partition_of_fp32_groups:
        if p in model.optimizer.state:
            for state_key in ['exp_avg', 'exp_avg_sq']:
                if state_key in model.optimizer.state[p]:
                    state_tensor = model.optimizer.state[p][state_key]
                    assert state_tensor.device == device, f"Optimizer state '{state_key}' is on {state_tensor.device}, expected {device}"

def validate_grad_device(model, device: torch.device) -> None:
    """Validates that the sharded gradients are on the specified device."""
    for p in model.optimizer.single_partition_of_fp32_groups:
        if p.grad is not None:
            assert p.grad.device == device, f"Gradient partition is on {p.grad.device}, expected {device}"


def run_model_zero2(model, param_groups, config_dict, hidden_dim, dtype, offloaded_states, pin_memory, non_blocking):
    """
    This function runs a training step, offloads states, reloads them, and verifies correctness for ZeRO-2.
    """
    offload_device = OffloadDeviceEnum.cpu
    offload_torch_device = torch.device(offload_device.value)
    accelerator_device = torch.device(get_accelerator().current_device_name())

    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=param_groups, config=config_dict)

    data_loader = random_dataloader(model=model,
                                      total_samples=10,
                                      hidden_dim=hidden_dim,
                                      device=model.device,
                                      dtype=dtype)
    dist.barrier()
    for batch in data_loader:
        loss = model(batch[0], batch[1])
        model.backward(loss)
        model.step()

        # --- Save state snapshots before offloading ---
        lp_params_expected = [p.clone().detach() for p in model.parameters()]
        fp32_param_tensors = model.optimizer.single_partition_of_fp32_groups
        fp32_params_expected = [p.clone().detach() for p in fp32_param_tensors]
        lp_grads_expected = [p.grad.clone().detach() if p.grad is not None else None for p in fp32_param_tensors]
        adam_exp_avg_expected = [model.optimizer.state[p]['exp_avg'].clone().detach() for p in fp32_param_tensors if p in model.optimizer.state]
        adam_exp_avg_sq_expected = [model.optimizer.state[p]['exp_avg_sq'].clone().detach() for p in fp32_param_tensors if p in model.optimizer.state]


        # --- Start offloading ---
        model.offload_states(include=offloaded_states,
                               device=offload_device,
                               pin_memory=pin_memory,
                               non_blocking=non_blocking)

        # --- Validate that states were moved to CPU ---
        if offloaded_states is None or OffloadStateTypeEnum.lp_params in offloaded_states:
            validate_lp_params_device(model, offload_torch_device)
        if offloaded_states is None or OffloadStateTypeEnum.hp_params in offloaded_states:
            validate_hp_params_device(model, offload_torch_device)
        if offloaded_states is None or OffloadStateTypeEnum.optim_states in offloaded_states:
            validate_adam_states_device(model, offload_torch_device)
        if offloaded_states is None or OffloadStateTypeEnum.lp_grads in offloaded_states:
            validate_grad_device(model, offload_torch_device)

        # --- Reload states back to GPU ---
        model.reload_states()

        # --- Verify restored states ---
        validate_lp_params_device(model, accelerator_device)
        validate_hp_params_device(model, accelerator_device)
        validate_adam_states_device(model, accelerator_device)
        validate_grad_device(model, accelerator_device)

        # NVerify data integrity for lp_params.
        for expected, restored in zip(lp_params_expected, model.parameters()):
            assert torch.equal(expected, restored)

        reloaded_fp32_param_tensors = model.optimizer.single_partition_of_fp32_groups
        for expected, restored in zip(fp32_params_expected, reloaded_fp32_param_tensors):
            assert torch.equal(expected, restored)
        for expected_grad, p in zip(lp_grads_expected, reloaded_fp32_param_tensors):
            if expected_grad is not None:
                assert torch.equal(expected_grad, p.grad)
            else:
                assert p.grad is None
        # Ensure the parameter exists in the state dict before iterating
        adam_params_in_state = [p for p in reloaded_fp32_param_tensors if p in model.optimizer.state]
        for expected, p in zip(adam_exp_avg_expected, adam_params_in_state):
            assert torch.equal(expected, model.optimizer.state[p]['exp_avg'])
        for expected, p in zip(adam_exp_avg_sq_expected, adam_params_in_state):
            assert torch.equal(expected, model.optimizer.state[p]['exp_avg_sq'])


@pytest.mark.parametrize("included_state", [
    OffloadStateTypeEnum.optim_states,
    OffloadStateTypeEnum.lp_grads,
    OffloadStateTypeEnum.hp_params,
    OffloadStateTypeEnum.lp_params,
    None
])
@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("non_blocking", [False, True])
@pytest.mark.parametrize("zero_stage", [1, 2])
class TestOffloadStatesZero2(DistributedTest):
    world_size = 2

    def test_offload_states_zero2(self, included_state, pin_memory, non_blocking):
        hidden_dim = 1024
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {"type": "Adam", "params": {"lr": 1e-6}},
            "zero_optimization": {"stage": zero_stage},
            "bf16": {"enabled": True}
        }
        model = SimpleModel(hidden_dim, nlayers=4)
        param_groups = [{
            "params": [p for n, p in model.named_parameters() if 'bias' not in n], "weight_decay": 0.1
        }, {
            "params": [p for n, p in model.named_parameters() if 'bias' in n], "weight_decay": 0.0
        }]
        offloaded_states = None if included_state is None else [included_state]
        run_model_zero2(model, param_groups, config_dict, hidden_dim, torch.bfloat16, offloaded_states, pin_memory,
                        non_blocking)

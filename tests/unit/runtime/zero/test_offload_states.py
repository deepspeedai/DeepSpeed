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
from deepspeed.utils import safe_get_local_fp32_param, safe_get_local_optimizer_state
from deepspeed.runtime.zero.offload_states import get_state_devices

# ==============================================================================
# ZeRO-1 and ZeRO-2 TESTS
# ==============================================================================


def validate_hp_params_device(model, device: torch.device):
    """Validates that the sharded FP32 parameters are on the specified device."""
    for p in model.optimizer.single_partition_of_fp32_groups:
        assert p.device.type == device.type, f"FP32 param partition is on {p.device}, expected {device}"


def validate_lp_params_device(model, device: torch.device):
    """Validates that the sharded LP parameters are on the specified device."""
    for p in model.parameters():
        assert p.device.type == device.type, f"LP param partition is on {p.device}, expected {device}"


def validate_adam_states_device(model, device: torch.device):
    """Validates that the sharded Adam optimizer states are on the specified device."""
    for p in model.optimizer.single_partition_of_fp32_groups:
        if p in model.optimizer.state:
            for state_key in ['exp_avg', 'exp_avg_sq']:
                if state_key in model.optimizer.state[p]:
                    state_tensor = model.optimizer.state[p][state_key]
                    assert state_tensor.device.type == device.type, f"Optimizer state '{state_key}' is on {state_tensor.device}, expected {device}"


def validate_grad_device(model, device: torch.device) -> None:
    """Validates that the sharded gradients are on the specified device."""
    # This path is for before step() where gradients are in averaged_gradients
    if model.optimizer.averaged_gradients:
        for grad_list in model.optimizer.averaged_gradients.values():
            if grad_list is not None:
                for grad_tensor in grad_list:
                    assert grad_tensor.device.type == device.type, f"Gradient partition in averaged_gradients is on {grad_tensor.device}, expected {device}"
    else:
        # This path is for after step() or if grads are not in averaged_gradients
        for p in model.optimizer.single_partition_of_fp32_groups:
            if p.grad is not None:
                assert p.grad.device.type == device.type, f"Gradient partition on hp_param.grad is on {p.grad.device}, expected {device}"


def run_model_zero12(model, param_groups, config_dict, hidden_dim, dtype, offloaded_states, pin_memory, non_blocking):
    """
    This function runs a training step, offloads states, reloads them, and verifies correctness for ZeRO-1/2.
    The logic is carefully structured to handle transient gradient states vs. persistent parameter/optimizer states.
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

    # We only need one step to verify the logic
    batch = next(iter(data_loader))

    loss = model(batch[0], batch[1])
    model.backward(loss)

    # Determine if we are testing a transient state (gradients) or a persistent state
    # REVERTED: Condition now only checks for lp_grads as it's the relevant transient state.
    is_grad_test = offloaded_states is not None and OffloadStateTypeEnum.lp_grads in offloaded_states

    if is_grad_test:
        # --- TEST PATH FOR TRANSIENT GRADIENT STATE ---
        # Gradients exist only between backward() and step(). We must test them here.
        grads_expected = [[g.clone().detach() for g in grad_list]
                          for grad_list in model.optimizer.averaged_gradients.values() if grad_list is not None]

        alloc_before_offload = get_accelerator().memory_allocated()
        model.offload_states(include=offloaded_states,
                             device=offload_device,
                             pin_memory=pin_memory,
                             non_blocking=non_blocking)
        alloc_after_offload = get_accelerator().memory_allocated()
        assert alloc_after_offload < alloc_before_offload, "FAIL: Allocated memory for grads should decrease after offload"
        validate_grad_device(model, offload_torch_device)

        model.reload_states()
        alloc_after_reload = get_accelerator().memory_allocated()

        assert alloc_after_reload > alloc_after_offload, "FAIL: Allocated memory for grads should increase after reload"
        validate_grad_device(model, accelerator_device)

        reloaded_grads = [
            grad_list for grad_list in model.optimizer.averaged_gradients.values() if grad_list is not None
        ]
        assert len(grads_expected) == len(reloaded_grads), "FAIL: Number of gradient groups changed after reload"
        for expected_list, reloaded_list in zip(grads_expected, reloaded_grads):
            for expected_g, reloaded_g in zip(expected_list, reloaded_list):
                assert torch.equal(expected_g, reloaded_g), "FAIL: Reloaded gradient data does not match original"

    model.step()

    if not is_grad_test:
        # --- TEST PATH FOR PERSISTENT STATES (Params, Optimizer States) ---
        # These states exist after step(), so we can test them here.

        # --- Save state snapshots before offloading for data integrity check ---
        lp_params_expected = [p.clone().detach() for p in model.parameters()]
        hp_params_expected = [p.clone().detach() for p in model.optimizer.single_partition_of_fp32_groups]

        adam_params_in_state_before = [
            p for p in model.optimizer.single_partition_of_fp32_groups if p in model.optimizer.state
        ]
        adam_exp_avg_expected = [
            model.optimizer.state[p]['exp_avg'].clone().detach() for p in adam_params_in_state_before
        ]
        adam_exp_avg_sq_expected = [
            model.optimizer.state[p]['exp_avg_sq'].clone().detach() for p in adam_params_in_state_before
        ]

        alloc_before_offload = get_accelerator().memory_allocated()
        model.offload_states(include=offloaded_states,
                             device=offload_device,
                             pin_memory=pin_memory,
                             non_blocking=non_blocking)
        alloc_after_offload = get_accelerator().memory_allocated()
        assert alloc_after_offload < alloc_before_offload, f"FAIL: Allocated memory for persistent state {offloaded_states} should decrease after offload"

        if offloaded_states is None or OffloadStateTypeEnum.lp_params in offloaded_states:
            validate_lp_params_device(model, offload_torch_device)
        if offloaded_states is None or OffloadStateTypeEnum.hp_params in offloaded_states:
            validate_hp_params_device(model, offload_torch_device)
        if offloaded_states is None or OffloadStateTypeEnum.optim_states in offloaded_states:
            validate_adam_states_device(model, offload_torch_device)

        model.reload_states()
        alloc_after_reload = get_accelerator().memory_allocated()
        assert alloc_after_reload > alloc_after_offload, f"FAIL: Allocated memory for persistent state {offloaded_states} should increase after reload"

        # --- Verify restored data integrity ---
        for expected, restored in zip(lp_params_expected, model.parameters()):
            assert torch.equal(expected, restored), "FAIL: Reloaded LP param data does not match original"

        for expected, restored in zip(hp_params_expected, model.optimizer.single_partition_of_fp32_groups):
            assert torch.equal(expected, restored), "FAIL: Reloaded HP param data does not match original"

        adam_params_in_state_after = [
            p for p in model.optimizer.single_partition_of_fp32_groups if p in model.optimizer.state
        ]
        assert len(adam_params_in_state_before) == len(
            adam_params_in_state_after), "FAIL: Number of params in optimizer state changed after reload"

        for expected, p in zip(adam_exp_avg_expected, adam_params_in_state_after):
            assert torch.equal(
                expected, model.optimizer.state[p]['exp_avg']), "FAIL: Reloaded 'exp_avg' data does not match original"
        for expected, p in zip(adam_exp_avg_sq_expected, adam_params_in_state_after):
            assert torch.equal(
                expected,
                model.optimizer.state[p]['exp_avg_sq']), "FAIL: Reloaded 'exp_avg_sq' data does not match original"

    # --- FINAL VALIDATION FOR ALL TESTS ---
    validate_lp_params_device(model, accelerator_device)
    validate_hp_params_device(model, accelerator_device)
    validate_adam_states_device(model, accelerator_device)

    assert torch.any(torch.ne(list(model.parameters())[0], 0.0))


@pytest.mark.parametrize("included_state", [
    OffloadStateTypeEnum.optim_states, OffloadStateTypeEnum.lp_grads, OffloadStateTypeEnum.hp_params,
    OffloadStateTypeEnum.lp_params, None
])
@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("non_blocking", [False, True])
@pytest.mark.parametrize("zero_stage", [1, 2])
class TestOffloadStatesZero12(DistributedTest):
    world_size = 2

    def test_offload_states_zero12(self, included_state, pin_memory, non_blocking, zero_stage):
        hidden_dim = 1024
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "zero_optimization": {
                "stage": zero_stage
            },
            "bf16": {
                "enabled": True
            }
        }
        model = SimpleModel(hidden_dim, nlayers=4)
        param_groups = [{
            "params": [p for n, p in model.named_parameters() if 'bias' not in n],
            "weight_decay": 0.1
        }, {
            "params": [p for n, p in model.named_parameters() if 'bias' in n],
            "weight_decay": 0.0
        }]
        offloaded_states = None if included_state is None else [included_state]
        run_model_zero12(model, param_groups, config_dict, hidden_dim, torch.bfloat16, offloaded_states, pin_memory,
                         non_blocking)


# ==============================================================================
# ZeRO-3 TESTS
# ==============================================================================


def validate_device(model, device: torch.device, offloaded_states) -> None:

    def compare_device(state) -> bool:
        devices = get_state_devices(model, state)
        return len(devices) == 1 and device in devices

    for state in OffloadStateTypeEnum:
        if offloaded_states is None or state in offloaded_states:
            if state == OffloadStateTypeEnum.contiguous_grad_buffer and device == torch.device("cpu"):
                assert len(get_state_devices(model,
                                             state)) == 0, f"State {state} must be removed after offload_states()"
            else:
                assert compare_device(state), f"State {state} is not on device {device}"


def run_model_zero3(model, param_groups, config_dict, hidden_dim, dtype, offloaded_states, pin_memory, non_blocking):
    # Currently we only support OffloadDeviceEnum.cpu
    offload_device = OffloadDeviceEnum.cpu

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

        hp_params_expected = [safe_get_local_fp32_param(p).clone() for p in model.parameters()]
        lp_params_expected = [p.ds_tensor.clone() for p in model.parameters()]
        lp_grads_expected = model.optimizer.grad_partitions_flat_buffer.clone()
        adam_exp_avg_expected = [safe_get_local_optimizer_state(p, "exp_avg").clone() for p in model.parameters()]
        adam_exp_avg_sq = [safe_get_local_optimizer_state(p, "exp_avg_sq").clone() for p in model.parameters()]

        # Start offloading
        alloc_before_offload = get_accelerator().memory_allocated()
        model.offload_states(include=offloaded_states,
                             device=offload_device,
                             pin_memory=pin_memory,
                             non_blocking=non_blocking)
        alloc_after_offload = get_accelerator().memory_allocated()
        assert alloc_after_offload < alloc_before_offload, "Allocated memory should decrease after offload"

        validate_device(model, torch.device(offload_device.value), offloaded_states)

        # Reload states
        model.reload_states()
        assert alloc_after_offload < get_accelerator().memory_allocated(
        ), "Allocated memory should increase after offload back"

        # Verify restored states
        hp_param_restored = [safe_get_local_fp32_param(p) for p in model.parameters()]
        for hp_param_expected, hp_param_restored in zip(hp_params_expected, hp_param_restored):
            assert torch.equal(hp_param_expected, hp_param_restored)

        lp_param_restored = [p.ds_tensor for p in model.parameters()]

        for lp_param_expected, lp_param_restored in zip(lp_params_expected, lp_param_restored):
            assert torch.equal(lp_param_expected, lp_param_restored)

        assert torch.equal(lp_grads_expected, model.optimizer.grad_partitions_flat_buffer)

        adam_exp_avg_restored = [safe_get_local_optimizer_state(p, "exp_avg") for p in model.parameters()]
        for adam_exp_avg_expected, adam_exp_avg_restored in zip(adam_exp_avg_expected, adam_exp_avg_restored):
            assert torch.equal(adam_exp_avg_expected, adam_exp_avg_restored)

        adam_exp_avg_sq_restored = [safe_get_local_optimizer_state(p, "exp_avg_sq") for p in model.parameters()]
        for adam_exp_avg_sq_expected, adam_exp_avg_sq_restored in zip(adam_exp_avg_sq, adam_exp_avg_sq_restored):
            assert torch.equal(adam_exp_avg_sq_expected, adam_exp_avg_sq_restored)

        validate_device(model, torch.device(get_accelerator().current_device_name()), offloaded_states)

    # Needed in ZeRO 3. Not doing so can give memory leak
    model.destroy()


@pytest.mark.parametrize("included_state", [
    OffloadStateTypeEnum.hp_params, OffloadStateTypeEnum.lp_params, OffloadStateTypeEnum.optim_states,
    OffloadStateTypeEnum.lp_grads, OffloadStateTypeEnum.contiguous_grad_buffer, None
])
@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("non_blocking", [False, True])
class TestOffloadStatesZero3(DistributedTest):
    # Need multiple gpus to test possible hanging
    world_size = 2

    def test_offload_states_zero3(self, included_state, pin_memory, non_blocking):
        hidden_dim = 1024

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "zero_optimization": {
                "stage": 3,
            }
        }
        config_dict["bf16"] = {"enabled": True}

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            model = SimpleModel(hidden_dim, nlayers=4)

        param_groups = [{
            "params": [p for n, p in model.named_parameters() if not 'bias' in n],
            "weight_decay": 0.1
        }, {
            "params": [p for n, p in model.named_parameters() if 'bias' in n],
            "weight_decay": 0.0
        }]
        offloaded_states = None if included_state is None else [included_state]
        run_model_zero3(model, param_groups, config_dict, hidden_dim, torch.bfloat16, offloaded_states, pin_memory,
                        non_blocking)

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


def is_offload_optimizer_enabled(config_dict):
    return config_dict.get("zero_optimization", {}).get("offload_optimizer", {}).get("device", None) is not None


def is_only_offload_optimizer_states(offloaded_states, optimizer_offload_states):
    if offloaded_states is None:
        return False
    offload_set = set(offloaded_states)
    optim_states_set = set(optimizer_offload_states)
    return offload_set - optim_states_set == set()


def run_model_zero12(model, param_groups, config_dict, hidden_dim, dtype, offloaded_states, pin_memory, non_blocking):
    """
    This function runs a training step, offloads states, reloads them, and verifies correctness for ZeRO-1/2.
    The logic is carefully structured to handle transient gradient states vs. persistent parameter/optimizer states.
    """
    offload_device = OffloadDeviceEnum.cpu
    offload_torch_device = torch.device(offload_device.value)
    accelerator_device = torch.device(get_accelerator().current_device_name())
    optimizer_device = offload_torch_device if is_offload_optimizer_enabled(config_dict) else accelerator_device
    offload_only_optimizer_states = is_only_offload_optimizer_states(
        offloaded_states, [OffloadStateTypeEnum.optim_states, OffloadStateTypeEnum.hp_params])
    expect_memory_change = not (is_offload_optimizer_enabled(config_dict) and offload_only_optimizer_states)

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
        grad_numel = sum(sum(g.numel() for g in grad_list) for grad_list in grads_expected)

        alloc_before_offload = get_accelerator().memory_allocated()
        model.offload_states(include=offloaded_states,
                             device=offload_device,
                             pin_memory=pin_memory,
                             non_blocking=non_blocking)
        alloc_after_offload = get_accelerator().memory_allocated()

        if grad_numel > 0:
            assert alloc_after_offload < alloc_before_offload, f"FAIL: Allocated memory for grads should decrease after offload {alloc_after_offload=} < {alloc_before_offload=}"
            validate_grad_device(model, offload_torch_device)

        model.reload_states()
        alloc_after_reload = get_accelerator().memory_allocated()

        if grad_numel > 0:
            assert alloc_after_reload > alloc_after_offload, f"FAIL: Allocated memory for grads should increase after reload {alloc_after_reload=} > {alloc_after_offload=}"
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

        if expect_memory_change:
            assert alloc_after_offload < alloc_before_offload, f"FAIL: Allocated memory for persistent state {offloaded_states} should decrease after offload"

        if offloaded_states is None or OffloadStateTypeEnum.lp_params in offloaded_states:
            validate_lp_params_device(model, offload_torch_device)
        if offloaded_states is None or OffloadStateTypeEnum.hp_params in offloaded_states:
            validate_hp_params_device(model, offload_torch_device)
        if offloaded_states is None or OffloadStateTypeEnum.optim_states in offloaded_states:
            validate_adam_states_device(model, offload_torch_device)

        model.reload_states()
        alloc_after_reload = get_accelerator().memory_allocated()

        if expect_memory_change:
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
    validate_hp_params_device(model, optimizer_device)
    validate_adam_states_device(model, optimizer_device)

    assert torch.any(torch.ne(list(model.parameters())[0], 0.0))


@pytest.mark.parametrize("included_state", [
    OffloadStateTypeEnum.optim_states, OffloadStateTypeEnum.lp_grads, OffloadStateTypeEnum.hp_params,
    OffloadStateTypeEnum.lp_params, None
])
@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("non_blocking", [False, True])
@pytest.mark.parametrize("zero_stage", [1, 2])
@pytest.mark.parametrize("static_offload_optimizer", [False, True])
class TestDynamicOffloadStatesZero12(DistributedTest):
    world_size = 2

    def test_dynamic_offload_states_zero12(self, included_state, pin_memory, non_blocking, zero_stage,
                                           static_offload_optimizer):
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
        if static_offload_optimizer:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}
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


def validate_device(model, state_device: dict[OffloadStateTypeEnum, torch.device], offloaded_states) -> None:

    def compare_device(state) -> bool:
        devices = get_state_devices(model, state)
        return len(devices) == 1 and state_device[state] in devices

    for state in OffloadStateTypeEnum:
        if offloaded_states is None or state in offloaded_states:
            if state == OffloadStateTypeEnum.contiguous_grad_buffer and state_device[state] == torch.device("cpu"):
                assert len(get_state_devices(model,
                                             state)) == 0, f"State {state} must be removed after offload_states()"
            else:
                assert compare_device(state), f"State {state} is not on device {state_device[state]}"


def run_model_zero3(model, param_groups, config_dict, hidden_dim, dtype, offloaded_states, pin_memory, non_blocking):
    # Currently we only support OffloadDeviceEnum.cpu
    offload_device = OffloadDeviceEnum.cpu
    offload_torch_device = torch.device(offload_device.value)
    accelerator_device = torch.device(get_accelerator().current_device_name())
    optimizer_device = offload_torch_device if is_offload_optimizer_enabled(config_dict) else accelerator_device
    offload_only_optimizer_states = is_only_offload_optimizer_states(
        offloaded_states,
        [OffloadStateTypeEnum.optim_states, OffloadStateTypeEnum.hp_params, OffloadStateTypeEnum.lp_grads])
    expect_memory_change = not (is_offload_optimizer_enabled(config_dict) and offload_only_optimizer_states)

    offload_state_device: dict[OffloadStateTypeEnum, torch.device] = {
        OffloadStateTypeEnum.hp_params: offload_torch_device,
        OffloadStateTypeEnum.lp_params: offload_torch_device,
        OffloadStateTypeEnum.optim_states: offload_torch_device,
        OffloadStateTypeEnum.lp_grads: offload_torch_device,
        OffloadStateTypeEnum.contiguous_grad_buffer: offload_torch_device,
    }

    reload_state_device: dict[OffloadStateTypeEnum, torch.device] = {
        OffloadStateTypeEnum.hp_params: optimizer_device,
        OffloadStateTypeEnum.lp_params: accelerator_device,
        OffloadStateTypeEnum.optim_states: optimizer_device,
        OffloadStateTypeEnum.lp_grads: optimizer_device,
        OffloadStateTypeEnum.contiguous_grad_buffer: accelerator_device,
    }

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

        if expect_memory_change:
            assert alloc_after_offload < alloc_before_offload, f"FAIL: Allocated memory should decrease after offload {alloc_after_offload=} < {alloc_before_offload=}"
            validate_device(model, offload_state_device, offloaded_states)

        # Reload states
        model.reload_states()
        alloc_after_reload = get_accelerator().memory_allocated()

        if expect_memory_change:
            assert alloc_after_reload > alloc_after_offload, f"FAIL: Allocated memory should increase after offload back {alloc_after_reload=} > {alloc_after_offload=}"

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

        validate_device(model, reload_state_device, offloaded_states)

    # Needed in ZeRO 3. Not doing so can give memory leak
    model.destroy()


@pytest.mark.parametrize("included_state", [
    OffloadStateTypeEnum.hp_params, OffloadStateTypeEnum.lp_params, OffloadStateTypeEnum.optim_states,
    OffloadStateTypeEnum.lp_grads, OffloadStateTypeEnum.contiguous_grad_buffer, None
])
@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("non_blocking", [False, True])
@pytest.mark.parametrize("static_offload_optimizer", [False, True])
class TestDynamicOffloadStatesZero3(DistributedTest):
    # Need multiple gpus to test possible hanging
    world_size = 2

    def test_dynamic_offload_states_zero3(self, included_state, pin_memory, non_blocking, static_offload_optimizer):
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
        if static_offload_optimizer:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}

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


class _InferenceOffloadModel(torch.nn.Module):

    def __init__(self, hidden_dim=1025):
        super().__init__()
        self.first = torch.nn.Linear(hidden_dim, hidden_dim)
        self.middle = torch.nn.Linear(hidden_dim, hidden_dim)
        self.last = torch.nn.Linear(hidden_dim, hidden_dim)
        self.last.weight = self.first.weight

    def forward(self, inputs):
        return self.last(torch.relu(self.middle(torch.relu(self.first(inputs)))))


def _init_inference_offload_model(dtype, persistent_parameters, zero_config=None):
    accelerator = get_accelerator()
    if accelerator.device_name() == "cpu":
        pytest.skip("Dynamic offload needs a separate accelerator and CPU device")
    if dtype == torch.bfloat16 and not accelerator.is_bf16_supported():
        pytest.skip("bf16 is not supported on this accelerator")
    if dtype == torch.float16 and not accelerator.is_fp16_supported():
        pytest.skip("fp16 is not supported on this accelerator")

    torch.manual_seed(1234)
    model = _InferenceOffloadModel().requires_grad_(False).eval()
    model = model.to(device=accelerator.current_device_name(), dtype=dtype)
    inputs = torch.linspace(-1, 1, 2050, device=accelerator.current_device_name(), dtype=dtype).reshape(2, 1025)
    parameter_snapshots = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
    with torch.no_grad():
        reference_output = model(inputs).cpu()

    config = {
        "train_micro_batch_size_per_gpu": 2,
        "zero_optimization": {
            "stage": 3,
            "stage3_param_persistence_threshold": 10_000_000 if persistent_parameters else 0,
        },
    }
    if zero_config is not None:
        config["zero_optimization"].update(zero_config)
    if dtype == torch.bfloat16:
        config["bf16"] = {"enabled": True}
    elif dtype == torch.float16:
        config["fp16"] = {"enabled": True}

    if config["zero_optimization"].get("zero_hpz_partition_size", 1) > 1:
        deepspeed.zero.Init(module=model, config_dict_or_path=config)

    engine, _, _, _ = deepspeed.initialize(model=model, config=config)
    engine.eval()
    return engine, inputs, reference_output, parameter_snapshots


def _assert_inference_offload_parameters(engine, parameter_snapshots):
    # The public gather context also covers a shared parameter used by two different modules.
    with deepspeed.zero.GatheredParameters(list(engine.module.parameters())):
        for name, param in engine.module.named_parameters():
            torch.testing.assert_close(param.detach().cpu(), parameter_snapshots[name], rtol=0, atol=0)


class TestDynamicOffloadStatesZero3Inference(DistributedTest):
    world_size = [1, 2]

    @pytest.mark.parametrize("dtype,pin_memory,non_blocking,persistent_parameters,included_state", [
        pytest.param(torch.bfloat16, False, False, False, None, id="bf16-pageable-sync-partitioned-all"),
        pytest.param(torch.bfloat16,
                     False,
                     True,
                     False,
                     OffloadStateTypeEnum.lp_params,
                     id="bf16-pageable-async-partitioned-lp"),
        pytest.param(
            torch.bfloat16, True, False, False, OffloadStateTypeEnum.lp_params, id="bf16-pinned-sync-partitioned-lp"),
        pytest.param(torch.bfloat16, True, True, False, None, id="bf16-pinned-async-partitioned-all"),
        pytest.param(
            torch.bfloat16, False, False, True, OffloadStateTypeEnum.lp_params, id="bf16-pageable-sync-persistent-lp"),
        pytest.param(torch.bfloat16, False, True, True, None, id="bf16-pageable-async-persistent-all"),
        pytest.param(torch.bfloat16, True, False, True, None, id="bf16-pinned-sync-persistent-all"),
        pytest.param(
            torch.bfloat16, True, True, True, OffloadStateTypeEnum.lp_params, id="bf16-pinned-async-persistent-lp"),
        pytest.param(torch.float32, True, True, True, None, id="fp32-pinned-async-persistent-all"),
        pytest.param(
            torch.float16, True, True, False, OffloadStateTypeEnum.lp_params, id="fp16-pinned-async-partitioned-lp"),
    ])
    def test_inference_offload_reload(self, dtype, pin_memory, non_blocking, persistent_parameters, included_state):
        """Frozen, optimizer-free inference must free accelerator memory and resume without changing the model."""
        engine, inputs, reference_output, parameter_snapshots = _init_inference_offload_model(
            dtype, persistent_parameters)
        accelerator = get_accelerator()
        accelerator_device = torch.device(accelerator.current_device_name())
        include = None if included_state is None else [included_state]
        try:
            with torch.no_grad():
                initial_output = engine(inputs).cpu()
                torch.testing.assert_close(initial_output, reference_output)

                for _ in range(3):
                    accelerator.synchronize()
                    allocated_before = accelerator.memory_allocated()
                    engine.offload_states(include=include,
                                          device=OffloadDeviceEnum.cpu,
                                          pin_memory=pin_memory,
                                          non_blocking=non_blocking)
                    accelerator.synchronize()
                    allocated_offloaded = accelerator.memory_allocated()
                    assert get_state_devices(engine, OffloadStateTypeEnum.lp_params) == {torch.device("cpu")}
                    assert allocated_offloaded < allocated_before, "Inference parameters must release accelerator memory"

                    engine.offload_states(include=include,
                                          device=OffloadDeviceEnum.cpu,
                                          pin_memory=pin_memory,
                                          non_blocking=non_blocking)
                    accelerator.synchronize()
                    assert get_state_devices(engine, OffloadStateTypeEnum.lp_params) == {torch.device("cpu")}
                    assert accelerator.memory_allocated() == allocated_offloaded

                    engine.reload_states(non_blocking=non_blocking)
                    accelerator.synchronize()
                    allocated_reloaded = accelerator.memory_allocated()
                    assert get_state_devices(engine, OffloadStateTypeEnum.lp_params) == {accelerator_device}
                    assert allocated_reloaded > allocated_offloaded, "Reload must restore accelerator parameter storage"

                    engine.reload_states(non_blocking=non_blocking)
                    accelerator.synchronize()
                    assert accelerator.memory_allocated() == allocated_reloaded
                    torch.testing.assert_close(engine(inputs).cpu(), initial_output, rtol=0, atol=0)
                    _assert_inference_offload_parameters(engine, parameter_snapshots)
        finally:
            engine.destroy()

    @pytest.mark.parametrize("included_states", [
        pytest.param([], id="empty"),
        pytest.param([OffloadStateTypeEnum.hp_params], id="hp-params"),
        pytest.param([OffloadStateTypeEnum.optim_states], id="optimizer-states"),
        pytest.param([OffloadStateTypeEnum.lp_grads], id="lp-grads"),
    ])
    def test_unselected_inference_states_are_noop(self, included_states):
        """Absent training states and an empty selection must leave inference parameters available."""
        engine, inputs, reference_output, parameter_snapshots = _init_inference_offload_model(torch.bfloat16, True)
        accelerator = get_accelerator()
        accelerator_device = torch.device(accelerator.current_device_name())
        try:
            with torch.no_grad():
                initial_output = engine(inputs).cpu()
                torch.testing.assert_close(initial_output, reference_output)
                accelerator.synchronize()
                allocated_before = accelerator.memory_allocated()
                # Reloading before the first offload must also leave the inference model untouched.
                engine.reload_states(non_blocking=True)
                accelerator.synchronize()
                assert accelerator.memory_allocated() == allocated_before
                engine.offload_states(include=included_states, pin_memory=True, non_blocking=True)
                accelerator.synchronize()
                assert get_state_devices(engine, OffloadStateTypeEnum.lp_params) == {accelerator_device}
                assert accelerator.memory_allocated() == allocated_before
                engine.reload_states(non_blocking=True)
                accelerator.synchronize()
                assert accelerator.memory_allocated() == allocated_before
                torch.testing.assert_close(engine(inputs).cpu(), initial_output, rtol=0, atol=0)
                _assert_inference_offload_parameters(engine, parameter_snapshots)
        finally:
            engine.destroy()


class TestDynamicInferenceOffloadLifecycle(DistributedTest):
    world_size = 1

    def test_destroy_while_offloaded_preserves_module_weights(self):
        """Destroying the engine must not leave retained module weights pointing to freed native host memory."""
        engine, inputs, reference_output, _ = _init_inference_offload_model(torch.bfloat16, True)
        module = engine.module
        destroyed = False
        try:
            with torch.no_grad():
                torch.testing.assert_close(engine(inputs).cpu(), reference_output)
            engine.offload_states(pin_memory=True, non_blocking=True)
            get_accelerator().synchronize()
            assert get_state_devices(engine, OffloadStateTypeEnum.lp_params) == {torch.device("cpu")}
            # Parameter views are empty while sharded. Their local weight tensors must remain readable
            # after destroy, even when a native pinned allocation previously owned their storage.
            expected_shards = {name: param.ds_tensor.detach().clone() for name, param in module.named_parameters()}
            engine.destroy()
            destroyed = True
            for name, param in module.named_parameters():
                assert param.ds_tensor.device == torch.device("cpu")
                torch.testing.assert_close(param.ds_tensor.detach().clone(), expected_shards[name], rtol=0, atol=0)
        finally:
            if not destroyed:
                engine.destroy()

    def test_static_parameter_offload_rejects_dynamic_offload(self):
        """Static parameter offload remains a separate mode and must reject dynamic offload without mutation."""
        engine, inputs, reference_output, _ = _init_inference_offload_model(
            torch.bfloat16, False, zero_config={"offload_param": {
                "device": "cpu",
                "pin_memory": False
            }})
        try:
            with torch.no_grad():
                initial_output = engine(inputs).cpu()
                torch.testing.assert_close(initial_output, reference_output)
                with pytest.raises(AssertionError, match="offloaded parameters"):
                    engine.offload_states()
                assert get_state_devices(engine, OffloadStateTypeEnum.lp_params) == {torch.device("cpu")}
                torch.testing.assert_close(engine(inputs).cpu(), initial_output, rtol=0, atol=0)
        finally:
            engine.destroy()

    @pytest.mark.world_size(2)
    def test_hpzero_rejects_dynamic_offload(self):
        """An unsupported secondary partition layout must fail before offloading any weights."""
        engine, inputs, reference_output, parameter_snapshots = _init_inference_offload_model(
            torch.bfloat16, False, zero_config={"zero_hpz_partition_size": 2})
        accelerator_device = torch.device(get_accelerator().current_device_name())
        try:
            with torch.no_grad():
                initial_output = engine(inputs).cpu()
                torch.testing.assert_close(initial_output, reference_output)
                with pytest.raises(NotImplementedError, match="hpZeRO"):
                    engine.offload_states()
                assert get_state_devices(engine, OffloadStateTypeEnum.lp_params) == {accelerator_device}
                torch.testing.assert_close(engine(inputs).cpu(), initial_output, rtol=0, atol=0)
                _assert_inference_offload_parameters(engine, parameter_snapshots)
        finally:
            engine.destroy()

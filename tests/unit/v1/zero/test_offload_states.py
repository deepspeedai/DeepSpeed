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

# The strict allocated-memory deltas asserted in this file assume memory_allocated()
# is allocator bookkeeping (cuda); on cpu it reports process RSS, which does not
# shrink when tensors are freed.
accelerator_device_mod = torch.get_device_module(get_accelerator().device_name())
allocator_backed_memory_stats = hasattr(accelerator_device_mod, 'memory_allocated')

# Offloading a state out of the accelerator presumes the offload target is a different
# memory tier; on the cpu accelerator the target is the accelerator itself.
offload_leaves_accelerator = get_accelerator().device_name() != 'cpu'

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
    expect_memory_change = allocator_backed_memory_stats and not (is_offload_optimizer_enabled(config_dict)
                                                                  and offload_only_optimizer_states)

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
            if allocator_backed_memory_stats:
                assert alloc_after_offload < alloc_before_offload, f"FAIL: Allocated memory for grads should decrease after offload {alloc_after_offload=} < {alloc_before_offload=}"
            validate_grad_device(model, offload_torch_device)

        model.reload_states()
        alloc_after_reload = get_accelerator().memory_allocated()

        if grad_numel > 0:
            if allocator_backed_memory_stats:
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
            if (offload_leaves_accelerator and state == OffloadStateTypeEnum.contiguous_grad_buffer
                    and state_device[state] == torch.device("cpu")):
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
    expect_memory_change = allocator_backed_memory_stats and not (is_offload_optimizer_enabled(config_dict)
                                                                  and offload_only_optimizer_states)

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


@pytest.mark.parametrize("precision", ["fp32", "autocast"])
@pytest.mark.parametrize("pin_memory", [False, True])
class TestOffloadStatesZero3AliasedMaster(DistributedTest):
    """A float32 model (with or without torch autocast) under ZeRO-3 without optimizer offload uses each weight
    partition as its own master copy. Offloading and reloading states must keep it one copy: after every round trip
    the weights are unchanged and training continues exactly as without offloading, and offloading lp_params frees the
    weights' device memory."""
    world_size = 2

    def test_round_trips_keep_one_copy(self, precision, pin_memory):
        hidden_dim = 256
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-2
                }
            },
            "zero_optimization": {
                "stage": 3
            },
        }
        if precision == "autocast":
            config_dict["torch_autocast"] = {"enabled": True, "dtype": "bfloat16"}

        def build():
            torch.manual_seed(0)
            model = SimpleModel(hidden_dim, nlayers=2)
            engine, _, _, _ = deepspeed.initialize(model=model,
                                                   model_parameters=model.parameters(),
                                                   config=config_dict)
            return engine

        def train(engine, batch):
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()

        def one_copy(engine):
            optimizer = engine.optimizer
            masters = optimizer.fp32_partitioned_groups_flat
            weights = optimizer.fp16_partitioned_groups_flat
            buffer = optimizer.lp_param_buffer.untyped_storage().data_ptr()
            return (all(master is weight for master, weight in zip(masters, weights))
                    and all(p.ds_tensor.untyped_storage().data_ptr() == buffer for p in engine.module.parameters()))

        engine = build()
        reference = build()
        assert one_copy(engine)
        batches = list(
            random_dataloader(model=engine,
                              total_samples=6,
                              hidden_dim=hidden_dim,
                              device=engine.device,
                              dtype=torch.float32))
        lp_bytes = engine.optimizer.lp_param_buffer.numel() * engine.optimizer.lp_param_buffer.element_size()
        # A weight round trip first (the master holds the same storage, which must not keep it on the device), then
        # a master round trip and a step before the next weight round trip (two copies would part there).
        rounds = [[OffloadStateTypeEnum.lp_params], [OffloadStateTypeEnum.hp_params], [OffloadStateTypeEnum.lp_params],
                  [OffloadStateTypeEnum.hp_params, OffloadStateTypeEnum.lp_params], None]
        for batch, include in zip(batches, rounds):
            train(engine, batch)
            train(reference, batch)
            weights_before = [p.ds_tensor.clone() for p in engine.module.parameters()]
            # offload_states first releases the gathered parameters ZeRO-3 keeps after a step; release them here so
            # that only the offload itself is measured.
            engine.empty_partition_cache()
            alloc_before = get_accelerator().memory_allocated()
            engine.offload_states(include=include, device=OffloadDeviceEnum.cpu, pin_memory=pin_memory)
            freed = alloc_before - get_accelerator().memory_allocated()
            if allocator_backed_memory_stats and offload_leaves_accelerator and (
                    include is None or OffloadStateTypeEnum.lp_params in include):
                assert freed >= lp_bytes, f"offloading lp_params freed {freed} bytes, the weights take {lp_bytes}"
            engine.reload_states()
            for before, param in zip(weights_before, engine.module.parameters()):
                assert torch.equal(before, param.ds_tensor), f"weights changed across offloading {include}"
            assert one_copy(engine), f"master and weights are two copies after offloading {include}"

        train(engine, batches[-1])
        train(reference, batches[-1])
        for param, reference_param in zip(engine.module.parameters(), reference.module.parameters()):
            assert torch.equal(param.ds_tensor, reference_param.ds_tensor)
        engine.destroy()
        reference.destroy()

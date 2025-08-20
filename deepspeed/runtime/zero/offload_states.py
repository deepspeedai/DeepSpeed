# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Set
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.offload_config import OffloadStateTypeEnum


def _make_offload_state_key(key):
    return f"{key}_offload_buffer"


def offload_optimizer_states(optimizer, device, pin_memory=False, non_blocking=False):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                if pin_memory and v.device.type != 'cpu':
                    pinned_buffer = torch.empty_like(v, device='cpu').pin_memory()
                    pinned_buffer.copy_(v, non_blocking=non_blocking)
                    state[k] = pinned_buffer
                else:
                    state[k] = v.to(device, non_blocking=non_blocking)


def reload_optimizer_states(optimizer, device, non_blocking=False):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=non_blocking)


def offload_adam_states(optimizer, device, pin_memory: bool = False, non_blocking: bool = False):
    """Move optimizer states to device. Note that this assumes the state structure of DeepSpeed Adam."""

    def move_key(state, key):
        offload_buf_key = _make_offload_state_key(key)
        if offload_buf_key not in state:
            state[offload_buf_key] = torch.empty_like(state[key], device=device)
            if pin_memory:
                state[offload_buf_key] = get_accelerator().pin_memory(state[offload_buf_key])
        state[offload_buf_key].copy_(state[key], non_blocking=non_blocking)
        state[key].data = state[offload_buf_key]

    for _, state in optimizer.state.items():
        if "exp_avg" in state:
            move_key(state, "exp_avg")
        if "exp_avg_sq" in state:
            move_key(state, "exp_avg_sq")


def reload_adam_states(optimizer, device, non_blocking: bool = False):
    """Move optimizer states to device. Note that this assumes the state structure of DeepSpeed Adam."""

    def move_back_key(state, key):
        state[key].data = state[_make_offload_state_key(key)].to(device, non_blocking=non_blocking)

    for _, state in optimizer.state.items():
        if "exp_avg" in state:
            move_back_key(state, "exp_avg")
        if "exp_avg_sq" in state:
            move_back_key(state, "exp_avg_sq")


def get_state_devices(model, state: OffloadStateTypeEnum) -> Set[torch.device]:
    """Retrieve the devices of the specified state of the model.

    Args:
        model (DeepSpeedEngine): The model whose device allocations are to be checked.
        state (OffloadStateTypeEnum): The specific state for which the devices should be retrieved.

    Returns:
        Set[torch.device]: A set of devices of the specified state.

    """
    if state == OffloadStateTypeEnum.hp_params:
        return set(model.optimizer.get_hp_param_device(p) for p in model.parameters())
    elif state == OffloadStateTypeEnum.lp_params:
        return set(p.ds_tensor.device for p in model.parameters())
    elif state == OffloadStateTypeEnum.lp_grads:
        return {model.optimizer.grad_partitions_flat_buffer.device}
    elif state == OffloadStateTypeEnum.optim_states:
        return set(model.optimizer.get_hp_param_device(p, "exp_avg") for p in model.parameters()) | \
               set(model.optimizer.get_hp_param_device(p, "exp_avg_sq") for p in model.parameters())
    elif state == OffloadStateTypeEnum.contiguous_grad_buffer:
        return set(bucket.buffer.device for bucket in model.optimizer.ipg_buckets.values()
                   if bucket.buffer is not None)

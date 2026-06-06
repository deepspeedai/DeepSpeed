# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

import torch

from deepspeed.accelerator import get_accelerator
from .passes import zero1_compile, zero3_compile
from .backend import make_backend, launch_compile_passes, init_schedule
from .util import get_deepcompile_handle, add_pre_backward_hook

WARMUP = 5


def _empty_grad_buffer(param):
    return torch.empty([0], dtype=param.dtype, device=param.device)


class _FlatPartitionGradBufferGroup(list):

    def __init__(self, grad_buffers, flat_partition, release_fn):
        super().__init__(grad_buffers)
        self.flat_partition = flat_partition
        self._release_fn = release_fn

    def release_grad_buffers(self):
        self._release_fn()


def _build_partition_grad_views(optimizer, group_idx):
    missing = object()
    original_all_grad_tensors = optimizer.all_grad_tensors.get(group_idx, missing)
    optimizer.all_grad_tensors[group_idx] = optimizer.get_all_grad_tensors(optimizer.params_in_partition[group_idx],
                                                                           optimizer.gradient_accumulation_dtype)
    try:
        return optimizer.get_flat_partition(optimizer.params_in_partition[group_idx],
                                            optimizer.first_offset[group_idx],
                                            optimizer.partition_size[group_idx],
                                            dtype=optimizer.gradient_accumulation_dtype,
                                            device=get_accelerator().current_device_name(),
                                            param_group_idx=group_idx,
                                            return_tensor_list=True)
    finally:
        if original_all_grad_tensors is missing:
            optimizer.all_grad_tensors.pop(group_idx, None)
        else:
            optimizer.all_grad_tensors[group_idx] = original_all_grad_tensors


def _build_flat_partition_grad_views(optimizer, group_idx):
    partition_size = int(optimizer.partition_size[group_idx])
    dtype = optimizer.gradient_accumulation_dtype
    device = get_accelerator().current_device_name()
    flat_buffer = torch.zeros(partition_size, dtype=dtype, device=device)

    views = []
    current_size = 0
    for i, tensor in enumerate(optimizer.params_in_partition[group_idx]):
        num_elements = tensor.numel()
        tensor_offset = 0

        if i == 0 and optimizer.first_offset[group_idx] > 0:
            tensor_offset = int(optimizer.first_offset[group_idx])
            num_elements -= tensor_offset

        if num_elements > partition_size - current_size:
            num_elements = partition_size - current_size

        if num_elements <= 0:
            continue

        view = flat_buffer.narrow(0, current_size, int(num_elements))
        if tensor_offset == 0 and num_elements == tensor.numel():
            view = view.view(tensor.shape)
        views.append(view)
        current_size += int(num_elements)

        if current_size >= partition_size:
            break

    if current_size < partition_size:
        views.append(flat_buffer.narrow(0, current_size, partition_size - current_size))

    return flat_buffer, views


def init_z1(engine, backend, compile_config, compile_kwargs, schedule=None, use_z2=False):

    optimizer = engine.optimizer
    optimizer.contiguous_gradients = False  # Avoid creating unnecessary buffer
    for hook in optimizer._grad_acc_hooks:
        hook.remove()
    optimizer._grad_acc_hooks.clear()

    dc = get_deepcompile_handle()
    dc.init(engine.data_parallel_group, compile_config, engine.zero_reduce_bucket_size())

    if use_z2:
        grad_buffer = {}
        for i, group in enumerate(optimizer.bit16_groups):
            grad_buffer[i] = [p.clone().detach() for p in _build_partition_grad_views(optimizer, i)]

            index_in_partition = 0
            first_in_partition = True
            for p in group:
                param_id = optimizer.get_param_id(p)
                p.param_id = param_id
                in_partition = optimizer.is_param_in_current_partition[param_id]

                if in_partition:
                    buf = grad_buffer[i][index_in_partition]
                    offset = optimizer.first_offset[i] if first_in_partition else 0
                    dc.register_param(p.param_id, p.shape, p, buf, int(offset))
                    index_in_partition += 1
                    first_in_partition = False
                else:
                    dc.register_param(p.param_id, p.shape, p, _empty_grad_buffer(p), 0)

        def set_z2_grad_buffer(_is_gradient_accumulation_boundary):
            optimizer.averaged_gradients = copy.copy(grad_buffer)

        add_pre_backward_hook(set_z2_grad_buffer)
    else:
        grad_buffer_metadata = {}

        for i, group in enumerate(optimizer.bit16_groups):
            grad_buffer_metadata[i] = []
            first_in_partition = True
            for p in group:
                param_id = optimizer.get_param_id(p)
                p.param_id = param_id
                in_partition = optimizer.is_param_in_current_partition[param_id]

                if in_partition:
                    offset = optimizer.first_offset[i] if first_in_partition else 0
                    grad_buffer_metadata[i].append((p.param_id, p, int(offset)))
                    dc.register_param(p.param_id, p.shape, p, _empty_grad_buffer(p), 0)
                    first_in_partition = False
                else:
                    dc.register_param(p.param_id, p.shape, p, _empty_grad_buffer(p), 0)

        current_grad_buffers = {}

        def set_z1_grad_buffer(is_gradient_accumulation_boundary):
            nonlocal current_grad_buffers
            if not is_gradient_accumulation_boundary:
                release_grad_buffer()
                current_grad_buffers = {}
                optimizer.averaged_gradients = {}
                return

            current_grad_buffers = {}
            for group_idx in range(len(optimizer.bit16_groups)):
                flat_grad_buffer, group_grad_buffers = _build_flat_partition_grad_views(optimizer, group_idx)
                current_grad_buffers[group_idx] = _FlatPartitionGradBufferGroup(
                    group_grad_buffers, flat_grad_buffer, lambda group_idx=group_idx: release_grad_buffer(group_idx))
                for (param_id, _, offset), grad_buffer in zip(grad_buffer_metadata[group_idx], group_grad_buffers):
                    dc.update_param_grad_buffer(param_id, grad_buffer, offset)
            optimizer.averaged_gradients = current_grad_buffers

        def release_grad_buffer(group_idx=None):
            group_indices = grad_buffer_metadata.keys() if group_idx is None else [group_idx]
            for idx in group_indices:
                for param_id, param, _ in grad_buffer_metadata[idx]:
                    dc.update_param_grad_buffer(param_id, _empty_grad_buffer(param), 0)
                if idx in current_grad_buffers:
                    current_grad_buffers[idx] = None

        add_pre_backward_hook(set_z1_grad_buffer)

    if schedule is None:
        schedule = []
        if use_z2:
            schedule.append((0, [zero1_compile.add_z2_reduce]))
        else:
            schedule.append((0, [zero1_compile.add_z1_reduce]))
    else:
        for opt in schedule:
            # avoid typical misconfiguration
            if zero3_compile.add_z3_gather_release in opt[1]:
                raise ValueError("A pass for ZeRO3 is not specified though ZeRO1 is enabled")

    init_schedule(schedule)

    engine.launch_compile_passes = launch_compile_passes
    return make_backend(backend, compile_config, compile_kwargs=compile_kwargs)

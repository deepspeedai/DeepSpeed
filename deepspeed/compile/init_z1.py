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


def init_z1(engine, backend, compile_config, compile_kwargs, schedule=None, use_z2=False):

    optimizer = engine.optimizer
    optimizer.contiguous_gradients = False  # Avoid creating unnecessary buffer
    for hook in optimizer._grad_acc_hooks:
        hook.remove()
    optimizer._grad_acc_hooks.clear()

    dc = get_deepcompile_handle()
    dc.init(engine.data_parallel_group, compile_config, engine.zero_reduce_bucket_size())

    grad_buffer = {}

    # Save original all_grad_tensors state as we temporarily modify it
    original_all_grad_tensors = optimizer.all_grad_tensors.copy() if hasattr(optimizer, 'all_grad_tensors') else {}

    for i, group in enumerate(optimizer.bit16_groups):
        # Temporarily populate all_grad_tensors for get_flat_partition call
        # This is needed because get_flat_partition accesses all_grad_tensors[param_group_idx][i]
        # but it's empty during initialization
        if i not in optimizer.all_grad_tensors or optimizer.all_grad_tensors[i] is None:
            optimizer.all_grad_tensors[i] = optimizer.get_all_grad_tensors(optimizer.params_in_partition[i],
                                                                           optimizer.gradient_accumulation_dtype)

        grad_buffer[i] = optimizer.get_flat_partition(optimizer.params_in_partition[i],
                                                      optimizer.first_offset[i],
                                                      optimizer.partition_size[i],
                                                      dtype=optimizer.gradient_accumulation_dtype,
                                                      device=get_accelerator().current_device_name(),
                                                      param_group_idx=i,
                                                      return_tensor_list=True)
        grad_buffer[i] = [p.clone().detach() for p in grad_buffer[i]]  # Maybe not necessary

        index_in_partition = 0
        first_in_partition = True
        for p in group:
            param_id = optimizer.get_param_id(p)
            p.param_id = param_id
            in_partition = optimizer.is_param_in_current_partition[param_id]

            if in_partition:
                buf = grad_buffer[i][index_in_partition]
                offset = optimizer.first_offset[i] if first_in_partition else 0
                # print(f"[r{dist.get_rank()}] Registering group {i} param {param_id} in_partition={in_partition} p={p.shape} buf={buf.shape} partition_offset={offset}")
                dc.register_param(p.param_id, p.shape, p, buf, int(offset))
                index_in_partition += 1
                first_in_partition = False
            else:
                # print(f"[r{dist.get_rank()}] Registering group {i} param {param_id} in_partition={in_partition} p={p.shape} buf=None")
                dc.register_param(p.param_id, p.shape, p, torch.empty([0], dtype=p.dtype, device=p.device), 0)

    # Restore original all_grad_tensors state
    optimizer.all_grad_tensors = original_all_grad_tensors

    def set_grad_buffer():
        optimizer.averaged_gradients = copy.copy(grad_buffer)

    add_pre_backward_hook(set_grad_buffer)

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
